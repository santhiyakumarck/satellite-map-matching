#!/usr/bin/env python3

import os
import re
import math
import time
import resource
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as rio_transform

import torch


# ============================================================
# USER INPUTS
# ============================================================

FRAME_FILE = "frame_06898.jpg"
GPS_FILE = "gps.csv"
MAP_FILE = "Satellite_Z21.tif"

# Fixed camera horizontal FOV
CAMERA_FOV_DEG = 90.0

# Camera frame rate
CAMERA_FPS = 30.0

# Ground altitude convention from your dataset
# altitude = -80 m means drone is on ground
GROUND_ALTITUDE = -80.0

# Extra ROI margin
ROI_MARGIN = 1.20

# Maximum image size used by SuperPoint/SuperGlue
# This prevents very large satellite ROIs from consuming
# too much RAM on Raspberry Pi 5.
MAX_INFERENCE_SIZE = 1600

# SuperPoint
MAX_KEYPOINTS = 2048
KEYPOINT_THRESHOLD = 0.005
NMS_RADIUS = 4

# SuperGlue
SUPERGLUE_WEIGHTS = "outdoor"
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.10

# RANSAC
RANSAC_REPROJ_THRESHOLD = 5.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRAME_PATH = BASE_DIR / FRAME_FILE
GPS_PATH = BASE_DIR / GPS_FILE
MAP_PATH = BASE_DIR / MAP_FILE

RESULTS_DIR = BASE_DIR / (
    "results_" + Path(FRAME_FILE).stem.replace("frame_", "")
)

SUPERGLUE_DIR = BASE_DIR / "SuperGluePretrainedNetwork"

SUPERPOINT_WEIGHT = (
    SUPERGLUE_DIR
    / "models"
    / "weights"
    / "superpoint_v1.pth"
)

SUPERGLUE_WEIGHT_FILE = (
    SUPERGLUE_DIR
    / "models"
    / "weights"
    / "superglue_outdoor.pth"
)


# ============================================================
# UTILITY
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def elapsed(start):
    return time.perf_counter() - start


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_peak_ram_mb():
    # Linux ru_maxrss is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2.0) ** 2
    )

    return 2.0 * R * math.asin(math.sqrt(a))


# ============================================================
# FRAME NUMBER
# ============================================================

def get_frame_number(filename):
    match = re.search(r"(\d+)", Path(filename).stem)

    if match is None:
        raise ValueError(
            "Could not determine frame number from filename: "
            + filename
        )

    return int(match.group(1))


# ============================================================
# GPS
# ============================================================

def load_gps():
    if not GPS_PATH.exists():
        raise FileNotFoundError(
            f"GPS file not found:\n{GPS_PATH}"
        )

    df = pd.read_csv(GPS_PATH)

    required = [
        "bag_timestamp",
        "latitude",
        "longitude",
        "altitude",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"GPS CSV does not contain required column: {col}"
            )

    df["bag_timestamp"] = pd.to_numeric(
        df["bag_timestamp"], errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df["altitude"] = pd.to_numeric(
        df["altitude"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "bag_timestamp",
            "latitude",
            "longitude",
            "altitude",
        ]
    ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("GPS CSV contains no valid GPS rows.")

    return df


def find_nearest_gps(df, frame_number):
    # The first GPS timestamp is treated as dataset start.
    gps_start_ns = int(df.iloc[0]["bag_timestamp"])

    frame_time_ns = (
        gps_start_ns
        + int((frame_number / CAMERA_FPS) * 1e9)
    )

    differences = np.abs(
        df["bag_timestamp"].astype(np.int64).values
        - frame_time_ns
    )

    nearest_index = int(np.argmin(differences))

    gps = df.iloc[nearest_index]

    difference_ms = differences[nearest_index] / 1e6

    return gps, frame_time_ns, difference_ms


# ============================================================
# CAMERA / GROUND FOOTPRINT
# ============================================================

def calculate_ground_footprint(
    frame_width,
    frame_height,
    height_agl,
):
    if height_agl <= 0.5:
        raise ValueError(
            f"Calculated AGL is only {height_agl:.3f} m. "
            "Ground footprint cannot be calculated."
        )

    fov_x = math.radians(CAMERA_FOV_DEG)

    aspect = frame_height / float(frame_width)

    fov_y = 2.0 * math.atan(
        aspect * math.tan(fov_x / 2.0)
    )

    ground_width = (
        2.0
        * height_agl
        * math.tan(fov_x / 2.0)
    )

    ground_height = (
        2.0
        * height_agl
        * math.tan(fov_y / 2.0)
    )

    return (
        fov_x,
        fov_y,
        ground_width,
        ground_height,
    )


# ============================================================
# GEO INFORMATION
# ============================================================

def get_map_gsd(src, latitude):
    transform = src.transform

    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

    if src.crs is None:
        raise ValueError(
            "GeoTIFF does not contain a CRS."
        )

    # Geographic CRS: degrees/pixel
    if src.crs.is_geographic:

        lat_rad = math.radians(latitude)

        meters_per_degree_lat = (
            111132.92
            - 559.82 * math.cos(2.0 * lat_rad)
            + 1.175 * math.cos(4.0 * lat_rad)
        )

        meters_per_degree_lon = (
            111412.84 * math.cos(lat_rad)
            - 93.5 * math.cos(3.0 * lat_rad)
        )

        gsd_x = pixel_x * meters_per_degree_lon
        gsd_y = pixel_y * meters_per_degree_lat

    else:
        # Projected CRS
        gsd_x = pixel_x
        gsd_y = pixel_y

    return gsd_x, gsd_y


# ============================================================
# GPS -> MAP PIXEL
# ============================================================

def gps_to_map_pixel(src, latitude, longitude):

    if src.crs.is_geographic:
        x = longitude
        y = latitude

    else:
        x_list, y_list = rio_transform(
            "EPSG:4326",
            src.crs,
            [longitude],
            [latitude],
        )

        x = x_list[0]
        y = y_list[0]

    row, col = rowcol(
        src.transform,
        x,
        y,
    )

    return int(row), int(col)


# ============================================================
# MAP PIXEL -> GPS
# ============================================================

def map_pixel_to_gps(src, row, col):

    x, y = xy(
        src.transform,
        row,
        col,
        offset="center",
    )

    if src.crs.is_geographic:
        longitude = x
        latitude = y

    else:
        lon_list, lat_list = rio_transform(
            src.crs,
            "EPSG:4326",
            [x],
            [y],
        )

        longitude = lon_list[0]
        latitude = lat_list[0]

    return float(latitude), float(longitude)


# ============================================================
# READ SATELLITE ROI
# ============================================================

def extract_satellite_roi(
    src,
    gps_lat,
    gps_lon,
    required_width_m,
    required_height_m,
    gsd_x,
    gsd_y,
):

    gps_row, gps_col = gps_to_map_pixel(
        src,
        gps_lat,
        gps_lon,
    )

    roi_width_px = int(
        math.ceil(required_width_m / gsd_x)
    )

    roi_height_px = int(
        math.ceil(required_height_m / gsd_y)
    )

    roi_width_px = max(32, roi_width_px)
    roi_height_px = max(32, roi_height_px)

    left = gps_col - roi_width_px // 2
    top = gps_row - roi_height_px // 2

    # Clamp ROI to map
    left = max(0, left)
    top = max(0, top)

    if left + roi_width_px > src.width:
        left = max(0, src.width - roi_width_px)

    if top + roi_height_px > src.height:
        top = max(0, src.height - roi_height_px)

    actual_width = min(
        roi_width_px,
        src.width - left,
    )

    actual_height = min(
        roi_height_px,
        src.height - top,
    )

    window = Window(
        left,
        top,
        actual_width,
        actual_height,
    )

    # Read RGB
    if src.count >= 3:
        data = src.read(
            [1, 2, 3],
            window=window,
        )
        image = np.transpose(
            data,
            (1, 2, 0),
        )

    else:
        data = src.read(
            1,
            window=window,
        )

        image = np.stack(
            [data, data, data],
            axis=2,
        )

    image = np.nan_to_num(image)

    if image.dtype != np.uint8:

        min_val = float(image.min())
        max_val = float(image.max())

        if max_val > min_val:
            image = (
                (image - min_val)
                / (max_val - min_val)
                * 255.0
            )

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    roi_info = {
        "left": int(left),
        "top": int(top),
        "width": int(actual_width),
        "height": int(actual_height),
        "gps_row": gps_row,
        "gps_col": gps_col,
    }

    return image, roi_info


# ============================================================
# RESIZE FOR SUPERPOINT / SUPERGLUE
# ============================================================

def resize_for_inference(image):
    h, w = image.shape[:2]

    largest = max(h, w)

    if largest <= MAX_INFERENCE_SIZE:
        return image.copy(), 1.0, 1.0

    scale = (
        MAX_INFERENCE_SIZE
        / float(largest)
    )

    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale, scale


# ============================================================
# TORCH IMAGE
# ============================================================

def image_to_tensor(image, device):

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:
        gray = image

    tensor = torch.from_numpy(
        gray.astype(np.float32) / 255.0
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    return tensor


# ============================================================
# SUPERPOINT + SUPERGLUE
# ============================================================

def run_superpoint_superglue(
    drone_image,
    satellite_image,
):

    import sys

    if str(SUPERGLUE_DIR) not in sys.path:
        sys.path.insert(
            0,
            str(SUPERGLUE_DIR),
        )

    from models.matching import Matching

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Inference device:", device)

    config = {
        "superpoint": {
            "nms_radius": NMS_RADIUS,
            "keypoint_threshold": KEYPOINT_THRESHOLD,
            "max_keypoints": MAX_KEYPOINTS,
        },
        "superglue": {
            "weights": SUPERGLUE_WEIGHTS,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "match_threshold": MATCH_THRESHOLD,
        },
    }

    print("Loading SuperPoint + SuperGlue...")

    load_start = time.perf_counter()

    matching = Matching(config).eval().to(device)

    synchronize()

    load_time = elapsed(load_start)

    print(
        f"Model loading time: {load_time:.3f} sec"
    )

    # --------------------------------------------------------
    # Prepare images
    # --------------------------------------------------------

    drone_small, drone_sx, drone_sy = (
        resize_for_inference(drone_image)
    )

    satellite_small, sat_sx, sat_sy = (
        resize_for_inference(satellite_image)
    )

    drone_tensor = image_to_tensor(
        drone_small,
        device,
    )

    satellite_tensor = image_to_tensor(
        satellite_small,
        device,
    )

    # --------------------------------------------------------
    # SuperPoint
    # --------------------------------------------------------

    synchronize()

    sp_start = time.perf_counter()

    with torch.no_grad():

        pred0 = matching.superpoint(
            {"image": drone_tensor}
        )

        pred1 = matching.superpoint(
            {"image": satellite_tensor}
        )

    synchronize()

    superpoint_time = elapsed(sp_start)

    # --------------------------------------------------------
    # SuperGlue
    # --------------------------------------------------------
# --------------------------------------------------------
# Convert SuperPoint list outputs to tensors
# --------------------------------------------------------

keypoints0_sp = pred0["keypoints"]
keypoints1_sp = pred1["keypoints"]

scores0_sp = pred0["scores"]
scores1_sp = pred1["scores"]

# SuperPoint returns variable-length keypoints as a list.
# SuperGlue expects tensors with batch dimension.

if isinstance(keypoints0_sp, list):
    keypoints0_sp = torch.stack(keypoints0_sp, dim=0)

if isinstance(keypoints1_sp, list):
    keypoints1_sp = torch.stack(keypoints1_sp, dim=0)

if isinstance(scores0_sp, list):
    scores0_sp = torch.stack(scores0_sp, dim=0)

if isinstance(scores1_sp, list):
    scores1_sp = torch.stack(scores1_sp, dim=0)

data = {
    "image0": drone_tensor,
    "image1": satellite_tensor,
    "keypoints0": keypoints0_sp,
    "keypoints1": keypoints1_sp,
    "scores0": scores0_sp,
    "scores1": scores1_sp,
    "descriptors0": pred0["descriptors"],
    "descriptors1": pred1["descriptors"],
}
    synchronize()

    sg_start = time.perf_counter()

    with torch.no_grad():

        pred_matches = matching.superglue(data)

    synchronize()

    superglue_time = elapsed(sg_start)

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

   keypoints0 = (
    keypoints0_sp[0]
    .detach()
    .cpu()
    .numpy()
)

keypoints1 = (
    keypoints1_sp[0]
    .detach()
    .cpu()
    .numpy()
)

scores0 = (
    scores0_sp[0]
    .detach()
    .cpu()
    .numpy()
)

scores1 = (
    scores1_sp[0]
    .detach()
    .cpu()
    .numpy()
)

    matches0 = (
        pred_matches["matches0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    confidence = (
        pred_matches["matching_scores0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    valid = matches0 > -1

    matched_kp0 = keypoints0[valid]
    matched_kp1 = keypoints1[
        matches0[valid]
    ]

    matched_confidence = confidence[valid]

    # --------------------------------------------------------
    # Convert keypoints back to original image coordinates
    # --------------------------------------------------------

    matched_kp0[:, 0] /= drone_sx
    matched_kp0[:, 1] /= drone_sy

    matched_kp1[:, 0] /= sat_sx
    matched_kp1[:, 1] /= sat_sy

    # All keypoints for visualization
    keypoints0_original = keypoints0.copy()
    keypoints1_original = keypoints1.copy()

    keypoints0_original[:, 0] /= drone_sx
    keypoints0_original[:, 1] /= drone_sy

    keypoints1_original[:, 0] /= sat_sx
    keypoints1_original[:, 1] /= sat_sy

    return {
        "keypoints0": keypoints0_original,
        "keypoints1": keypoints1_original,
        "scores0": scores0,
        "scores1": scores1,
        "matched_kp0": matched_kp0,
        "matched_kp1": matched_kp1,
        "confidence": matched_confidence,
        "superpoint_time": superpoint_time,
        "superglue_time": superglue_time,
        "model_load_time": load_time,
        "device": str(device),
    }


# ============================================================
# DRAW SUPERPOINT KEYPOINTS
# ============================================================

def draw_keypoints(image, keypoints):

    output = image.copy()

    for point in keypoints:

        x = int(round(point[0]))
        y = int(round(point[1]))

        if (
            0 <= x < output.shape[1]
            and 0 <= y < output.shape[0]
        ):
            cv2.circle(
                output,
                (x, y),
                2,
                (0, 255, 0),
                -1,
            )

    return output


# ============================================================
# DRAW SUPERGLUE MATCHES
# ============================================================

def draw_matches(
    drone,
    satellite,
    kp0,
    kp1,
    max_draw=100,
):

    h0, w0 = drone.shape[:2]
    h1, w1 = satellite.shape[:2]

    canvas_h = max(h0, h1)
    canvas_w = w0 + w1

    canvas = np.zeros(
        (canvas_h, canvas_w, 3),
        dtype=np.uint8,
    )

    canvas[:h0, :w0] = drone
    canvas[:h1, w0:w0 + w1] = satellite

    count = min(
        len(kp0),
        max_draw,
    )

    for i in range(count):

        p0 = kp0[i]
        p1 = kp1[i]

        x0 = int(round(p0[0]))
        y0 = int(round(p0[1]))

        x1 = int(round(p1[0] + w0))
        y1 = int(round(p1[1]))

        if not (
            0 <= x0 < w0
            and 0 <= y0 < h0
            and 0 <= p1[0] < w1
            and 0 <= p1[1] < h1
        ):
            continue

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            1,
        )

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            (0, 0, 255),
            -1,
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            (0, 0, 255),
            -1,
        )

    return canvas


# ============================================================
# RANSAC HOMOGRAPHY
# ============================================================

def run_ransac(kp0, kp1):

    if len(kp0) < 4:

        return {
            "H": None,
            "mask": None,
            "inliers": 0,
            "outliers": 0,
            "ratio": 0.0,
        }

    H, mask = cv2.findHomography(
        kp0,
        kp1,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_CONFIDENCE,
        maxIters=RANSAC_MAX_ITERS,
    )

    if H is None or mask is None:

        return {
            "H": None,
            "mask": None,
            "inliers": 0,
            "outliers": len(kp0),
            "ratio": 0.0,
        }

    mask = mask.ravel().astype(bool)

    inliers = int(np.sum(mask))
    outliers = int(len(mask) - inliers)

    ratio = (
        inliers / float(len(mask))
        if len(mask) > 0
        else 0.0
    )

    return {
        "H": H,
        "mask": mask,
        "inliers": inliers,
        "outliers": outliers,
        "ratio": ratio,
    }


# ============================================================
# DRAW RANSAC INLIERS
# ============================================================

def draw_ransac_matches(
    drone,
    satellite,
    kp0,
    kp1,
    mask,
):

    if mask is None:
        return draw_matches(
            drone,
            satellite,
            kp0,
            kp1,
        )

    inlier_kp0 = kp0[mask]
    inlier_kp1 = kp1[mask]

    return draw_matches(
        drone,
        satellite,
        inlier_kp0,
        inlier_kp1,
        max_draw=200,
    )


# ============================================================
# ESTIMATE POSITION
# ============================================================

def estimate_position(
    H,
    frame_width,
    frame_height,
    roi_info,
    src,
):

    center = np.array(
        [
            [
                [
                    frame_width / 2.0,
                    frame_height / 2.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    projected = cv2.perspectiveTransform(
        center,
        H,
    )

    roi_x = float(
        projected[0, 0, 0]
    )

    roi_y = float(
        projected[0, 0, 1]
    )

    full_x = (
        roi_info["left"]
        + roi_x
    )

    full_y = (
        roi_info["top"]
        + roi_y
    )

    estimated_lat, estimated_lon = (
        map_pixel_to_gps(
            src,
            full_y,
            full_x,
        )
    )

    return {
        "roi_x": roi_x,
        "roi_y": roi_y,
        "map_x": full_x,
        "map_y": full_y,
        "latitude": estimated_lat,
        "longitude": estimated_lon,
    }


# ============================================================
# FINAL POSITION VISUALIZATION
# ============================================================

def draw_final_position(
    satellite,
    roi_info,
    gps_lat,
    gps_lon,
    estimated_lat,
    estimated_lon,
    src,
):

    output = satellite.copy()

    # Reference GPS pixel inside ROI
    ref_row, ref_col = gps_to_map_pixel(
        src,
        gps_lat,
        gps_lon,
    )

    ref_x = (
        ref_col
        - roi_info["left"]
    )

    ref_y = (
        ref_row
        - roi_info["top"]
    )

    # Estimated pixel
    est_row, est_col = gps_to_map_pixel(
        src,
        estimated_lat,
        estimated_lon,
    )

    est_x = (
        est_col
        - roi_info["left"]
    )

    est_y = (
        est_row
        - roi_info["top"]
    )

    ref_x = int(round(ref_x))
    ref_y = int(round(ref_y))

    est_x = int(round(est_x))
    est_y = int(round(est_y))

    # GPS reference
    if (
        0 <= ref_x < output.shape[1]
        and 0 <= ref_y < output.shape[0]
    ):

        cv2.circle(
            output,
            (ref_x, ref_y),
            10,
            (255, 0, 0),
            3,
        )

        cv2.putText(
            output,
            "GPS",
            (ref_x + 12, ref_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    # Estimated position
    if (
        0 <= est_x < output.shape[1]
        and 0 <= est_y < output.shape[0]
    ):

        cv2.circle(
            output,
            (est_x, est_y),
            10,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            output,
            "ESTIMATED",
            (est_x + 12, est_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Connect reference and estimate
    if (
        0 <= ref_x < output.shape[1]
        and 0 <= ref_y < output.shape[0]
        and 0 <= est_x < output.shape[1]
        and 0 <= est_y < output.shape[0]
    ):

        cv2.line(
            output,
            (ref_x, ref_y),
            (est_x, est_y),
            (255, 255, 0),
            2,
        )

    return output


# ============================================================
# SAVE HOMOGRAPHY
# ============================================================

def save_homography(H):

    np.savetxt(
        RESULTS_DIR / "homography.txt",
        H,
        fmt="%.10f",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_header(
        "OFFLINE SATELLITE MAP MATCHING"
    )

    print("Frame :", FRAME_PATH)
    print("GPS   :", GPS_PATH)
    print("Map   :", MAP_PATH)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        FRAME_PATH,
        GPS_PATH,
        MAP_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    if not SUPERGLUE_DIR.exists():

        raise FileNotFoundError(
            "SuperGluePretrainedNetwork not found:\n"
            + str(SUPERGLUE_DIR)
        )

    if not SUPERPOINT_WEIGHT.exists():

        raise FileNotFoundError(
            "SuperPoint weight not found:\n"
            + str(SUPERPOINT_WEIGHT)
        )

    if not SUPERGLUE_WEIGHT_FILE.exists():

        raise FileNotFoundError(
            "SuperGlue Outdoor weight not found:\n"
            + str(SUPERGLUE_WEIGHT_FILE)
        )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    frame_start = time.perf_counter()

    frame = cv2.imread(
        str(FRAME_PATH),
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        raise RuntimeError(
            "Could not read frame image."
        )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    frame_number = get_frame_number(
        FRAME_FILE
    )

    frame_time = elapsed(frame_start)

    print_header("FRAME INFORMATION")

    print(
        "Frame number :", frame_number
    )

    print(
        "Frame size   : "
        f"{frame_width} x {frame_height}"
    )

    print(
        "Camera FPS   :", CAMERA_FPS
    )

    print(
        "Camera FOV   :", CAMERA_FOV_DEG,
        "degrees"
    )

    cv2.imwrite(
        str(
            RESULTS_DIR
            / "drone_frame.png"
        ),
        frame,
    )

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    gps_start = time.perf_counter()

    gps_df = load_gps()

    gps_row, frame_timestamp_ns, gps_diff_ms = (
        find_nearest_gps(
            gps_df,
            frame_number,
        )
    )

    gps_lat = float(
        gps_row["latitude"]
    )

    gps_lon = float(
        gps_row["longitude"]
    )

    altitude = float(
        gps_row["altitude"]
    )

    # Dataset convention:
    # -80 m = ground
    height_agl = (
        altitude
        - GROUND_ALTITUDE
    )

    gps_time = elapsed(gps_start)

    print_header("GPS INFORMATION")

    print(
        "Frame timestamp :",
        frame_timestamp_ns
    )

    print(
        "Nearest GPS     :",
        gps_row["bag_timestamp"]
    )

    print(
        "GPS time diff   :",
        f"{gps_diff_ms:.3f} ms"
    )

    print(
        "Latitude        :",
        f"{gps_lat:.9f}"
    )

    print(
        "Longitude       :",
        f"{gps_lon:.9f}"
    )

    print(
        "Altitude        :",
        f"{altitude:.3f} m"
    )

    print(
        "Ground altitude :",
        f"{GROUND_ALTITUDE:.3f} m"
    )

    print(
        "Height AGL      :",
        f"{height_agl:.3f} m"
    )

    # --------------------------------------------------------
    # Ground footprint
    # --------------------------------------------------------

    footprint_start = time.perf_counter()

    (
        fov_x,
        fov_y,
        ground_width,
        ground_height,
    ) = calculate_ground_footprint(
        frame_width,
        frame_height,
        height_agl,
    )

    required_width = (
        ground_width
        * ROI_MARGIN
    )

    required_height = (
        ground_height
        * ROI_MARGIN
    )

    footprint_time = elapsed(
        footprint_start
    )

    print_header(
        "CAMERA GROUND FOOTPRINT"
    )

    print(
        "Horizontal FOV :",
        f"{math.degrees(fov_x):.2f} deg"
    )

    print(
        "Vertical FOV   :",
        f"{math.degrees(fov_y):.2f} deg"
    )

    print(
        "Ground width   :",
        f"{ground_width:.2f} m"
    )

    print(
        "Ground height  :",
        f"{ground_height:.2f} m"
    )

    print(
        "ROI margin     :",
        f"{ROI_MARGIN:.2f}x"
    )

    print(
        "Required width :",
        f"{required_width:.2f} m"
    )

    print(
        "Required height:",
        f"{required_height:.2f} m"
    )

    # --------------------------------------------------------
    # Satellite map / ROI
    # --------------------------------------------------------

    roi_start = time.perf_counter()

    with rasterio.open(
        str(MAP_PATH)
    ) as src:

        print_header(
            "SATELLITE MAP INFORMATION"
        )

        print(
            "Map size :",
            src.width,
            "x",
            src.height
        )

        print(
            "Bands    :",
            src.count
        )

        print(
            "CRS      :",
            src.crs
        )

        gsd_x, gsd_y = get_map_gsd(
            src,
            gps_lat,
        )

        print(
            "GSD X    :",
            f"{gsd_x:.4f} m/pixel"
        )

        print(
            "GSD Y    :",
            f"{gsd_y:.4f} m/pixel"
        )

        satellite_roi, roi_info = (
            extract_satellite_roi(
                src,
                gps_lat,
                gps_lon,
                required_width,
                required_height,
                gsd_x,
                gsd_y,
            )
        )

        print_header("ROI INFORMATION")

        print(
            "ROI left   :",
            roi_info["left"]
        )

        print(
            "ROI top    :",
            roi_info["top"]
        )

        print(
            "ROI width  :",
            roi_info["width"],
            "pixels"
        )

        print(
            "ROI height :",
            roi_info["height"],
            "pixels"
        )

        print(
            "GPS pixel row :",
            roi_info["gps_row"]
        )

        print(
            "GPS pixel col :",
            roi_info["gps_col"]
        )

        # ----------------------------------------------------
        # Convert RGB satellite to BGR for OpenCV
        # ----------------------------------------------------

        if satellite_roi.shape[2] == 3:

            satellite_bgr = cv2.cvtColor(
                satellite_roi,
                cv2.COLOR_RGB2BGR,
            )

        else:

            satellite_bgr = satellite_roi

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "satellite_roi.png"
            ),
            satellite_bgr,
        )

        roi_time = elapsed(
            roi_start
        )

        # ----------------------------------------------------
        # SuperPoint / SuperGlue
        # ----------------------------------------------------

        sg_result = (
            run_superpoint_superglue(
                frame,
                satellite_bgr,
            )
        )

        # ----------------------------------------------------
        # Save SuperPoint keypoints
        # ----------------------------------------------------

        drone_kp_img = draw_keypoints(
            frame,
            sg_result["keypoints0"],
        )

        satellite_kp_img = draw_keypoints(
            satellite_bgr,
            sg_result["keypoints1"],
        )

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "superpoint_keypoints_drone.png"
            ),
            drone_kp_img,
        )

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "superpoint_keypoints_satellite.png"
            ),
            satellite_kp_img,
        )

        print_header(
            "SUPERPOINT / SUPERGLUE"
        )

        print(
            "SuperPoint drone keypoints :",
            len(sg_result["keypoints0"])
        )

        print(
            "SuperPoint satellite keypoints :",
            len(sg_result["keypoints1"])
        )

        print(
            "SuperGlue valid matches :",
            len(sg_result["matched_kp0"])
        )

        print(
            "SuperPoint time :",
            f"{sg_result['superpoint_time']:.3f} sec"
        )

        print(
            "SuperGlue time  :",
            f"{sg_result['superglue_time']:.3f} sec"
        )

        # ----------------------------------------------------
        # SuperGlue visualization
        # ----------------------------------------------------

        matches_img = draw_matches(
            frame,
            satellite_bgr,
            sg_result["matched_kp0"],
            sg_result["matched_kp1"],
            max_draw=200,
        )

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "superglue_matches.png"
            ),
            matches_img,
        )

        # ----------------------------------------------------
        # RANSAC
        # ----------------------------------------------------

        ransac_start = time.perf_counter()

        ransac_result = run_ransac(
            sg_result["matched_kp0"],
            sg_result["matched_kp1"],
        )

        ransac_time = elapsed(
            ransac_start
        )

        print_header(
            "RANSAC HOMOGRAPHY"
        )

        print(
            "Total matches :",
            len(sg_result["matched_kp0"])
        )

        print(
            "Inliers       :",
            ransac_result["inliers"]
        )

        print(
            "Outliers      :",
            ransac_result["outliers"]
        )

        print(
            "Inlier ratio  :",
            f"{ransac_result['ratio']:.4f}"
        )

        print(
            "RANSAC time   :",
            f"{ransac_time:.3f} sec"
        )

        H = ransac_result["H"]

        if H is None:

            print()
            print(
                "ERROR: Homography could not be estimated."
            )

            print(
                "Need at least 4 valid matches."
            )

            total_time = elapsed(
                total_start
            )

            print(
                f"\nTotal processing time: "
                f"{total_time:.3f} sec"
            )

            return

        save_homography(H)

        # ----------------------------------------------------
        # RANSAC visualization
        # ----------------------------------------------------

        ransac_img = draw_ransac_matches(
            frame,
            satellite_bgr,
            sg_result["matched_kp0"],
            sg_result["matched_kp1"],
            ransac_result["mask"],
        )

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "ransac_inliers.png"
            ),
            ransac_img,
        )

        # ----------------------------------------------------
        # Position estimation
        # ----------------------------------------------------

        position_start = time.perf_counter()

        position = estimate_position(
            H,
            frame_width,
            frame_height,
            roi_info,
            src,
        )

        estimated_lat = position[
            "latitude"
        ]

        estimated_lon = position[
            "longitude"
        ]

        position_error = haversine_m(
            gps_lat,
            gps_lon,
            estimated_lat,
            estimated_lon,
        )

        position_time = elapsed(
            position_start
        )

        print_header(
            "POSITION ESTIMATION"
        )

        print(
            "Estimated ROI X :",
            f"{position['roi_x']:.2f}"
        )

        print(
            "Estimated ROI Y :",
            f"{position['roi_y']:.2f}"
        )

        print(
            "Estimated map X :",
            f"{position['map_x']:.2f}"
        )

        print(
            "Estimated map Y :",
            f"{position['map_y']:.2f}"
        )

        print(
            "Reference latitude :",
            f"{gps_lat:.9f}"
        )

        print(
            "Reference longitude:",
            f"{gps_lon:.9f}"
        )

        print(
            "Estimated latitude :",
            f"{estimated_lat:.9f}"
        )

        print(
            "Estimated longitude:",
            f"{estimated_lon:.9f}"
        )

        print(
            "Position error      :",
            f"{position_error:.3f} m"
        )

        print(
            "Position time       :",
            f"{position_time:.3f} sec"
        )

        # ----------------------------------------------------
        # Final visualization
        # ----------------------------------------------------

        final_img = draw_final_position(
            satellite_bgr,
            roi_info,
            gps_lat,
            gps_lon,
            estimated_lat,
            estimated_lon,
            src,
        )

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "final_estimated_position.png"
            ),
            final_img,
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    total_time = elapsed(
        total_start
    )

    peak_ram = get_peak_ram_mb()

    matching_time = (
        sg_result["superpoint_time"]
        + sg_result["superglue_time"]
        + ransac_time
        + position_time
    )

    # --------------------------------------------------------
    # Results file
    # --------------------------------------------------------

    results_text = []

    results_text.append(
        "OFFLINE SATELLITE MAP MATCHING RESULTS"
    )

    results_text.append(
        "========================================"
    )

    results_text.append(
        f"Frame file: {FRAME_FILE}"
    )

    results_text.append(
        f"Frame number: {frame_number}"
    )

    results_text.append(
        f"Frame size: {frame_width} x {frame_height}"
    )

    results_text.append(
        f"GPS file: {GPS_FILE}"
    )

    results_text.append(
        f"Map file: {MAP_FILE}"
    )

    results_text.append(
        f"GPS latitude: {gps_lat:.9f}"
    )

    results_text.append(
        f"GPS longitude: {gps_lon:.9f}"
    )

    results_text.append(
        f"Altitude: {altitude:.6f} m"
    )

    results_text.append(
        f"Ground altitude: {GROUND_ALTITUDE:.6f} m"
    )

    results_text.append(
        f"Height AGL: {height_agl:.6f} m"
    )

    results_text.append(
        f"Camera FOV: {CAMERA_FOV_DEG:.2f} deg"
    )

    results_text.append(
        f"Ground footprint width: "
        f"{ground_width:.3f} m"
    )

    results_text.append(
        f"Ground footprint height: "
        f"{ground_height:.3f} m"
    )

    results_text.append(
        f"GSD X: {gsd_x:.6f} m/pixel"
    )

    results_text.append(
        f"GSD Y: {gsd_y:.6f} m/pixel"
    )

    results_text.append(
        f"ROI size: "
        f"{roi_info['width']} x "
        f"{roi_info['height']}"
    )

    results_text.append(
        f"SuperPoint drone keypoints: "
        f"{len(sg_result['keypoints0'])}"
    )

    results_text.append(
        f"SuperPoint satellite keypoints: "
        f"{len(sg_result['keypoints1'])}"
    )

    results_text.append(
        f"SuperGlue matches: "
        f"{len(sg_result['matched_kp0'])}"
    )

    results_text.append(
        f"RANSAC inliers: "
        f"{ransac_result['inliers']}"
    )

    results_text.append(
        f"RANSAC outliers: "
        f"{ransac_result['outliers']}"
    )

    results_text.append(
        f"Inlier ratio: "
        f"{ransac_result['ratio']:.6f}"
    )

    results_text.append(
        f"Estimated latitude: "
        f"{estimated_lat:.9f}"
    )

    results_text.append(
        f"Estimated longitude: "
        f"{estimated_lon:.9f}"
    )

    results_text.append(
        f"Position error: "
        f"{position_error:.3f} m"
    )

    results_text.append("")
    results_text.append(
        "TIMING"
    )
    results_text.append(
        "------"
    )

    results_text.append(
        f"Frame loading: "
        f"{frame_time:.4f} sec"
    )

    results_text.append(
        f"GPS processing: "
        f"{gps_time:.4f} sec"
    )

    results_text.append(
        f"Footprint calculation: "
        f"{footprint_time:.4f} sec"
    )

    results_text.append(
        f"ROI extraction: "
        f"{roi_time:.4f} sec"
    )

    results_text.append(
        f"Model loading: "
        f"{sg_result['model_load_time']:.4f} sec"
    )

    results_text.append(
        f"SuperPoint: "
        f"{sg_result['superpoint_time']:.4f} sec"
    )

    results_text.append(
        f"SuperGlue: "
        f"{sg_result['superglue_time']:.4f} sec"
    )

    results_text.append(
        f"RANSAC: "
        f"{ransac_time:.4f} sec"
    )

    results_text.append(
        f"Position estimation: "
        f"{position_time:.4f} sec"
    )

    results_text.append(
        f"Matching pipeline: "
        f"{matching_time:.4f} sec"
    )

    results_text.append(
        f"TOTAL: "
        f"{total_time:.4f} sec"
    )

    results_text.append(
        f"Peak RAM: "
        f"{peak_ram:.2f} MB"
    )

    results_text.append(
        f"Device: "
        f"{sg_result['device']}"
    )

    with open(
        RESULTS_DIR / "results.txt",
        "w",
    ) as f:

        f.write(
            "\n".join(results_text)
        )

    # --------------------------------------------------------
    # Final terminal summary
    # --------------------------------------------------------

    print_header(
        "FINAL PERFORMANCE SUMMARY"
    )

    print(
        "Frame                  :",
        FRAME_FILE
    )

    print(
        "Height AGL             :",
        f"{height_agl:.2f} m"
    )

    print(
        "Dynamic ROI            :",
        f"{roi_info['width']} x "
        f"{roi_info['height']}"
    )

    print(
        "SuperPoint keypoints   :",
        len(sg_result["keypoints0"]),
        "/",
        len(sg_result["keypoints1"]),
    )

    print(
        "SuperGlue matches      :",
        len(sg_result["matched_kp0"])
    )

    print(
        "RANSAC inliers         :",
        ransac_result["inliers"]
    )

    print(
        "RANSAC inlier ratio    :",
        f"{ransac_result['ratio']:.3f}"
    )

    print(
        "Reference GPS          :",
        f"{gps_lat:.9f}, {gps_lon:.9f}"
    )

    print(
        "Estimated GPS          :",
        f"{estimated_lat:.9f}, "
        f"{estimated_lon:.9f}"
    )

    print(
        "Position error         :",
        f"{position_error:.3f} m"
    )

    print(
        "SuperPoint time        :",
        f"{sg_result['superpoint_time']:.3f} sec"
    )

    print(
        "SuperGlue time         :",
        f"{sg_result['superglue_time']:.3f} sec"
    )

    print(
        "RANSAC time            :",
        f"{ransac_time:.3f} sec"
    )

    print(
        "Total time             :",
        f"{total_time:.3f} sec"
    )

    print(
        "Peak RAM               :",
        f"{peak_ram:.2f} MB"
    )

    print(
        "Device                 :",
        sg_result["device"]
    )

    print()
    print(
        "Results saved to:"
    )
    print(
        RESULTS_DIR
    )

    print()
    print(
        "Done."
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Program interrupted by user."
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(type(e).__name__ + ":", e)
        print()
        raise
