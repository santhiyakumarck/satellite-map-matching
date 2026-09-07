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
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform

import torch


# ============================================================
# ONLY CHANGE THESE 3 VALUES
# ============================================================

FRAME_FILE = "frame_06898.jpg"
GPS_FILE = "gps.csv"
MAP_FILE = "Satellite_Z21.tif"


# ============================================================
# FIXED PROJECT SETTINGS
# ============================================================

CAMERA_FPS = 30.0
CAMERA_FOV_DEG = 90.0

# Your dataset:
# -80 m = ground level
GROUND_ALTITUDE = -80.0

# ROI safety margin
ROI_MARGIN = 1.20

# Pi 5 memory protection
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
RANSAC_THRESHOLD = 5.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRAME_PATH = BASE_DIR / FRAME_FILE
GPS_PATH = BASE_DIR / GPS_FILE
MAP_PATH = BASE_DIR / MAP_FILE

RESULT_DIR = (
    BASE_DIR /
    ("results_" + Path(FRAME_FILE).stem)
)

SUPERGLUE_DIR = BASE_DIR / "SuperGluePretrainedNetwork"

SUPERPOINT_WEIGHT = (
    SUPERGLUE_DIR /
    "models" /
    "weights" /
    "superpoint_v1.pth"
)

SUPERGLUE_WEIGHT = (
    SUPERGLUE_DIR /
    "models" /
    "weights" /
    "superglue_outdoor.pth"
)


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def sync_cuda():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_ram_mb():

    return (
        resource
        .getrusage(resource.RUSAGE_SELF)
        .ru_maxrss
        / 1024.0
    )


def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2 * R *
        math.asin(math.sqrt(a))
    )


# ============================================================
# FRAME NUMBER
# ============================================================

def get_frame_number():

    name = Path(FRAME_FILE).stem

    numbers = re.findall(
        r"\d+",
        name
    )

    if not numbers:

        raise ValueError(
            "Frame number cannot be found from filename."
        )

    return int(numbers[-1])


# ============================================================
# LOAD GPS
# ============================================================

def load_gps():

    if not GPS_PATH.exists():

        raise FileNotFoundError(
            f"GPS file not found:\n{GPS_PATH}"
        )

    df = pd.read_csv(
        GPS_PATH
    )

    required_columns = [
        "bag_timestamp",
        "latitude",
        "longitude",
        "altitude"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing GPS column: {column}"
            )

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if len(df) == 0:

        raise ValueError(
            "No valid GPS data found."
        )

    return df


# ============================================================
# FIND GPS FOR FRAME
# ============================================================

def find_gps_for_frame(
    gps,
    frame_number
):

    # First GPS timestamp is dataset start
    start_timestamp = int(
        gps.iloc[0]["bag_timestamp"]
    )

    frame_timestamp = (
        start_timestamp
        +
        int(
            frame_number
            / CAMERA_FPS
            * 1e9
        )
    )

    gps_timestamps = (
        gps["bag_timestamp"]
        .astype(np.int64)
        .values
    )

    difference = np.abs(
        gps_timestamps
        -
        frame_timestamp
    )

    index = int(
        np.argmin(difference)
    )

    row = gps.iloc[index]

    difference_ms = (
        difference[index]
        / 1e6
    )

    return (
        row,
        frame_timestamp,
        difference_ms
    )


# ============================================================
# CAMERA FOOTPRINT
# ============================================================

def calculate_footprint(
    width,
    height,
    altitude_agl
):

    if altitude_agl <= 0.5:

        raise ValueError(
            f"AGL is only {altitude_agl:.2f} m."
        )

    fov_x = math.radians(
        CAMERA_FOV_DEG
    )

    aspect_ratio = (
        height / float(width)
    )

    fov_y = (
        2 *
        math.atan(
            aspect_ratio
            *
            math.tan(fov_x / 2)
        )
    )

    ground_width = (
        2 *
        altitude_agl *
        math.tan(fov_x / 2)
    )

    ground_height = (
        2 *
        altitude_agl *
        math.tan(fov_y / 2)
    )

    return (
        fov_y,
        ground_width,
        ground_height
    )


# ============================================================
# MAP GSD
# ============================================================

def calculate_gsd(
    src,
    latitude
):

    pixel_x = abs(
        src.transform.a
    )

    pixel_y = abs(
        src.transform.e
    )

    if src.crs.is_geographic:

        lat = math.radians(
            latitude
        )

        meters_lat = (
            111132.92
            -
            559.82 *
            math.cos(2 * lat)
            +
            1.175 *
            math.cos(4 * lat)
        )

        meters_lon = (
            111412.84 *
            math.cos(lat)
            -
            93.5 *
            math.cos(3 * lat)
        )

        gsd_x = (
            pixel_x *
            meters_lon
        )

        gsd_y = (
            pixel_y *
            meters_lat
        )

    else:

        gsd_x = pixel_x
        gsd_y = pixel_y

    return gsd_x, gsd_y


# ============================================================
# GPS -> MAP PIXEL
# ============================================================

def gps_to_pixel(
    src,
    latitude,
    longitude
):

    if src.crs.is_geographic:

        x = longitude
        y = latitude

    else:

        x_list, y_list = rio_transform(
            "EPSG:4326",
            src.crs,
            [longitude],
            [latitude]
        )

        x = x_list[0]
        y = y_list[0]

    row, col = rowcol(
        src.transform,
        x,
        y
    )

    return (
        int(row),
        int(col)
    )


# ============================================================
# MAP PIXEL -> GPS
# ============================================================

def pixel_to_gps(
    src,
    row,
    col
):

    x, y = xy(
        src.transform,
        row,
        col,
        offset="center"
    )

    if src.crs.is_geographic:

        longitude = x
        latitude = y

    else:

        lon, lat = rio_transform(
            src.crs,
            "EPSG:4326",
            [x],
            [y]
        )

        longitude = lon[0]
        latitude = lat[0]

    return (
        float(latitude),
        float(longitude)
    )


# ============================================================
# EXTRACT MAP ROI
# ============================================================

def extract_roi(
    src,
    latitude,
    longitude,
    footprint_width,
    footprint_height,
    gsd_x,
    gsd_y
):

    gps_row, gps_col = gps_to_pixel(
        src,
        latitude,
        longitude
    )

    roi_width = int(
        math.ceil(
            footprint_width
            * ROI_MARGIN
            / gsd_x
        )
    )

    roi_height = int(
        math.ceil(
            footprint_height
            * ROI_MARGIN
            / gsd_y
        )
    )

    roi_width = min(
        roi_width,
        src.width
    )

    roi_height = min(
        roi_height,
        src.height
    )

    left = (
        gps_col -
        roi_width // 2
    )

    top = (
        gps_row -
        roi_height // 2
    )

    left = max(
        0,
        left
    )

    top = max(
        0,
        top
    )

    if left + roi_width > src.width:

        left = (
            src.width -
            roi_width
        )

    if top + roi_height > src.height:

        top = (
            src.height -
            roi_height
        )

    window = Window(
        left,
        top,
        roi_width,
        roi_height
    )

    if src.count >= 3:

        data = src.read(
            [1, 2, 3],
            window=window
        )

        image = np.transpose(
            data,
            (1, 2, 0)
        )

    else:

        band = src.read(
            1,
            window=window
        )

        image = np.stack(
            [band, band, band],
            axis=2
        )

    image = np.nan_to_num(
        image
    )

    if image.dtype != np.uint8:

        minimum = image.min()
        maximum = image.max()

        if maximum > minimum:

            image = (
                (image - minimum)
                /
                (maximum - minimum)
                *
                255
            )

        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    return (
        image,
        {
            "left": int(left),
            "top": int(top),
            "width": int(roi_width),
            "height": int(roi_height),
            "gps_row": gps_row,
            "gps_col": gps_col
        }
    )


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_image(
    image
):

    height, width = image.shape[:2]

    largest = max(
        height,
        width
    )

    if largest <= MAX_INFERENCE_SIZE:

        return (
            image.copy(),
            1.0,
            1.0
        )

    scale = (
        MAX_INFERENCE_SIZE
        /
        float(largest)
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return (
        resized,
        scale,
        scale
    )


# ============================================================
# IMAGE -> TORCH
# ============================================================

def image_tensor(
    image,
    device
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    tensor = torch.from_numpy(
        gray.astype(
            np.float32
        )
        /
        255.0
    )

    tensor = tensor[
        None,
        None,
        ...
    ]

    return tensor.to(
        device
    )


# ============================================================
# SUPERPOINT + SUPERGLUE
# ============================================================

def run_matching(
    drone,
    satellite
):

    import sys

    sys.path.insert(
        0,
        str(SUPERGLUE_DIR)
    )

    from models.matching import Matching

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Inference device:",
        device
    )

    config = {

        "superpoint": {

            "nms_radius":
                NMS_RADIUS,

            "keypoint_threshold":
                KEYPOINT_THRESHOLD,

            "max_keypoints":
                MAX_KEYPOINTS
        },

        "superglue": {

            "weights":
                SUPERGLUE_WEIGHTS,

            "sinkhorn_iterations":
                SINKHORN_ITERATIONS,

            "match_threshold":
                MATCH_THRESHOLD
        }
    }

    print(
        "Loading SuperPoint + SuperGlue..."
    )

    load_start = time.perf_counter()

    matching = Matching(
        config
    ).eval().to(
        device
    )

    sync_cuda()

    load_time = (
        time.perf_counter()
        -
        load_start
    )

    print(
        f"Model loading time: "
        f"{load_time:.3f} sec"
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    drone_small, drone_sx, drone_sy = (
        resize_image(drone)
    )

    satellite_small, sat_sx, sat_sy = (
        resize_image(satellite)
    )

    tensor0 = image_tensor(
        drone_small,
        device
    )

    tensor1 = image_tensor(
        satellite_small,
        device
    )

    # --------------------------------------------------------
    # SuperPoint
    # --------------------------------------------------------

    sync_cuda()

    start = time.perf_counter()

    with torch.no_grad():

        pred0 = matching.superpoint(
            {
                "image":
                    tensor0
            }
        )

        pred1 = matching.superpoint(
            {
                "image":
                    tensor1
            }
        )

    sync_cuda()

    superpoint_time = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Get SuperPoint outputs
    # --------------------------------------------------------

     # --------------------------------------------------------
    # Get SuperPoint outputs
    # --------------------------------------------------------

    kp0 = pred0["keypoints"]
    kp1 = pred1["keypoints"]

    scores0 = pred0["scores"]
    scores1 = pred1["scores"]

    # --------------------------------------------------------
    # Convert SuperPoint outputs to tensors
    #
    # Different versions of SuperPoint may return:
    # tensor / list / tuple
    # --------------------------------------------------------

    def unwrap_output(value):

        # Keep unwrapping list/tuple until the actual tensor
        # is reached.
        while isinstance(value, (list, tuple)):

            if len(value) == 0:
                raise RuntimeError(
                    "SuperPoint returned an empty list/tuple."
                )

            value = value[0]

        return value

    kp0_tensor = unwrap_output(kp0)
    kp1_tensor = unwrap_output(kp1)

    scores0_tensor = unwrap_output(scores0)
    scores1_tensor = unwrap_output(scores1)

    # --------------------------------------------------------
    # Check output types
    # --------------------------------------------------------

    if not isinstance(kp0_tensor, torch.Tensor):
        raise TypeError(
            f"SuperPoint keypoints0 type is "
            f"{type(kp0_tensor)}"
        )

    if not isinstance(kp1_tensor, torch.Tensor):
        raise TypeError(
            f"SuperPoint keypoints1 type is "
            f"{type(kp1_tensor)}"
        )

    if not isinstance(scores0_tensor, torch.Tensor):
        raise TypeError(
            f"SuperPoint scores0 type is "
            f"{type(scores0_tensor)}"
        )

    if not isinstance(scores1_tensor, torch.Tensor):
        raise TypeError(
            f"SuperPoint scores1 type is "
            f"{type(scores1_tensor)}"
        )

    # --------------------------------------------------------
    # Make sure batch dimension exists
    # --------------------------------------------------------

    if kp0_tensor.ndim == 2:
        kp0_tensor = kp0_tensor.unsqueeze(0)

    if kp1_tensor.ndim == 2:
        kp1_tensor = kp1_tensor.unsqueeze(0)

    if scores0_tensor.ndim == 1:
        scores0_tensor = scores0_tensor.unsqueeze(0)

    if scores1_tensor.ndim == 1:
        scores1_tensor = scores1_tensor.unsqueeze(0)
    if isinstance(kp0, list):
        kp0_tensor = kp0[0]
    else:
        kp0_tensor = kp0

    if isinstance(kp1, list):
        kp1_tensor = kp1[0]
    else:
        kp1_tensor = kp1

    if isinstance(scores0, list):
        scores0_tensor = scores0[0]
    else:
        scores0_tensor = scores0

    if isinstance(scores1, list):
        scores1_tensor = scores1[0]
    else:
        scores1_tensor = scores1

    # Make sure batch dimension exists
    if kp0_tensor.ndim == 2:
        kp0_tensor = kp0_tensor.unsqueeze(0)

    if kp1_tensor.ndim == 2:
        kp1_tensor = kp1_tensor.unsqueeze(0)

    if scores0_tensor.ndim == 1:
        scores0_tensor = scores0_tensor.unsqueeze(0)

    if scores1_tensor.ndim == 1:
        scores1_tensor = scores1_tensor.unsqueeze(0)

    # --------------------------------------------------------
    # SuperGlue
    # --------------------------------------------------------

    data = {

        "image0":
            tensor0,

        "image1":
            tensor1,

        "keypoints0":
            kp0_tensor,

        "keypoints1":
            kp1_tensor,

        "scores0":
            scores0_tensor,

        "scores1":
            scores1_tensor,

        "descriptors0":
            pred0["descriptors"],

        "descriptors1":
            pred1["descriptors"]
    }

    sync_cuda()

    start = time.perf_counter()

    with torch.no_grad():

        matches = matching.superglue(
            data
        )

    sync_cuda()

    superglue_time = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # NumPy conversion
    # --------------------------------------------------------

    kp0_np = (
        kp0_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    kp1_np = (
        kp1_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Match indexes
    # --------------------------------------------------------

    matches0 = (
        matches["matches0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    if "matching_scores0" in matches:

        match_scores = (
            matches[
                "matching_scores0"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

    else:

        match_scores = np.ones(
            len(matches0),
            dtype=np.float32
        )

    valid = (
        matches0 > -1
    )

    matched0 = kp0_np[
        valid
    ].copy()

    matched1 = kp1_np[
        matches0[valid]
    ].copy()

    confidence = match_scores[
        valid
    ]

    # --------------------------------------------------------
    # Convert back to original image size
    # --------------------------------------------------------

    if len(matched0) > 0:

        matched0[:, 0] /= drone_sx
        matched0[:, 1] /= drone_sy

        matched1[:, 0] /= sat_sx
        matched1[:, 1] /= sat_sy

    kp0_original = kp0_np.copy()
    kp1_original = kp1_np.copy()

    kp0_original[:, 0] /= drone_sx
    kp0_original[:, 1] /= drone_sy

    kp1_original[:, 0] /= sat_sx
    kp1_original[:, 1] /= sat_sy

    return {

        "device":
            str(device),

        "model_time":
            load_time,

        "superpoint_time":
            superpoint_time,

        "superglue_time":
            superglue_time,

        "keypoints0":
            kp0_original,

        "keypoints1":
            kp1_original,

        "matched0":
            matched0,

        "matched1":
            matched1,

        "confidence":
            confidence
    }


# ============================================================
# DRAW KEYPOINTS
# ============================================================

def draw_keypoints(
    image,
    keypoints
):

    output = image.copy()

    for point in keypoints:

        x = int(point[0])
        y = int(point[1])

        if (
            0 <= x < output.shape[1]
            and
            0 <= y < output.shape[0]
        ):

            cv2.circle(
                output,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    return output


# ============================================================
# DRAW MATCHES
# ============================================================

def draw_matches(
    image0,
    image1,
    points0,
    points1
):

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    height = max(
        h0,
        h1
    )

    canvas = np.zeros(
        (
            height,
            w0 + w1,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :h0,
        :w0
    ] = image0

    canvas[
        :h1,
        w0:w0 + w1
    ] = image1

    count = min(
        len(points0),
        200
    )

    for i in range(count):

        p0 = points0[i]
        p1 = points1[i]

        x0 = int(p0[0])
        y0 = int(p0[1])

        x1 = int(
            p1[0] + w0
        )
        y1 = int(p1[1])

        if (
            0 <= x0 < w0
            and
            0 <= y0 < h0
            and
            0 <= p1[0] < w1
            and
            0 <= p1[1] < h1
        ):

            cv2.line(
                canvas,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                1
            )

            cv2.circle(
                canvas,
                (x0, y0),
                3,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                canvas,
                (x1, y1),
                3,
                (0, 0, 255),
                -1
            )

    return canvas


# ============================================================
# RANSAC
# ============================================================

def calculate_homography(
    points0,
    points1
):

    if len(points0) < 4:

        return (
            None,
            None,
            0,
            len(points0),
            0.0
        )

    H, mask = cv2.findHomography(
        points0,
        points1,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        confidence=RANSAC_CONFIDENCE,
        maxIters=RANSAC_MAX_ITERS
    )

    if H is None:

        return (
            None,
            None,
            0,
            len(points0),
            0.0
        )

    mask = mask.ravel().astype(
        bool
    )

    inliers = int(
        np.sum(mask)
    )

    outliers = (
        len(mask)
        -
        inliers
    )

    ratio = (
        inliers
        /
        len(mask)
    )

    return (
        H,
        mask,
        inliers,
        outliers,
        ratio
    )


# ============================================================
# ESTIMATE POSITION
# ============================================================

def estimate_position(
    H,
    frame,
    roi_info,
    src
):

    height, width = (
        frame.shape[:2]
    )

    center = np.array(
        [
            [
                [
                    width / 2.0,
                    height / 2.0
                ]
            ]
        ],
        dtype=np.float32
    )

    projected = (
        cv2.perspectiveTransform(
            center,
            H
        )
    )

    roi_x = float(
        projected[0, 0, 0]
    )

    roi_y = float(
        projected[0, 0, 1]
    )

    map_x = (
        roi_info["left"]
        +
        roi_x
    )

    map_y = (
        roi_info["top"]
        +
        roi_y
    )

    latitude, longitude = (
        pixel_to_gps(
            src,
            map_y,
            map_x
        )
    )

    return (
        roi_x,
        roi_y,
        map_x,
        map_y,
        latitude,
        longitude
    )


# ============================================================
# FINAL VISUALIZATION
# ============================================================

def draw_final_result(
    satellite,
    roi_info,
    src,
    gps_lat,
    gps_lon,
    est_lat,
    est_lon
):

    output = satellite.copy()

    ref_row, ref_col = (
        gps_to_pixel(
            src,
            gps_lat,
            gps_lon
        )
    )

    est_row, est_col = (
        gps_to_pixel(
            src,
            est_lat,
            est_lon
        )
    )

    ref_x = (
        ref_col -
        roi_info["left"]
    )

    ref_y = (
        ref_row -
        roi_info["top"]
    )

    est_x = (
        est_col -
        roi_info["left"]
    )

    est_y = (
        est_row -
        roi_info["top"]
    )

    ref_x = int(ref_x)
    ref_y = int(ref_y)

    est_x = int(est_x)
    est_y = int(est_y)

    if (
        0 <= ref_x < output.shape[1]
        and
        0 <= ref_y < output.shape[0]
    ):

        cv2.circle(
            output,
            (ref_x, ref_y),
            12,
            (255, 0, 0),
            3
        )

        cv2.putText(
            output,
            "GPS",
            (ref_x + 15, ref_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    if (
        0 <= est_x < output.shape[1]
        and
        0 <= est_y < output.shape[0]
    ):

        cv2.circle(
            output,
            (est_x, est_y),
            12,
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            "EST",
            (est_x + 15, est_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    RESULT_DIR.mkdir(
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("SATELLITE MAP MATCHING")
    print("=" * 70)

    print(
        "Frame :",
        FRAME_PATH
    )

    print(
        "GPS   :",
        GPS_PATH
    )

    print(
        "Map   :",
        MAP_PATH
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        FRAME_PATH,
        GPS_PATH,
        MAP_PATH
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File not found:\n{path}"
            )

    if not SUPERGLUE_DIR.exists():

        raise FileNotFoundError(
            "SuperGluePretrainedNetwork not found."
        )

    if not SUPERPOINT_WEIGHT.exists():

        raise FileNotFoundError(
            "SuperPoint weight not found."
        )

    if not SUPERGLUE_WEIGHT.exists():

        raise FileNotFoundError(
            "SuperGlue outdoor weight not found."
        )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    start = time.perf_counter()

    frame = cv2.imread(
        str(FRAME_PATH)
    )

    if frame is None:

        raise RuntimeError(
            "Could not read frame."
        )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    frame_number = (
        get_frame_number()
    )

    frame_time = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=" * 70)
    print("FRAME")
    print("=" * 70)

    print(
        "Frame number :",
        frame_number
    )

    print(
        "Frame size   :",
        f"{frame_width} x {frame_height}"
    )

    cv2.imwrite(
        str(
            RESULT_DIR /
            "drone_frame.png"
        ),
        frame
    )

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    start = time.perf_counter()

    gps = load_gps()

    gps_row, frame_timestamp, gps_difference = (
        find_gps_for_frame(
            gps,
            frame_number
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

    height_agl = (
        altitude
        -
        GROUND_ALTITUDE
    )

    gps_time = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=" * 70)
    print("GPS")
    print("=" * 70)

    print(
        "Frame timestamp :",
        frame_timestamp
    )

    print(
        "GPS timestamp   :",
        gps_row["bag_timestamp"]
    )

    print(
        "Time difference :",
        f"{gps_difference:.3f} ms"
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
        "Height AGL      :",
        f"{height_agl:.3f} m"
    )

    # --------------------------------------------------------
    # Footprint
    # --------------------------------------------------------

    start = time.perf_counter()

    vertical_fov, footprint_width, footprint_height = (
        calculate_footprint(
            frame_width,
            frame_height,
            height_agl
        )
    )

    required_width = (
        footprint_width
        *
        ROI_MARGIN
    )

    required_height = (
        footprint_height
        *
        ROI_MARGIN
    )

    footprint_time = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=" * 70)
    print("CAMERA FOOTPRINT")
    print("=" * 70)

    print(
        "Horizontal FOV :",
        f"{CAMERA_FOV_DEG:.2f} deg"
    )

    print(
        "Vertical FOV   :",
        f"{math.degrees(vertical_fov):.2f} deg"
    )

    print(
        "Ground width   :",
        f"{footprint_width:.2f} m"
    )

    print(
        "Ground height  :",
        f"{footprint_height:.2f} m"
    )

    print(
        "ROI width      :",
        f"{required_width:.2f} m"
    )

    print(
        "ROI height     :",
        f"{required_height:.2f} m"
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    start = time.perf_counter()

    with rasterio.open(
        str(MAP_PATH)
    ) as src:

        gsd_x, gsd_y = (
            calculate_gsd(
                src,
                gps_lat
            )
        )

        print()
        print("=" * 70)
        print("SATELLITE MAP")
        print("=" * 70)

        print(
            "Map size :",
            f"{src.width} x {src.height}"
        )

        print(
            "CRS      :",
            src.crs
        )

        print(
            "GSD X    :",
            f"{gsd_x:.4f} m/pixel"
        )

        print(
            "GSD Y    :",
            f"{gsd_y:.4f} m/pixel"
        )

        satellite_rgb, roi_info = (
            extract_roi(
                src,
                gps_lat,
                gps_lon,
                required_width,
                required_height,
                gsd_x,
                gsd_y
            )
        )

        satellite = cv2.cvtColor(
            satellite_rgb,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "satellite_roi.png"
            ),
            satellite
        )

        roi_time = (
            time.perf_counter()
            -
            start
        )

        print()
        print("=" * 70)
        print("ROI")
        print("=" * 70)

        print(
            "Left   :",
            roi_info["left"]
        )

        print(
            "Top    :",
            roi_info["top"]
        )

        print(
            "Width  :",
            roi_info["width"]
        )

        print(
            "Height :",
            roi_info["height"]
        )

        # ----------------------------------------------------
        # SuperPoint + SuperGlue
        # ----------------------------------------------------

        matching = run_matching(
            frame,
            satellite
        )

        # ----------------------------------------------------
        # Keypoint visualization
        # ----------------------------------------------------

        drone_kp = draw_keypoints(
            frame,
            matching["keypoints0"]
        )

        satellite_kp = draw_keypoints(
            satellite,
            matching["keypoints1"]
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "superpoint_drone.png"
            ),
            drone_kp
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "superpoint_satellite.png"
            ),
            satellite_kp
        )

        # ----------------------------------------------------
        # Match visualization
        # ----------------------------------------------------

        match_image = draw_matches(
            frame,
            satellite,
            matching["matched0"],
            matching["matched1"]
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "superglue_matches.png"
            ),
            match_image
        )

        print()
        print("=" * 70)
        print("SUPERPOINT / SUPERGLUE")
        print("=" * 70)

        print(
            "Drone keypoints     :",
            len(matching["keypoints0"])
        )

        print(
            "Satellite keypoints:",
            len(matching["keypoints1"])
        )

        print(
            "Valid matches       :",
            len(matching["matched0"])
        )

        print(
            "SuperPoint time     :",
            f"{matching['superpoint_time']:.3f} sec"
        )

        print(
            "SuperGlue time      :",
            f"{matching['superglue_time']:.3f} sec"
        )

        # ----------------------------------------------------
        # RANSAC
        # ----------------------------------------------------

        start = time.perf_counter()

        H, mask, inliers, outliers, ratio = (
            calculate_homography(
                matching["matched0"],
                matching["matched1"]
            )
        )

        ransac_time = (
            time.perf_counter()
            -
            start
        )

        print()
        print("=" * 70)
        print("RANSAC")
        print("=" * 70)

        print(
            "Matches       :",
            len(matching["matched0"])
        )

        print(
            "Inliers       :",
            inliers
        )

        print(
            "Outliers      :",
            outliers
        )

        print(
            "Inlier ratio  :",
            f"{ratio:.4f}"
        )

        if H is None:

            print()
            print(
                "Homography could not be calculated."
            )

            print(
                "At least 4 valid matches are required."
            )

            return

        np.savetxt(
            RESULT_DIR /
            "homography.txt",
            H,
            fmt="%.10f"
        )

        # ----------------------------------------------------
        # RANSAC image
        # ----------------------------------------------------

        inlier0 = (
            matching["matched0"][mask]
        )

        inlier1 = (
            matching["matched1"][mask]
        )

        ransac_image = draw_matches(
            frame,
            satellite,
            inlier0,
            inlier1
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "ransac_inliers.png"
            ),
            ransac_image
        )

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        start = time.perf_counter()

        (
            roi_x,
            roi_y,
            map_x,
            map_y,
            estimated_lat,
            estimated_lon
        ) = estimate_position(
            H,
            frame,
            roi_info,
            src
        )

        position_error = haversine(
            gps_lat,
            gps_lon,
            estimated_lat,
            estimated_lon
        )

        position_time = (
            time.perf_counter()
            -
            start
        )

        # ----------------------------------------------------
        # Final image
        # ----------------------------------------------------

        final_image = draw_final_result(
            satellite,
            roi_info,
            src,
            gps_lat,
            gps_lon,
            estimated_lat,
            estimated_lon
        )

        cv2.imwrite(
            str(
                RESULT_DIR /
                "final_position.png"
            ),
            final_image
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    total_time = (
        time.perf_counter()
        -
        total_start
    )

    ram = peak_ram_mb()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "Reference GPS :",
        f"{gps_lat:.9f}, {gps_lon:.9f}"
    )

    print(
        "Estimated GPS :",
        f"{estimated_lat:.9f}, "
        f"{estimated_lon:.9f}"
    )

    print(
        "Position error:",
        f"{position_error:.3f} m"
    )

    print(
        "SuperGlue matches:",
        len(matching["matched0"])
    )

    print(
        "RANSAC inliers:",
        inliers
    )

    print(
        "Inlier ratio:",
        f"{ratio:.4f}"
    )

    print()
    print("=" * 70)
    print("TIMING")

======================================================================
SATELLITE MAP MATCHING
======================================================================
Frame : /home/tunga/pi5/frame_06898.jpg
GPS   : /home/tunga/pi5/gps.csv
Map   : /home/tunga/pi5/Satellite_Z21.tif

======================================================================
FRAME
======================================================================
Frame number : 6898
Frame size   : 1280 x 720

======================================================================
GPS
======================================================================
Frame timestamp : 1788426873722440021
GPS timestamp   : 1.7884268725825477e+18
Time difference : 1139.892 ms
Latitude        : 12.651060500
Longitude       : 80.139528200
Altitude        : 329.918 m
Height AGL      : 409.918 m

======================================================================
CAMERA FOOTPRINT
======================================================================
Horizontal FOV : 90.00 deg
Vertical FOV   : 58.72 deg
Ground width   : 819.84 m
Ground height  : 461.16 m
ROI width      : 983.80 m
ROI height     : 553.39 m

======================================================================
SATELLITE MAP
======================================================================
Map size : 17978 x 8249
CRS      : EPSG:4326
GSD X    : 0.0720 m/pixel
GSD Y    : 0.0733 m/pixel

======================================================================
ROI
======================================================================
Left   : 839
Top    : 0
Width  : 16406
Height : 8249

Inference device: cpu
Loading SuperPoint + SuperGlue...
Loaded SuperPoint model
Loaded SuperGlue model ("outdoor" weights)
Model loading time: 0.248 sec

======================================================================
ERROR(pi5_env) tunga@tunga-desktop:~/pi5$ python3 ssm_major_code.py

======================================================================
SATELLITE MAP MATCHING
======================================================================
Frame : /home/tunga/pi5/frame_06898.jpg
GPS   : /home/tunga/pi5/gps.csv
Map   : /home/tunga/pi5/Satellite_Z21.tif

======================================================================
FRAME
======================================================================
Frame number : 6898
Frame size   : 1280 x 720

======================================================================
GPS
======================================================================
Frame timestamp : 1788426873722440021
GPS timestamp   : 1.7884268725825477e+18
Time difference : 1139.892 ms
Latitude        : 12.651060500
Longitude       : 80.139528200
Altitude        : 329.918 m
Height AGL      : 409.918 m

======================================================================
CAMERA FOOTPRINT
======================================================================
Horizontal FOV : 90.00 deg
Vertical FOV   : 58.72 deg
Ground width   : 819.84 m
Ground height  : 461.16 m
ROI width      : 983.80 m
ROI height     : 553.39 m

======================================================================
SATELLITE MAP
======================================================================
Map size : 17978 x 8249
CRS      : EPSG:4326
GSD X    : 0.0720 m/pixel
GSD Y    : 0.0733 m/pixel

======================================================================
ROI
======================================================================
Left   : 839
Top    : 0
Width  : 16406
Height : 8249

Inference device: cpu
Loading SuperPoint + SuperGlue...
Loaded SuperPoint mod(pi5_env) tunga@tunga-desktop:~/pi5$ python3 ssm_major_code.py

======================================================================
SATELLITE MAP MATCHING
======================================================================
Frame : /home/tunga/pi5/frame_06898.jpg
GPS   : /home/tunga/pi5/gps.csv
Map   : /home/tunga/pi5/Satellite_Z21.tif

======================================================================
FRAME
======================================================================
Frame number : 6898
Frame size   : 1280 x 720

======================================================================
GPS
======================================================================
Frame timestamp : 1788426873722440021
GPS timestamp   : 1.7884268725825477e+18
Time difference : 1139.892 ms
Latitude        : 12.651060500
Longitude       : 80.139528200
Altitude        : 329.918 m
Height AGL      : 409.918 m

======================================================================
CAMERA FOOTPRINT
======================================================================
Horizontal FOV : 90.00 deg
Vertical FOV   : 58.72 deg
Ground width   : 819.84 m
Ground height  : 461.16 m
ROI width      : 983.80 m
ROI height     : 553.39 m

======================================================================
SATELLITE MAP
======================================================================
Map size : 17978 x 8249
CRS      : EPSG:4326
GSD X    : 0.0720 m/pixel
GSD Y    : 0.0733 m/pixel

======================================================================
ROI
======================================================================
Left   : 839
Top    : 0
Width  : 16406
Height : 8249

Inference device: cpu
Loading SuperPoint + SuperGlue...
Loaded SuperPoint model
Loaded SuperGlue model ("outdoor" weights)
Model loading time: 0.248 sec

======================================================================
ERROR
======================================================================
AttributeError : 'tuple' object has no attribute 'ndim'
Traceback (most recent call last):
  File "/home/tunga/pi5/ssm_major_code.py", line 2324, in <module>
    main()
  File "/home/tunga/pi5/ssm_major_code.py", line 1782, in main
    matching = run_matching(
               ^^^^^^^^^^^^^
  File "/home/tunga/pi5/ssm_major_code.py", line 871, in run_matching
    if scores0_tensor.ndim == 1:
       ^^^^^^^^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'ndim'
el
Loaded SuperGlue model ("outdoor" weights)
Model loading time: 0.248 sec

======================================================================
ERROR
======================================================================
AttributeError : 'tuple' object has no attribute 'ndim'
Traceback (most recent call last):
  File "/home/tunga/pi5/ssm_major_code.py", line 2324, in <module>
    main()
  File "/home/tunga/pi5/ssm_major_code.py", line 1782, in main
    matching = run_matching(
               ^^^^^^^^^^^^^
  File "/home/tunga/pi5/ssm_major_code.py", line 871, in run_matching
    if scores0_tensor.ndim == 1:
       ^^^^^^^^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'ndim'

======================================================================
AttributeError : 'tuple' object has no attribute 'ndim'
Traceback (most recent call last):
  File "/home/tunga/pi5/ssm_major_code.py", line 2324, in <module>
    main()
  File "/home/tunga/pi5/ssm_major_code.py", line 1782, in main
    matching = run_matching(
               ^^^^^^^^^^^^^
  File "/home/tunga/pi5/ssm_major_code.py", line 871, in run_matching
    if scores0_tensor.ndim == 1:
       ^^^^^^^^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'ndim'

    print("=" * 70)

    print(
        "Frame loading :",
        f"{frame_time:.4f} sec"
    )

    print(
        "GPS processing:",
        f"{gps_time:.4f} sec"
    )

    print(
        "Footprint     :",
        f"{footprint_time:.4f} sec"
    )

    print(
        "ROI extraction:",
        f"{roi_time:.4f} sec"
    )

    print(
        "Model loading :",
        f"{matching['model_time']:.4f} sec"
    )

    print(
        "SuperPoint    :",
        f"{matching['superpoint_time']:.4f} sec"
    )

    print(
        "SuperGlue     :",
        f"{matching['superglue_time']:.4f} sec"
    )

    print(
        "RANSAC        :",
        f"{ransac_time:.4f} sec"
    )

    print(
        "Position      :",
        f"{position_time:.4f} sec"
    )

    print(
        "TOTAL         :",
        f"{total_time:.4f} sec"
    )

    print(
        "Peak RAM      :",
        f"{ram:.2f} MB"
    )

    print(
        "Device        :",
        matching["device"]
    )

    # --------------------------------------------------------
    # Save text results
    # --------------------------------------------------------

    with open(
        RESULT_DIR /
        "results.txt",
        "w"
    ) as f:

        f.write(
            "SATELLITE MAP MATCHING RESULTS\n"
        )

        f.write(
            "===============================\n\n"
        )

        f.write(
            f"Frame: {FRAME_FILE}\n"
        )

        f.write(
            f"GPS: {GPS_FILE}\n"
        )

        f.write(
            f"Map: {MAP_FILE}\n\n"
        )

        f.write(
            f"Frame number: {frame_number}\n"
        )

        f.write(
            f"Frame size: "
            f"{frame_width} x "
            f"{frame_height}\n"
        )

        f.write(
            f"Latitude: "
            f"{gps_lat:.9f}\n"
        )

        f.write(
            f"Longitude: "
            f"{gps_lon:.9f}\n"
        )

        f.write(
            f"Altitude: "
            f"{altitude:.3f} m\n"
        )

        f.write(
            f"AGL: "
            f"{height_agl:.3f} m\n"
        )

        f.write(
            f"GSD X: "
            f"{gsd_x:.6f} m/pixel\n"
        )

        f.write(
            f"GSD Y: "
            f"{gsd_y:.6f} m/pixel\n"
        )

        f.write(
            f"ROI: "
            f"{roi_info['width']} x "
            f"{roi_info['height']}\n"
        )

        f.write(
            f"SuperPoint drone: "
            f"{len(matching['keypoints0'])}\n"
        )

        f.write(
            f"SuperPoint satellite: "
            f"{len(matching['keypoints1'])}\n"
        )

        f.write(
            f"SuperGlue matches: "
            f"{len(matching['matched0'])}\n"
        )

        f.write(
            f"RANSAC inliers: "
            f"{inliers}\n"
        )

        f.write(
            f"RANSAC outliers: "
            f"{outliers}\n"
        )

        f.write(
            f"Inlier ratio: "
            f"{ratio:.6f}\n"
        )

        f.write(
            f"Estimated latitude: "
            f"{estimated_lat:.9f}\n"
        )

        f.write(
            f"Estimated longitude: "
            f"{estimated_lon:.9f}\n"
        )

        f.write(
            f"Position error: "
            f"{position_error:.3f} m\n"
        )

        f.write("\nTIMING\n")

        f.write(
            f"Frame loading: "
            f"{frame_time:.4f} sec\n"
        )

        f.write(
            f"GPS: "
            f"{gps_time:.4f} sec\n"
        )

        f.write(
            f"Footprint: "
            f"{footprint_time:.4f} sec\n"
        )

        f.write(
            f"ROI: "
            f"{roi_time:.4f} sec\n"
        )

        f.write(
            f"Model loading: "
            f"{matching['model_time']:.4f} sec\n"
        )

        f.write(
            f"SuperPoint: "
            f"{matching['superpoint_time']:.4f} sec\n"
        )

        f.write(
            f"SuperGlue: "
            f"{matching['superglue_time']:.4f} sec\n"
        )

        f.write(
            f"RANSAC: "
            f"{ransac_time:.4f} sec\n"
        )

        f.write(
            f"Position: "
            f"{position_time:.4f} sec\n"
        )

        f.write(
            f"TOTAL: "
            f"{total_time:.4f} sec\n"
        )

        f.write(
            f"Peak RAM: "
            f"{ram:.2f} MB\n"
        )

    print()
    print(
        "Results saved in:"
    )

    print(
        RESULT_DIR
    )

    print()
    print("DONE")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            type(error).__name__,
            ":",
            error
        )

        raise
