#!/usr/bin/env python3

import os
import math
import time

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy

import torch

from models.matching import Matching


# ============================================================
# CONFIGURATION
# ============================================================
# ONLY CHANGE THESE INPUT VALUES WHEN USING ANOTHER DATASET
# ============================================================

BASE_DIR = "/home/sandiya/sept 3-Drone-map-test"

FRAME_FILE = os.path.join(
    BASE_DIR,
    "frame_06898.jpg"
)

GPS_CSV = os.path.join(
    BASE_DIR,
    "gps.csv"
)

FRAME_TIMESTAMP_CSV = os.path.join(
    BASE_DIR,
    "frame_timestamps.csv"
)

SATELLITE_MAP = os.path.join(
    BASE_DIR,
    "Satellite_Z21.tif"
)


# ============================================================
# CAMERA PARAMETERS
# ============================================================

HFOV_DEG = 90.0

# Drone frame is assumed to be looking approximately nadir.
# Altitude from GPS CSV is assumed to be AGL.
#
# If the camera is tilted significantly, this simple footprint
# calculation should later be replaced by pose-based geometry.


# ============================================================
# ROI PARAMETERS
# ============================================================

ROI_MARGIN_PX = 1000

# Maximum dimension used by SuperPoint/SuperGlue.
#
# The original satellite ROI is NEVER modified.
# Only a temporary copy is resized for neural-network matching.

MAX_MATCHING_DIM = 2048


# ============================================================
# SUPERPOINT / SUPERGLUE PARAMETERS
# ============================================================

SUPERGLUE_WEIGHTS = "outdoor"

MAX_KEYPOINTS = 2048

KEYPOINT_THRESHOLD = 0.005

MATCH_THRESHOLD = 0.2


# ============================================================
# RANSAC PARAMETERS
# ============================================================

RANSAC_REPROJ_THRESHOLD = 5.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.999


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "offline_map_matching_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

ORIGINAL_ROI_FILE = os.path.join(
    OUTPUT_DIR,
    "satellite_roi_original.png"
)

MATCHING_ROI_FILE = os.path.join(
    OUTPUT_DIR,
    "satellite_roi_for_matching.png"
)

MATCHES_NPZ = os.path.join(
    OUTPUT_DIR,
    "superglue_matches.npz"
)

MATCHES_VIS = os.path.join(
    OUTPUT_DIR,
    "superglue_matches.jpg"
)

RANSAC_NPZ = os.path.join(
    OUTPUT_DIR,
    "ransac_results.npz"
)

RANSAC_VIS = os.path.join(
    OUTPUT_DIR,
    "ransac_inliers.jpg"
)

POSITION_VIS = os.path.join(
    OUTPUT_DIR,
    "estimated_position.jpg"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "map_matching_report.txt"
)


# ============================================================
# UTILITY
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# STEP 1
# FRAME TIMESTAMP -> NEAREST GPS
# ============================================================

def get_frame_timestamp():

    timestamp_df = pd.read_csv(
        FRAME_TIMESTAMP_CSV
    )

    required = [
        "frame_filename",
        "timestamp_ns"
    ]

    for col in required:
        if col not in timestamp_df.columns:
            raise RuntimeError(
                f"Missing column in frame timestamp CSV: {col}"
            )

    frame_name = os.path.basename(
        FRAME_FILE
    )

    rows = timestamp_df[
        timestamp_df["frame_filename"].astype(str)
        == frame_name
    ]

    if len(rows) == 0:

        raise RuntimeError(
            f"Frame {frame_name} not found in "
            f"{FRAME_TIMESTAMP_CSV}"
        )

    timestamp_ns = int(
        rows.iloc[0]["timestamp_ns"]
    )

    return timestamp_ns


def get_nearest_gps(frame_timestamp_ns):

    gps_df = pd.read_csv(
        GPS_CSV
    )

    required = [
        "bag_timestamp",
        "latitude",
        "longitude",
        "altitude"
    ]

    for col in required:
        if col not in gps_df.columns:
            raise RuntimeError(
                f"Missing column in GPS CSV: {col}"
            )

    gps_timestamps = pd.to_numeric(
        gps_df["bag_timestamp"],
        errors="coerce"
    ).values

    differences = np.abs(
        gps_timestamps.astype(np.int64)
        - np.int64(frame_timestamp_ns)
    )

    nearest_index = int(
        np.argmin(differences)
    )

    row = gps_df.iloc[nearest_index]

    gps_timestamp_ns = int(
        gps_timestamps[nearest_index]
    )

    time_difference_ms = (
        abs(
            gps_timestamp_ns
            - frame_timestamp_ns
        )
        / 1e6
    )

    latitude = float(
        row["latitude"]
    )

    longitude = float(
        row["longitude"]
    )

    altitude = float(
        row["altitude"]
    )

    return (
        latitude,
        longitude,
        altitude,
        gps_timestamp_ns,
        time_difference_ms
    )


# ============================================================
# STEP 2
# OPEN SATELLITE MAP
# ============================================================

def get_map_information():

    src = rasterio.open(
        SATELLITE_MAP
    )

    return src


# ============================================================
# STEP 3
# GPS -> SATELLITE PIXEL
# ============================================================

def gps_to_pixel(
    src,
    latitude,
    longitude
):

    row, col = rowcol(
        src.transform,
        longitude,
        latitude
    )

    row = float(row)
    col = float(col)

    return row, col


# ============================================================
# STEP 4
# CALCULATE CAMERA FOV
# ============================================================

def calculate_vfov(
    hfov_deg,
    frame_width,
    frame_height
):

    hfov_rad = math.radians(
        hfov_deg
    )

    aspect_ratio = (
        frame_height
        / frame_width
    )

    vfov_rad = 2.0 * math.atan(
        math.tan(hfov_rad / 2.0)
        * aspect_ratio
    )

    return math.degrees(
        vfov_rad
    )


# ============================================================
# STEP 4
# CALCULATE GROUND FOOTPRINT
# ============================================================

def calculate_ground_footprint(
    altitude,
    hfov_deg,
    vfov_deg
):

    width_m = (
        2.0
        * altitude
        * math.tan(
            math.radians(hfov_deg) / 2.0
        )
    )

    height_m = (
        2.0
        * altitude
        * math.tan(
            math.radians(vfov_deg) / 2.0
        )
    )

    return width_m, height_m


# ============================================================
# STEP 4
# CALCULATE APPROXIMATE MAP METERS / PIXEL
# ============================================================

def calculate_map_meters_per_pixel(
    src,
    latitude
):

    # GeoTIFF is EPSG:4326.
    #
    # Convert degree resolution to approximate meters.
    #
    # 1 degree latitude is approximately 111320 m.
    #
    # Longitude distance depends on latitude.

    x_deg = abs(
        src.transform.a
    )

    y_deg = abs(
        src.transform.e
    )

    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0
        * math.cos(
            math.radians(latitude)
        )
    )

    x_m_per_px = (
        x_deg
        * meters_per_degree_lon
    )

    y_m_per_px = (
        y_deg
        * meters_per_degree_lat
    )

    return (
        x_m_per_px,
        y_m_per_px
    )


# ============================================================
# STEP 4
# CREATE ROI
# ============================================================

def create_roi(
    src,
    center_row,
    center_col,
    footprint_width_m,
    footprint_height_m,
    x_m_per_px,
    y_m_per_px
):

    footprint_width_px = (
        footprint_width_m
        / x_m_per_px
    )

    footprint_height_px = (
        footprint_height_m
        / y_m_per_px
    )

    roi_width = int(
        math.ceil(
            footprint_width_px
            + 2 * ROI_MARGIN_PX
        )
    )

    roi_height = int(
        math.ceil(
            footprint_height_px
            + 2 * ROI_MARGIN_PX
        )
    )

    x1 = int(
        math.floor(
            center_col
            - roi_width / 2.0
        )
    )

    y1 = int(
        math.floor(
            center_row
            - roi_height / 2.0
        )
    )

    x2 = x1 + roi_width
    y2 = y1 + roi_height

    # Clip to map boundaries.

    x1_clip = max(
        0,
        x1
    )

    y1_clip = max(
        0,
        y1
    )

    x2_clip = min(
        src.width,
        x2
    )

    y2_clip = min(
        src.height,
        y2
    )

    return (
        x1_clip,
        y1_clip,
        x2_clip,
        y2_clip,
        footprint_width_px,
        footprint_height_px
    )


# ============================================================
# READ GEO TIFF ROI
# ============================================================

def read_roi(
    src,
    x1,
    y1,
    x2,
    y2
):

    width = x2 - x1
    height = y2 - y1

    window = rasterio.windows.Window(
        x1,
        y1,
        width,
        height
    )

    data = src.read(
        window=window
    )

    # Rasterio:
    # bands, height, width
    #
    # OpenCV:
    # height, width, channels

    if data.shape[0] >= 3:

        image = np.transpose(
            data[:3],
            (1, 2, 0)
        )

    else:

        image = data[0]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    # Convert datatype safely.

    if image.dtype != np.uint8:

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

    return image


# ============================================================
# STEP 5
# RESIZE SATELLITE ROI
# ============================================================

def resize_for_matching(
    image
):

    height, width = image.shape[:2]

    largest_dimension = max(
        width,
        height
    )

    if largest_dimension <= MAX_MATCHING_DIM:

        scale = 1.0

        resized = image.copy()

    else:

        scale = (
            MAX_MATCHING_DIM
            / largest_dimension
        )

        new_width = int(
            round(
                width * scale
            )
        )

        new_height = int(
            round(
                height * scale
            )
        )

        resized = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    return resized, scale


# ============================================================
# CONVERT BGR -> TORCH GRAYSCALE
# ============================================================

def image_to_tensor(
    image,
    device
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    tensor = (
        torch.from_numpy(
            gray
        )
        .float()
        / 255.0
    )

    tensor = tensor[
        None,
        None
    ]

    tensor = tensor.to(
        device
    )

    return tensor


# ============================================================
# STEP 5
# SUPERPOINT + SUPERGLUE
# ============================================================

def run_superglue(
    frame,
    satellite_matching
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device              : {device}"
    )

    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold":
                KEYPOINT_THRESHOLD,
            "max_keypoints":
                MAX_KEYPOINTS
        },

        "superglue": {
            "weights":
                SUPERGLUE_WEIGHTS,
            "sinkhorn_iterations": 20,
            "match_threshold":
                MATCH_THRESHOLD
        }
    }

    matching = Matching(
        config
    ).eval().to(
        device
    )

    frame_tensor = image_to_tensor(
        frame,
        device
    )

    satellite_tensor = image_to_tensor(
        satellite_matching,
        device
    )

    print()
    print("Running SuperPoint + SuperGlue...")
    print()

    with torch.no_grad():

        pred = matching({
            "image0": frame_tensor,
            "image1": satellite_tensor
        })

    keypoints0 = (
        pred["keypoints0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    keypoints1 = (
        pred["keypoints1"][0]
        .detach()
        .cpu()
        .numpy()
    )

    matches0 = (
        pred["matches0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    matching_scores0 = (
        pred["matching_scores0"][0]
        .detach()
        .cpu()
        .numpy()
    )

    valid = matches0 > -1

    matched_kpts0 = (
        keypoints0[valid]
    )

    matched_indices1 = (
        matches0[valid]
        .astype(np.int32)
    )

    matched_kpts1 = (
        keypoints1[
            matched_indices1
        ]
    )

    matched_confidence = (
        matching_scores0[valid]
    )

    return (
        matched_kpts0,
        matched_kpts1,
        matched_confidence,
        device
    )


# ============================================================
# DRAW SUPERGLUE MATCHES
# ============================================================

def create_match_visualization(
    frame,
    satellite,
    kpts0,
    kpts1,
    confidence
):

    h0, w0 = frame.shape[:2]
    h1, w1 = satellite.shape[:2]

    canvas_height = max(
        h0,
        h1
    )

    canvas_width = (
        w0 + w1
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :h0,
        :w0
    ] = frame

    canvas[
        :h1,
        w0:w0 + w1
    ] = satellite

    # Draw up to 300 matches for readability.

    number_to_draw = min(
        300,
        len(kpts0)
    )

    if len(kpts0) > number_to_draw:

        indices = np.linspace(
            0,
            len(kpts0) - 1,
            number_to_draw
        ).astype(int)

    else:

        indices = np.arange(
            len(kpts0)
        )

    for i in indices:

        x0, y0 = kpts0[i]

        x1, y1 = kpts1[i]

        x1_canvas = (
            x1 + w0
        )

        cv2.line(
            canvas,
            (
                int(x0),
                int(y0)
            ),
            (
                int(x1_canvas),
                int(y1)
            ),
            (0, 255, 0),
            1
        )

        cv2.circle(
            canvas,
            (
                int(x0),
                int(y0)
            ),
            3,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            canvas,
            (
                int(x1_canvas),
                int(y1)
            ),
            3,
            (0, 0, 255),
            -1
        )

    return canvas


# ============================================================
# STEP 6
# RANSAC
# ============================================================

def run_ransac(
    matched_kpts0,
    matched_kpts1,
    matched_confidence
):

    if len(matched_kpts0) < 4:

        raise RuntimeError(
            "Not enough matches for homography."
        )

    H, mask = cv2.findHomography(
        matched_kpts0,
        matched_kpts1,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD,
        None,
        RANSAC_MAX_ITERS,
        RANSAC_CONFIDENCE
    )

    if H is None or mask is None:

        raise RuntimeError(
            "RANSAC failed to calculate homography."
        )

    mask = mask.ravel().astype(
        bool
    )

    inlier_kpts0 = (
        matched_kpts0[mask]
    )

    inlier_kpts1 = (
        matched_kpts1[mask]
    )

    inlier_confidence = (
        matched_confidence[mask]
    )

    total_matches = (
        len(matched_kpts0)
    )

    inliers = int(
        np.sum(mask)
    )

    outliers = (
        total_matches
        - inliers
    )

    inlier_ratio = (
        inliers
        / total_matches
        if total_matches > 0
        else 0.0
    )

    mean_inlier_confidence = (
        float(
            np.mean(
                inlier_confidence
            )
        )
        if inliers > 0
        else 0.0
    )

    return (
        H,
        mask,
        inlier_kpts0,
        inlier_kpts1,
        inlier_confidence,
        total_matches,
        inliers,
        outliers,
        inlier_ratio,
        mean_inlier_confidence
    )


# ============================================================
# DRAW RANSAC INLIERS
# ============================================================

def create_inlier_visualization(
    frame,
    satellite,
    inlier_kpts0,
    inlier_kpts1
):

    h0, w0 = frame.shape[:2]
    h1, w1 = satellite.shape[:2]

    canvas_height = max(
        h0,
        h1
    )

    canvas_width = (
        w0 + w1
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :h0,
        :w0
    ] = frame

    canvas[
        :h1,
        w0:w0 + w1
    ] = satellite

    for p0, p1 in zip(
        inlier_kpts0,
        inlier_kpts1
    ):

        x0, y0 = p0

        x1, y1 = p1

        x1_canvas = (
            x1 + w0
        )

        cv2.line(
            canvas,
            (
                int(x0),
                int(y0)
            ),
            (
                int(x1_canvas),
                int(y1)
            ),
            (0, 255, 0),
            1
        )

        cv2.circle(
            canvas,
            (
                int(x0),
                int(y0)
            ),
            3,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            canvas,
            (
                int(x1_canvas),
                int(y1)
            ),
            3,
            (0, 0, 255),
            -1
        )

    return canvas


# ============================================================
# STEP 7
# VISUAL POSITION ESTIMATION
# ============================================================

def estimate_position(
    H,
    frame_width,
    frame_height,
    resize_scale,
    roi_x1,
    roi_y1,
    src
):

    center_x = (
        frame_width
        / 2.0
    )

    center_y = (
        frame_height
        / 2.0
    )

    point = np.array(
        [
            [
                [
                    center_x,
                    center_y
                ]
            ]
        ],
        dtype=np.float32
    )

    projected = cv2.perspectiveTransform(
        point,
        H
    )

    matching_x = float(
        projected[0, 0, 0]
    )

    matching_y = float(
        projected[0, 0, 1]
    )

    # Undo satellite ROI resize.

    original_roi_x = (
        matching_x
        / resize_scale
    )

    original_roi_y = (
        matching_y
        / resize_scale
    )

    # Convert ROI coordinate to
    # full GeoTIFF coordinate.

    map_column = (
        roi_x1
        + original_roi_x
    )

    map_row = (
        roi_y1
        + original_roi_y
    )

    # Convert GeoTIFF pixel to
    # longitude / latitude.

    longitude, latitude = xy(
        src.transform,
        map_row,
        map_column
    )

    return (
        center_x,
        center_y,
        matching_x,
        matching_y,
        original_roi_x,
        original_roi_y,
        map_column,
        map_row,
        float(latitude),
        float(longitude)
    )


# ============================================================
# POSITION VISUALIZATION
# ============================================================

def create_position_visualization(
    satellite_matching,
    matching_x,
    matching_y,
    latitude,
    longitude
):

    visual = (
        satellite_matching.copy()
    )

    x = int(
        round(matching_x)
    )

    y = int(
        round(matching_y)
    )

    cv2.circle(
        visual,
        (x, y),
        15,
        (0, 0, 255),
        -1
    )

    cv2.line(
        visual,
        (x - 30, y),
        (x + 30, y),
        (0, 0, 255),
        3
    )

    cv2.line(
        visual,
        (x, y - 30),
        (x, y + 30),
        (0, 0, 255),
        3
    )

    text = (
        f"Lat: {latitude:.7f} "
        f"Lon: {longitude:.7f}"
    )

    cv2.putText(
        visual,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return visual


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    print_section(
        "OFFLINE SATELLITE MAP MATCHING"
    )

    # --------------------------------------------------------
    # LOAD FRAME
    # --------------------------------------------------------

    frame = cv2.imread(
        FRAME_FILE
    )

    if frame is None:

        raise RuntimeError(
            f"Could not read frame:\n"
            f"{FRAME_FILE}"
        )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    print()
    print("FRAME")
    print("-" * 70)

    print(
        f"File                : {FRAME_FILE}"
    )

    print(
        f"Size                : "
        f"{frame_width} x {frame_height}"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print_section(
        "STEP 1 - FRAME TIMESTAMP -> GPS"
    )

    frame_timestamp_ns = (
        get_frame_timestamp()
    )

    (
        gps_latitude,
        gps_longitude,
        altitude,
        gps_timestamp_ns,
        gps_difference_ms
    ) = get_nearest_gps(
        frame_timestamp_ns
    )

    print(
        f"Frame timestamp     : "
        f"{frame_timestamp_ns}"
    )

    print(
        f"Nearest GPS         : "
        f"{gps_timestamp_ns}"
    )

    print(
        f"Time difference     : "
        f"{gps_difference_ms:.3f} ms"
    )

    print(
        f"GPS latitude        : "
        f"{gps_latitude:.10f}"
    )

    print(
        f"GPS longitude       : "
        f"{gps_longitude:.10f}"
    )

    print(
        f"Altitude AGL        : "
        f"{altitude:.6f} m"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print_section(
        "STEP 2 - SATELLITE MAP"
    )

    src = get_map_information()

    print(
        f"Map size            : "
        f"{src.width} x {src.height}"
    )

    print(
        f"CRS                 : "
        f"{src.crs}"
    )

    print(
        f"Bounds              : "
        f"{src.bounds}"
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print_section(
        "STEP 3 - GPS -> MAP PIXEL"
    )

    center_row, center_col = (
        gps_to_pixel(
            src,
            gps_latitude,
            gps_longitude
        )
    )

    print(
        f"Map center column   : "
        f"{center_col:.3f}"
    )

    print(
        f"Map center row      : "
        f"{center_row:.3f}"
    )

    if not (
        0 <= center_col < src.width
        and
        0 <= center_row < src.height
    ):

        raise RuntimeError(
            "GPS position is outside satellite map."
        )

    print(
        "GPS is inside satellite map."
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print_section(
        "STEP 4 - AUTOMATIC ROI CALCULATION"
    )

    vfov_deg = calculate_vfov(
        HFOV_DEG,
        frame_width,
        frame_height
    )

    (
        footprint_width_m,
        footprint_height_m
    ) = calculate_ground_footprint(
        altitude,
        HFOV_DEG,
        vfov_deg
    )

    (
        x_m_per_px,
        y_m_per_px
    ) = calculate_map_meters_per_pixel(
        src,
        gps_latitude
    )

    (
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        footprint_width_px,
        footprint_height_px
    ) = create_roi(
        src,
        center_row,
        center_col,
        footprint_width_m,
        footprint_height_m,
        x_m_per_px,
        y_m_per_px
    )

    print(
        f"Horizontal FOV     : "
        f"{HFOV_DEG:.3f} deg"
    )

    print(
        f"Vertical FOV       : "
        f"{vfov_deg:.3f} deg"
    )

    print(
        f"Ground width       : "
        f"{footprint_width_m:.3f} m"
    )

    print(
        f"Ground height      : "
        f"{footprint_height_m:.3f} m"
    )

    print(
        f"Map X m/pixel      : "
        f"{x_m_per_px:.8f}"
    )

    print(
        f"Map Y m/pixel      : "
        f"{y_m_per_px:.8f}"
    )

    print(
        f"Footprint width    : "
        f"{footprint_width_px:.3f} px"
    )

    print(
        f"Footprint height   : "
        f"{footprint_height_px:.3f} px"
    )

    print(
        f"ROI margin         : "
        f"{ROI_MARGIN_PX} px"
    )

    print(
        f"ROI x1             : {roi_x1}"
    )

    print(
        f"ROI y1             : {roi_y1}"
    )

    print(
        f"ROI x2             : {roi_x2}"
    )

    print(
        f"ROI y2             : {roi_y2}"
    )

    roi_width = (
        roi_x2 - roi_x1
    )

    roi_height = (
        roi_y2 - roi_y1
    )

    print(
        f"Actual ROI size    : "
        f"{roi_width} x {roi_height}"
    )

    # --------------------------------------------------------
    # READ ORIGINAL ROI
    # --------------------------------------------------------

    original_roi = read_roi(
        src,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2
    )

    cv2.imwrite(
        ORIGINAL_ROI_FILE,
        original_roi
    )

    print()
    print(
        f"Original ROI saved : "
        f"{ORIGINAL_ROI_FILE}"
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print_section(
        "STEP 5 - SUPERPOINT + SUPERGLUE"
    )

    (
        satellite_matching,
        resize_scale
    ) = resize_for_matching(
        original_roi
    )

    matching_height, matching_width = (
        satellite_matching.shape[:2]
    )

    print(
        f"Original ROI size  : "
        f"{roi_width} x {roi_height}"
    )

    print(
        f"Matching ROI size  : "
        f"{matching_width} x {matching_height}"
    )

    print(
        f"Resize scale       : "
        f"{resize_scale:.10f}"
    )

    cv2.imwrite(
        MATCHING_ROI_FILE,
        satellite_matching
    )

    matching_start = time.perf_counter()

    (
        matched_kpts0,
        matched_kpts1,
        matched_confidence,
        device
    ) = run_superglue(
        frame,
        satellite_matching
    )

    matching_time = (
        time.perf_counter()
        - matching_start
    )

    total_matches = (
        len(matched_kpts0)
    )

    mean_confidence = (
        float(
            np.mean(
                matched_confidence
            )
        )
        if total_matches > 0
        else 0.0
    )

    print()
    print(
        f"SuperPoint keypoints "
        f"(frame)             : "
        f"{min(MAX_KEYPOINTS, 2048)}"
    )

    print(
        f"Valid SuperGlue matches: "
        f"{total_matches}"
    )

    print(
        f"Mean match confidence : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Matching time        : "
        f"{matching_time:.4f} s"
    )

    np.savez(
        MATCHES_NPZ,
        matched_kpts0=matched_kpts0,
        matched_kpts1=matched_kpts1,
        matched_confidence=matched_confidence,
        resize_scale=resize_scale
    )

    match_visual = (
        create_match_visualization(
            frame,
            satellite_matching,
            matched_kpts0,
            matched_kpts1,
            matched_confidence
        )
    )

    cv2.imwrite(
        MATCHES_VIS,
        match_visual
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print_section(
        "STEP 6 - RANSAC INLIER CALCULATION"
    )

    ransac_start = time.perf_counter()

    (
        H,
        ransac_mask,
        inlier_kpts0,
        inlier_kpts1,
        inlier_confidence,
        total_matches,
        inliers,
        outliers,
        inlier_ratio,
        mean_inlier_confidence
    ) = run_ransac(
        matched_kpts0,
        matched_kpts1,
        matched_confidence
    )

    ransac_time = (
        time.perf_counter()
        - ransac_start
    )

    print(
        f"Total matches       : "
        f"{total_matches}"
    )

    print(
        f"RANSAC inliers      : "
        f"{inliers}"
    )

    print(
        f"RANSAC outliers     : "
        f"{outliers}"
    )

    print(
        f"Inlier ratio        : "
        f"{inlier_ratio * 100:.2f}%"
    )

    print(
        f"Mean inlier confidence: "
        f"{mean_inlier_confidence:.4f}"
    )

    print()
    print("HOMOGRAPHY")
    print("-" * 70)

    print(H)

    print(
        f"\nRANSAC time         : "
        f"{ransac_time:.4f} s"
    )

    np.savez(
        RANSAC_NPZ,
        homography=H,
        ransac_mask=ransac_mask,
        inlier_kpts0=inlier_kpts0,
        inlier_kpts1=inlier_kpts1,
        inlier_confidence=inlier_confidence
    )

    inlier_visual = (
        create_inlier_visualization(
            frame,
            satellite_matching,
            inlier_kpts0,
            inlier_kpts1
        )
    )

    cv2.imwrite(
        RANSAC_VIS,
        inlier_visual
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    print_section(
        "STEP 7 - VISUAL POSITION ESTIMATION"
    )

    (
        frame_center_x,
        frame_center_y,
        matching_x,
        matching_y,
        original_roi_x,
        original_roi_y,
        map_column,
        map_row,
        estimated_latitude,
        estimated_longitude
    ) = estimate_position(
        H,
        frame_width,
        frame_height,
        resize_scale,
        roi_x1,
        roi_y1,
        src
    )

    print()
    print(
        f"Frame center X      : "
        f"{frame_center_x:.3f}"
    )

    print(
        f"Frame center Y      : "
        f"{frame_center_y:.3f}"
    )

    print()
    print(
        f"Matching ROI X      : "
        f"{matching_x:.3f}"
    )

    print(
        f"Matching ROI Y      : "
        f"{matching_y:.3f}"
    )

    print()
    print(
        f"Original ROI X      : "
        f"{original_roi_x:.3f}"
    )

    print(
        f"Original ROI Y      : "
        f"{original_roi_y:.3f}"
    )

    print()
    print(
        f"Map column          : "
        f"{map_column:.3f}"
    )

    print(
        f"Map row             : "
        f"{map_row:.3f}"
    )

    print()
    print(
        f"Estimated Latitude  : "
        f"{estimated_latitude:.10f}"
    )

    print(
        f"Estimated Longitude : "
        f"{estimated_longitude:.10f}"
    )

    position_visual = (
        create_position_visualization(
            satellite_matching,
            matching_x,
            matching_y,
            estimated_latitude,
            estimated_longitude
        )
    )

    cv2.imwrite(
        POSITION_VIS,
        position_visual
    )

    # --------------------------------------------------------
    # GPS VS VISUAL DIAGNOSTIC
    # --------------------------------------------------------

    gps_column_difference = (
        map_column - center_col
    )

    gps_row_difference = (
        map_row - center_row
    )

    # Approximate ground difference.

    east_difference_m = (
        gps_column_difference
        * x_m_per_px
    )

    north_difference_m = (
        -gps_row_difference
        * y_m_per_px
    )

    horizontal_difference_m = math.sqrt(
        east_difference_m ** 2
        +
        north_difference_m ** 2
    )

    print_section(
        "GPS VS VISUAL DIAGNOSTIC"
    )

    print(
        f"GPS latitude       : "
        f"{gps_latitude:.10f}"
    )

    print(
        f"GPS longitude      : "
        f"{gps_longitude:.10f}"
    )

    print(
        f"Visual latitude    : "
        f"{estimated_latitude:.10f}"
    )

    print(
        f"Visual longitude   : "
        f"{estimated_longitude:.10f}"
    )

    print()
    print(
        f"East difference    : "
        f"{east_difference_m:.3f} m"
    )

    print(
        f"North difference   : "
        f"{north_difference_m:.3f} m"
    )

    print(
        f"Horizontal difference: "
        f"{horizontal_difference_m:.3f} m"
    )

    # --------------------------------------------------------
    # TOTAL PROCESSING TIME
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    print_section(
        "FINAL RESULT"
    )

    print()
    print(
        f"Estimated Latitude  : "
        f"{estimated_latitude:.10f}"
    )

    print(
        f"Estimated Longitude : "
        f"{estimated_longitude:.10f}"
    )

    print()
    print(
        f"Total processing time: "
        f"{total_time:.4f} s"
    )

    # --------------------------------------------------------
    # SAVE COMPLETE REPORT
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "OFFLINE SATELLITE MAP MATCHING REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "INPUT\n"
        )

        f.write(
            f"Frame              : {FRAME_FILE}\n"
        )

        f.write(
            f"GPS CSV            : {GPS_CSV}\n"
        )

        f.write(
            f"Frame timestamp CSV: "
            f"{FRAME_TIMESTAMP_CSV}\n"
        )

        f.write(
            f"Satellite map      : "
            f"{SATELLITE_MAP}\n\n"
        )

        f.write(
            "FRAME\n"
        )

        f.write(
            f"Width              : "
            f"{frame_width}\n"
        )

        f.write(
            f"Height             : "
            f"{frame_height}\n\n"
        )

        f.write(
            "GPS\n"
        )

        f.write(
            f"Frame timestamp    : "
            f"{frame_timestamp_ns}\n"
        )

        f.write(
            f"GPS timestamp      : "
            f"{gps_timestamp_ns}\n"
        )

        f.write(
            f"Timestamp difference: "
            f"{gps_difference_ms:.6f} ms\n"
        )

        f.write(
            f"Latitude           : "
            f"{gps_latitude:.10f}\n"
        )

        f.write(
            f"Longitude          : "
            f"{gps_longitude:.10f}\n"
        )

        f.write(
            f"Altitude AGL       : "
            f"{altitude:.6f} m\n\n"
        )

        f.write(
            "ROI\n"
        )

        f.write(
            f"HFOV               : "
            f"{HFOV_DEG:.6f} deg\n"
        )

        f.write(
            f"VFOV               : "
            f"{vfov_deg:.6f} deg\n"
        )

        f.write(
            f"Margin             : "
            f"{ROI_MARGIN_PX} px\n"
        )

        f.write(
            f"ROI x1             : "
            f"{roi_x1}\n"
        )

        f.write(
            f"ROI y1             : "
            f"{roi_y1}\n"
        )

        f.write(
            f"ROI x2             : "
            f"{roi_x2}\n"
        )

        f.write(
            f"ROI y2             : "
            f"{roi_y2}\n"
        )

        f.write(
            f"ROI width          : "
            f"{roi_width}\n"
        )

        f.write(
            f"ROI height         : "
            f"{roi_height}\n\n"
        )

        f.write(
            "SUPERPOINT / SUPERGLUE\n"
        )

        f.write(
            f"Device             : "
            f"{device}\n"
        )

        f.write(
            f"Matching ROI width : "
            f"{matching_width}\n"
        )

        f.write(
            f"Matching ROI height: "
            f"{matching_height}\n"
        )

        f.write(
            f"Resize scale       : "
            f"{resize_scale:.10f}\n"
        )

        f.write(
            f"Valid matches      : "
            f"{total_matches}\n"
        )

        f.write(
            f"Mean confidence    : "
            f"{mean_confidence:.6f}\n"
        )

        f.write(
            f"Matching time      : "
            f"{matching_time:.6f} s\n\n"
        )

        f.write(
            "RANSAC\n"
        )

        f.write(
            f"Inliers            : "
            f"{inliers}\n"
        )

        f.write(
            f"Outliers           : "
            f"{outliers}\n"
        )

        f.write(
            f"Inlier ratio       : "
            f"{inlier_ratio * 100:.6f} %\n"
        )

        f.write(
            f"Mean inlier confidence: "
            f"{mean_inlier_confidence:.6f}\n"
        )

        f.write(
            f"RANSAC time        : "
            f"{ransac_time:.6f} s\n\n"
        )

        f.write(
            "HOMOGRAPHY\n"
        )

        f.write(
            f"{H}\n\n"
        )

        f.write(
            "VISUAL POSITION\n"
        )

        f.write(
            f"Frame center X     : "
            f"{frame_center_x:.6f}\n"
        )

        f.write(
            f"Frame center Y     : "
            f"{frame_center_y:.6f}\n"
        )

        f.write(
            f"Matching ROI X     : "
            f"{matching_x:.6f}\n"
        )

        f.write(
            f"Matching ROI Y     : "
            f"{matching_y:.6f}\n"
        )

        f.write(
            f"Original ROI X     : "
            f"{original_roi_x:.6f}\n"
        )

        f.write(
            f"Original ROI Y     : "
            f"{original_roi_y:.6f}\n"
        )

        f.write(
            f"Map column         : "
            f"{map_column:.6f}\n"
        )

        f.write(
            f"Map row            : "
            f"{map_row:.6f}\n"
        )

        f.write(
            f"Latitude           : "
            f"{estimated_latitude:.10f}\n"
        )

        f.write(
            f"Longitude          : "
            f"{estimated_longitude:.10f}\n\n"
        )

        f.write(
            "GPS VS VISUAL\n"
        )

        f.write(
            f"East difference    : "
            f"{east_difference_m:.6f} m\n"
        )

        f.write(
            f"North difference   : "
            f"{north_difference_m:.6f} m\n"
        )

        f.write(
            f"Horizontal difference: "
            f"{horizontal_difference_m:.6f} m\n\n"
        )

        f.write(
            "PROCESSING TIME\n"
        )

        f.write(
            f"SuperGlue          : "
            f"{matching_time:.6f} s\n"
        )

        f.write(
            f"RANSAC             : "
            f"{ransac_time:.6f} s\n"
        )

        f.write(
            f"Total              : "
            f"{total_time:.6f} s\n"
        )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_FILE}"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  {ORIGINAL_ROI_FILE}"
    )

    print(
        f"  {MATCHING_ROI_FILE}"
    )

    print(
        f"  {MATCHES_NPZ}"
    )

    print(
        f"  {MATCHES_VIS}"
    )

    print(
        f"  {RANSAC_NPZ}"
    )

    print(
        f"  {RANSAC_VIS}"
    )

    print(
        f"  {POSITION_VIS}"
    )

    print(
        f"  {REPORT_FILE}"
    )

    src.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
