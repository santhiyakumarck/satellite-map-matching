```python
#!/usr/bin/env python3

# ================================================================
# MASTER REAL-TIME SATELLITE MAP MATCHING SCRIPT
#
# CHANGE ONLY THESE 3 INPUTS:
#   FRAME_FILE
#   GPS_FILE
#   MAP_FILE
#
# Everything else is automatic:
#   1. Load camera frame
#   2. Detect frame number
#   3. Read GPS CSV
#   4. Match frame timestamp with GPS
#   5. Automatically find relative altitude / AGL
#   6. Calculate camera FOV footprint
#   7. Read GeoTIFF GSD
#   8. Calculate dynamic satellite ROI
#   9. SuperPoint
#  10. SuperGlue Outdoor
#  11. RANSAC Homography
#  12. Estimate GPS position
#  13. Calculate GPS error
#  14. Save visualizations
#  15. Measure processing time
#
# Designed for:
#   Raspberry Pi 5 / Ubuntu / Python 3
# ================================================================


import os
import re
import sys
import math
import time
import traceback

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch

from rasterio.transform import rowcol, xy
from rasterio.warp import transform


# ================================================================
#                    ONLY CHANGE THESE
# ================================================================

FRAME_FILE = "frame_06898.jpg"
GPS_FILE   = "gps.csv"
MAP_FILE   = "Satellite_Z21.tif"


# ================================================================
#                    FIXED PARAMETERS
# ================================================================

CAMERA_FOV_DEG = 90.0

CAMERA_FPS = 30.0

# Frame numbering starts from zero.
# Change this ONLY if your extracted frame numbering is different.
FRAME_NUMBER_OVERRIDE = None

# Dynamic ROI margin.
#
# 1.00 = exact camera footprint
# 1.20 = 20% extra satellite area
#
# A small margin is useful because GPS is not perfectly accurate.
ROI_MARGIN = 1.20

# SuperPoint / SuperGlue settings
MAX_KEYPOINTS = 1024

SUPERGLUE_WEIGHTS = "outdoor"

MATCH_THRESHOLD = 0.10

# RANSAC parameters
RANSAC_REPROJ_THRESHOLD = 5.0

# Minimum acceptable result
MIN_RANSAC_INLIERS = 5
MIN_INLIER_RATIO = 0.20


# ================================================================
#                    DIRECTORY SETUP
# ================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FRAME_PATH = os.path.join(SCRIPT_DIR, FRAME_FILE)
GPS_PATH = os.path.join(SCRIPT_DIR, GPS_FILE)
MAP_PATH = os.path.join(SCRIPT_DIR, MAP_FILE)

# Automatically create result directory
frame_stem = os.path.splitext(os.path.basename(FRAME_FILE))[0]

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    f"results_{frame_stem}"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ================================================================
#                    SUPERGLUE IMPORT
# ================================================================

SUPERGLUE_DIR = os.path.join(
    SCRIPT_DIR,
    "SuperGluePretrainedNetwork"
)

if SUPERGLUE_DIR not in sys.path:
    sys.path.insert(0, SUPERGLUE_DIR)

try:
    from models.matching import Matching
except Exception as e:

    print("\nERROR: Could not import SuperGlue.")
    print("Expected folder:")
    print(SUPERGLUE_DIR)
    print("\nOriginal error:")
    print(e)

    sys.exit(1)


# ================================================================
#                    GPU SYNCHRONIZATION
# ================================================================

def synchronize_cuda():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


# ================================================================
#                    TIMESTAMP HELPERS
# ================================================================

def get_frame_number():

    global FRAME_NUMBER_OVERRIDE

    if FRAME_NUMBER_OVERRIDE is not None:
        return int(FRAME_NUMBER_OVERRIDE)

    # Example:
    # frame_06898.jpg
    # frame_1654.jpg

    match = re.search(
        r"(\d+)",
        os.path.splitext(
            os.path.basename(FRAME_FILE)
        )[0]
    )

    if match:
        return int(match.group(1))

    return None


def get_frame_timestamp():

    frame_number = get_frame_number()

    if frame_number is None:
        return None

    timestamp_sec = frame_number / CAMERA_FPS

    return timestamp_sec


# ================================================================
#                    FIND TIME COLUMN
# ================================================================

def find_timestamp_column(df):

    candidates = [

        "bag_timestamp",

        "timestamp",

        "timestamp_ns",

        "time_ns",

        "gps_timestamp",

    ]

    for c in candidates:

        if c in df.columns:
            return c

    return None


# ================================================================
#                    FIND GPS COLUMNS
# ================================================================

def find_column(df, candidates):

    lower_map = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


def find_lat_lon_columns(df):

    lat_col = find_column(
        df,
        [
            "latitude",
            "lat",
            "gps_latitude",
            "gps_lat",
        ]
    )

    lon_col = find_column(
        df,
        [
            "longitude",
            "lon",
            "lng",
            "gps_longitude",
            "gps_lon",
        ]
    )

    return lat_col, lon_col


# ================================================================
#              AUTOMATIC ALTITUDE COLUMN SEARCH
# ================================================================

def find_relative_altitude_column(df):

    """
    Search for a REAL relative altitude / AGL column.

    Priority is given to names such as:

        rel_alt
        relative_altitude
        altitude_agl
        height_agl
        agl
        camera_height
        height
        relative_height

    The normal 'altitude' GPS column is deliberately NOT
    selected here because it may represent absolute altitude.
    """

    priority = [

        "rel_alt",
        "relative_altitude",
        "relative_alt",
        "relalt",

        "altitude_agl",
        "height_agl",
        "agl",

        "camera_height",
        "camera_altitude",

        "relative_height",
        "height",

        "altitude_relative",

    ]

    lower_map = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for name in priority:

        if name in lower_map:

            return lower_map[name]

    # Fuzzy search
    for c in df.columns:

        name = str(c).lower().strip()

        if (
            "rel_alt" in name
            or "relative_alt" in name
            or "alt_agl" in name
            or "height_agl" in name
            or name == "agl"
        ):

            return c

    return None


# ================================================================
#           AUTOMATIC SEARCH FOR ALTITUDE CSV
# ================================================================

def find_altitude_from_other_csv(
    main_gps_path
):

    """
    Search other CSV files in the same directory.

    This allows:

        gps.csv

    to contain latitude/longitude while another CSV such as:

        gps_data.csv

    contains relative altitude.

    The user does NOT need to change another input.
    """

    csv_files = []

    for filename in os.listdir(SCRIPT_DIR):

        if filename.lower().endswith(".csv"):

            full_path = os.path.join(
                SCRIPT_DIR,
                filename
            )

            if os.path.abspath(full_path) != os.path.abspath(
                main_gps_path
            ):

                csv_files.append(full_path)

    # Prefer files containing altitude-related names
    csv_files.sort(
        key=lambda x: (
            0 if any(
                word in os.path.basename(x).lower()
                for word in [
                    "alt",
                    "rel",
                    "gps",
                    "global",
                ]
            ) else 1,
            x
        )
    )

    for csv_path in csv_files:

        try:

            df = pd.read_csv(csv_path)

            alt_col = find_relative_altitude_column(df)

            if alt_col is not None:

                return csv_path, df, alt_col

        except Exception:
            continue

    return None, None, None


# ================================================================
#                    CONVERT TIMESTAMPS
# ================================================================

def create_time_seconds(df):

    """
    Convert available timestamps to seconds.

    Supports nanosecond ROS bag timestamps and
    normal second timestamps.
    """

    ts_col = find_timestamp_column(df)

    if ts_col is None:

        return None, None

    values = pd.to_numeric(
        df[ts_col],
        errors="coerce"
    )

    valid = values.dropna()

    if len(valid) == 0:

        return None, None

    median_value = float(
        valid.median()
    )

    # Nanoseconds
    if median_value > 1e16:

        seconds = values / 1e9

    # Microseconds
    elif median_value > 1e13:

        seconds = values / 1e6

    # Milliseconds
    elif median_value > 1e10:

        seconds = values / 1e3

    else:

        seconds = values.astype(float)

    return seconds, ts_col


# ================================================================
#                    GPS MATCHING
# ================================================================

def get_nearest_gps(
    gps_df,
    frame_time_sec
):

    time_seconds, time_col = create_time_seconds(
        gps_df
    )

    if time_seconds is None:

        raise RuntimeError(
            "No usable timestamp column found in GPS CSV."
        )

    lat_col, lon_col = find_lat_lon_columns(
        gps_df
    )

    if lat_col is None or lon_col is None:

        raise RuntimeError(
            "Latitude / longitude columns not found."
        )

    valid_mask = (
        time_seconds.notna()
        & pd.to_numeric(
            gps_df[lat_col],
            errors="coerce"
        ).notna()
        & pd.to_numeric(
            gps_df[lon_col],
            errors="coerce"
        ).notna()
    )

    if valid_mask.sum() == 0:

        raise RuntimeError(
            "No valid GPS records."
        )

    valid_indices = np.where(
        valid_mask.values
    )[0]

    valid_times = time_seconds.iloc[
        valid_indices
    ].values

    nearest_local = int(
        np.argmin(
            np.abs(
                valid_times - frame_time_sec
            )
        )
    )

    nearest_index = valid_indices[
        nearest_local
    ]

    row = gps_df.iloc[
        nearest_index
    ]

    gps_time = float(
        time_seconds.iloc[
            nearest_index
        ]
    )

    latitude = float(
        row[lat_col]
    )

    longitude = float(
        row[lon_col]
    )

    time_difference = abs(
        gps_time - frame_time_sec
    )

    return (
        latitude,
        longitude,
        gps_time,
        time_difference,
        nearest_index
    )


# ================================================================
#                  ALTITUDE MATCHING
# ================================================================

def get_nearest_altitude(
    alt_df,
    alt_col,
    frame_time_sec
):

    time_seconds, time_col = create_time_seconds(
        alt_df
    )

    if time_seconds is None:

        raise RuntimeError(
            "Altitude CSV does not contain a usable timestamp."
        )

    values = pd.to_numeric(
        alt_df[alt_col],
        errors="coerce"
    )

    valid_mask = (
        time_seconds.notna()
        & values.notna()
    )

    if valid_mask.sum() == 0:

        raise RuntimeError(
            "No valid relative altitude values."
        )

    valid_indices = np.where(
        valid_mask.values
    )[0]

    valid_times = time_seconds.iloc[
        valid_indices
    ].values

    nearest_local = int(
        np.argmin(
            np.abs(
                valid_times - frame_time_sec
            )
        )
    )

    nearest_index = valid_indices[
        nearest_local
    ]

    altitude = float(
        values.iloc[
            nearest_index
        ]
    )

    altitude_time = float(
        time_seconds.iloc[
            nearest_index
        ]
    )

    time_difference = abs(
        altitude_time - frame_time_sec
    )

    return (
        altitude,
        altitude_time,
        time_difference
    )


# ================================================================
#                    GEO TIFF INFORMATION
# ================================================================

def get_map_information(src):

    width = src.width
    height = src.height

    transform_map = src.transform

    bounds = src.bounds

    crs = src.crs

    # Pixel size in CRS units
    pixel_x = abs(
        transform_map.a
    )

    pixel_y = abs(
        transform_map.e
    )

    return (
        width,
        height,
        transform_map,
        bounds,
        crs,
        pixel_x,
        pixel_y
    )


# ================================================================
#          CONVERT GEO TIFF PIXEL SIZE TO METERS
# ================================================================

def calculate_gsd_meters(
    src,
    latitude
):

    """
    Convert GeoTIFF pixel size to meters.

    If CRS is projected in meters:
        direct conversion.

    If CRS is EPSG:4326:
        degree/pixel -> meters/pixel.
    """

    pixel_x = abs(
        src.transform.a
    )

    pixel_y = abs(
        src.transform.e
    )

    crs = src.crs

    if crs is None:

        raise RuntimeError(
            "GeoTIFF has no CRS."
        )

    # Geographic CRS
    if crs.is_geographic:

        lat_rad = math.radians(
            latitude
        )

        meters_per_degree_lat = (
            111132.92
            - 559.82 * math.cos(2 * lat_rad)
            + 1.175 * math.cos(4 * lat_rad)
        )

        meters_per_degree_lon = (
            111412.84 * math.cos(lat_rad)
            - 93.5 * math.cos(3 * lat_rad)
        )

        gsd_x = (
            pixel_x
            * meters_per_degree_lon
        )

        gsd_y = (
            pixel_y
            * meters_per_degree_lat
        )

    else:

        # Projected CRS
        gsd_x = pixel_x
        gsd_y = pixel_y

    return gsd_x, gsd_y


# ================================================================
#              GPS -> MAP PIXEL COORDINATE
# ================================================================

def gps_to_pixel(
    src,
    latitude,
    longitude
):

    map_crs = src.crs

    if map_crs is None:

        raise RuntimeError(
            "GeoTIFF CRS missing."
        )

    # GPS coordinates are WGS84
    if str(map_crs).upper() not in [
        "EPSG:4326"
    ]:

        xs, ys = transform(
            "EPSG:4326",
            map_crs,
            [longitude],
            [latitude]
        )

        x = xs[0]
        y = ys[0]

    else:

        x = longitude
        y = latitude

    row, col = rowcol(
        src.transform,
        x,
        y
    )

    return int(row), int(col)


# ================================================================
#                    PIXEL -> GPS
# ================================================================

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

    map_crs = src.crs

    if map_crs.is_geographic:

        longitude = x
        latitude = y

    else:

        lons, lats = transform(
            map_crs,
            "EPSG:4326",
            [x],
            [y]
        )

        longitude = lons[0]
        latitude = lats[0]

    return (
        float(latitude),
        float(longitude)
    )


# ================================================================
#                    DYNAMIC ROI
# ================================================================

def calculate_dynamic_roi(
    frame_width,
    frame_height,
    altitude_m,
    fov_deg,
    gsd_x,
    gsd_y,
    gps_row,
    gps_col,
    map_width,
    map_height
):

    if altitude_m <= 0:

        raise RuntimeError(
            f"Invalid relative altitude: {altitude_m:.3f} m"
        )

    fov_x_rad = math.radians(
        fov_deg
    )

    # Camera vertical FOV calculated from
    # horizontal FOV and image aspect ratio.
    fov_y_rad = 2.0 * math.atan(
        (
            frame_height
            / frame_width
        )
        * math.tan(
            fov_x_rad / 2.0
        )
    )

    # Ground footprint
    ground_width_m = (
        2.0
        * altitude_m
        * math.tan(
            fov_x_rad / 2.0
        )
    )

    ground_height_m = (
        2.0
        * altitude_m
        * math.tan(
            fov_y_rad / 2.0
        )
    )

    # Apply margin
    ground_width_m *= ROI_MARGIN
    ground_height_m *= ROI_MARGIN

    roi_width_px = int(
        math.ceil(
            ground_width_m / gsd_x
        )
    )

    roi_height_px = int(
        math.ceil(
            ground_height_m / gsd_y
        )
    )

    # Ensure ROI is at least reasonable
    roi_width_px = max(
        roi_width_px,
        64
    )

    roi_height_px = max(
        roi_height_px,
        64
    )

    # Center around GPS
    x1 = int(
        gps_col
        - roi_width_px / 2
    )

    y1 = int(
        gps_row
        - roi_height_px / 2
    )

    x2 = x1 + roi_width_px
    y2 = y1 + roi_height_px

    # Clip to map boundaries
    x1 = max(
        0,
        min(
            x1,
            map_width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            map_height - 1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            x2,
            map_width
        )
    )

    y2 = max(
        y1 + 1,
        min(
            y2,
            map_height
        )
    )

    actual_width = x2 - x1
    actual_height = y2 - y1

    return {
        "fov_x_deg": fov_deg,
        "fov_y_deg": math.degrees(
            fov_y_rad
        ),

        "ground_width_m": ground_width_m,
        "ground_height_m": ground_height_m,

        "roi_width_px": actual_width,
        "roi_height_px": actual_height,

        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }


# ================================================================
#                    READ SATELLITE ROI
# ================================================================

def read_satellite_roi(
    src,
    roi
):

    window = rasterio.windows.Window(
        roi["x1"],
        roi["y1"],
        roi["roi_width_px"],
        roi["roi_height_px"]
    )

    data = src.read(
        window=window
    )

    # RGB
    if data.shape[0] >= 3:

        image = np.transpose(
            data[:3],
            (1, 2, 0)
        )

    elif data.shape[0] == 1:

        image = data[0]

    else:

        raise RuntimeError(
            "Satellite image has unsupported band count."
        )

    # Convert to uint8
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


# ================================================================
#                    IMAGE PREPARATION
# ================================================================

def prepare_gray(image):

    if len(image.shape) == 2:

        gray = image

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    return gray


def resize_for_matching(
    image,
    max_dimension=1200
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / max(
            h,
            w
        )
    )

    if scale == 1.0:

        return image

    new_w = int(
        w * scale
    )

    new_h = int(
        h * scale
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ================================================================
#                    SUPERPOINT
# ================================================================

def run_superpoint(
    matching,
    image0,
    image1,
    device
):

    image0_gray = prepare_gray(
        image0
    )

    image1_gray = prepare_gray(
        image1
    )

    # Convert to tensor
    inp0 = torch.from_numpy(
        image0_gray
    ).float() / 255.0

    inp1 = torch.from_numpy(
        image1_gray
    ).float() / 255.0

    inp0 = inp0[
        None,
        None
    ].to(device)

    inp1 = inp1[
        None,
        None
    ].to(device)

    synchronize_cuda()

    start = time.perf_counter()

    with torch.no_grad():

        pred0 = matching.superpoint({
            "image": inp0
        })

        pred1 = matching.superpoint({
            "image": inp1
        })

    synchronize_cuda()

    elapsed = (
        time.perf_counter()
        - start
    )

    kpts0 = pred0[
        "keypoints"
    ][0].detach().cpu().numpy()

    kpts1 = pred1[
        "keypoints"
    ][0].detach().cpu().numpy()

    scores0 = pred0[
        "scores"
    ][0].detach().cpu().numpy()

    scores1 = pred1[
        "scores"
    ][0].detach().cpu().numpy()

    desc0 = pred0[
        "descriptors"
    ][0].detach().cpu()

    desc1 = pred1[
        "descriptors"
    ][0].detach().cpu()

    return (
        elapsed,
        kpts0,
        kpts1,
        scores0,
        scores1,
        desc0,
        desc1
    )


# ================================================================
#                    SUPERGLUE
# ================================================================

def run_superglue(
    matching,
    kpts0,
    kpts1,
    scores0,
    scores1,
    desc0,
    desc1,
    image0_shape,
    image1_shape,
    device
):

    # Convert keypoints/descriptors
    data = {

        "keypoints0": torch.from_numpy(
            kpts0
        ).float().unsqueeze(0).to(device),

        "keypoints1": torch.from_numpy(
            kpts1
        ).float().unsqueeze(0).to(device),

        "scores0": torch.from_numpy(
            scores0
        ).float().unsqueeze(0).to(device),

        "scores1": torch.from_numpy(
            scores1
        ).float().unsqueeze(0).to(device),

        "descriptors0": desc0.unsqueeze(0).to(device),

        "descriptors1": desc1.unsqueeze(0).to(device),

    }

    synchronize_cuda()

    start = time.perf_counter()

    with torch.no_grad():

        pred = matching.superglue(
            data
        )

    synchronize_cuda()

    elapsed = (
        time.perf_counter()
        - start
    )

    matches0 = pred[
        "matches0"
    ][0].detach().cpu().numpy()

    matching_scores0 = pred[
        "matching_scores0"
    ][0].detach().cpu().numpy()

    return (
        elapsed,
        matches0,
        matching_scores0
    )


# ================================================================
#                    RANSAC
# ================================================================

def run_ransac(
    kpts0,
    kpts1,
    matches0
):

    valid = (
        matches0 > -1
    )

    matched_kpts0 = kpts0[
        valid
    ]

    matched_kpts1 = kpts1[
        matches0[valid]
    ]

    if len(matched_kpts0) < 4:

        return (
            None,
            matched_kpts0,
            matched_kpts1,
            np.zeros(
                len(matched_kpts0),
                dtype=bool
            ),
            0.0
        )

    start = time.perf_counter()

    H, mask = cv2.findHomography(
        matched_kpts0,
        matched_kpts1,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if mask is None:

        mask = np.zeros(
            len(matched_kpts0),
            dtype=np.uint8
        )

    mask = mask.ravel().astype(bool)

    inlier_count = int(
        np.sum(mask)
    )

    if len(mask) > 0:

        ratio = (
            inlier_count
            / len(mask)
        )

    else:

        ratio = 0.0

    return (
        H,
        matched_kpts0,
        matched_kpts1,
        mask,
        elapsed
    )


# ================================================================
#                POSITION FROM HOMOGRAPHY
# ================================================================

def estimate_satellite_position(
    H,
    frame_shape,
    roi
):

    if H is None:

        return None, None

    h, w = frame_shape[:2]

    center = np.array(
        [
            [
                [
                    w / 2.0,
                    h / 2.0
                ]
            ]
        ],
        dtype=np.float32
    )

    projected = cv2.perspectiveTransform(
        center,
        H
    )

    x = float(
        projected[0, 0, 0]
    )

    y = float(
        projected[0, 0, 1]
    )

    # Satellite ROI coordinates -> full GeoTIFF
    full_col = (
        roi["x1"]
        + x
    )

    full_row = (
        roi["y1"]
        + y
    )

    return (
        full_row,
        full_col
    )


# ================================================================
#                    HAVERSINE ERROR
# ================================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2.0) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2.0) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(1.0 - a)
    )

    return R * c


# ================================================================
#                    VISUALIZATION
# ================================================================

def save_keypoints(
    image,
    keypoints,
    output_path
):

    if len(image.shape) == 2:

        vis = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        vis = image.copy()

    for point in keypoints:

        x, y = point

        cv2.circle(
            vis,
            (
                int(round(x)),
                int(round(y))
            ),
            2,
            (0, 255, 0),
            -1
        )

    cv2.imwrite(
        output_path,
        vis
    )


def save_matches(
    image0,
    image1,
    kpts0,
    kpts1,
    matches0,
    output_path,
    inlier_mask=None
):

    img0 = image0.copy()
    img1 = image1.copy()

    if len(img0.shape) == 2:

        img0 = cv2.cvtColor(
            img0,
            cv2.COLOR_GRAY2BGR
        )

    if len(img1.shape) == 2:

        img1 = cv2.cvtColor(
            img1,
            cv2.COLOR_GRAY2BGR
        )

    h0, w0 = img0.shape[:2]

    h1, w1 = img1.shape[:2]

    canvas_h = max(
        h0,
        h1
    )

    canvas_w = w0 + w1

    canvas = np.zeros(
        (
            canvas_h,
            canvas_w,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :h0,
        :w0
    ] = img0

    canvas[
        :h1,
        w0:w0 + w1
    ] = img1

    valid_indices = np.where(
        matches0 > -1
    )[0]

    for counter, idx0 in enumerate(
        valid_indices
    ):

        idx1 = matches0[
            idx0
        ]

        if (
            inlier_mask is not None
            and counter < len(inlier_mask)
        ):

            is_inlier = bool(
                inlier_mask[counter]
            )

        else:

            is_inlier = True

        if is_inlier:

            thickness = 1

        else:

            thickness = 1

        p0 = kpts0[
            idx0
        ]

        p1 = kpts1[
            idx1
        ]

        x0 = int(
            round(p0[0])
        )

        y0 = int(
            round(p0[1])
        )

        x1 = int(
            round(p1[0])
        ) + w0

        y1 = int(
            round(p1[1])
        )

        # Green for inliers, red for outliers
        if is_inlier:

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                0,
                0,
                255
            )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            color,
            thickness
        )

        cv2.circle(
            canvas,
            (x0, y0),
            2,
            color,
            -1
        )

        cv2.circle(
            canvas,
            (x1, y1),
            2,
            color,
            -1
        )

    cv2.imwrite(
        output_path,
        canvas
    )


def save_final_position(
    satellite_image,
    gps_pixel,
    estimated_pixel,
    output_path
):

    if len(satellite_image.shape) == 2:

        vis = cv2.cvtColor(
            satellite_image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        vis = satellite_image.copy()

    if gps_pixel is not None:

        gps_row, gps_col = gps_pixel

        cv2.circle(
            vis,
            (
                int(gps_col),
                int(gps_row)
            ),
            8,
            (255, 0, 0),
            3
        )

        cv2.putText(
            vis,
            "GPS",
            (
                int(gps_col) + 10,
                int(gps_row)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    if estimated_pixel is not None:

        est_row, est_col = estimated_pixel

        cv2.drawMarker(
            vis,
            (
                int(est_col),
                int(est_row)
            ),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            30,
            3
        )

        cv2.putText(
            vis,
            "ESTIMATED",
            (
                int(est_col) + 10,
                int(est_row)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        output_path,
        vis
    )


# ================================================================
#                    MAIN
# ================================================================

def main():

    total_start = time.perf_counter()

    print("\n")
    print("=" * 70)
    print("        SATELLITE MAP MATCHING")
    print("        RASPBERRY PI 5")
    print("=" * 70)

    print("\nINPUTS")
    print("-" * 70)

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

    print(
        "FOV   :",
        CAMERA_FOV_DEG,
        "degrees"
    )

    print(
        "Result:",
        RESULT_DIR
    )


    # ============================================================
    # CHECK FILES
    # ============================================================

    for path in [
        FRAME_PATH,
        GPS_PATH,
        MAP_PATH
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile not found:\n{path}"
            )


    # ============================================================
    # LOAD FRAME
    # ============================================================

    frame = cv2.imread(
        FRAME_PATH,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise RuntimeError(
            "Could not read camera frame."
        )

    frame_h, frame_w = frame.shape[:2]

    print("\nFRAME")
    print("-" * 70)

    print(
        "Resolution:",
        frame_w,
        "x",
        frame_h
    )

    frame_number = get_frame_number()

    print(
        "Frame number:",
        frame_number
    )

    frame_time_sec = get_frame_timestamp()

    if frame_time_sec is None:

        raise RuntimeError(
            "Could not determine frame timestamp."
        )

    print(
        "Frame time:",
        f"{frame_time_sec:.6f}",
        "sec"
    )


    # ============================================================
    # LOAD GPS CSV
    # ============================================================

    print("\nGPS")
    print("-" * 70)

    gps_df = pd.read_csv(
        GPS_PATH
    )

    print(
        "GPS rows:",
        len(gps_df)
    )

    (
        gps_lat,
        gps_lon,
        gps_time,
        gps_dt,
        gps_index
    ) = get_nearest_gps(
        gps_df,
        frame_time_sec
    )

    print(
        "Latitude :",
        f"{gps_lat:.8f}"
    )

    print(
        "Longitude:",
        f"{gps_lon:.8f}"
    )

    print(
        "GPS index:",
        gps_index
    )

    print(
        "GPS time difference:",
        f"{gps_dt * 1000:.3f} ms"
    )


    # ============================================================
    # FIND RELATIVE ALTITUDE
    # ============================================================

    print("\nALTITUDE")
    print("-" * 70)

    altitude_df = None
    altitude_col = None
    altitude_source = None

    # First search the selected GPS CSV
    altitude_col = find_relative_altitude_column(
        gps_df
    )

    if altitude_col is not None:

        altitude_df = gps_df
        altitude_source = GPS_PATH

    else:

        # Search another CSV automatically
        (
            altitude_source,
            altitude_df,
            altitude_col
        ) = find_altitude_from_other_csv(
            GPS_PATH
        )

    if altitude_df is None:

        print(
            "\nERROR: No relative altitude / AGL column found."
        )

        print(
            "\nAvailable CSV files:"
        )

        for filename in os.listdir(
            SCRIPT_DIR
        ):

            if filename.lower().endswith(
                ".csv"
            ):

                print(
                    "  ",
                    filename
                )

        print(
            "\nThe normal 'altitude' column is NOT used"
        )

        print(
            "because it is not guaranteed to be camera AGL."
        )

        print(
            "\nRequired column examples:"
        )

        print(
            "  rel_alt"
        )

        print(
            "  relative_altitude"
        )

        print(
            "  altitude_agl"
        )

        print(
            "  height_agl"
        )

        print(
            "  agl"
        )

        sys.exit(1)

    print(
        "Altitude CSV:",
        altitude_source
    )

    print(
        "Altitude column:",
        altitude_col
    )

    (
        relative_altitude,
        altitude_time,
        altitude_dt
    ) = get_nearest_altitude(
        altitude_df,
        altitude_col,
        frame_time_sec
    )

    print(
        "Relative altitude:",
        f"{relative_altitude:.3f} m"
    )

    print(
        "Altitude time difference:",
        f"{altitude_dt * 1000:.3f} ms"
    )

    if relative_altitude <= 0:

        raise RuntimeError(
            "\nRelative altitude is <= 0 m.\n"
            f"Detected value: {relative_altitude:.3f} m\n"
            "Cannot calculate camera ground footprint."
        )


    # ============================================================
    # OPEN SATELLITE MAP
    # ============================================================

    print("\nSATELLITE MAP")
    print("-" * 70)

    src = rasterio.open(
        MAP_PATH
    )

    (
        map_width,
        map_height,
        map_transform,
        map_bounds,
        map_crs,
        pixel_x,
        pixel_y
    ) = get_map_information(
        src
    )

    print(
        "Map size:",
        map_width,
        "x",
        map_height
    )

    print(
        "CRS:",
        map_crs
    )

    print(
        "Bounds:",
        map_bounds
    )

    print(
        "Pixel size:",
        pixel_x,
        pixel_y
    )


    # ============================================================
    # GSD
    # ============================================================

    gsd_x, gsd_y = calculate_gsd_meters(
        src,
        gps_lat
    )

    print(
        "GSD X:",
        f"{gsd_x:.4f} m/pixel"
    )

    print(
        "GSD Y:",
        f"{gsd_y:.4f} m/pixel"
    )


    # ============================================================
    # GPS -> SATELLITE PIXEL
    # ============================================================

    gps_row, gps_col = gps_to_pixel(
        src,
        gps_lat,
        gps_lon
    )

    print(
        "GPS map pixel:",
        gps_row,
        gps_col
    )


    # ============================================================
    # CHECK GPS INSIDE MAP
    # ============================================================

    if not (
        0 <= gps_row < map_height
        and
        0 <= gps_col < map_width
    ):

        raise RuntimeError(
            "\nGPS location is outside the GeoTIFF."
        )


    # ============================================================
    # DYNAMIC ROI
    # ============================================================

    print("\nDYNAMIC ROI")
    print("-" * 70)

    roi_start = time.perf_counter()

    roi = calculate_dynamic_roi(
        frame_w,
        frame_h,
        relative_altitude,
        CAMERA_FOV_DEG,
        gsd_x,
        gsd_y,
        gps_row,
        gps_col,
        map_width,
        map_height
    )

    satellite_roi = read_satellite_roi(
        src,
        roi
    )

    roi_time = (
        time.perf_counter()
        - roi_start
    )

    print(
        "Horizontal FOV:",
        f"{roi['fov_x_deg']:.2f}",
        "deg"
    )

    print(
        "Vertical FOV:",
        f"{roi['fov_y_deg']:.2f}",
        "deg"
    )

    print(
        "Ground width:",
        f"{roi['ground_width_m']:.2f}",
        "m"
    )

    print(
        "Ground height:",
        f"{roi['ground_height_m']:.2f}",
        "m"
    )

    print(
        "ROI size:",
        roi["roi_width_px"],
        "x",
        roi["roi_height_px"],
        "pixels"
    )

    print(
        "ROI full-map:",
        "(",
        roi["x1"],
        ",",
        roi["y1"],
        ") -> (",
        roi["x2"],
        ",",
        roi["y2"],
        ")"
    )

    print(
        "ROI extraction:",
        f"{roi_time:.4f}",
        "sec"
    )


    # ============================================================
    # SAVE ROI
    # ============================================================

    cv2.imwrite(
        os.path.join(
            RESULT_DIR,
            "satellite_roi.png"
        ),
        satellite_roi
    )


    # ============================================================
    # PREPARE MATCHING IMAGES
    # ============================================================

    # Keep aspect ratio but reduce very large satellite ROI
    # for Raspberry Pi processing.
    #
    # Camera frame is kept at original size.

    frame_match = resize_for_matching(
        frame,
        max_dimension=1200
    )

    satellite_match = resize_for_matching(
        satellite_roi,
        max_dimension=1200
    )


    # ============================================================
    # DEVICE
    # ============================================================

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print("\nDEVICE")
    print("-" * 70)

    print(
        "PyTorch device:",
        device
    )


    # ============================================================
    # LOAD SUPERPOINT + SUPERGLUE
    # ============================================================

    print("\nLOADING MODELS")
    print("-" * 70)

    model_start = time.perf_counter()

    config = {

        "superpoint": {

            "nms_radius": 4,

            "keypoint_threshold": 0.005,

            "max_keypoints": MAX_KEYPOINTS,

        },

        "superglue": {

            "weights": SUPERGLUE_WEIGHTS,

            "sinkhorn_iterations": 20,

            "match_threshold": MATCH_THRESHOLD,

        }

    }

    matching = Matching(
        config
    ).eval().to(device)

    synchronize_cuda()

    model_time = (
        time.perf_counter()
        - model_start
    )

    print(
        "Model initialization:",
        f"{model_time:.4f}",
        "sec"
    )


    # ============================================================
    # SUPERPOINT
    # ============================================================

    print("\nSUPERPOINT")
    print("-" * 70)

    (
        superpoint_time,
        kpts0,
        kpts1,
        scores0,
        scores1,
        desc0,
        desc1
    ) = run_superpoint(
        matching,
        frame_match,
        satellite_match,
        device
    )

    print(
        "Drone keypoints:",
        len(kpts0)
    )

    print(
        "Satellite keypoints:",
        len(kpts1)
    )

    print(
        "SuperPoint time:",
        f"{superpoint_time:.4f}",
        "sec"
    )


    # Save keypoints
    save_keypoints(
        frame_match,
        kpts0,
        os.path.join(
            RESULT_DIR,
            "superpoint_keypoints_drone.png"
        )
    )

    save_keypoints(
        satellite_match,
        kpts1,
        os.path.join(
            RESULT_DIR,
            "superpoint_keypoints_satellite.png"
        )
    )


    # ============================================================
    # SUPERGLUE
    # ============================================================

    print("\nSUPERGLUE")
    print("-" * 70)

    (
        superglue_time,
        matches0,
        matching_scores0
    ) = run_superglue(
        matching,
        kpts0,
        kpts1,
        scores0,
        scores1,
        desc0,
        desc1,
        frame_match.shape,
        satellite_match.shape,
        device
    )

    valid_matches = (
        matches0 > -1
    )

    number_matches = int(
        np.sum(valid_matches)
    )

    print(
        "Valid matches:",
        number_matches
    )

    print(
        "SuperGlue time:",
        f"{superglue_time:.4f}",
        "sec"
    )


    # ============================================================
    # RANSAC
    # ============================================================

    print("\nRANSAC")
    print("-" * 70)

    (
        H,
        matched_kpts0,
        matched_kpts1,
        inlier_mask,
        ransac_time
    ) = run_ransac(
        kpts0,
        kpts1,
        matches0
    )

    inliers = int(
        np.sum(inlier_mask)
    )

    outliers = (
        len(inlier_mask)
        - inliers
    )

    if len(inlier_mask) > 0:

        inlier_ratio = (
            inliers
            / len(inlier_mask)
        )

    else:

        inlier_ratio = 0.0

    print(
        "Matches:",
        len(inlier_mask)
    )

    print(
        "Inliers:",
        inliers
    )

    print(
        "Outliers:",
        outliers
    )

    print(
        "Inlier ratio:",
        f"{inlier_ratio:.4f}"
    )

    print(
        "RANSAC time:",
        f"{ransac_time:.4f}",
        "sec"
    )


    # ============================================================
    # SAVE MATCH VISUALIZATION
    # ============================================================

    save_matches(
        frame_match,
        satellite_match,
        kpts0,
        kpts1,
        matches0,
        os.path.join(
            RESULT_DIR,
            "superglue_matches.png"
        ),
        None
    )

    save_matches(
        frame_match,
        satellite_match,
        kpts0,
        kpts1,
        matches0,
        os.path.join(
            RESULT_DIR,
            "ransac_inliers.png"
        ),
        inlier_mask
    )


    # ============================================================
    # HOMOGRAPHY
    # ============================================================

    if H is not None:

        np.savetxt(
            os.path.join(
                RESULT_DIR,
                "homography.txt"
            ),
            H
        )

        print(
            "\nHomography:"
        )

        print(H)

    else:

        print(
            "\nHomography could not be calculated."
        )


    # ============================================================
    # POSITION ESTIMATION
    # ============================================================
alignment
ChatGPT can make mistakes. Check important
    print("\nPOSITION")
    print("-" * 70)

    position_start = time.perf_counter()

    estimated_full_pixel = (
        None
    )

    estimated_lat = None
    estimated_lon = None
    position_error = None

    if (
        H is not None
        and inliers >= MIN_RANSAC_INLIERS
        and inlier_ratio >= MIN_INLIER_RATIO
    ):

        estimated_full_pixel = (
            estimate_satellite_position(
                H,
                frame_match.shape,
                roi
            )
        )

        if estimated_full_pixel is not None:

            (
                est_row,
                est_col
            ) = estimated_full_pixel

            # Make sure position stays inside map
            est_row = max(
                0,
                min(
                    est_row,
                    map_height - 1
                )
            )

            est_col = max(
                0,
                min(
                    est_col,
                    map_width - 1
                )
            )

            estimated_full_pixel = (
                est_row,
                est_col
            )

            (
                estimated_lat,
                estimated_lon
            ) = pixel_to_gps(
                src,
                est_row,
                est_col
            )

            position_error = haversine(
                gps_lat,
                gps_lon,
                estimated_lat,
                estimated_lon
            )

    position_time = (
        time.perf_counter()
        - position_start
    )


    # ============================================================
    # SAVE FINAL POSITION IMAGE
    # ============================================================

    # GPS pixel relative to ROI
    gps_roi_row = (
        gps_row
        - roi["y1"]
    )

    gps_roi_col = (
        gps_col
        - roi["x1"]
    )

    estimated_roi_pixel = None

    if estimated_full_pixel is not None:

        estimated_roi_pixel = (
            estimated_full_pixel[0]
            - roi["y1"],

            estimated_full_pixel[1]
            - roi["x1"]
        )

    save_final_position(
        satellite_roi,
        (
            gps_roi_row,
            gps_roi_col
        ),
        estimated_roi_pixel,
        os.path.join(
            RESULT_DIR,
            "final_estimated_position.png"
        )
    )


    # ============================================================
    # CLOSE MAP
    # ============================================================

    src.close()alignment
ChatGPT can make mistakes. Check important


    # ============================================================
    # TOTAL TIME
    # ============================================================

    total_time = (
        time.perf_counter()
        - total_start
    )


    # ============================================================
    # FINAL RESULT
    # ============================================================
alignment
ChatGPT can make mistakes. Check important
    print("\n")
    print("=" * 70)
    print("                    FINAL RESULT")
    print("=" * 70)

    print(
        "\nGround truth GPS:"
    )

    print(
        "Latitude :",
        f"{gps_lat:.8f}"
    )

    print(
        "Longitude:",
        f"{gps_lon:.8f}"
    )

    print(
        "\nRelative altitude:",
        f"{relative_altitude:.3f} m"
    )

    print(
        "\nDynamic ROI:"
    )

    print(
        "Width :",
        roi["roi_width_px"],
        "pixels"
    )

    print(
        "Height:",
        roi["roi_height_px"],
        "pixels"
    )

    print(
        "\nSuperPoint keypoints:"
    )

    print(
        "Drone    :",
        len(kpts0)
    )

    print(
        "Satellite:",
        len(kpts1)
    )alignment
ChatGPT can make mistakes. Check important

    print(
        "\nSuperGlue matches:",
        number_matches
    )

    print(
        "RANSAC inliers:",
        inliers
    )

    print(
        "RANSAC outliers:",alignment
ChatGPT can make mistakes. Check important
        outliers
    )

    print(
        "Inlier ratio:",
        f"{inlier_ratio:.4f}"
    )

    if estimated_lat is not None:

        print(
            "\nEstimated GPS:"
        )

        print(
            "Latitude :",
            f"{estimated_lat:.8f}"
        )

        print(
            "Longitude:",
            f"{estimated_lon:.8f}"
        )

        print(
            "\nPosition error:",
            f"{position_error:.3f} m"
        )
alignment
ChatGPT can make mistakes. Check important
        if position_error <= 10:

            print(
                "STATUS: GOOD"
            )

        elif position_error <= 30:

            print(
                "STATUS: ACCEPTABLE"
            )

        else:

            print(
                "STATUS: HIGH POSITION ERROR"
            )

    else:

        print(
            "\nEstimated GPS: FAILED"
        )

        print(
            "Reason: insufficient valid RANSAC result."
        )


    # ============================================================
    # PERFORMANCE
    # ============================================================

    print("\n")
    print("=" * 70)
    print("                    PERFORMANCE")
    print("=" * 70)

    print(
        f"{'ROI extraction':20s}",
        f"{roi_time:.4f} sec"
    )

    print(
        f"{'SuperPoint':20s}",
        f"{superpoint_time:.4f} sec"
    )

    print(
        f"{'SuperGlue':20s}",
        f"{superglue_time:.4f} sec"
    )

    print(
        f"{'RANSAC':20s}",
        f"{ransac_time:.4f} sec"
    )alignment
ChatGPT can make mistakes. Check important

    print(
        f"{'Position':20s}",
        f"{position_time:.4f} sec"
    )

    print(
        "-" * 50
    )

    print(
        f"{'Model loading':20s}",
        f"{model_time:.4f} sec"
    )

    print(
        f"{'TOTAL':20s}",
        f"{total_time:.4f} sec"
    )


    # ============================================================
    # SAVE TEXT RESULTS
    # ============================================================

    results_file = os.path.join(
        RESULT_DIR,
        "results.txt"
    )

    with open(
        results_file,
        "w"
    ) as f:

        f.write(
            "SATELLITE MAP MATCHING RESULTS\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Frame: {FRAME_FILE}\n"
        )

        f.write(
            f"GPS CSV: {GPS_FILE}\n"
        )

        f.write(
            f"Map: {MAP_FILE}\n"
        )

        f.write(
            f"Frame number: {frame_number}\n"
        )

        f.write(
            f"Frame time: {frame_time_sec:.9f} sec\n"
        )

        f.write(
            f"GPS latitude: {gps_lat:.10f}\n"
        )

        f.write(
            f"GPS longitude: {gps_lon:.10f}\n"
        )

        f.write(
            f"Relative altitude: "
            f"{relative_altitude:.6f} m\n"
        )

        f.write(
            f"FOV: {CAMERA_FOV_DEG:.3f} deg\n"
        )

        f.write(
            f"Ground footprint width: "
            f"{roi['ground_width_m']:.6f} m\n"
        )

        f.write(
            f"Ground footprint height: "
            f"{roi['ground_height_m']:.6f} m\n"
        )

        f.write(
            f"ROI width: "
            f"{roi['roi_width_px']} px\n"
        )

        f.write(
            f"ROI height: "
            f"{roi['roi_height_px']} px\n"
        )

        f.write(
            f"SuperPoint drone keypoints: "
            f"{len(kpts0)}\n"
        )

        f.write(
            f"SuperPoint satellite keypoints: "
            f"{len(kpts1)}\n"
        )

        f.write(
            f"SuperGlue matches: "
            f"{number_matches}\n"
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
            f"{inlier_ratio:.6f}\n"
        )

        if estimated_lat is not None:

            f.write(
                f"Estimated latitude: "
                f"{estimated_lat:.10f}\n"
            )

            f.write(
                f"Estimated longitude: "
                f"{estimated_lon:.10f}\n"
            )

            f.write(
                f"Position error: "
                f"{position_error:.6f} m\n"
            )

        else:

            f.write(
                "Estimated position: FAILED\n"
            )

        f.write(
            "\nPERFORMANCE\n"
        )

        f.write(
            f"ROI extraction: "
            f"{roi_time:.6f} sec\n"
        )

        f.write(
            f"SuperPoint: "
            f"{superpoint_time:.6f} sec\n"
        )

        f.write(
            f"SuperGlue: "
            f"{superglue_time:.6f} sec\n"
        )

        f.write(
            f"RANSAC: "
            f"{ransac_time:.6f} sec\n"
        )

        f.write(
            f"Position: "
            f"{position_time:.6f} sec\n"
        )

        f.write(
            f"Model loading: "
            f"{model_time:.6f} sec\n"
        )

        f.write(
            f"TOTAL: "
            f"{total_time:.6f} sec\n"
        )


    # ============================================================
    # FINISH
    # ============================================================

    print("\n")
    print("=" * 70)

    print(
        "RESULTS SAVED TO:"
    )

    print(
        RESULT_DIR
    )

    print("\nFiles:")

    for filename in sorted(
        os.listdir(
            RESULT_DIR
        )
    ):

        print(
            "  ",
            filename
        )

    print(
        "\nDONE"
    )

    print("=" * 70)


# ================================================================
#                    ERROR HANDLING
# ================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as e:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(e)
        )

        print(
            "\nTraceback:"
        )

        traceback.print_exc()

        sys.exit(1)
```

The only three lines you need to change for another test are:

```python
FRAME_FILE = "frame_06898.jpg"
GPS_FILE   = "gps.csv"
MAP_FILE   = "Satellite_Z21.tif"
```

**Important:** this script deliberately refuses to use the current `gps.csv` `altitude` column for the FOV ROI, because your shown values (`-80` to `922 m`) are not safely interpretable as camera AGL. It will automatically look in `gps_data.csv` and other CSVs for a relative-altitude column. If none exists, it stops instead of producing a physically incorrect ROI.

Also, the `Gdk-CRITICAL` messages from Shotwell are unrelated to the map-matching algorithm; they are a GUI/display issue from running the application through SSH.
