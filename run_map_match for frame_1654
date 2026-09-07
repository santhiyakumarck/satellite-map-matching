#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import rasterio

from rasterio.windows import Window
from rasterio.warp import transform


# ============================================================
# MASTER DRONE -> SATELLITE MAP MATCHING PIPELINE
# ============================================================
#
# Input:
#   frame_1654.jpg
#   gps_data.csv
#   Map_Export_2026_08_28_162236.tif
#
# Process:
#
#   Frame
#      ↓
#   Frame timestamp
#      ↓
#   Nearest GPS
#      ↓
#   Latitude + Longitude
#      ↓
#   GPS -> Satellite TIFF pixel
#      ↓
#   Extract ROI
#      ↓
#   SuperPoint
#      ↓
#   SuperGlue
#      ↓
#   RANSAC
#      ↓
#   Homography
#      ↓
#   Estimated map position
#      ↓
#   Estimated latitude + longitude
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = "/home/sandiya/Documents/pi5"

FRAME_FILE = os.path.join(
    BASE_DIR,
    "frame_1654.jpg"
)

GPS_FILE = os.path.join(
    BASE_DIR,
    "gps_data.csv"
)

# Satellite map to use
MAP_FILE = os.path.join(
    BASE_DIR,
    "Map_Export_2026_08_28_162236.tif"
)

SUPERGLUE_DIR = os.path.join(
    BASE_DIR,
    "SuperGluePretrainedNetwork"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results_1654"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. FRAME SETTINGS
# ============================================================

FRAME_NUMBER = 1654

# Camera frame rate
CAMERA_FPS = 30.0

# ROS bag start timestamp
BAG_START_TIMESTAMP_NS = 1787901742201901848


# ============================================================
# 3. MAP MATCHING SETTINGS
# ============================================================

# ROI size around GPS position
ROI_SIZE = 2000

# Maximum SuperPoint keypoints
MAX_KEYPOINTS = 2048

# RANSAC reprojection threshold
RANSAC_THRESHOLD = 5.0

# Minimum acceptable inliers
MIN_RANSAC_INLIERS = 8

# Minimum acceptable inlier ratio
MIN_INLIER_RATIO = 0.20


# ============================================================
# 4. START
# ============================================================

print()
print("=" * 70)
print(" MASTER DRONE -> SATELLITE MAP MATCHING")
print("=" * 70)


# ============================================================
# 5. CHECK FILES
# ============================================================

print()
print("-" * 70)
print("Checking input files")
print("-" * 70)

required_files = [
    FRAME_FILE,
    GPS_FILE,
    MAP_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print()
        print("ERROR: File not found:")
        print(file_path)

        sys.exit(1)

    print("FOUND:", file_path)


# ============================================================
# 6. LOAD DRONE FRAME
# ============================================================

print()
print("-" * 70)
print("1. Loading drone frame")
print("-" * 70)

drone_bgr = cv2.imread(
    FRAME_FILE,
    cv2.IMREAD_COLOR
)

if drone_bgr is None:

    print("ERROR: Could not read drone image.")

    sys.exit(1)


drone_gray = cv2.cvtColor(
    drone_bgr,
    cv2.COLOR_BGR2GRAY
)

drone_height, drone_width = drone_gray.shape

print("Frame :", os.path.basename(FRAME_FILE))
print("Width :", drone_width)
print("Height:", drone_height)


# ============================================================
# 7. CALCULATE FRAME TIMESTAMP
# ============================================================

print()
print("-" * 70)
print("2. Calculating frame timestamp")
print("-" * 70)

frame_timestamp_ns = int(
    BAG_START_TIMESTAMP_NS
    +
    (
        FRAME_NUMBER / CAMERA_FPS
    )
    *
    1_000_000_000
)

frame_timestamp_sec = (
    frame_timestamp_ns /
    1_000_000_000
)

print(
    "Frame number:",
    FRAME_NUMBER
)

print(
    "Frame timestamp ns:",
    frame_timestamp_ns
)

print(
    "Frame timestamp sec:",
    frame_timestamp_sec
)


# ============================================================
# 8. LOAD GPS CSV
# ============================================================

print()
print("-" * 70)
print("3. Loading GPS CSV")
print("-" * 70)

gps = pd.read_csv(
    GPS_FILE
)

print(
    "GPS rows:",
    len(gps)
)


required_columns = [
    "bag_timestamp",
    "gps_timestamp_sec",
    "gps_timestamp_nanosec",
    "latitude",
    "longitude",
    "altitude"
]

for column in required_columns:

    if column not in gps.columns:

        print()
        print(
            "ERROR: Missing GPS column:",
            column
        )

        sys.exit(1)


# ============================================================
# 9. GPS TIMESTAMP -> NANOSECONDS
# ============================================================

gps_timestamp_ns = (
    gps["gps_timestamp_sec"].astype(np.int64)
    *
    1_000_000_000
    +
    gps["gps_timestamp_nanosec"].astype(np.int64)
)


# ============================================================
# 10. FIND NEAREST GPS
# ============================================================

print()
print("-" * 70)
print("4. Finding nearest GPS")
print("-" * 70)

time_difference_ns = np.abs(
    gps_timestamp_ns -
    frame_timestamp_ns
)

nearest_index = int(
    np.argmin(time_difference_ns)
)

nearest_gps = gps.iloc[
    nearest_index
]

gps_lat = float(
    nearest_gps["latitude"]
)

gps_lon = float(
    nearest_gps["longitude"]
)

gps_alt = float(
    nearest_gps["altitude"]
)

nearest_gps_timestamp = int(
    gps_timestamp_ns.iloc[
        nearest_index
    ]
)

gps_time_difference_ms = (
    abs(
        nearest_gps_timestamp -
        frame_timestamp_ns
    )
    /
    1_000_000
)


print(
    "Nearest GPS row:",
    nearest_index
)

print(
    "Latitude:",
    gps_lat
)

print(
    "Longitude:",
    gps_lon
)

print(
    "Altitude:",
    gps_alt
)

print(
    "GPS timestamp:",
    nearest_gps_timestamp
)

print(
    "Time difference:",
    gps_time_difference_ms,
    "ms"
)


# ============================================================
# 11. OPEN SATELLITE TIFF
# ============================================================

print()
print("-" * 70)
print("5. Opening satellite GeoTIFF")
print("-" * 70)

src = rasterio.open(
    MAP_FILE
)

print(
    "Map:",
    os.path.basename(MAP_FILE)
)

print(
    "Width :",
    src.width
)

print(
    "Height:",
    src.height
)

print(
    "Bands :",
    src.count
)

print(
    "CRS   :",
    src.crs
)

print()
print("Map bounds:")

print(
    "Left  :",
    src.bounds.left
)

print(
    "Bottom:",
    src.bounds.bottom
)

print(
    "Right :",
    src.bounds.right
)

print(
    "Top   :",
    src.bounds.top
)

print()
print(
    "Pixel size:",
    src.res
)


# ============================================================
# 12. GPS -> MAP CRS
# ============================================================

print()
print("-" * 70)
print("6. GPS -> Satellite map coordinates")
print("-" * 70)

GPS_CRS = "EPSG:4326"

MAP_CRS = src.crs

if MAP_CRS is None:

    print(
        "ERROR: Satellite TIFF has no CRS."
    )

    src.close()

    sys.exit(1)


map_x_list, map_y_list = transform(
    GPS_CRS,
    MAP_CRS,
    [gps_lon],
    [gps_lat]
)

map_x = float(
    map_x_list[0]
)

map_y = float(
    map_y_list[0]
)

print(
    "GPS longitude:",
    gps_lon
)

print(
    "GPS latitude :",
    gps_lat
)

print()
print("Map coordinates:")

print(
    "X:",
    map_x
)

print(
    "Y:",
    map_y
)


# ============================================================
# 13. MAP COORDINATES -> PIXEL
# ============================================================

print()
print("-" * 70)
print("7. GPS -> TIFF pixel")
print("-" * 70)

pixel_row, pixel_col = src.index(
    map_x,
    map_y
)

pixel_row = int(
    pixel_row
)

pixel_col = int(
    pixel_col
)

print(
    "GPS pixel:"
)

print(
    "Column:",
    pixel_col
)

print(
    "Row   :",
    pixel_row
)


# ============================================================
# 14. CHECK GPS INSIDE MAP
# ============================================================

if (
    pixel_col < 0
    or pixel_col >= src.width
    or
    pixel_row < 0
    or pixel_row >= src.height
):

    print()
    print(
        "ERROR: GPS position is OUTSIDE the satellite TIFF."
    )

    print()
    print(
        "The selected satellite map does not cover"
    )

    print(
        "the GPS position of frame_1654."
    )

    src.close()

    sys.exit(1)


print()
print(
    "GPS position is INSIDE the satellite map."
)


# ============================================================
# 15. CREATE ROI
# ============================================================

print()
print("-" * 70)
print("8. Extracting satellite ROI")
print("-" * 70)

half_roi = ROI_SIZE // 2

roi_left = pixel_col - half_roi

roi_top = pixel_row - half_roi

# Keep ROI inside map

roi_left = max(
    0,
    roi_left
)

roi_top = max(
    0,
    roi_top
)

if (
    roi_left + ROI_SIZE
    >
    src.width
):

    roi_left = max(
        0,
        src.width - ROI_SIZE
    )


if (
    roi_top + ROI_SIZE
    >
    src.height
):

    roi_top = max(
        0,
        src.height - ROI_SIZE
    )


roi_width = min(
    ROI_SIZE,
    src.width - roi_left
)

roi_height = min(
    ROI_SIZE,
    src.height - roi_top
)


print(
    "ROI left  :",
    roi_left
)

print(
    "ROI top   :",
    roi_top
)

print(
    "ROI width :",
    roi_width
)

print(
    "ROI height:",
    roi_height
)


window = Window(
    roi_left,
    roi_top,
    roi_width,
    roi_height
)


# ============================================================
# 16. READ ROI FROM TIFF
# ============================================================

roi_data = src.read(
    window=window
)

print()
print(
    "ROI array shape:",
    roi_data.shape
)


# ============================================================
# 17. CONVERT TIFF ROI
# ============================================================

if src.count >= 3:

    satellite_rgb = np.stack(
        [
            roi_data[0],
            roi_data[1],
            roi_data[2]
        ],
        axis=2
    )

elif src.count == 1:

    satellite_rgb = roi_data[0]

else:

    print(
        "ERROR: Unsupported TIFF band count."
    )

    src.close()

    sys.exit(1)


# ============================================================
# 18. NORMALIZE SATELLITE IMAGE
# ============================================================

if satellite_rgb.dtype != np.uint8:

    satellite_rgb = cv2.normalize(
        satellite_rgb,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )


# ============================================================
# 19. RGB -> GRAY
# ============================================================

if len(
    satellite_rgb.shape
) == 3:

    satellite_gray = cv2.cvtColor(
        satellite_rgb,
        cv2.COLOR_RGB2GRAY
    )

else:

    satellite_gray = satellite_rgb


# ============================================================
# 20. SAVE SATELLITE ROI
# ============================================================

satellite_roi_file = os.path.join(
    RESULT_DIR,
    "satellite_roi.png"
)

cv2.imwrite(
    satellite_roi_file,
    satellite_gray
)

print()
print(
    "Satellite ROI saved:"
)

print(
    satellite_roi_file
)


# ============================================================
# 21. LOAD SUPERPOINT + SUPERGLUE
# ============================================================

print()
print("-" * 70)
print("9. Loading SuperPoint + SuperGlue")
print("-" * 70)

sys.path.insert(
    0,
    SUPERGLUE_DIR
)

try:

    from models.matching import Matching

except Exception as error:

    print()
    print(
        "ERROR: Could not import SuperGlue."
    )

    print(error)

    src.close()

    sys.exit(1)


# ============================================================
# 22. SELECT DEVICE
# ============================================================

if torch.cuda.is_available():

    device = "cuda"

else:

    device = "cpu"


print(
    "Device:",
    device
)


# ============================================================
# 23. SUPERPOINT + SUPERGLUE CONFIG
# ============================================================

config = {

    "superpoint": {

        "nms_radius": 4,

        "keypoint_threshold": 0.005,

        "max_keypoints": MAX_KEYPOINTS
    },

    "superglue": {

        "weights": "outdoor",

        "sinkhorn_iterations": 20,

        "match_threshold": 0.2
    }
}


# ============================================================
# 24. LOAD MODEL
# ============================================================

matching = Matching(
    config
).eval().to(
    device
)


# ============================================================
# 25. IMAGE -> TORCH TENSOR
# ============================================================

def image_to_tensor(
    image
):

    image_float = (
        image.astype(
            np.float32
        )
        /
        255.0
    )

    tensor = torch.from_numpy(
        image_float
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        device
    )


drone_tensor = image_to_tensor(
    drone_gray
)

satellite_tensor = image_to_tensor(
    satellite_gray
)


# ============================================================
# 26. RUN SUPERPOINT + SUPERGLUE
# ============================================================

print()
print("-" * 70)
print("10. Running SuperPoint + SuperGlue")
print("-" * 70)

with torch.no_grad():

    prediction = matching({

        "image0": drone_tensor,

        "image1": satellite_tensor

    })


# ============================================================
# 27. GET KEYPOINTS
# ============================================================

keypoints0 = (
    prediction[
        "keypoints0"
    ][0]
    .cpu()
    .numpy()
)

keypoints1 = (
    prediction[
        "keypoints1"
    ][0]
    .cpu()
    .numpy()
)


matches0 = (
    prediction[
        "matches0"
    ][0]
    .cpu()
    .numpy()
)


matching_scores0 = (
    prediction[
        "matching_scores0"
    ][0]
    .cpu()
    .numpy()
)


# ============================================================
# 28. VALID MATCHES
# ============================================================

valid_matches = (
    matches0 > -1
)

matched_kp0 = keypoints0[
    valid_matches
]

matched_kp1 = keypoints1[
    matches0[
        valid_matches
    ]
]

matched_scores = matching_scores0[
    valid_matches
]


print()
print(
    "SuperPoint keypoints:"
)

print(
    "Drone    :",
    len(keypoints0)
)

print(
    "Satellite:",
    len(keypoints1)
)

print()
print(
    "SuperGlue valid matches:"
)

print(
    len(matched_kp0)
)


# ============================================================
# 29. SAVE SUPERPOINT KEYPOINT VISUALIZATION
# ============================================================

print()
print("-" * 70)
print("11. Saving SuperPoint keypoint visualization")
print("-" * 70)


# Drone keypoints

drone_keypoint_vis = cv2.cvtColor(
    drone_gray,
    cv2.COLOR_GRAY2BGR
)

for point in keypoints0:

    x = int(
        round(
            point[0]
        )
    )

    y = int(
        round(
            point[1]
        )
    )

    if (
        0 <= x < drone_width
        and
        0 <= y < drone_height
    ):

        cv2.circle(
            drone_keypoint_vis,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )


drone_keypoint_file = os.path.join(
    RESULT_DIR,
    "superpoint_keypoints_drone.png"
)

cv2.imwrite(
    drone_keypoint_file,
    drone_keypoint_vis
)


# Satellite keypoints

satellite_height, satellite_width = (
    satellite_gray.shape
)

satellite_keypoint_vis = cv2.cvtColor(
    satellite_gray,
    cv2.COLOR_GRAY2BGR
)

for point in keypoints1:

    x = int(
        round(
            point[0]
        )
    )

    y = int(
        round(
            point[1]
        )
    )

    if (
        0 <= x < satellite_width
        and
        0 <= y < satellite_height
    ):

        cv2.circle(
            satellite_keypoint_vis,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )


satellite_keypoint_file = os.path.join(
    RESULT_DIR,
    "superpoint_keypoints_satellite.png"
)

cv2.imwrite(
    satellite_keypoint_file,
    satellite_keypoint_vis
)


print(
    "Drone keypoints saved."
)

print(
    "Satellite keypoints saved."
)


# ============================================================
# 30. SUPERGLUE MATCH VISUALIZATION
# ============================================================

print()
print("-" * 70)
print("12. Saving SuperGlue match visualization")
print("-" * 70)


if len(matched_kp0) > 0:

    canvas_height = max(
        drone_height,
        satellite_height
    )

    canvas_width = (
        drone_width +
        satellite_width
    )

    match_canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3
        ),
        dtype=np.uint8
    )


    # Drone image

    match_canvas[
        :drone_height,
        :drone_width
    ] = cv2.cvtColor(
        drone_gray,
        cv2.COLOR_GRAY2BGR
    )


    # Satellite image

    match_canvas[
        :satellite_height,
        drone_width:
        drone_width + satellite_width
    ] = cv2.cvtColor(
        satellite_gray,
        cv2.COLOR_GRAY2BGR
    )


    # Draw matches

    for point0, point1 in zip(
        matched_kp0,
        matched_kp1
    ):

        x0 = int(
            round(
                point0[0]
            )
        )

        y0 = int(
            round(
                point0[1]
            )
        )

        x1 = int(
            round(
                point1[0]
            )
        ) + drone_width

        y1 = int(
            round(
                point1[1]
            )
        )


        cv2.line(
            match_canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            1
        )


        cv2.circle(
            match_canvas,
            (x0, y0),
            3,
            (0, 0, 255),
            -1
        )


        cv2.circle(
            match_canvas,
            (x1, y1),
            3,
            (0, 0, 255),
            -1
        )


    superglue_match_file = os.path.join(
        RESULT_DIR,
        "superglue_matches.png"
    )

    cv2.imwrite(
        superglue_match_file,
        match_canvas
    )


    print(
        "Saved:",
        superglue_match_file
    )

else:

    print(
        "No SuperGlue matches available."
    )


# ============================================================
# 31. RANSAC HOMOGRAPHY
# ============================================================

print()
print("-" * 70)
print("13. RANSAC Homography")
print("-" * 70)


if len(matched_kp0) < 4:

    print()
    print(
        "ERROR: Not enough matches for homography."
    )

    src.close()

    sys.exit(1)


homography, ransac_mask = cv2.findHomography(
    matched_kp0,
    matched_kp1,
    cv2.RANSAC,
    RANSAC_THRESHOLD
)


if homography is None:

    print()
    print(
        "ERROR: Homography estimation failed."
    )

    src.close()

    sys.exit(1)


ransac_mask = (
    ransac_mask
    .ravel()
    .astype(bool)
)


inlier_kp0 = matched_kp0[
    ransac_mask
]

inlier_kp1 = matched_kp1[
    ransac_mask
]


inliers = len(
    inlier_kp0
)

outliers = (
    len(matched_kp0)
    -
    inliers
)

inlier_ratio = (
    inliers /
    len(matched_kp0)
)


print()
print(
    "Total matches:",
    len(matched_kp0)
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
    inlier_ratio
)

print()
print(
    "Homography:"
)

print(
    homography
)


# ============================================================
# 32. SAVE HOMOGRAPHY
# ============================================================

homography_file = os.path.join(
    RESULT_DIR,
    "homography.txt"
)

np.savetxt(
    homography_file,
    homography
)


# ============================================================
# 33. RANSAC INLIER VISUALIZATION
# ============================================================

print()
print("-" * 70)
print("14. Saving RANSAC inlier visualization")
print("-" * 70)


ransac_canvas_height = max(
    drone_height,
    satellite_height
)

ransac_canvas_width = (
    drone_width +
    satellite_width
)


ransac_canvas = np.zeros(
    (
        ransac_canvas_height,
        ransac_canvas_width,
        3
    ),
    dtype=np.uint8
)


# Drone

ransac_canvas[
    :drone_height,
    :drone_width
] = cv2.cvtColor(
    drone_gray,
    cv2.COLOR_GRAY2BGR
)


# Satellite

ransac_canvas[
    :satellite_height,
    drone_width:
    drone_width + satellite_width
] = cv2.cvtColor(
    satellite_gray,
    cv2.COLOR_GRAY2BGR
)


# Draw ONLY RANSAC INLIERS

for point0, point1 in zip(
    inlier_kp0,
    inlier_kp1
):

    x0 = int(
        round(
            point0[0]
        )
    )

    y0 = int(
        round(
            point0[1]
        )
    )

    x1 = int(
        round(
            point1[0]
        )
    ) + drone_width

    y1 = int(
        round(
            point1[1]
        )
    )


    cv2.line(
        ransac_canvas,
        (x0, y0),
        (x1, y1),
        (0, 255, 0),
        2
    )


    cv2.circle(
        ransac_canvas,
        (x0, y0),
        4,
        (0, 0, 255),
        -1
    )


    cv2.circle(
        ransac_canvas,
        (x1, y1),
        4,
        (0, 0, 255),
        -1
    )


ransac_file = os.path.join(
    RESULT_DIR,
    "ransac_inliers.png"
)

cv2.imwrite(
    ransac_file,
    ransac_canvas
)


print(
    "Saved:",
    ransac_file
)


# ============================================================
# 34. MATCHING QUALITY
# ============================================================

print()
print("-" * 70)
print("15. Matching quality")
print("-" * 70)


matching_success = (
    inliers >= MIN_RANSAC_INLIERS
    and
    inlier_ratio >= MIN_INLIER_RATIO
)


if matching_success:

    print(
        "MATCHING STATUS: PASS"
    )

else:

    print(
        "MATCHING STATUS: FAILED"
    )


# ============================================================
# 35. ESTIMATE DRONE CENTER IN SATELLITE ROI
# ============================================================

print()
print("-" * 70)
print("16. Estimating position using homography")
print("-" * 70)


drone_center_x = (
    drone_width /
    2.0
)

drone_center_y = (
    drone_height /
    2.0
)


drone_center = np.array(
    [
        [
            [
                drone_center_x,
                drone_center_y
            ]
        ]
    ],
    dtype=np.float32
)


estimated_satellite_point = (
    cv2.perspectiveTransform(
        drone_center,
        homography
    )
)


estimated_roi_x = float(
    estimated_satellite_point[
        0,
        0,
        0
    ]
)

estimated_roi_y = float(
    estimated_satellite_point[
        0,
        0,
        1
    ]
)


print(
    "Drone center:"
)

print(
    "X:",
    drone_center_x
)

print(
    "Y:",
    drone_center_y
)


print()
print(
    "Estimated position inside ROI:"
)

print(
    "X:",
    estimated_roi_x
)

print(
    "Y:",
    estimated_roi_y
)


# ============================================================
# 36. ROI PIXEL -> FULL TIFF PIXEL
# ============================================================

estimated_full_pixel_col = (
    roi_left +
    estimated_roi_x
)

estimated_full_pixel_row = (
    roi_top +
    estimated_roi_y
)


print()
print(
    "Estimated full satellite pixel:"
)

print(
    "Column:",
    estimated_full_pixel_col
)

print(
    "Row:",
    estimated_full_pixel_row
)


# ============================================================
# 37. FULL TIFF PIXEL -> MAP COORDINATE
# ============================================================

estimated_map_x, estimated_map_y = src.xy(
    int(
        round(
            estimated_full_pixel_row
        )
    ),
    int(
        round(
            estimated_full_pixel_col
        )
    )
)


print()
print(
    "Estimated map coordinates:"
)

print(
    "X:",
    estimated_map_x
)

print(
    "Y:",
    estimated_map_y
)


# ============================================================
# 38. MAP COORDINATE -> GPS
# ============================================================

estimated_lon_list, estimated_lat_list = transform(
    MAP_CRS,
    GPS_CRS,
    [estimated_map_x],
    [estimated_map_y]
)


estimated_lon = float(
    estimated_lon_list[0]
)

estimated_lat = float(
    estimated_lat_list[0]
)


print()
print(
    "Estimated GPS:"
)

print(
    "Latitude :",
    estimated_lat
)

print(
    "Longitude:",
    estimated_lon
)


# ============================================================
# 39. POSITION ERROR
# ============================================================

print()
print("-" * 70)
print("17. Position error")
print("-" * 70)


latitude_error = (
    estimated_lat -
    gps_lat
)

longitude_error = (
    estimated_lon -
    gps_lon
)


meters_per_degree_lat = 111320.0

meters_per_degree_lon = (
    111320.0 *
    np.cos(
        np.radians(
            gps_lat
        )
    )
)


north_error_m = (
    latitude_error *
    meters_per_degree_lat
)


east_error_m = (
    longitude_error *
    meters_per_degree_lon
)


position_error_m = np.sqrt(
    north_error_m ** 2
    +
    east_error_m ** 2
)


print(
    "North error:",
    north_error_m,
    "m"
)

print(
    "East error:",
    east_error_m,
    "m"
)

print(
    "Total position error:",
    position_error_m,
    "m"
)


# ============================================================
# 40. FINAL POSITION VISUALIZATION
# ============================================================

print()
print("-" * 70)
print("18. Final position visualization")
print("-" * 70)


final_map = cv2.cvtColor(
    satellite_gray,
    cv2.COLOR_GRAY2BGR
)


# GPS ground-truth position inside ROI

gps_roi_x = (
    pixel_col -
    roi_left
)

gps_roi_y = (
    pixel_row -
    roi_top
)


# Draw GPS position

if (
    0 <= gps_roi_x < satellite_width
    and
    0 <= gps_roi_y < satellite_height
):

    cv2.circle(
        final_map,
        (
            int(
                round(
                    gps_roi_x
                )
            ),
            int(
                round(
                    gps_roi_y
                )
            )
        ),
        10,
        (255, 0, 0),
        3
    )


    cv2.putText(
        final_map,
        "GPS",
        (
            int(
                round(
                    gps_roi_x
                )
            ) + 15,
            int(
                round(
                    gps_roi_y
                )
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )


# Draw estimated position

if (
    0 <= estimated_roi_x < satellite_width
    and
    0 <= estimated_roi_y < satellite_height
):

    cv2.circle(
        final_map,
        (
            int(
                round(
                    estimated_roi_x
                )
            ),
            int(
                round(
                    estimated_roi_y
                )
            )
        ),
        10,
        (0, 255, 0),
        3
    )


    cv2.putText(
        final_map,
        "EST",
        (
            int(
                round(
                    estimated_roi_x
                )
            ) + 15,
            int(
                round(
                    estimated_roi_y
                )
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


final_position_file = os.path.join(
    RESULT_DIR,
    "final_estimated_position.png"
)

cv2.imwrite(
    final_position_file,
    final_map
)


print(
    "Saved:",
    final_position_file
)


# ============================================================
# 41. SAVE TEXT RESULTS
# ============================================================

results_file = os.path.join(
    RESULT_DIR,
    "results.txt"
)


with open(
    results_file,
    "w"
) as result:

    result.write(
        "SATELLITE MAP MATCHING RESULTS\n"
    )

    result.write(
        "========================================\n\n"
    )

    result.write(
        f"Frame: frame_{FRAME_NUMBER:04d}.jpg\n"
    )

    result.write(
        f"Frame number: {FRAME_NUMBER}\n"
    )

    result.write(
        f"Frame timestamp ns: "
        f"{frame_timestamp_ns}\n"
    )

    result.write(
        f"Nearest GPS row: "
        f"{nearest_index}\n"
    )

    result.write(
        f"GPS latitude: "
        f"{gps_lat}\n"
    )

    result.write(
        f"GPS longitude: "
        f"{gps_lon}\n"
    )

    result.write(
        f"GPS altitude: "
        f"{gps_alt}\n"
    )

    result.write(
        f"GPS time difference ms: "
        f"{gps_time_difference_ms}\n\n"
    )

    result.write(
        f"Satellite map: "
        f"{os.path.basename(MAP_FILE)}\n"
    )

    result.write(
        f"Map CRS: "
        f"{src.crs}\n"
    )

    result.write(
        f"Map width: "
        f"{src.width}\n"
    )

    result.write(
        f"Map height: "
        f"{src.height}\n"
    )

    result.write(
        f"Map pixel size: "
        f"{src.res}\n\n"
    )

    result.write(
        f"GPS map pixel column: "
        f"{pixel_col}\n"
    )

    result.write(
        f"GPS map pixel row: "
        f"{pixel_row}\n"
    )

    result.write(
        f"ROI left: "
        f"{roi_left}\n"
    )

    result.write(
        f"ROI top: "
        f"{roi_top}\n"
    )

    result.write(
        f"ROI width: "
        f"{roi_width}\n"
    )

    result.write(
        f"ROI height: "
        f"{roi_height}\n\n"
    )

    result.write(
        f"SuperPoint drone keypoints: "
        f"{len(keypoints0)}\n"
    )

    result.write(
        f"SuperPoint satellite keypoints: "
        f"{len(keypoints1)}\n"
    )

    result.write(
        f"SuperGlue matches: "
        f"{len(matched_kp0)}\n"
    )

    result.write(
        f"RANSAC inliers: "
        f"{inliers}\n"
    )

    result.write(
        f"RANSAC outliers: "
        f"{outliers}\n"
    )

    result.write(
        f"Inlier ratio: "
        f"{inlier_ratio}\n\n"
    )

    result.write(
        f"Estimated ROI X: "
        f"{estimated_roi_x}\n"
    )

    result.write(
        f"Estimated ROI Y: "
        f"{estimated_roi_y}\n"
    )

    result.write(
        f"Estimated map pixel column: "
        f"{estimated_full_pixel_col}\n"
    )

    result.write(
        f"Estimated map pixel row: "
        f"{estimated_full_pixel_row}\n"
    )

    result.write(
        f"Estimated latitude: "
        f"{estimated_lat}\n"
    )

    result.write(
        f"Estimated longitude: "
        f"{estimated_lon}\n\n"
    )

    result.write(
        f"North error meters: "
        f"{north_error_m}\n"
    )

    result.write(
        f"East error meters: "
        f"{east_error_m}\n"
    )

    result.write(
        f"Total position error meters: "
        f"{position_error_m}\n"
    )

    result.write(
        f"Matching status: "
        f"{'PASS' if matching_success else 'FAILED'}\n"
    )


# ============================================================
# 42. CLOSE TIFF
# ============================================================

src.close()


# ============================================================
# 43. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print(" PIPELINE COMPLETE")
print("=" * 70)

print()
print(
    "Results directory:"
)

print(
    RESULT_DIR
)

print()
print(
    "Ground-truth GPS:"
)

print(
    "Latitude :",
    gps_lat
)

print(
    "Longitude:",
    gps_lon
)

print()
print(
    "Estimated GPS:"
)

print(
    "Latitude :",
    estimated_lat
)

print(
    "Longitude:",
    estimated_lon
)

print()
print(
    "SuperPoint drone keypoints:",
    len(keypoints0)
)

print(
    "SuperPoint satellite keypoints:",
    len(keypoints1)
)

print(
    "SuperGlue matches:",
    len(matched_kp0)
)

print(
    "RANSAC inliers:",
    inliers
)

print(
    "RANSAC outliers:",
    outliers
)

print(
    "Inlier ratio:",
    inlier_ratio
)

print()
print(
    "Position error:",
    position_error_m,
    "meters"
)

print()
print(
    "Matching status:",
    "PASS"
    if matching_success
    else
    "FAILED"
)

print()
print(
    "Generated output files:"
)

for filename in sorted(
    os.listdir(
        RESULT_DIR
    )
):

    print(
        " ",
        filename
    )

print()
print("=" * 70)
print(" Done.")
print("=" * 70)
