#!/usr/bin/env python3

import os
import math
import time
import sys

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy

sys.path.insert(
    0,
    "/home/sandiya/sept 3-Drone-map-test/SuperGluePretrainedNetwork"
)

import torch
from models.matching import Matching


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/sandiya/sept 3-Drone-map-test"

# ------------------------------------------------------------
# MULTIPLE FRAME INPUT DIRECTORY
# ------------------------------------------------------------

FRAME_DIR = os.path.join(
    BASE_DIR,
    "multiple_frames"
)

# ------------------------------------------------------------
# CSV / SATELLITE MAP
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "offline_multi_frame_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TEST / PROCESSING CONTROL
# ============================================================

# 1 = process every selected frame
# 2 = process every second frame
# etc.

FRAME_STEP = 1


# ============================================================
# CAMERA PARAMETERS
# ============================================================

HFOV_DEG = 90.0


# ============================================================
# ROI PARAMETERS
# ============================================================

ROI_MARGIN_PX = 1000

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
# UTILITY
# ============================================================

def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CUDA SYNCHRONIZATION
# ============================================================

def synchronize_cuda(device):

    if device == "cuda":

        torch.cuda.synchronize()


# ============================================================
# STEP 1
# LOAD FRAME TIMESTAMP CSV ONCE
# ============================================================

def load_frame_timestamp_data():

    print_section(
        "LOADING FRAME TIMESTAMP DATA"
    )

    df = pd.read_csv(
        FRAME_TIMESTAMP_CSV
    )

    required = [
        "frame_filename",
        "timestamp_ns"
    ]

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"Missing column in frame timestamp CSV: {col}"
            )

    # Create normalized basename column.
    #
    # This allows both:
    #
    # frame_00863.jpg
    #
    # and
    #
    # /some/path/frame_00863.jpg
    #
    # to match correctly.

    df["frame_basename"] = (
        df["frame_filename"]
        .astype(str)
        .map(os.path.basename)
    )

    print(
        f"Frame timestamp records : {len(df)}"
    )

    return df


# ============================================================
# STEP 2
# LOAD GPS CSV ONCE
# ============================================================

def load_gps_data():

    print_section(
        "LOADING GPS DATA"
    )

    df = pd.read_csv(
        GPS_CSV
    )

    required = [
        "bag_timestamp",
        "latitude",
        "longitude",
        "altitude"
    ]

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"Missing column in GPS CSV: {col}"
            )

    df["bag_timestamp"] = pd.to_numeric(
        df["bag_timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["bag_timestamp"]
    ).copy()

    df["bag_timestamp"] = (
        df["bag_timestamp"]
        .astype(np.int64)
    )

    print(
        f"GPS records             : {len(df)}"
    )

    return df


# ============================================================
# STEP 3
# GET SELECTED FRAME LIST
# ============================================================

def get_frame_list(
    timestamp_df
):

    print_section(
        "LOADING SELECTED FRAMES"
    )

    if not os.path.isdir(
        FRAME_DIR
    ):

        raise RuntimeError(
            f"Frame directory does not exist:\n{FRAME_DIR}"
        )

    selected_frames = sorted(
        [
            f
            for f in os.listdir(FRAME_DIR)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    if len(selected_frames) == 0:

        raise RuntimeError(
            f"No image frames found in:\n{FRAME_DIR}"
        )

    # Apply frame step.

    selected_frames = selected_frames[
        ::FRAME_STEP
    ]

    print(
        f"Frames found            : "
        f"{len(selected_frames)}"
    )

    print()

    valid_frames = []

    for frame_name in selected_frames:

        frame_path = os.path.join(
            FRAME_DIR,
            frame_name
        )

        if not os.path.isfile(
            frame_path
        ):

            print(
                f"WARNING: File does not exist: "
                f"{frame_path}"
            )

            continue

        matches = timestamp_df[
            timestamp_df["frame_basename"]
            == frame_name
        ]

        if len(matches) == 0:

            print(
                f"WARNING: No timestamp found for "
                f"{frame_name}"
            )

            continue

        timestamp_ns = int(
            matches.iloc[0]["timestamp_ns"]
        )

        valid_frames.append(
            (
                frame_name,
                frame_path,
                timestamp_ns
            )
        )

        print(
            f"OK  {frame_name:20s}"
            f" timestamp_ns = {timestamp_ns}"
        )

    print()

    if len(valid_frames) == 0:

        raise RuntimeError(
            "None of the selected frames have "
            "valid timestamp information."
        )

    print(
        f"Valid frames to process : "
        f"{len(valid_frames)}"
    )

    return valid_frames


# ============================================================
# STEP 4
# FIND NEAREST GPS
# ============================================================

def get_nearest_gps(
    gps_df,
    frame_timestamp_ns
):

    gps_timestamps = (
        gps_df["bag_timestamp"]
        .values
    )

    differences = np.abs(
        gps_timestamps
        - np.int64(frame_timestamp_ns)
    )

    nearest_index = int(
        np.argmin(differences)
    )

    row = gps_df.iloc[
        nearest_index
    ]

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
# STEP 5
# OPEN SATELLITE MAP
# ============================================================

def get_map_information():

    src = rasterio.open(
        SATELLITE_MAP
    )

    return src


# ============================================================
# STEP 6
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

    return (
        float(row),
        float(col)
    )


# ============================================================
# STEP 7
# CALCULATE VFOV
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

    vfov_rad = (
        2.0
        * math.atan(
            math.tan(
                hfov_rad / 2.0
            )
            * aspect_ratio
        )
    )

    return math.degrees(
        vfov_rad
    )


# ============================================================
# STEP 8
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
            math.radians(
                hfov_deg
            ) / 2.0
        )
    )

    height_m = (
        2.0
        * altitude
        * math.tan(
            math.radians(
                vfov_deg
            ) / 2.0
        )
    )

    return (
        width_m,
        height_m
    )


# ============================================================
# STEP 9
# MAP METERS / PIXEL
# ============================================================

def calculate_map_meters_per_pixel(
    src,
    latitude
):

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
            math.radians(
                latitude
            )
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
# STEP 10
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
# RESIZE SATELLITE ROI
# ============================================================

def resize_for_matching(
    image
):

    height, width = (
        image.shape[:2]
    )

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

    return (
        resized,
        scale
    )


# ============================================================
# IMAGE -> TORCH GRAYSCALE
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
# LOAD SUPERPOINT + SUPERGLUE MODEL ONCE
# ============================================================

def load_matching_model():

    print_section(
        "LOADING SUPERPOINT + SUPERGLUE MODEL"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device                  : {device}"
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

            "sinkhorn_iterations":
                20,

            "match_threshold":
                MATCH_THRESHOLD
        }
    }

    model_start = time.perf_counter()

    matching = Matching(
        config
    ).eval().to(
        device
    )

    synchronize_cuda(
        device
    )

    model_load_time = (
        time.perf_counter()
        - model_start
    )

    print(
        f"Model load time         : "
        f"{model_load_time:.4f} s"
    )

    print(
        "Model loaded successfully."
    )

    return (
        matching,
        device,
        model_load_time
    )


# ============================================================
# SUPERPOINT + SUPERGLUE
#
# IMPORTANT:
# MODEL IS PASSED IN.
# IT IS NOT CREATED AGAIN.
# ============================================================

def run_superglue(
    matching,
    device,
    frame,
    satellite_matching
):

    frame_tensor = image_to_tensor(
        frame,
        device
    )

    satellite_tensor = image_to_tensor(
        satellite_matching,
        device
    )

    synchronize_cuda(
        device
    )

    inference_start = time.perf_counter()

    with torch.no_grad():

        pred = matching({

            "image0": frame_tensor,

            "image1": satellite_tensor
        })

    synchronize_cuda(
        device
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # ALL SUPERPOINT KEYPOINTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SUPERGLUE MATCHES
    # --------------------------------------------------------

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

    valid = (
        matches0 > -1
    )

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
        keypoints0,
        keypoints1,
        matched_kpts0,
        matched_kpts1,
        matched_confidence,
        inference_time
    )


# ============================================================
# DRAW ALL SUPERPOINT KEYPOINTS
# ============================================================

def create_keypoint_visualization(
    image,
    keypoints,
    title
):

    visual = image.copy()

    for point in keypoints:

        x = int(
            round(point[0])
        )

        y = int(
            round(point[1])
        )

        if (
            0 <= x < visual.shape[1]
            and
            0 <= y < visual.shape[0]
        ):

            cv2.circle(
                visual,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    cv2.putText(
        visual,
        f"{title} - Keypoints: {len(keypoints)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return visual


# ============================================================
# SUPERGLUE MATCH VISUALIZATION
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

    cv2.putText(
        canvas,
        f"SuperGlue matches: {len(kpts0)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
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

    mask = (
        mask
        .ravel()
        .astype(bool)
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
# RANSAC INLIER VISUALIZATION
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

    cv2.putText(
        canvas,
        f"RANSAC inliers: {len(inlier_kpts0)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# POSITION ESTIMATION
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

    # Convert ROI coordinate
    # to full map coordinate.

    map_column = (
        roi_x1
        + original_roi_x
    )

    map_row = (
        roi_y1
        + original_roi_y
    )

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

    # Only draw if projected point
    # is inside the matching ROI.

    if (
        0 <= x < visual.shape[1]
        and
        0 <= y < visual.shape[0]
    ):

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

    else:

        print(
            "WARNING: Estimated position is "
            "outside matching ROI."
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
# SAVE REPORT
# ============================================================

def save_frame_report(
    report_file,
    frame_name,
    frame_width,
    frame_height,
    frame_timestamp_ns,
    gps_timestamp_ns,
    gps_difference_ms,
    gps_latitude,
    gps_longitude,
    altitude,
    center_row,
    center_col,
    vfov_deg,
    footprint_width_m,
    footprint_height_m,
    x_m_per_px,
    y_m_per_px,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
    roi_width,
    roi_height,
    matching_width,
    matching_height,
    resize_scale,
    device,
    keypoints0_count,
    keypoints1_count,
    total_matches,
    mean_confidence,
    matching_time,
    H,
    inliers,
    outliers,
    inlier_ratio,
    mean_inlier_confidence,
    ransac_time,
    frame_center_x,
    frame_center_y,
    matching_x,
    matching_y,
    original_roi_x,
    original_roi_y,
    map_column,
    map_row,
    estimated_latitude,
    estimated_longitude,
    east_difference_m,
    north_difference_m,
    horizontal_difference_m,
    total_time
):

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "OFFLINE SATELLITE MAP MATCHING REPORT\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "INPUT\n"
        )

        f.write(
            f"Frame              : {frame_name}\n"
        )

        f.write(
            f"Frame directory    : {FRAME_DIR}\n"
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
            "MAP PIXEL\n"
        )

        f.write(
            f"GPS map column     : "
            f"{center_col:.6f}\n"
        )

        f.write(
            f"GPS map row        : "
            f"{center_row:.6f}\n\n"
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
            f"Ground width       : "
            f"{footprint_width_m:.6f} m\n"
        )

        f.write(
            f"Ground height      : "
            f"{footprint_height_m:.6f} m\n"
        )

        f.write(
            f"Map X m/pixel      : "
            f"{x_m_per_px:.10f}\n"
        )

        f.write(
            f"Map Y m/pixel      : "
            f"{y_m_per_px:.10f}\n"
        )

        f.write(
            f"ROI x1             : {roi_x1}\n"
        )

        f.write(
            f"ROI y1             : {roi_y1}\n"
        )

        f.write(
            f"ROI x2             : {roi_x2}\n"
        )

        f.write(
            f"ROI y2             : {roi_y2}\n"
        )

        f.write(
            f"ROI width          : {roi_width}\n"
        )

        f.write(
            f"ROI height         : {roi_height}\n\n"
        )

        f.write(
            "SUPERPOINT / SUPERGLUE\n"
        )

        f.write(
            f"Device             : "
            f"{device}\n"
        )

        f.write(
            f"Camera keypoints   : "
            f"{keypoints0_count}\n"
        )

        f.write(
            f"Satellite keypoints: "
            f"{keypoints1_count}\n"
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
            f"SuperPoint/SuperGlue: "
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


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame_name,
    frame_path,
    frame_timestamp_ns,
    gps_df,
    src,
    matching,
    device,
    model_load_time
):

    frame_start = time.perf_counter()

    print_section(
        f"PROCESSING {frame_name}"
    )

    # --------------------------------------------------------
    # CREATE FRAME OUTPUT DIRECTORY
    # --------------------------------------------------------

    frame_output_dir = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(
            frame_name
        )[0]
    )

    os.makedirs(
        frame_output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    camera_frame_file = os.path.join(
        frame_output_dir,
        "camera_frame.jpg"
    )

    keypoints_camera_file = os.path.join(
        frame_output_dir,
        "keypoints_camera.jpg"
    )

    keypoints_satellite_file = os.path.join(
        frame_output_dir,
        "keypoints_satellite.jpg"
    )

    original_roi_file = os.path.join(
        frame_output_dir,
        "satellite_roi_original.png"
    )

    matching_roi_file = os.path.join(
        frame_output_dir,
        "satellite_roi_for_matching.png"
    )

    matches_npz = os.path.join(
        frame_output_dir,
        "superglue_matches.npz"
    )

    matches_vis = os.path.join(
        frame_output_dir,
        "superglue_matches.jpg"
    )

    ransac_npz = os.path.join(
        frame_output_dir,
        "ransac_results.npz"
    )

    ransac_vis = os.path.join(
        frame_output_dir,
        "ransac_inliers.jpg"
    )

    position_vis = os.path.join(
        frame_output_dir,
        "estimated_position.jpg"
    )

    report_file = os.path.join(
        frame_output_dir,
        "report.txt"
    )

    try:

        # ====================================================
        # LOAD FRAME
        # ====================================================

        frame = cv2.imread(
            frame_path
        )

        if frame is None:

            raise RuntimeError(
                f"Could not read frame:\n{frame_path}"
            )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        cv2.imwrite(
            camera_frame_file,
            frame
        )

        print(
            f"Frame size             : "
            f"{frame_width} x {frame_height}"
        )

        # ====================================================
        # FRAME -> GPS
        # ====================================================

        print_section(
            "STEP 1 - FRAME TIMESTAMP -> GPS"
        )

        (
            gps_latitude,
            gps_longitude,
            altitude,
            gps_timestamp_ns,
            gps_difference_ms
        ) = get_nearest_gps(
            gps_df,
            frame_timestamp_ns
        )

        print(
            f"Frame timestamp       : "
            f"{frame_timestamp_ns}"
        )

        print(
            f"Nearest GPS timestamp : "
            f"{gps_timestamp_ns}"
        )

        print(
            f"Time difference       : "
            f"{gps_difference_ms:.3f} ms"
        )

        print(
            f"GPS latitude          : "
            f"{gps_latitude:.10f}"
        )

        print(
            f"GPS longitude         : "
            f"{gps_longitude:.10f}"
        )

        print(
            f"Altitude AGL          : "
            f"{altitude:.6f} m"
        )

        # ====================================================
        # GPS -> MAP PIXEL
        # ====================================================

        print_section(
            "STEP 2 - GPS -> MAP PIXEL"
        )

        (
            center_row,
            center_col
        ) = gps_to_pixel(
            src,
            gps_latitude,
            gps_longitude
        )

        print(
            f"Map center column     : "
            f"{center_col:.3f}"
        )

        print(
            f"Map center row        : "
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

        # ====================================================
        # ROI
        # ====================================================

        print_section(
            "STEP 3 - AUTOMATIC ROI CALCULATION"
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

        roi_width = (
            roi_x2
            - roi_x1
        )

        roi_height = (
            roi_y2
            - roi_y1
        )

        print(
            f"Horizontal FOV       : "
            f"{HFOV_DEG:.3f} deg"
        )

        print(
            f"Vertical FOV         : "
            f"{vfov_deg:.3f} deg"
        )

        print(
            f"Ground width         : "
            f"{footprint_width_m:.3f} m"
        )

        print(
            f"Ground height        : "
            f"{footprint_height_m:.3f} m"
        )

        print(
            f"Map X m/pixel        : "
            f"{x_m_per_px:.8f}"
        )

        print(
            f"Map Y m/pixel        : "
            f"{y_m_per_px:.8f}"
        )

        print(
            f"ROI x1               : {roi_x1}"
        )

        print(
            f"ROI y1               : {roi_y1}"
        )

        print(
            f"ROI x2               : {roi_x2}"
        )

        print(
            f"ROI y2               : {roi_y2}"
        )

        print(
            f"Actual ROI size      : "
            f"{roi_width} x {roi_height}"
        )

        # ====================================================
        # READ ROI
        # ====================================================

        original_roi = read_roi(
            src,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        )

        cv2.imwrite(
            original_roi_file,
            original_roi
        )

        # ====================================================
        # RESIZE ROI
        # ====================================================

        print_section(
            "STEP 4 - SATELLITE ROI RESIZE"
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
            f"Original ROI         : "
            f"{roi_width} x {roi_height}"
        )

        print(
            f"Matching ROI         : "
            f"{matching_width} x {matching_height}"
        )

        print(
            f"Resize scale         : "
            f"{resize_scale:.10f}"
        )

        cv2.imwrite(
            matching_roi_file,
            satellite_matching
        )

        # ====================================================
        # SUPERPOINT + SUPERGLUE
        # ====================================================

        print_section(
            "STEP 5 - SUPERPOINT + SUPERGLUE"
        )

        (
            keypoints0,
            keypoints1,
            matched_kpts0,
            matched_kpts1,
            matched_confidence,
            matching_time
        ) = run_superglue(
            matching,
            device,
            frame,
            satellite_matching
        )

        keypoints0_count = len(
            keypoints0
        )

        keypoints1_count = len(
            keypoints1
        )

        total_matches = len(
            matched_kpts0
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

        print(
            f"Camera keypoints     : "
            f"{keypoints0_count}"
        )

        print(
            f"Satellite keypoints  : "
            f"{keypoints1_count}"
        )

        print(
            f"Valid matches         : "
            f"{total_matches}"
        )

        print(
            f"Mean match confidence: "
            f"{mean_confidence:.4f}"
        )

        print(
            f"Matching time         : "
            f"{matching_time:.4f} s"
        )

        # ====================================================
        # SAVE ALL KEYPOINT VISUALIZATIONS
        # ====================================================

        camera_keypoint_visual = (
            create_keypoint_visualization(
                frame,
                keypoints0,
                "Camera"
            )
        )

        satellite_keypoint_visual = (
            create_keypoint_visualization(
                satellite_matching,
                keypoints1,
                "Satellite ROI"
            )
        )

        cv2.imwrite(
            keypoints_camera_file,
            camera_keypoint_visual
        )

        cv2.imwrite(
            keypoints_satellite_file,
            satellite_keypoint_visual
        )

        # ====================================================
        # SAVE MATCH DATA
        # ====================================================

        np.savez(
            matches_npz,

            # ALL SuperPoint keypoints
            keypoints0=keypoints0,
            keypoints1=keypoints1,

            # SuperGlue matched points
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
            matches_vis,
            match_visual
        )

        # ====================================================
        # RANSAC
        # ====================================================

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
            f"Total matches        : "
            f"{total_matches}"
        )

        print(
            f"RANSAC inliers       : "
            f"{inliers}"
        )

        print(
            f"RANSAC outliers      : "
            f"{outliers}"
        )

        print(
            f"Inlier ratio         : "
            f"{inlier_ratio * 100:.2f}%"
        )

        print(
            f"Mean inlier confidence: "
            f"{mean_inlier_confidence:.4f}"
        )

        print()
        print("HOMOGRAPHY")
        print("-" * 80)
        print(H)

        print(
            f"\nRANSAC time          : "
            f"{ransac_time:.4f} s"
        )

        np.savez(
            ransac_npz,

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
            ransac_vis,
            inlier_visual
        )

        # ====================================================
        # POSITION ESTIMATION
        # ====================================================

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

        print(
            f"Frame center X      : "
            f"{frame_center_x:.3f}"
        )

        print(
            f"Frame center Y      : "
            f"{frame_center_y:.3f}"
        )

        print(
            f"Matching ROI X      : "
            f"{matching_x:.3f}"
        )

        print(
            f"Matching ROI Y      : "
            f"{matching_y:.3f}"
        )

        print(
            f"Original ROI X      : "
            f"{original_roi_x:.3f}"
        )

        print(
            f"Original ROI Y      : "
            f"{original_roi_y:.3f}"
        )

        print(
            f"Map column          : "
            f"{map_column:.3f}"
        )

        print(
            f"Map row             : "
            f"{map_row:.3f}"
        )

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
            position_vis,
            position_visual
        )

        # ====================================================
        # GPS VS VISUAL
        # ====================================================

        print_section(
            "GPS VS VISUAL DIAGNOSTIC"
        )

        gps_column_difference = (
            map_column
            - center_col
        )

        gps_row_difference = (
            map_row
            - center_row
        )

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

        print(
            f"GPS latitude        : "
            f"{gps_latitude:.10f}"
        )

        print(
            f"GPS longitude       : "
            f"{gps_longitude:.10f}"
        )

        print(
            f"Visual latitude     : "
            f"{estimated_latitude:.10f}"
        )

        print(
            f"Visual longitude    : "
            f"{estimated_longitude:.10f}"
        )

        print()

        print(
            f"East difference     : "
            f"{east_difference_m:.3f} m"
        )

        print(
            f"North difference    : "
            f"{north_difference_m:.3f} m"
        )

        print(
            f"Horizontal difference: "
            f"{horizontal_difference_m:.3f} m"
        )

        # ====================================================
        # TOTAL FRAME TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - frame_start
        )

        # ====================================================
        # SAVE REPORT
        # ====================================================

        save_frame_report(

            report_file,

            frame_name,

            frame_width,
            frame_height,

            frame_timestamp_ns,

            gps_timestamp_ns,

            gps_difference_ms,

            gps_latitude,
            gps_longitude,
            altitude,

            center_row,
            center_col,

            vfov_deg,

            footprint_width_m,
            footprint_height_m,

            x_m_per_px,
            y_m_per_px,

            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,

            roi_width,
            roi_height,

            matching_width,
            matching_height,

            resize_scale,

            device,

            keypoints0_count,
            keypoints1_count,

            total_matches,

            mean_confidence,

            matching_time,

            H,

            inliers,
            outliers,

            inlier_ratio,

            mean_inlier_confidence,

            ransac_time,

            frame_center_x,
            frame_center_y,

            matching_x,
            matching_y,

            original_roi_x,
            original_roi_y,

            map_column,
            map_row,

            estimated_latitude,
            estimated_longitude,

            east_difference_m,
            north_difference_m,

            horizontal_difference_m,

            total_time
        )

        # ====================================================
        # FINAL FRAME RESULT
        # ====================================================

        print_section(
            f"FRAME COMPLETE - {frame_name}"
        )

        print(
            f"Estimated latitude  : "
            f"{estimated_latitude:.10f}"
        )

        print(
            f"Estimated longitude : "
            f"{estimated_longitude:.10f}"
        )

        print(
            f"Inliers             : "
            f"{inliers}"
        )

        print(
            f"Inlier ratio        : "
            f"{inlier_ratio * 100:.2f}%"
        )

        print(
            f"Horizontal error    : "
            f"{horizontal_difference_m:.3f} m"
        )

        print(
            f"Frame processing time: "
            f"{total_time:.4f} s"
        )

        print(
            f"Output directory    : "
            f"{frame_output_dir}"
        )

        return {

            "frame": frame_name,

            "timestamp_ns":
                frame_timestamp_ns,

            "gps_timestamp_ns":
                gps_timestamp_ns,

            "gps_difference_ms":
                gps_difference_ms,

            "gps_latitude":
                gps_latitude,

            "gps_longitude":
                gps_longitude,

            "altitude":
                altitude,

            "camera_keypoints":
                keypoints0_count,

            "satellite_keypoints":
                keypoints1_count,

            "matches":
                total_matches,

            "inliers":
                inliers,

            "outliers":
                outliers,

            "inlier_ratio":
                inlier_ratio,

            "mean_match_confidence":
                mean_confidence,

            "mean_inlier_confidence":
                mean_inlier_confidence,

            "estimated_latitude":
                estimated_latitude,

            "estimated_longitude":
                estimated_longitude,

            "east_error_m":
                east_difference_m,

            "north_error_m":
                north_difference_m,

            "horizontal_error_m":
                horizontal_difference_m,

            "matching_time_s":
                matching_time,

            "ransac_time_s":
                ransac_time,

            "total_time_s":
                total_time,

            "status":
                "SUCCESS",

            "error":
                ""
        }

    except Exception as e:

        total_time = (
            time.perf_counter()
            - frame_start
        )

        print()
        print(
            f"ERROR processing {frame_name}"
        )

        print(
            f"Reason: {e}"
        )

        # Save failure information.

        with open(
            report_file,
            "w"
        ) as f:

            f.write(
                "FRAME PROCESSING FAILED\n"
            )

            f.write(
                "=" * 80 + "\n\n"
            )

            f.write(
                f"Frame      : {frame_name}\n"
            )

            f.write(
                f"Timestamp  : {frame_timestamp_ns}\n"
            )

            f.write(
                f"Error      : {e}\n"
            )

            f.write(
                f"Time       : {total_time:.6f} s\n"
            )

        return {

            "frame": frame_name,

            "timestamp_ns":
                frame_timestamp_ns,

            "gps_timestamp_ns":
                np.nan,

            "gps_difference_ms":
                np.nan,

            "gps_latitude":
                np.nan,

            "gps_longitude":
                np.nan,

            "altitude":
                np.nan,

            "camera_keypoints":
                np.nan,

            "satellite_keypoints":
                np.nan,

            "matches":
                np.nan,

            "inliers":
                np.nan,

            "outliers":
                np.nan,

            "inlier_ratio":
                np.nan,

            "mean_match_confidence":
                np.nan,

            "mean_inlier_confidence":
                np.nan,

            "estimated_latitude":
                np.nan,

            "estimated_longitude":
                np.nan,

            "east_error_m":
                np.nan,

            "north_error_m":
                np.nan,

            "horizontal_error_m":
                np.nan,

            "matching_time_s":
                np.nan,

            "ransac_time_s":
                np.nan,

            "total_time_s":
                total_time,

            "status":
                "FAILED",

            "error":
                str(e)
        }


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.perf_counter()

    print_section(
        "OFFLINE MULTI-FRAME SATELLITE MAP MATCHING"
    )

    print(
        f"Base directory        : {BASE_DIR}"
    )

    print(
        f"Frame directory       : {FRAME_DIR}"
    )

    print(
        f"Output directory      : {OUTPUT_DIR}"
    )

    print(
        f"Frame step            : {FRAME_STEP}"
    )

    # ========================================================
    # LOAD CSV DATA ONCE
    # ========================================================

    timestamp_df = (
        load_frame_timestamp_data()
    )

    gps_df = (
        load_gps_data()
    )

    # ========================================================
    # GET SELECTED FRAMES
    # ========================================================

    selected_frames = (
        get_frame_list(
            timestamp_df
        )
    )

    # ========================================================
    # OPEN SATELLITE MAP ONCE
    # ========================================================

    print_section(
        "OPENING SATELLITE MAP"
    )

    src = get_map_information()

    print(
        f"Map size              : "
        f"{src.width} x {src.height}"
    )

    print(
        f"CRS                   : "
        f"{src.crs}"
    )

    print(
        f"Bounds                : "
        f"{src.bounds}"
    )

    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    (
        matching,
        device,
        model_load_time
    ) = load_matching_model()

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    results = []

    total_frames = (
        len(selected_frames)
    )

    print_section(
        "STARTING MULTI-FRAME PROCESSING"
    )

    print(
        f"Total frames          : "
        f"{total_frames}"
    )

    print(
        f"Model will be reused  : YES"
    )

    print()

    for index, (
        frame_name,
        frame_path,
        frame_timestamp_ns
    ) in enumerate(
        selected_frames,
        start=1
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"# FRAME {index} / {total_frames}"
        )

        print(
            f"# {frame_name}"
        )

        print(
            "#" * 80
        )

        result = process_frame(

            frame_name,

            frame_path,

            frame_timestamp_ns,

            gps_df,

            src,

            matching,

            device,

            model_load_time
        )

        results.append(
            result
        )

        # ----------------------------------------------------
        # SAVE CSV AFTER EVERY FRAME
        #
        # This means if the program stops after frame 4,
        # results from frames 1-4 are already saved.
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        results_csv = os.path.join(
            OUTPUT_DIR,
            "multi_frame_results.csv"
        )

        results_df.to_csv(
            results_csv,
            index=False
        )

        print()
        print(
            f"Progress: "
            f"{index}/{total_frames} frames completed."
        )

    # ========================================================
    # CLOSE SATELLITE MAP
    # ========================================================

    src.close()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    total_processing_time = (
        time.perf_counter()
        - overall_start
    )

    results_df = pd.DataFrame(
        results
    )

    results_csv = os.path.join(
        OUTPUT_DIR,
        "multi_frame_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    successful = int(
        np.sum(
            results_df["status"]
            == "SUCCESS"
        )
    )

    failed = int(
        np.sum(
            results_df["status"]
            == "FAILED"
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "MULTI-FRAME PROCESSING COMPLETE"
    )

    print()
    print(
        f"Total frames processed : "
        f"{total_frames}"
    )

    print(
        f"Successful             : "
        f"{successful}"
    )

    print(
        f"Failed                 : "
        f"{failed}"
    )

    print(
        f"Model load time        : "
        f"{model_load_time:.4f} s"
    )

    print(
        f"Total processing time  : "
        f"{total_processing_time:.4f} s"
    )

    print()
    print(
        "RESULTS"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['frame']:20s} "
            f"status={result['status']:7s}"
        )

        if result["status"] == "SUCCESS":

            print(
                f"    GPS     : "
                f"{result['gps_latitude']:.7f}, "
                f"{result['gps_longitude']:.7f}"
            )

            print(
                f"    Visual  : "
                f"{result['estimated_latitude']:.7f}, "
                f"{result['estimated_longitude']:.7f}"
            )

            print(
                f"    Matches : "
                f"{result['matches']}"
            )

            print(
                f"    Inliers : "
                f"{result['inliers']} "
                f"({result['inlier_ratio'] * 100:.2f}%)"
            )

            print(
                f"    Error   : "
                f"{result['horizontal_error_m']:.3f} m"
            )

        else:

            print(
                f"    Error   : "
                f"{result['error']}"
            )

    print()
    print(
        "=" * 80
    )

    print(
        f"Final CSV saved to:"
    )

    print(
        results_csv
    )

    print()
    print(
        f"Per-frame results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
