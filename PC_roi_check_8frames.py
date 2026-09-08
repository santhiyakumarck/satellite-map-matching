#!/usr/bin/env python3

import os
import math
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = "/home/sandiya/sept 3-Drone-map-test"


# ============================================================
# INPUT FILES
# ============================================================

GPS_FILE = os.path.join(BASE_DIR, "gps.csv")

FRAME_TIMESTAMP_FILE = os.path.join(
    BASE_DIR,
    "frame_timestamps.csv"
)

SATELLITE_FILE = os.path.join(
    BASE_DIR,
    "Satellite_Z21.tif"
)


# ============================================================
# SELECT 8 FRAMES HERE
# ============================================================

FRAME_FILES = FRAME_FILE = os.path.join(
    BASE_DIR,
    "frame_06898.jpg"
)


# ============================================================
# CAMERA / ROI PARAMETERS
# ============================================================

HFOV_DEG = 90.0

ROI_MARGIN_PX = 1000

# Drone altitude is AGL
USE_AGL = True


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "roi_check_8frames"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CSV FILES
# ============================================================

gps_df = pd.read_csv(GPS_FILE)

frame_ts_df = pd.read_csv(
    FRAME_TIMESTAMP_FILE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_frame_timestamp(frame_name):

    # Try to find frame using common possible column names

    possible_frame_columns = [
        "frame",
        "frame_name",
        "filename",
        "file",
        "image",
        "image_name"
    ]

    frame_column = None

    for col in possible_frame_columns:
        if col in frame_ts_df.columns:
            frame_column = col
            break

    if frame_column is None:
        raise RuntimeError(
            "Could not find frame-name column in frame_timestamps.csv"
        )

    rows = frame_ts_df[
        frame_ts_df[frame_column].astype(str) == str(frame_name)
    ]

    if len(rows) == 0:

        # Also try without extension

        frame_stem = os.path.splitext(frame_name)[0]

        rows = frame_ts_df[
            frame_ts_df[frame_column]
            .astype(str)
            .str.replace(".jpg", "", regex=False)
            .str.replace(".png", "", regex=False)
            == frame_stem
        ]

    if len(rows) == 0:
        raise RuntimeError(
            f"No timestamp found for {frame_name}"
        )

    row = rows.iloc[0]

    possible_timestamp_columns = [
        "timestamp",
        "frame_timestamp",
        "bag_timestamp",
        "timestamp_ns"
    ]

    timestamp_column = None

    for col in possible_timestamp_columns:
        if col in frame_ts_df.columns:
            timestamp_column = col
            break

    if timestamp_column is None:
        raise RuntimeError(
            "Could not find timestamp column in frame_timestamps.csv"
        )

    return int(row[timestamp_column])


# ============================================================

def get_nearest_gps(frame_timestamp_ns):

    gps_timestamps = (
        gps_df["bag_timestamp"]
        .astype(np.int64)
        .values
    )

    differences = np.abs(
        gps_timestamps - frame_timestamp_ns
    )

    index = np.argmin(differences)

    gps_row = gps_df.iloc[index]

    latitude = float(gps_row["latitude"])
    longitude = float(gps_row["longitude"])
    altitude = float(gps_row["altitude"])

    difference_ms = (
        differences[index] / 1e6
    )

    return (
        latitude,
        longitude,
        altitude,
        difference_ms
    )


# ============================================================

def gps_to_pixel(src, latitude, longitude):

    row, col = rowcol(
        src.transform,
        longitude,
        latitude
    )

    return int(row), int(col)


# ============================================================

def calculate_roi(
    center_row,
    center_col,
    altitude,
    frame_width,
    frame_height,
    pixel_size_x,
    pixel_size_y
):

    # --------------------------------------------------------
    # Camera vertical FOV
    # --------------------------------------------------------

    vfov_rad = 2.0 * math.atan(
        math.tan(
            math.radians(HFOV_DEG) / 2.0
        )
        * (
            frame_height /
            frame_width
        )
    )

    vfov_deg = math.degrees(vfov_rad)

    # --------------------------------------------------------
    # Ground footprint
    # --------------------------------------------------------

    footprint_width_m = (
        2.0
        * altitude
        * math.tan(
            math.radians(HFOV_DEG) / 2.0
        )
    )

    footprint_height_m = (
        2.0
        * altitude
        * math.tan(
            vfov_rad / 2.0
        )
    )

    # --------------------------------------------------------
    # Convert ground footprint to map pixels
    # --------------------------------------------------------

    footprint_width_px = (
        footprint_width_m /
        pixel_size_x
    )

    footprint_height_px = (
        footprint_height_m /
        pixel_size_y
    )

    # --------------------------------------------------------
    # ROI including 1000 px margin
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ROI coordinates
    # --------------------------------------------------------

    x1 = int(
        center_col
        - roi_width / 2
    )

    y1 = int(
        center_row
        - roi_height / 2
    )

    x2 = x1 + roi_width
    y2 = y1 + roi_height

    return (
        x1,
        y1,
        x2,
        y2,
        roi_width,
        roi_height,
        vfov_deg,
        footprint_width_m,
        footprint_height_m,
        footprint_width_px,
        footprint_height_px
    )


# ============================================================

def extract_roi(
    src,
    x1,
    y1,
    x2,
    y2
):

    # Clip ROI to actual GeoTIFF dimensions

    clipped_x1 = max(0, x1)
    clipped_y1 = max(0, y1)

    clipped_x2 = min(src.width, x2)
    clipped_y2 = min(src.height, y2)

    width = clipped_x2 - clipped_x1
    height = clipped_y2 - clipped_y1

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "ROI is completely outside the GeoTIFF"
        )

    # Rasterio window

    window = rasterio.windows.Window(
        clipped_x1,
        clipped_y1,
        width,
        height
    )

    data = src.read(
        [1, 2, 3],
        window=window
    )

    # Rasterio: C,H,W
    # OpenCV: H,W,C

    roi = np.transpose(
        data,
        (1, 2, 0)
    )

    roi = np.clip(
        roi,
        0,
        255
    ).astype(np.uint8)

    return (
        roi,
        clipped_x1,
        clipped_y1,
        clipped_x2,
        clipped_y2
    )


# ============================================================

def create_roi_visualization(
    src,
    center_row,
    center_col,
    x1,
    y1,
    x2,
    y2,
    output_file
):

    # Read the whole map for visualization only.
    # This is NOT used for ROI calculation.

    data = src.read(
        [1, 2, 3]
    )

    image = np.transpose(
        data,
        (1, 2, 0)
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # Resize only visualization

    max_dimension = 2000

    scale = min(
        1.0,
        max_dimension /
        max(image.shape[:2])
    )

    if scale < 1.0:

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # Coordinates need same scale

    vx1 = int(x1 * scale)
    vy1 = int(y1 * scale)
    vx2 = int(x2 * scale)
    vy2 = int(y2 * scale)

    vcx = int(center_col * scale)
    vcy = int(center_row * scale)

    # ROI rectangle

    cv2.rectangle(
        image,
        (vx1, vy1),
        (vx2, vy2),
        (255, 0, 0),
        4
    )

    # GPS/map center

    cv2.circle(
        image,
        (vcx, vcy),
        10,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        output_file,
        image
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("8-FRAME SATELLITE ROI CHECK")
    print("=" * 70)

    print()
    print("Satellite :", SATELLITE_FILE)
    print("GPS CSV   :", GPS_FILE)
    print("Timestamp :", FRAME_TIMESTAMP_FILE)
    print("Frames    :", len(FRAME_FILES))
    print("ROI margin:", ROI_MARGIN_PX, "px")
    print()

    # --------------------------------------------------------
    # Open GeoTIFF once
    # --------------------------------------------------------

    src = rasterio.open(
        SATELLITE_FILE
    )

    print("GeoTIFF information")
    print("--------------------")
    print("Width  :", src.width)
    print("Height :", src.height)
    print("CRS    :", src.crs)

    bounds = src.bounds

    print("Left   :", bounds.left)
    print("Right  :", bounds.right)
    print("Bottom :", bounds.bottom)
    print("Top    :", bounds.top)

    # Pixel size

    pixel_size_x = abs(
        src.transform.a
    )

    pixel_size_y = abs(
        src.transform.e
    )

    print(
        "Pixel size X :",
        pixel_size_x
    )

    print(
        "Pixel size Y :",
        pixel_size_y
    )

    print()

    results = []

    # ========================================================
    # PROCESS 8 FRAMES
    # ========================================================

    for number, frame_name in enumerate(
        FRAME_FILES,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"FRAME {number}/8 : {frame_name}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Load frame
        # ----------------------------------------------------

        frame_path = os.path.join(
            BASE_DIR,
            frame_name
        )

        frame = cv2.imread(
            frame_path
        )

        if frame is None:

            print(
                "ERROR: Could not read:",
                frame_path
            )

            continue

        frame_height, frame_width = (
            frame.shape[:2]
        )

        print(
            "Frame size :",
            frame_width,
            "x",
            frame_height
        )

        # ----------------------------------------------------
        # Frame timestamp
        # ----------------------------------------------------

        frame_timestamp_ns = (
            get_frame_timestamp(
                frame_name
            )
        )

        print(
            "Frame timestamp :",
            frame_timestamp_ns
        )

        # ----------------------------------------------------
        # Nearest GPS
        # ----------------------------------------------------

        (
            latitude,
            longitude,
            altitude,
            gps_difference_ms
        ) = get_nearest_gps(
            frame_timestamp_ns
        )

        print(
            f"GPS latitude  : {latitude:.10f}"
        )

        print(
            f"GPS longitude : {longitude:.10f}"
        )

        print(
            f"Altitude AGL  : {altitude:.3f} m"
        )

        print(
            f"GPS time diff : {gps_difference_ms:.3f} ms"
        )

        # ----------------------------------------------------
        # GPS → GeoTIFF pixel
        # ----------------------------------------------------

        center_row, center_col = (
            gps_to_pixel(
                src,
                latitude,
                longitude
            )
        )

        print()
        print(
            "GPS → GeoTIFF pixel"
        )

        print(
            "Column :",
            center_col
        )

        print(
            "Row    :",
            center_row
        )

        # ----------------------------------------------------
        # ROI calculation
        # ----------------------------------------------------

        (
            x1,
            y1,
            x2,
            y2,
            roi_width,
            roi_height,
            vfov_deg,
            footprint_width_m,
            footprint_height_m,
            footprint_width_px,
            footprint_height_px
        ) = calculate_roi(
            center_row,
            center_col,
            altitude,
            frame_width,
            frame_height,
            pixel_size_x,
            pixel_size_y
        )

        print()
        print(
            "ROI calculation"
        )

        print(
            f"Vertical FOV       : {vfov_deg:.3f} deg"
        )

        print(
            f"Ground width       : {footprint_width_m:.3f} m"
        )

        print(
            f"Ground height      : {footprint_height_m:.3f} m"
        )

        print(
            f"Footprint width    : {footprint_width_px:.1f} px"
        )

        print(
            f"Footprint height   : {footprint_height_px:.1f} px"
        )

        print()

        print(
            "Requested ROI"
        )

        print(
            f"x1={x1}, y1={y1}, "
            f"x2={x2}, y2={y2}"
        )

        print(
            f"Requested size : "
            f"{roi_width} x {roi_height}"
        )

        # ----------------------------------------------------
        # Extract ROI
        # ----------------------------------------------------

        (
            roi,
            clipped_x1,
            clipped_y1,
            clipped_x2,
            clipped_y2
        ) = extract_roi(
            src,
            x1,
            y1,
            x2,
            y2
        )

        print()
        print(
            "Clipped ROI"
        )

        print(
            f"x1={clipped_x1}, "
            f"y1={clipped_y1}, "
            f"x2={clipped_x2}, "
            f"y2={clipped_y2}"
        )

        print(
            f"Actual ROI size : "
            f"{roi.shape[1]} x {roi.shape[0]}"
        )

        # ----------------------------------------------------
        # Save ROI
        # ----------------------------------------------------

        frame_stem = os.path.splitext(
            frame_name
        )[0]

        roi_file = os.path.join(
            OUTPUT_DIR,
            f"{frame_stem}_roi.png"
        )

        cv2.imwrite(
            roi_file,
            roi
        )

        print()
        print(
            "ROI saved :",
            roi_file
        )

        # ----------------------------------------------------
        # Save ROI visualization
        # ----------------------------------------------------

        visualization_file = os.path.join(
            OUTPUT_DIR,
            f"{frame_stem}_roi_location.jpg"
        )

        create_roi_visualization(
            src,
            center_row,
            center_col,
            clipped_x1,
            clipped_y1,
            clipped_x2,
            clipped_y2,
            visualization_file
        )

        print(
            "ROI visualization :",
            visualization_file
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "frame": frame_name,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude,
            "gps_time_diff_ms": gps_difference_ms,
            "map_col": center_col,
            "map_row": center_row,
            "roi_x1": clipped_x1,
            "roi_y1": clipped_y1,
            "roi_x2": clipped_x2,
            "roi_y2": clipped_y2,
            "roi_width": roi.shape[1],
            "roi_height": roi.shape[0]
        })

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_file = os.path.join(
        OUTPUT_DIR,
        "roi_summary.csv"
    )

    summary_df = pd.DataFrame(
        results
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print()
    print("=" * 70)
    print("ROI SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary saved :",
        summary_file
    )

    src.close()

    print()
    print("=" * 70)
    print("ROI CHECK COMPLETED")
    print("=" * 70)


# ============================================================

if __name__ == "__main__":
    main()
