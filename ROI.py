#!/usr/bin/env python3

import os
import math
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# ONLY CHANGE THIS FRAME
# ============================================================

FRAME_FILE = "frame_00858.jpg"


# ============================================================
# FIXED SETTINGS
# ============================================================

BASE_DIR = "/home/sandiya/sept 3-Drone-map-test"

FRAME_DIR = os.path.join(
    BASE_DIR,
    "drone_data_20260903_144040",
    "frames"
)

GPS_CSV = os.path.join(BASE_DIR, "gps.csv")

FRAME_TIMESTAMP_CSV = os.path.join(
    BASE_DIR,
    "frame_timestamps.csv"
)

GEOTIFF_FILE = os.path.join(
    BASE_DIR,
    "Satellite_Z21.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "roi_results"
)

HFOV_DEG = 90.0

ROI_MARGIN_PX = 1000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("GPS-BASED SATELLITE ROI GENERATION")
print("=" * 70)


# ============================================================
# 1. FRAME PATH
# ============================================================

frame_path = os.path.join(FRAME_DIR, FRAME_FILE)

print("\nFrame:")
print(frame_path)

if not os.path.exists(frame_path):
    raise FileNotFoundError(
        f"\nFrame not found:\n{frame_path}"
    )


# ============================================================
# 2. READ FRAME
# ============================================================

frame = cv2.imread(frame_path)

if frame is None:
    raise RuntimeError("Could not read frame.")

frame_height, frame_width = frame.shape[:2]

print("\nFrame size:")
print(f"Width  : {frame_width}")
print(f"Height : {frame_height}")


# ============================================================
# 3. READ FRAME TIMESTAMP CSV
# ============================================================

frame_df = pd.read_csv(FRAME_TIMESTAMP_CSV)

print("\nFrame timestamp CSV columns:")
print(list(frame_df.columns))


# Find the requested frame
frame_rows = frame_df[
    frame_df["frame_filename"].astype(str) == FRAME_FILE
]

if len(frame_rows) == 0:

    # Try without extension
    frame_stem = os.path.splitext(FRAME_FILE)[0]

    frame_rows = frame_df[
        frame_df["frame_filename"]
        .astype(str)
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".png", "", regex=False)
        == frame_stem
    ]


if len(frame_rows) == 0:
    raise RuntimeError(
        f"Frame {FRAME_FILE} not found in frame_timestamps.csv"
    )


frame_row = frame_rows.iloc[0]

frame_timestamp_ns = int(frame_row["timestamp_ns"])

frame_timestamp_sec = float(
    frame_row["timestamp_sec"]
)


print("\nFrame timestamp:")
print(f"timestamp_ns  : {frame_timestamp_ns}")
print(f"timestamp_sec : {frame_timestamp_sec:.9f}")


# ============================================================
# 4. READ GPS CSV
# ============================================================

gps_df = pd.read_csv(GPS_CSV)

print("\nGPS CSV columns:")
print(list(gps_df.columns))


# ============================================================
# 5. RECONSTRUCT GPS TIMESTAMP IN SECONDS
# ============================================================

gps_df["gps_time_full_sec"] = (
    gps_df["gps_timestamp_sec"].astype(float)
    +
    gps_df["gps_timestamp_nanosec"].astype(float) / 1e9
)


# ============================================================
# 6. FIND NEAREST GPS TO FRAME
# ============================================================

gps_time_difference = np.abs(
    gps_df["gps_time_full_sec"].values
    -
    frame_timestamp_sec
)

nearest_index = np.argmin(gps_time_difference)

gps_row = gps_df.iloc[nearest_index]

nearest_gps_time_sec = float(
    gps_row["gps_time_full_sec"]
)

time_difference_sec = abs(
    frame_timestamp_sec
    -
    nearest_gps_time_sec
)

time_difference_ms = time_difference_sec * 1000.0


# ============================================================
# 7. GPS VALUES
# ============================================================

latitude = float(gps_row["latitude"])
longitude = float(gps_row["longitude"])
altitude = float(gps_row["altitude"])


print("\n" + "=" * 70)
print("GPS MATCH")
print("=" * 70)

print(f"Frame timestamp : {frame_timestamp_sec:.9f}")
print(f"GPS timestamp   : {nearest_gps_time_sec:.9f}")

print(f"Time difference : {time_difference_ms:.3f} ms")

print(f"\nLatitude  : {latitude}")
print(f"Longitude : {longitude}")
print(f"Altitude  : {altitude} m")


# ============================================================
# 8. READ SATELLITE GEOTIFF
# ============================================================

with rasterio.open(GEOTIFF_FILE) as src:

    map_width = src.width
    map_height = src.height

    bounds = src.bounds

    transform = src.transform

    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)


    print("\n" + "=" * 70)
    print("SATELLITE MAP")
    print("=" * 70)

    print(f"Width  : {map_width}")
    print(f"Height : {map_height}")

    print(f"Left   : {bounds.left}")
    print(f"Right  : {bounds.right}")
    print(f"Bottom : {bounds.bottom}")
    print(f"Top    : {bounds.top}")

    print(f"\nPixel size X : {pixel_size_x}")
    print(f"Pixel size Y : {pixel_size_y}")


    # ========================================================
    # 9. GPS -> MAP PIXEL
    # ========================================================

    gps_col, gps_row = ~transform * (
        longitude,
        latitude
    )

    gps_col = float(gps_col)
    gps_row = float(gps_row)


    print("\nGPS map pixel:")
    print(f"Column : {gps_col:.2f}")
    print(f"Row    : {gps_row:.2f}")


    # ========================================================
    # 10. CALCULATE VERTICAL FOV
    # ========================================================

    aspect_ratio = frame_height / frame_width

    hfov_rad = math.radians(HFOV_DEG)

    vfov_rad = 2.0 * math.atan(
        math.tan(hfov_rad / 2.0)
        * aspect_ratio
    )

    vfov_deg = math.degrees(vfov_rad)


    print("\nCamera FOV:")
    print(f"HFOV : {HFOV_DEG:.2f} degrees")
    print(f"VFOV : {vfov_deg:.2f} degrees")


    # ========================================================
    # 11. GROUND FOOTPRINT
    # ========================================================

    # Assuming altitude is height above ground
    ground_width_m = (
        2.0
        * altitude
        * math.tan(hfov_rad / 2.0)
    )

    ground_height_m = (
        2.0
        * altitude
        * math.tan(vfov_rad / 2.0)
    )


    print("\nGround footprint:")
    print(f"Width  : {ground_width_m:.3f} m")
    print(f"Height : {ground_height_m:.3f} m")


    # ========================================================
    # 12. GROUND METERS -> MAP PIXELS
    # ========================================================

    # Approximate conversion:
    # latitude degrees -> meters
    # longitude degrees -> meters

    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0
        * math.cos(math.radians(latitude))
    )


    ground_width_deg = (
        ground_width_m
        / meters_per_degree_lon
    )

    ground_height_deg = (
        ground_height_m
        / meters_per_degree_lat
    )


    footprint_width_px = (
        ground_width_deg
        / pixel_size_x
    )

    footprint_height_px = (
        ground_height_deg
        / pixel_size_y
    )


    print("\nMap footprint:")
    print(
        f"Width  : {footprint_width_px:.2f} px"
    )

    print(
        f"Height : {footprint_height_px:.2f} px"
    )


    # ========================================================
    # 13. INITIAL ROI
    # ========================================================

    half_width = footprint_width_px / 2.0
    half_height = footprint_height_px / 2.0


    left = int(
        math.floor(
            gps_col - half_width
        )
    )

    right = int(
        math.ceil(
            gps_col + half_width
        )
    )

    top = int(
        math.floor(
            gps_row - half_height
        )
    )

    bottom = int(
        math.ceil(
            gps_row + half_height
        )
    )


    print("\nInitial footprint ROI:")
    print(f"Left   : {left}")
    print(f"Top    : {top}")
    print(f"Right  : {right}")
    print(f"Bottom : {bottom}")


    # ========================================================
    # 14. ADD 1000 PX MARGIN
    # ========================================================

    left -= ROI_MARGIN_PX
    right += ROI_MARGIN_PX

    top -= ROI_MARGIN_PX
    bottom += ROI_MARGIN_PX


    print("\nROI after 1000 px margin:")
    print(f"Left   : {left}")
    print(f"Top    : {top}")
    print(f"Right  : {right}")
    print(f"Bottom : {bottom}")


    # ========================================================
    # 15. CLIP ROI TO MAP
    # ========================================================

    left_clipped = max(0, left)
    top_clipped = max(0, top)

    right_clipped = min(
        map_width,
        right
    )

    bottom_clipped = min(
        map_height,
        bottom
    )


    roi_width = (
        right_clipped
        -
        left_clipped
    )

    roi_height = (
        bottom_clipped
        -
        top_clipped
    )


    print("\nClipped ROI:")
    print(f"Left   : {left_clipped}")
    print(f"Top    : {top_clipped}")
    print(f"Right  : {right_clipped}")
    print(f"Bottom : {bottom_clipped}")

    print("\nROI size:")
    print(f"Width  : {roi_width}")
    print(f"Height : {roi_height}")


    # ========================================================
    # 16. EXTRACT ROI
    # ========================================================

    window = Window(
        left_clipped,
        top_clipped,
        roi_width,
        roi_height
    )

    roi = src.read(
        window=window
    )


    # ========================================================
    # 17. CONVERT TO IMAGE
    # ========================================================

    if roi.shape[0] >= 3:

        roi_image = np.transpose(
            roi[:3],
            (1, 2, 0)
        )

    elif roi.shape[0] == 1:

        roi_image = roi[0]

    else:

        roi_image = np.transpose(
            roi,
            (1, 2, 0)
        )


# ============================================================
# 18. SAVE ROI
# ============================================================

frame_stem = os.path.splitext(
    FRAME_FILE
)[0]

roi_output = os.path.join(
    OUTPUT_DIR,
    f"{frame_stem}_roi.png"
)

cv2.imwrite(
    roi_output,
    cv2.cvtColor(
        roi_image,
        cv2.COLOR_RGB2BGR
    )
)


# ============================================================
# 19. SAVE ROI INFORMATION
# ============================================================

info_output = os.path.join(
    OUTPUT_DIR,
    f"{frame_stem}_roi_info.csv"
)


info = pd.DataFrame([{

    "frame": FRAME_FILE,

    "frame_timestamp_sec":
        frame_timestamp_sec,

    "gps_timestamp_sec":
        nearest_gps_time_sec,

    "time_difference_ms":
        time_difference_ms,

    "latitude":
        latitude,

    "longitude":
        longitude,

    "altitude_m":
        altitude,

    "gps_pixel_column":
        gps_col,

    "gps_pixel_row":
        gps_row,

    "hfov_deg":
        HFOV_DEG,

    "vfov_deg":
        vfov_deg,

    "ground_width_m":
        ground_width_m,

    "ground_height_m":
        ground_height_m,

    "footprint_width_px":
        footprint_width_px,

    "footprint_height_px":
        footprint_height_px,

    "margin_px":
        ROI_MARGIN_PX,

    "roi_left":
        left_clipped,

    "roi_top":
        top_clipped,

    "roi_right":
        right_clipped,

    "roi_bottom":
        bottom_clipped,

    "roi_width":
        roi_width,

    "roi_height":
        roi_height,

}])

info.to_csv(
    info_output,
    index=False
)


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("ROI GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nROI image:")
print(roi_output)

print(f"\nROI information:")
print(info_output)

print("\nDone.")
