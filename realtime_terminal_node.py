#!/usr/bin/env python3
import os
import sys
import time
import threading
import traceback
from collections import deque

import cv2
import numpy as np
import rasterio
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
)
from sensor_msgs.msg import CompressedImage, NavSatFix

SMM_PATH = "/home/sugashini/pi5"
sys.path.insert(0, SMM_PATH)
import smm

CAMERA_TOPIC = "/cam0/image_raw/compressed"
# Match the GPS input of the working map_matching_node.py.
GPS_TOPIC = "/mavros/global_position/global"
GPS_MAX_TIME_DIFF = 0.5
GPS_BUFFER_SIZE = 100
USE_BENCH_TEST_GPS = False
BENCH_TEST_LATITUDE = 12.6510608
BENCH_TEST_LONGITUDE = 80.1395289
BENCH_TEST_ALTITUDE = 343.707782
RESULTS_DIR = os.path.join(SMM_PATH, "realtime_results", "ransac_inliers")


class RealtimeTerminalNode(Node):
    def __init__(self):
        super().__init__("realtime_terminal_map_matching")
        self.frame_lock = threading.Lock()
        self.gps_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_timestamp_ns = None
        self.latest_frame_sequence = 0
        self.gps_buffer = deque(maxlen=GPS_BUFFER_SIZE)
        self.stop_event = threading.Event()
        self.processed_count = 0
        self.last_gps_warning_time = 0.0
        os.makedirs(RESULTS_DIR, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("=" * 65)
        self.get_logger().info("REALTIME TERMINAL SATELLITE MAP MATCHING")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Camera topic: {CAMERA_TOPIC}")
        self.get_logger().info(f"GPS topic: {GPS_TOPIC}")
        self.get_logger().info(f"Image output: {RESULTS_DIR}")
        self.get_logger().info("=" * 65)

        self.get_logger().info("Loading SuperPoint/SuperGlue model...")
        self.matching = smm.create_matching_model(self.device)
        self.get_logger().info("Opening satellite map...")
        self.map_src = rasterio.open(smm.SATELLITE_MAP)
        self.get_logger().info("Model and satellite map loaded.")

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.camera_sub = self.create_subscription(
            CompressedImage, CAMERA_TOPIC, self.camera_callback, camera_qos
        )
        self.gps_sub = self.create_subscription(
            NavSatFix, GPS_TOPIC, self.gps_callback, qos_profile_sensor_data
        )
        self.worker_thread = threading.Thread(
            target=self.matching_worker, daemon=True
        )
        self.worker_thread.start()
        self.get_logger().info("Waiting for camera frames and GPS...")

    def get_timestamp_ns(self, header):
        stamp = header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if timestamp_ns == 0:
            timestamp_ns = self.get_clock().now().nanoseconds
        return timestamp_ns

    def camera_callback(self, msg):
        try:
            compressed = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warning("Could not decode camera frame.")
                return
            timestamp_ns = self.get_timestamp_ns(msg.header)
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_timestamp_ns = timestamp_ns
                self.latest_frame_sequence += 1
        except Exception as exc:
            self.get_logger().error(f"Camera callback error: {exc}")

    def gps_callback(self, msg):
        try:
            lat, lon, alt = float(msg.latitude), float(msg.longitude), float(msg.altitude)
            if not (
                np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)
                and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
                and not (lat == 0.0 and lon == 0.0)
            ):
                return
            gps_data = {
                "timestamp_ns": self.get_timestamp_ns(msg.header),
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
            }
            with self.gps_lock:
                self.gps_buffer.append(gps_data)
        except Exception as exc:
            self.get_logger().error(f"GPS callback error: {exc}")

    def find_nearest_gps(self, frame_timestamp_ns):
        if USE_BENCH_TEST_GPS:
            return {
                "timestamp_ns": frame_timestamp_ns,
                "latitude": BENCH_TEST_LATITUDE,
                "longitude": BENCH_TEST_LONGITUDE,
                "altitude": BENCH_TEST_ALTITUDE,
            }, 0.0

        with self.gps_lock:
            gps_samples = list(self.gps_buffer)

        if not gps_samples:
            return None, None

        nearest_gps = min(
            gps_samples,
            key=lambda gps: abs(
                gps["timestamp_ns"] - frame_timestamp_ns
            ),
        )

        time_difference = abs(
            nearest_gps["timestamp_ns"] - frame_timestamp_ns
        ) / 1e9

        if time_difference > GPS_MAX_TIME_DIFF:
            return None, time_difference

        return nearest_gps, time_difference
    def matching_worker(self):
        last_sequence = 0

        while not self.stop_event.is_set():
            with self.frame_lock:
                if (
                    self.latest_frame is None
                    or self.latest_frame_sequence == last_sequence
                ):
                    frame = None
                else:
                    frame = self.latest_frame.copy()
                    frame_timestamp_ns = self.latest_frame_timestamp_ns
                    sequence = self.latest_frame_sequence

            if frame is None:
                self.stop_event.wait(0.1)
                continue

            gps, gps_time_difference = self.find_nearest_gps(
                frame_timestamp_ns
            )

            if gps is None:
                now = time.monotonic()
                if now - self.last_gps_warning_time >= 5.0:
                    self.get_logger().warning(
                        "No GPS sample within "
                        f"{GPS_MAX_TIME_DIFF:.1f}s of camera timestamp. "
                        "Waiting for synchronized GPS."
                    )
                    self.last_gps_warning_time = now

                self.stop_event.wait(0.2)
                continue

            # Mark this frame selected only after finding its GPS.
            last_sequence = sequence
            self.processed_count += 1

            self.get_logger().info(
                f"\n{'=' * 65}\n"
                f"Processing frame #{self.processed_count}\n"
                f"Frame timestamp: {frame_timestamp_ns}\n"
                f"GPS timestamp difference: {gps_time_difference:.3f} s"
            )
            self.get_logger().info(
                f"Reference GPS: {gps['latitude']:.8f}, "
                f"{gps['longitude']:.8f}, "
                f"altitude={gps['altitude']:.3f} m"
            )

            start = time.perf_counter()

            try:
                result = smm.run_map_matching(
                    frame=frame,
                    gps_latitude=gps["latitude"],
                    gps_longitude=gps["longitude"],
                    altitude=gps["altitude"],
                    src=self.map_src,
                    matching=self.matching,
                    device=self.device,
                )

                elapsed = time.perf_counter() - start

                estimated_lat = result.get("estimated_latitude")
                estimated_lon = result.get("estimated_longitude")

                self.get_logger().info("-" * 65)
                self.get_logger().info("SATELLITE MAP MATCHING RESULT")

                if estimated_lat is not None and estimated_lon is not None:
                    self.get_logger().info(
                        f"Estimated latitude : {estimated_lat:.10f}"
                    )
                    self.get_logger().info(
                        f"Estimated longitude: {estimated_lon:.10f}"
                    )
                else:
                    self.get_logger().warning(
                        "Estimated coordinates are missing from the result."
                    )

                self.get_logger().info(
                    f"Reference latitude  : {gps['latitude']:.10f}"
                )
                self.get_logger().info(
                    f"Reference longitude : {gps['longitude']:.10f}"
                )

                for key, label, fmt in [
                    ("frame_keypoints", "Frame keypoints", "d"),
                    ("satellite_keypoints", "Satellite keypoints", "d"),
                    ("total_matches", "Total matches", "d"),
                    ("inliers", "RANSAC inliers", "d"),
                    ("outliers", "RANSAC outliers", "d"),
                    ("inlier_ratio", "Inlier ratio", ".4f"),
                    ("mean_inlier_confidence", "Mean inlier confidence", ".4f"),
                    ("east_error_m", "East error", ".3f"),
                    ("north_error_m", "North error", ".3f"),
                    ("horizontal_error_m", "Horizontal error", ".3f"),
                    ("total_time", "Algorithm time", ".3f"),
                ]:
                    if key in result and result[key] is not None:
                        self.get_logger().info(
                            f"{label:24s}: {format(result[key], fmt)}"
                            + (
                                " m"
                                if key.endswith("_error_m")
                                else " s"
                                if key == "total_time"
                                else ""
                            )
                        )

                self.get_logger().info(
                    f"{'Terminal elapsed time':24s}: {elapsed:.3f} s"
                )

                image = result.get("ransac_inlier_image")
                if image is not None:
                    filename = os.path.join(
                        RESULTS_DIR,
                        f"ransac_inliers_{frame_timestamp_ns}.jpg",
                    )
                    if cv2.imwrite(filename, image):
                        self.get_logger().info(
                            f"RANSAC inlier image saved: {filename}"
                        )
                    else:
                        self.get_logger().error(
                            f"Failed to save RANSAC image: {filename}"
                        )
                else:
                    self.get_logger().warning(
                        "No RANSAC inlier image returned."
                    )

                self.get_logger().info("=" * 65)

            except Exception as exc:
                self.get_logger().error(
                    f"Map matching failed: {exc}\n{traceback.format_exc()}"
                )

    def destroy_node(self):
        self.stop_event.set()

        # Wait for any active map-matching call to finish before
        # closing the satellite map it is using.
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join()

        if hasattr(self, "map_src") and self.map_src is not None:
            self.map_src.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RealtimeTerminalNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(
            f"Failed to start realtime terminal node: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
