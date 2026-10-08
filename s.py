#!/usr/bin/env python3

import threading
import time
from collections import deque

import cv2
import numpy as np
import rasterio
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import CompressedImage, NavSatFix

from cv_bridge import CvBridge

import sys

# ============================================================
# SMM PATH
# ============================================================

SMM_PATH = "/home/sugashini/pi5"

if SMM_PATH not in sys.path:
    sys.path.insert(0, SMM_PATH)

import smm


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_TOPIC = "/cam0/image_raw/compressed"
GPS_TOPIC = "/mavros/global_position/raw/fix"

GPS_MAX_TIME_DIFF = 0.5

GPS_BUFFER_SIZE = 100


# ============================================================
# ROS 2 NODE
# ============================================================

class SatelliteMapMatchingNode(Node):

    def __init__(self):

        super().__init__("satellite_map_matching")

        self.get_logger().info("========================================")
        self.get_logger().info("Satellite Map Matching Node")
        self.get_logger().info("========================================")

        # ----------------------------------------------------
        # CV Bridge
        # ----------------------------------------------------

        self.bridge = CvBridge()

        # ----------------------------------------------------
        # Latest camera frame
        # ----------------------------------------------------

        self.latest_frame = None
        self.latest_frame_timestamp = None

        self.frame_lock = threading.Lock()

        self.frame_count = 0

        # ----------------------------------------------------
        # GPS buffer
        # ----------------------------------------------------

        self.gps_buffer = deque(maxlen=GPS_BUFFER_SIZE)

        self.gps_lock = threading.Lock()

        self.gps_count = 0

        # ----------------------------------------------------
        # Worker control
        # ----------------------------------------------------

        self.worker_running = True

        self.worker_thread = threading.Thread(
            target=self.smm_worker,
            daemon=True
        )

        # ----------------------------------------------------
        # Load PyTorch device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.get_logger().info(
            f"PyTorch device: {self.device}"
        )

        # ----------------------------------------------------
        # Load SuperPoint + SuperGlue ONCE
        # ----------------------------------------------------

        self.get_logger().info(
            "Loading SuperPoint + SuperGlue..."
        )

        start = time.perf_counter()

        self.matching = smm.create_matching_model(
            self.device
        )

        elapsed = time.perf_counter() - start

        self.get_logger().info(
            f"Models loaded in {elapsed:.3f} sec"
        )

        # ----------------------------------------------------
        # Open satellite map ONCE
        # ----------------------------------------------------

        self.get_logger().info(
            "Opening satellite map..."
        )

        self.map_src = rasterio.open(
            smm.SATELLITE_MAP
        )

        self.get_logger().info(
            f"Satellite map: {smm.SATELLITE_MAP}"
        )

        self.get_logger().info(
            f"Map size: "
            f"{self.map_src.width} x {self.map_src.height}"
        )

        # ----------------------------------------------------
        # Camera QoS
        # ----------------------------------------------------

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ----------------------------------------------------
        # Camera subscriber
        # ----------------------------------------------------

        self.camera_subscription = self.create_subscription(
            CompressedImage,
            CAMERA_TOPIC,
            self.camera_callback,
            camera_qos,
        )

        # ----------------------------------------------------
        # GPS subscriber
        # ----------------------------------------------------

        self.gps_subscription = self.create_subscription(
            NavSatFix,
            GPS_TOPIC,
            self.gps_callback,
            qos_profile_sensor_data,
        )

        # ----------------------------------------------------
        # Start worker
        # ----------------------------------------------------

        self.worker_thread.start()

        self.get_logger().info(
            "ROS 2 subscriptions started"
        )

        self.get_logger().info(
            f"Camera topic: {CAMERA_TOPIC}"
        )

        self.get_logger().info(
            f"GPS topic: {GPS_TOPIC}"
        )

        self.get_logger().info(
            "SMM worker started"
        )

    # ========================================================
    # CAMERA CALLBACK
    # ========================================================

    def camera_callback(self, msg):

        try:

            # Decode compressed image
            np_arr = np.frombuffer(
                msg.data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if frame is None:

                self.get_logger().warning(
                    "Failed to decode camera frame"
                )

                return

            # ROS timestamp
            timestamp = (
                float(msg.header.stamp.sec)
                +
                float(msg.header.stamp.nanosec) * 1e-9
            )

            # ------------------------------------------------
            # Store ONLY latest frame
            # ------------------------------------------------

            with self.frame_lock:

                self.latest_frame = frame

                self.latest_frame_timestamp = timestamp

            self.frame_count += 1

            # Print every 10 frames
            if self.frame_count % 10 == 0:

                height, width = frame.shape[:2]

                self.get_logger().info(
                    f"Camera frames received: "
                    f"{self.frame_count} | "
                    f"resolution: {width}x{height}"
                )

        except Exception as e:

            self.get_logger().error(
                f"Camera callback error: {e}"
            )

    # ========================================================
    # GPS CALLBACK
    # ========================================================

    def gps_callback(self, msg):

        try:

            latitude = float(msg.latitude)
            longitude = float(msg.longitude)
            altitude = float(msg.altitude)

            timestamp = (
                float(msg.header.stamp.sec)
                +
                float(msg.header.stamp.nanosec) * 1e-9
            )

            # ------------------------------------------------
            # Ignore invalid GPS
            # ------------------------------------------------

            if latitude == 0.0 and longitude == 0.0:

                return

            # ------------------------------------------------
            # Store GPS sample
            # ------------------------------------------------

            with self.gps_lock:

                self.gps_buffer.append(
                    {
                        "timestamp": timestamp,
                        "latitude": latitude,
                        "longitude": longitude,
                        "altitude": altitude,
                    }
                )

            self.gps_count += 1

        except Exception as e:

            self.get_logger().error(
                f"GPS callback error: {e}"
            )

    # ========================================================
    # FIND NEAREST GPS
    # ========================================================

    def get_nearest_gps(self, frame_timestamp):

        with self.gps_lock:

            if not self.gps_buffer:

                return None

            gps_samples = list(self.gps_buffer)

        # Find GPS sample closest to frame timestamp

        nearest = min(
            gps_samples,
            key=lambda gps: abs(
                gps["timestamp"] - frame_timestamp
            )
        )

        time_difference = abs(
            nearest["timestamp"] - frame_timestamp
        )

        if time_difference > GPS_MAX_TIME_DIFF:

            return None

        return nearest

    # ========================================================
    # SMM WORKER
    # ========================================================

    def smm_worker(self):

        self.get_logger().info(
            "SMM worker waiting for camera + GPS..."
        )

        last_processed_timestamp = None

        while self.worker_running and rclpy.ok():

            try:

                # ------------------------------------------------
                # Get latest frame
                # ------------------------------------------------

                with self.frame_lock:

                    frame = self.latest_frame

                    frame_timestamp = (
                        self.latest_frame_timestamp
                    )

                if frame is None:

                    time.sleep(0.1)

                    continue

                # ------------------------------------------------
                # Don't process same frame twice
                # ------------------------------------------------

                if (
                    last_processed_timestamp is not None
                    and
                    frame_timestamp == last_processed_timestamp
                ):

                    time.sleep(0.1)

                    continue

                # ------------------------------------------------
                # Find synchronized GPS
                # ------------------------------------------------

                gps = self.get_nearest_gps(
                    frame_timestamp
                )

                if gps is None:

                    time.sleep(0.1)

                    continue

                # ------------------------------------------------
                # Mark frame as selected
                # ------------------------------------------------

                last_processed_timestamp = (
                    frame_timestamp
                )

                self.get_logger().info(
                    "----------------------------------------"
                )

                self.get_logger().info(
                    "Starting satellite map matching"
                )

                self.get_logger().info(
                    f"Frame timestamp: "
                    f"{frame_timestamp:.6f}"
                )

                self.get_logger().info(
                    f"GPS timestamp: "
                    f"{gps['timestamp']:.6f}"
                )

                self.get_logger().info(
                    f"GPS time difference: "
                    f"{abs(gps['timestamp'] - frame_timestamp):.3f} sec"
                )

                self.get_logger().info(
                    f"GPS: "
                    f"{gps['latitude']:.7f}, "
                    f"{gps['longitude']:.7f}, "
                    f"alt={gps['altitude']:.3f} m"
                )

                # ------------------------------------------------
                # Run SMM
                # ------------------------------------------------

                start = time.perf_counter()

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

                # ------------------------------------------------
                # Print result
                # ------------------------------------------------

                self.get_logger().info(
                    "========== SMM RESULT =========="
                )

                self.get_logger().info(
                    f"Estimated latitude: "
                    f"{result['estimated_latitude']:.10f}"
                )

                self.get_logger().info(
                    f"Estimated longitude: "
                    f"{result['estimated_longitude']:.10f}"
                )

                self.get_logger().info(
                    f"Matches: "
                    f"{result['total_matches']}"
                )

                self.get_logger().info(
                    f"Inliers: "
                    f"{result['inliers']}"
                )

                self.get_logger().info(
                    f"Inlier ratio: "
                    f"{result['inlier_ratio']:.4f}"
                )

                self.get_logger().info(
                    f"Horizontal error: "
                    f"{result['horizontal_error_m']:.3f} m"
                )

                self.get_logger().info(
                    f"SMM time: "
                    f"{elapsed:.3f} sec"
                )

                self.get_logger().info(
                    "================================"
                )

            except Exception as e:

                self.get_logger().error(
                    f"SMM worker error: {e}"
                )

                # Prevent rapid error loop
                time.sleep(1.0)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            "Stopping SMM worker..."
        )

        self.worker_running = False

        if self.worker_thread.is_alive():

            self.worker_thread.join(
                timeout=2.0
            )

        # Close satellite map
        if self.map_src is not None:

            self.map_src.close()

        self.get_logger().info(
            "SMM worker stopped"
        )

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = SatelliteMapMatchingNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
