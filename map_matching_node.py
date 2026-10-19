#!/usr/bin/env python3

import threading
import time
import sys
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

from sensor_msgs.msg import (
    CompressedImage,
    NavSatFix,
    NavSatStatus,
)

from cv_bridge import CvBridge


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

# Maximum allowed time difference between camera and GPS.
GPS_MAX_TIME_DIFF = 0.5

GPS_BUFFER_SIZE = 100


# ============================================================
# BENCH-TEST CONFIGURATION
# ============================================================

# TEMPORARY CABIN TEST ONLY.
#
# True:
#   Uses fixed test GPS coordinates.
#   Allows SMM to run without a real GPS fix.
#
# False:
#   Uses real GPS data synchronized with the camera frame.
#   REQUIRED for flight testing.

USE_BENCH_TEST_GPS = False

BENCH_TEST_LATITUDE = 12.6510608
BENCH_TEST_LONGITUDE = 80.1395289
BENCH_TEST_ALTITUDE = 343.707782


# ============================================================
# OUTPUT TOPICS
# ============================================================

RANSAC_IMAGE_TOPIC = (
    "/satellite_map_matching/ransac_inlier_image"
)

ESTIMATED_POSITION_TOPIC = (
    "/satellite_map_matching/estimated_position"
)


# ============================================================
# ROS 2 NODE
# ============================================================

class SatelliteMapMatchingNode(Node):

    def __init__(self):

        super().__init__("satellite_map_matching")

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "Satellite Map Matching Node"
        )
        self.get_logger().info(
            "========================================"
        )

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

        self.gps_buffer = deque(
            maxlen=GPS_BUFFER_SIZE
        )

        self.gps_lock = threading.Lock()

        self.gps_count = 0

        # ----------------------------------------------------
        # Worker control
        # ----------------------------------------------------

        self.worker_running = True

        # ----------------------------------------------------
        # RANSAC INLIER IMAGE PUBLISHER
        # ----------------------------------------------------

        self.ransac_inlier_image_pub = (
            self.create_publisher(
                CompressedImage,
                RANSAC_IMAGE_TOPIC,
                10,
            )
        )

        # ----------------------------------------------------
        # ESTIMATED POSITION PUBLISHER
        # ----------------------------------------------------

        self.estimated_position_pub = (
            self.create_publisher(
                NavSatFix,
                ESTIMATED_POSITION_TOPIC,
                10,
            )
        )

        # ----------------------------------------------------
        # Load PyTorch device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
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
            f"{self.map_src.width} x "
            f"{self.map_src.height}"
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

        self.camera_subscription = (
            self.create_subscription(
                CompressedImage,
                CAMERA_TOPIC,
                self.camera_callback,
                camera_qos,
            )
        )

        # ----------------------------------------------------
        # GPS subscriber
        # ----------------------------------------------------

        self.gps_subscription = (
            self.create_subscription(
                NavSatFix,
                GPS_TOPIC,
                self.gps_callback,
                qos_profile_sensor_data,
            )
        )

        # ----------------------------------------------------
        # Start SMM worker
        # ----------------------------------------------------

        self.worker_thread = threading.Thread(
            target=self.smm_worker,
            daemon=True,
        )

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
            f"RANSAC image output: "
            f"{RANSAC_IMAGE_TOPIC}"
        )

        self.get_logger().info(
            f"Estimated position output: "
            f"{ESTIMATED_POSITION_TOPIC}"
        )

        if USE_BENCH_TEST_GPS:

            self.get_logger().warning(
                "BENCH TEST MODE ENABLED: "
                "using fixed test GPS coordinates. "
                "Disable this mode before flight!"
            )

        else:

            self.get_logger().info(
                "FLIGHT MODE: using real GPS data"
            )

        self.get_logger().info(
            "SMM worker started"
        )

    # ========================================================
    # CAMERA CALLBACK
    # ========================================================

    def camera_callback(self, msg):

        try:

            # Decode compressed image.
            np_arr = np.frombuffer(
                msg.data,
                dtype=np.uint8,
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR,
            )

            if frame is None:

                self.get_logger().warning(
                    "Failed to decode camera frame"
                )

                return

            # Use the original ROS camera timestamp.
            timestamp = (
                float(msg.header.stamp.sec)
                + float(
                    msg.header.stamp.nanosec
                ) * 1e-9
            )

            # Store ONLY the latest frame.
            with self.frame_lock:

                self.latest_frame = frame

                self.latest_frame_timestamp = (
                    timestamp
                )

            self.frame_count += 1

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
                + float(
                    msg.header.stamp.nanosec
                ) * 1e-9
            )

            # Ignore invalid coordinates.
            if (
                latitude == 0.0
                and longitude == 0.0
            ):

                return

            # Store GPS sample.
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

            gps_samples = list(
                self.gps_buffer
            )

        # Find GPS sample closest to the frame timestamp.
        nearest = min(
            gps_samples,
            key=lambda gps: abs(
                gps["timestamp"] - frame_timestamp
            ),
        )

        time_difference = abs(
            nearest["timestamp"]
            - frame_timestamp
        )

        if time_difference > GPS_MAX_TIME_DIFF:

            return None

        return nearest

    # ========================================================
    # PUBLISH RANSAC INLIER IMAGE
    # ========================================================

    def publish_ransac_inlier_image(
        self,
        result,
        frame_timestamp,
    ):

        try:

            inlier_image = result.get(
                "ransac_inlier_image"
            )

            if inlier_image is None:

                self.get_logger().error(
                    "RANSAC inlier image is missing "
                    "from the SMM result"
                )

                return False

            # Encode the result image as JPEG.
            success, encoded_image = cv2.imencode(
                ".jpg",
                inlier_image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80,
                ],
            )

            if not success:

                self.get_logger().error(
                    "Failed to encode RANSAC image"
                )

                return False

            # Construct compressed ROS image.
            image_msg = CompressedImage()

            # Preserve the timestamp of the camera frame
            # used for this SMM result.
            image_msg.header.stamp = (
                self.get_clock().now().to_msg()
            )

            image_msg.header.frame_id = (
                "satellite_map"
            )

            image_msg.format = "jpeg"

            image_msg.data = (
                encoded_image.tobytes()
            )

            self.ransac_inlier_image_pub.publish(
                image_msg
            )

            self.get_logger().info(
                "Published RANSAC inlier image"
            )

            return True

        except Exception as e:

            self.get_logger().error(
                f"RANSAC image publishing error: {e}"
            )

            return False

    # ========================================================
    # PUBLISH ESTIMATED POSITION
    # ========================================================

    def publish_estimated_position(
        self,
        result,
        altitude,
        frame_timestamp,
    ):

        try:

            estimated_latitude = float(
                result["estimated_latitude"]
            )

            estimated_longitude = float(
                result["estimated_longitude"]
            )

            # Check that the estimated coordinates
            # are valid geographic coordinates.
            if not (
                -90.0 <= estimated_latitude <= 90.0
            ):

                self.get_logger().error(
                    "Invalid estimated latitude"
                )

                return False

            if not (
                -180.0 <= estimated_longitude <= 180.0
            ):

                self.get_logger().error(
                    "Invalid estimated longitude"
                )

                return False

            position_msg = NavSatFix()

            # Preserve the camera frame timestamp.
            position_msg.header.stamp = (
                self.get_clock().now().to_msg()
            )

            position_msg.header.frame_id = (
                "satellite_map"
            )

            # Report a valid estimated position.
            position_msg.status.status = (
                NavSatStatus.STATUS_FIX
            )

            position_msg.status.service = (
                NavSatStatus.SERVICE_GPS
            )

            # These coordinates come from satellite
            # map matching, not directly from the GPS.
            position_msg.latitude = (
                estimated_latitude
            )

            position_msg.longitude = (
                estimated_longitude
            )

            # SMM estimates latitude and longitude.
            # Altitude remains the reference altitude
            # supplied to the algorithm.
            position_msg.altitude = float(
                altitude
            )

            # Position uncertainty is not calculated
            # here, so do not report a fabricated value.
            position_msg.position_covariance_type = (
                NavSatFix.COVARIANCE_TYPE_UNKNOWN
            )

            self.estimated_position_pub.publish(
                position_msg
            )

            self.get_logger().info(
                "Published estimated position: "
                f"lat={estimated_latitude:.10f}, "
                f"lon={estimated_longitude:.10f}"
            )

            return True

        except Exception as e:

            self.get_logger().error(
                f"Estimated position publishing error: {e}"
            )

            return False

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
                # Don't process the same frame twice
                # ------------------------------------------------

                if (
                    last_processed_timestamp
                    is not None
                    and frame_timestamp
                    == last_processed_timestamp
                ):

                    time.sleep(0.1)

                    continue

                # ------------------------------------------------
                # Find synchronized GPS or use bench coordinates
                # ------------------------------------------------

                if USE_BENCH_TEST_GPS:

                    # TEMPORARY CABIN TEST ONLY.
                    # The fixed coordinates are used to
                    # select the satellite ROI.

                    gps = {
                        "timestamp": frame_timestamp,
                        "latitude": (
                            BENCH_TEST_LATITUDE
                        ),
                        "longitude": (
                            BENCH_TEST_LONGITUDE
                        ),
                        "altitude": (
                            BENCH_TEST_ALTITUDE
                        ),
                    }

                else:

                    # Flight mode: require real synchronized GPS.
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

                if USE_BENCH_TEST_GPS:

                    self.get_logger().warning(
                        "Using BENCH TEST GPS coordinates"
                    )

                else:

                    self.get_logger().info(
                        f"GPS timestamp: "
                        f"{gps['timestamp']:.6f}"
                    )

                    self.get_logger().info(
                        f"GPS time difference: "
                        f"{abs(gps['timestamp'] - frame_timestamp):.3f} sec"
                    )

                self.get_logger().info(
                    f"Reference GPS: "
                    f"{gps['latitude']:.7f}, "
                    f"{gps['longitude']:.7f}, "
                    f"alt={gps['altitude']:.3f} m"
                )

                # ------------------------------------------------
                # Run satellite map matching
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

                elapsed = (
                    time.perf_counter() - start
                )

                # ------------------------------------------------
                # Print SMM result
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
                    f"SMM time: {elapsed:.3f} sec"
                )

                # ------------------------------------------------
                # Publish RANSAC inlier image
                # ------------------------------------------------

                image_published = (
                    self.publish_ransac_inlier_image(
                        result,
                        frame_timestamp,
                    )
                )

                # ------------------------------------------------
                # Publish estimated position
                # ------------------------------------------------

                position_published = (
                    self.publish_estimated_position(
                        result,
                        gps["altitude"],
                        frame_timestamp,
                    )
                )

                # ------------------------------------------------
                # Publication status
                # ------------------------------------------------

                self.get_logger().info(
                    "========== PUBLISH STATUS =========="
                )

                self.get_logger().info(
                    f"RANSAC image published: "
                    f"{image_published}"
                )

                self.get_logger().info(
                    f"Estimated position published: "
                    f"{position_published}"
                )

                self.get_logger().info(
                    "===================================="
                )

                self.get_logger().info(
                    "================================"
                )

            except Exception as e:

                self.get_logger().error(
                    f"SMM worker error: {e}"
                )

                # Prevent rapid error loop.
                time.sleep(1.0)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            "Stopping SMM worker..."
        )

        self.worker_running = False

        if (
            hasattr(self, "worker_thread")
            and self.worker_thread.is_alive()
        ):

            self.worker_thread.join(
                timeout=5.0
            )

        # Close satellite map.
        if hasattr(self, "map_src"):

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

    node = None

    try:

        node = SatelliteMapMatchingNode()

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        if node is not None:

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()
