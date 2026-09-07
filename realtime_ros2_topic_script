

#!/usr/bin/env python3

import os
import time
import threading

import cv2
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.transform import xy, rowcol

import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge

# ============================================================
# SUPERPOINT / SUPERGLUE
# ============================================================

from models.matching import Matching


class RealtimeMapMatching(Node):

    def __init__(self):

        super().__init__('realtime_map_matching')

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            'map_file',
            '/home/tunga/pi5/Map_Export_2026_08_28_162236.tif'
        )

        self.declare_parameter(
            'superpoint_weights',
            '/home/tunga/pi5/weights/superpoint_v1.pth'
        )

        self.declare_parameter(
            'superglue_weights',
            '/home/tunga/pi5/weights/superglue_outdoor.pth'
        )

        self.declare_parameter(
            'map_matching_hz',
            2.0
        )

        self.declare_parameter(
            'roi_size_pixels',
            300
        )

        self.declare_parameter(
            'max_keypoints',
            1024
        )

        self.declare_parameter(
            'min_inliers',
            10
        )

        self.declare_parameter(
            'min_inlier_ratio',
            0.25
        )

        # ====================================================
        # GET PARAMETERS
        # ====================================================

        self.map_file = self.get_parameter(
            'map_file'
        ).value

        self.superpoint_weights = self.get_parameter(
            'superpoint_weights'
        ).value

        self.superglue_weights = self.get_parameter(
            'superglue_weights'
        ).value

        self.map_matching_hz = self.get_parameter(
            'map_matching_hz'
        ).value

        self.roi_size = self.get_parameter(
            'roi_size_pixels'
        ).value

        self.max_keypoints = self.get_parameter(
            'max_keypoints'
        ).value

        self.min_inliers = self.get_parameter(
            'min_inliers'
        ).value

        self.min_inlier_ratio = self.get_parameter(
            'min_inlier_ratio'
        ).value

        # ====================================================
        # ROS
        # ====================================================

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/cam0/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_callback,
            qos_profile_sensor_data
        )

        # Map matched GPS output
        self.map_fix_pub = self.create_publisher(
            NavSatFix,
            '/map_matching/fix',
            10
        )

        # Map matched pose for RViz
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/map_matching/pose',
            10
        )

        # ====================================================
        # VARIABLES
        # ====================================================

        self.current_lat = None
        self.current_lon = None
        self.current_alt = None

        self.latest_frame = None

        self.gps_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.last_matching_time = 0.0

        # Initial position for RViz local coordinate system
        self.reference_lat = None
        self.reference_lon = None

        # ====================================================
        # LOAD GEOTIFF
        # ====================================================

        self.get_logger().info(
            'Loading GeoTIFF...'
        )

        if not os.path.exists(self.map_file):

            self.get_logger().error(
                f'GeoTIFF not found: {self.map_file}'
            )

            raise FileNotFoundError(self.map_file)

        self.map_dataset = rasterio.open(
            self.map_file
        )

        self.get_logger().info(
            f'Map width  : {self.map_dataset.width}'
        )

        self.get_logger().info(
            f'Map height : {self.map_dataset.height}'
        )

        self.get_logger().info(
            f'Map CRS    : {self.map_dataset.crs}'
        )

        self.get_logger().info(
            f'Map bounds : {self.map_dataset.bounds}'
        )

        self.transform = self.map_dataset.transform

        # ====================================================
        # LOAD SUPERPOINT + SUPERGLUE
        # ====================================================

        self.get_logger().info(
            'Loading SuperPoint + SuperGlue...'
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device

        self.get_logger().info(
            f'Inference device: {self.device}'
        )

        matching_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': self.max_keypoints
            },

            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2
            }
        }

        self.matching = Matching(
            matching_config
        ).eval().to(
            self.device
        )

        self.get_logger().info(
            'SuperPoint + SuperGlue loaded.'
        )

        # ====================================================
        # START PROCESSING TIMER
        # ====================================================

        timer_period = 1.0 / self.map_matching_hz

        self.timer = self.create_timer(
            timer_period,
            self.process_frame
        )

        self.get_logger().info(
            '=========================================='
        )

        self.get_logger().info(
            'REAL-TIME MAP MATCHING NODE STARTED'
        )

        self.get_logger().info(
            '=========================================='

        )

    # ========================================================
    # GPS CALLBACK
    # ========================================================

    def gps_callback(self, msg):

        with self.gps_lock:

            self.current_lat = msg.latitude
            self.current_lon = msg.longitude
            self.current_alt = msg.altitude

            if self.reference_lat is None:

                self.reference_lat = msg.latitude
                self.reference_lon = msg.longitude

                self.get_logger().info(
                    f'RViz reference GPS: '
                    f'{self.reference_lat:.8f}, '
                    f'{self.reference_lon:.8f}'
                )

    # ========================================================
    # CAMERA CALLBACK
    # ========================================================

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            with self.frame_lock:

                self.latest_frame = frame.copy()

        except Exception as e:

            self.get_logger().error(
                f'Image conversion error: {e}'
            )

    # ========================================================
    # MAIN PROCESSING
    # ========================================================

    def process_frame(self):

        # ----------------------------------------------------
        # GET LATEST GPS
        # ----------------------------------------------------

        with self.gps_lock:

            if self.current_lat is None:

                return

            gps_lat = self.current_lat
            gps_lon = self.current_lon

        # ----------------------------------------------------
        # GET LATEST CAMERA FRAME
        # ----------------------------------------------------

        with self.frame_lock:

            if self.latest_frame is None:

                return

            frame = self.latest_frame.copy()

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # GPS → SATELLITE MAP PIXEL
        # ----------------------------------------------------

        try:

            gps_row, gps_col = rowcol(
                self.transform,
                gps_lon,
                gps_lat
            )

        except Exception as e:

            self.get_logger().error(
                f'GPS to map conversion failed: {e}'
            )

            return

        # ----------------------------------------------------
        # CHECK MAP BOUNDARY
        # ----------------------------------------------------

        if (
            gps_col < 0 or
            gps_col >= self.map_dataset.width or
            gps_row < 0 or
            gps_row >= self.map_dataset.height
        ):

            self.get_logger().warn(
                'GPS position is outside GeoTIFF.'
            )

            return

        # ----------------------------------------------------
        # EXTRACT SATELLITE ROI
        # ----------------------------------------------------

        half = self.roi_size // 2

        col_start = max(
            0,
            gps_col - half
        )

        row_start = max(
            0,
            gps_row - half
        )

        col_end = min(
            self.map_dataset.width,
            gps_col + half
        )

        row_end = min(
            self.map_dataset.height,
            gps_row + half
        )

        width = col_end - col_start
        height = row_end - row_start

        if width <= 20 or height <= 20:

            self.get_logger().warn(
                'Satellite ROI too small.'
            )

            return

        # ----------------------------------------------------
        # READ GEO TIFF ROI
        # ----------------------------------------------------

        try:

            window = Window(
                col_start,
                row_start,
                width,
                height
            )

            satellite = self.map_dataset.read(
                window=window
            )

        except Exception as e:

            self.get_logger().error(
                f'GeoTIFF ROI read error: {e}'
            )

            return

        # ----------------------------------------------------
        # CONVERT SATELLITE TO BGR
        # ----------------------------------------------------

        if satellite.shape[0] >= 3:

            satellite = np.transpose(
                satellite[:3],
                (1, 2, 0)
            )

            satellite = cv2.cvtColor(
                satellite,
                cv2.COLOR_RGB2BGR
            )

        elif satellite.shape[0] == 1:

            satellite = satellite[0]

        else:

            self.get_logger().error(
                'Unsupported satellite image bands.'
            )

            return

        # ----------------------------------------------------
        # CAMERA IMAGE
        # ----------------------------------------------------

        drone_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        satellite_gray = cv2.cvtColor(
            satellite,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # CONVERT TO TORCH
        # ----------------------------------------------------

        drone_tensor = (
            torch.from_numpy(
                drone_gray
            )
            .float()
            / 255.0
        )

        satellite_tensor = (
            torch.from_numpy(
                satellite_gray
            )
            .float()
            / 255.0
        )

        drone_tensor = drone_tensor[
            None,
            None
        ].to(self.device)

        satellite_tensor = satellite_tensor[
            None,
            None
        ].to(self.device)

        # ----------------------------------------------------
        # SUPERPOINT + SUPERGLUE
        # ----------------------------------------------------

        try:

            with torch.no_grad():

                prediction = self.matching({

                    'image0': drone_tensor,

                    'image1': satellite_tensor

                })

        except Exception as e:

            self.get_logger().error(
                f'SuperPoint/SuperGlue error: {e}'
            )

            return

        # ----------------------------------------------------
        # GET RESULTS
        # ----------------------------------------------------

        keypoints0 = prediction[
            'keypoints0'
        ][0].cpu().numpy()

        keypoints1 = prediction[
            'keypoints1'
        ][0].cpu().numpy()

        matches0 = prediction[
            'matches0'
        ][0].cpu().numpy()

        confidence = prediction[
            'matching_scores0'
        ][0].cpu().numpy()

        # ----------------------------------------------------
        # VALID MATCHES
        # ----------------------------------------------------

        valid = matches0 > -1

        valid_indices = np.where(
            valid
        )[0]

        if len(valid_indices) < 4:

            self.get_logger().warn(
                f'Not enough matches: '
                f'{len(valid_indices)}'
            )

            return

        mkpts0 = keypoints0[
            valid_indices
        ]

        mkpts1 = keypoints1[
            matches0[valid_indices]
        ]

        mconf = confidence[
            valid_indices
        ]

        # ----------------------------------------------------
        # RANSAC HOMOGRAPHY
        # ----------------------------------------------------

        try:

            H, mask = cv2.findHomography(
                mkpts0,
                mkpts1,
                cv2.RANSAC,
                5.0
            )

        except Exception as e:

            self.get_logger().warn(
                f'RANSAC error: {e}'
            )

            return

        if H is None or mask is None:

            self.get_logger().warn(
                'Homography could not be estimated.'
            )

            return

        mask = mask.ravel()

        inliers = int(
            np.sum(mask)
        )

        total_matches = len(
            mkpts0
        )

        inlier_ratio = (
            inliers / total_matches
        )

        # ----------------------------------------------------
        # QUALITY CHECK
        # ----------------------------------------------------

        if (
            inliers < self.min_inliers or
            inlier_ratio < self.min_inlier_ratio
        ):

            self.get_logger().warn(
                f'MAP MATCH REJECTED | '
                f'Matches={total_matches} | '
                f'Inliers={inliers} | '
                f'Ratio={inlier_ratio:.2f}'
            )

            return

        # ----------------------------------------------------
        # ESTIMATE CENTER OF DRONE IMAGE
        # ----------------------------------------------------

        h, w = drone_gray.shape

        center_point = np.array(
            [
                [
                    [w / 2.0, h / 2.0]
                ]
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # DRONE CENTER → SATELLITE ROI
        # ----------------------------------------------------

        mapped_point = cv2.perspectiveTransform(
            center_point,
            H
        )

        satellite_x = float(
            mapped_point[0, 0, 0]
        )

        satellite_y = float(
            mapped_point[0, 0, 1]
        )

        # ----------------------------------------------------
        # ROI PIXEL → FULL GEOTIFF PIXEL
        # ----------------------------------------------------

        map_col = (
            col_start +
            satellite_x
        )

        map_row = (
            row_start +
            satellite_y
        )

        # ----------------------------------------------------
        # CHECK MAP PIXEL
        # ----------------------------------------------------

        if (
            map_col < 0 or
            map_col >= self.map_dataset.width or
            map_row < 0 or
            map_row >= self.map_dataset.height
        ):

            self.get_logger().warn(
                'Matched point outside GeoTIFF.'
            )

            return

        # ----------------------------------------------------
        # FULL MAP PIXEL → LAT/LON
        # ----------------------------------------------------

        matched_lon, matched_lat = xy(
            self.transform,
            map_row,
            map_col
        )

        # ----------------------------------------------------
        # PROCESSING TIME
        # ----------------------------------------------------

        processing_time = (
            time.perf_counter()
            - start_time
        )

        processing_ms = (
            processing_time * 1000.0
        )

        # ----------------------------------------------------
        # PUBLISH MAP-MATCHED GPS
        # ----------------------------------------------------

        self.publish_map_fix(
            matched_lat,
            matched_lon
        )

        # ----------------------------------------------------
        # PUBLISH RVIZ POSE
        # ----------------------------------------------------

        self.publish_rviz_pose(
            matched_lat,
            matched_lon
        )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        self.get_logger().info(
            f'MAP MATCH SUCCESS | '
            f'GPS=({gps_lat:.7f}, {gps_lon:.7f}) | '
            f'Matched=({matched_lat:.7f}, '
            f'{matched_lon:.7f}) | '
            f'Matches={total_matches} | '
            f'Inliers={inliers} | '
            f'Ratio={inlier_ratio:.2f} | '
            f'Time={processing_ms:.1f} ms'
        )

    # ========================================================
    # PUBLISH NAVSATFIX
    # ========================================================

    def publish_map_fix(
        self,
        latitude,
        longitude
    ):

        msg = NavSatFix()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = 'map'

        msg.latitude = latitude
        msg.longitude = longitude

        if self.current_alt is not None:

            msg.altitude = self.current_alt

        self.map_fix_pub.publish(
            msg
        )

    # ========================================================
    # LAT/LON → LOCAL XY FOR RVIZ
    # ========================================================

    def publish_rviz_pose(
        self,
        latitude,
        longitude
    ):

        if self.reference_lat is None:

            return

        # ----------------------------------------------------
        # Approximate local ENU conversion
        # ----------------------------------------------------

        earth_radius = 6378137.0

        lat0 = np.radians(
            self.reference_lat
        )

        dlat = np.radians(
            latitude -
            self.reference_lat
        )

        dlon = np.radians(
            longitude -
            self.reference_lon
        )

        north = (
            dlat *
            earth_radius
        )

        east = (
            dlon *
            earth_radius *
            np.cos(lat0)
        )

        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = 'map'

        msg.pose.position.x = east
        msg.pose.position.y = north
        msg.pose.position.z = 0.0

        msg.pose.orientation.w = 1.0

        self.pose_pub.publish(
            msg
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):

        try:

            self.map_dataset.close()

        except Exception:

            pass

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = RealtimeMapMatching()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
