#!/usr/bin/env python3

import os
import math
import time
import threading
from collections import deque

import cv2
import numpy as np
import rasterio

from rasterio.transform import rowcol, xy

import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix

from geometry_msgs.msg import PoseStamped

from std_msgs.msg import Float32
from std_msgs.msg import String

from cv_bridge import CvBridge


# ============================================================
# SUPERGLUE IMPORT
# ============================================================

SUPERGLUE_REPO = (
    "/home/sandiya/sept 3-Drone-map-test/"
    "SuperGluePretrainedNetwork"
)

import sys

sys.path.insert(
    0,
    SUPERGLUE_REPO
)

from models.matching import Matching


# ============================================================
# MAP MATCHING NODE
# ============================================================

class MapMatchingNode(Node):

    def __init__(self):

        super().__init__(
            "satellite_map_matching"
        )

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            "camera_topic",
            "/camera/image_raw"
        )

        self.declare_parameter(
            "ins_topic",
            "/ins/fix"
        )

        self.declare_parameter(
            "satellite_map",
            "/home/sandiya/sept 3-Drone-map-test/"
            "Satellite_Z21.tif"
        )

        self.declare_parameter(
            "hfov_deg",
            90.0
        )

        self.declare_parameter(
            "roi_margin_px",
            1000
        )

        self.declare_parameter(
            "max_matching_dim",
            2048
        )

        self.declare_parameter(
            "max_keypoints",
            2048
        )

        self.declare_parameter(
            "keypoint_threshold",
            0.005
        )

        self.declare_parameter(
            "match_threshold",
            0.2
        )

        self.declare_parameter(
            "ransac_reproj_threshold",
            5.0
        )

        self.declare_parameter(
            "ransac_max_iters",
            5000
        )

        self.declare_parameter(
            "ransac_confidence",
            0.999
        )

        self.declare_parameter(
            "process_hz",
            1.0
        )

        self.declare_parameter(
            "publish_debug_image",
            True
        )

        # ====================================================
        # READ PARAMETERS
        # ====================================================

        self.camera_topic = self.get_parameter(
            "camera_topic"
        ).value

        self.ins_topic = self.get_parameter(
            "ins_topic"
        ).value

        self.satellite_map = self.get_parameter(
            "satellite_map"
        ).value

        self.hfov_deg = float(
            self.get_parameter(
                "hfov_deg"
            ).value
        )

        self.roi_margin_px = int(
            self.get_parameter(
                "roi_margin_px"
            ).value
        )

        self.max_matching_dim = int(
            self.get_parameter(
                "max_matching_dim"
            ).value
        )

        self.max_keypoints = int(
            self.get_parameter(
                "max_keypoints"
            ).value
        )

        self.keypoint_threshold = float(
            self.get_parameter(
                "keypoint_threshold"
            ).value
        )

        self.match_threshold = float(
            self.get_parameter(
                "match_threshold"
            ).value
        )

        self.ransac_reproj_threshold = float(
            self.get_parameter(
                "ransac_reproj_threshold"
            ).value
        )

        self.ransac_max_iters = int(
            self.get_parameter(
                "ransac_max_iters"
            ).value
        )

        self.ransac_confidence = float(
            self.get_parameter(
                "ransac_confidence"
            ).value
        )

        self.process_hz = float(
            self.get_parameter(
                "process_hz"
            ).value
        )

        self.publish_debug_image = bool(
            self.get_parameter(
                "publish_debug_image"
            ).value
        )

        # ====================================================
        # CV BRIDGE
        # ====================================================

        self.bridge = CvBridge()

        # ====================================================
        # DATA STORAGE
        # ====================================================

        self.latest_frame = None
        self.latest_frame_stamp = None

        self.ins_buffer = deque(
            maxlen=100
        )

        self.data_lock = threading.Lock()

        # ====================================================
        # OPEN SATELLITE MAP
        # ====================================================

        self.get_logger().info(
            "Opening satellite map..."
        )

        self.src = rasterio.open(
            self.satellite_map
        )

        self.get_logger().info(
            f"Satellite map: "
            f"{self.src.width} x {self.src.height}"
        )

        self.get_logger().info(
            f"CRS: {self.src.crs}"
        )

        # ====================================================
        # LOAD SUPERPOINT / SUPERGLUE
        # ====================================================

        self.get_logger().info(
            "Loading SuperPoint + SuperGlue..."
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.get_logger().info(
            f"Device: {self.device}"
        )

        config = {

            "superpoint": {

                "nms_radius": 4,

                "keypoint_threshold":
                    self.keypoint_threshold,

                "max_keypoints":
                    self.max_keypoints
            },

            "superglue": {

                "weights":
                    "outdoor",

                "sinkhorn_iterations":
                    20,

                "match_threshold":
                    self.match_threshold
            }
        }

        self.matching = (
            Matching(config)
            .eval()
            .to(self.device)
        )

        self.get_logger().info(
            "SuperPoint + SuperGlue loaded."
        )

        # ====================================================
        # ROS2 SUBSCRIBERS
        # ====================================================

        self.camera_subscriber = (
            self.create_subscription(
                Image,
                self.camera_topic,
                self.camera_callback,
                10
            )
        )

        self.ins_subscriber = (
            self.create_subscription(
                NavSatFix,
                self.ins_topic,
                self.ins_callback,
                50
            )
        )

        # ====================================================
        # ROS2 PUBLISHERS
        # ====================================================

        self.estimated_position_pub = (
            self.create_publisher(
                NavSatFix,
                "/map_matching/estimated_position",
                10
            )
        )

        self.pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/map_matching/pose",
                10
            )
        )

        self.inlier_ratio_pub = (
            self.create_publisher(
                Float32,
                "/map_matching/inlier_ratio",
                10
            )
        )

        self.processing_time_pub = (
            self.create_publisher(
                Float32,
                "/map_matching/processing_time",
                10
            )
        )

        self.status_pub = (
            self.create_publisher(
                String,
                "/map_matching/status",
                10
            )
        )

        if self.publish_debug_image:

            self.debug_image_pub = (
                self.create_publisher(
                    Image,
                    "/map_matching/debug_image",
                    2
                )
            )

        # ====================================================
        # PROCESSING TIMER
        # ====================================================

        timer_period = (
            1.0 / self.process_hz
        )

        self.processing_timer = (
            self.create_timer(
                timer_period,
                self.process_latest_frame
            )
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.frame_counter = 0

        self.process_counter = 0

        self.origin_latitude = None
        self.origin_longitude = None

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            "Satellite Map Matching Node started."
        )

        self.get_logger().info(
            f"Camera topic : {self.camera_topic}"
        )

        self.get_logger().info(
            f"INS topic    : {self.ins_topic}"
        )

        self.get_logger().info(
            f"Processing   : {self.process_hz} Hz"
        )

        self.get_logger().info(
            "========================================"
        )

    # ========================================================
    # CAMERA CALLBACK
    # ========================================================

    def camera_callback(
        self,
        msg
    ):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Camera conversion failed: {e}"
            )

            return

        with self.data_lock:

            self.latest_frame = frame

            self.latest_frame_stamp = (
                msg.header.stamp.sec
                +
                msg.header.stamp.nanosec * 1e-9
            )

    # ========================================================
    # INS CALLBACK
    # ========================================================

    def ins_callback(
        self,
        msg
    ):

        latitude = float(
            msg.latitude
        )

        longitude = float(
            msg.longitude
        )

        altitude = float(
            msg.altitude
        )

        timestamp = (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec * 1e-9
        )

        # Ignore invalid values.

        if not np.isfinite(latitude):
            return

        if not np.isfinite(longitude):
            return

        if not np.isfinite(altitude):
            return

        with self.data_lock:

            self.ins_buffer.append(
                (
                    timestamp,
                    latitude,
                    longitude,
                    altitude
                )
            )

    # ========================================================
    # FIND INS CLOSEST TO CAMERA TIMESTAMP
    # ========================================================

    def get_nearest_ins(
        self,
        camera_timestamp
    ):

        with self.data_lock:

            if len(self.ins_buffer) == 0:

                return None

            samples = list(
                self.ins_buffer
            )

        best_sample = min(
            samples,
            key=lambda sample:
                abs(
                    sample[0]
                    -
                    camera_timestamp
                )
        )

        return best_sample

    # ========================================================
    # CALCULATE VFOV
    # ========================================================

    def calculate_vfov(
        self,
        frame_width,
        frame_height
    ):

        hfov_rad = math.radians(
            self.hfov_deg
        )

        aspect_ratio = (
            frame_height
            /
            frame_width
        )

        vfov_rad = (
            2.0
            *
            math.atan(
                math.tan(
                    hfov_rad / 2.0
                )
                *
                aspect_ratio
            )
        )

        return math.degrees(
            vfov_rad
        )

    # ========================================================
    # GROUND FOOTPRINT
    # ========================================================

    def calculate_ground_footprint(
        self,
        altitude,
        vfov_deg
    ):

        width_m = (
            2.0
            *
            altitude
            *
            math.tan(
                math.radians(
                    self.hfov_deg
                ) / 2.0
            )
        )

        height_m = (
            2.0
            *
            altitude
            *
            math.tan(
                math.radians(
                    vfov_deg
                ) / 2.0
            )
        )

        return (
            width_m,
            height_m
        )

    # ========================================================
    # MAP METERS / PIXEL
    # ========================================================

    def calculate_map_meters_per_pixel(
        self,
        latitude
    ):

        x_deg = abs(
            self.src.transform.a
        )

        y_deg = abs(
            self.src.transform.e
        )

        meters_per_degree_lat = (
            111320.0
        )

        meters_per_degree_lon = (
            111320.0
            *
            math.cos(
                math.radians(
                    latitude
                )
            )
        )

        x_m_per_px = (
            x_deg
            *
            meters_per_degree_lon
        )

        y_m_per_px = (
            y_deg
            *
            meters_per_degree_lat
        )

        return (
            x_m_per_px,
            y_m_per_px
        )

    # ========================================================
    # CREATE ROI
    # ========================================================

    def create_roi(
        self,
        center_row,
        center_col,
        footprint_width_m,
        footprint_height_m,
        x_m_per_px,
        y_m_per_px
    ):

        footprint_width_px = (
            footprint_width_m
            /
            x_m_per_px
        )

        footprint_height_px = (
            footprint_height_m
            /
            y_m_per_px
        )

        roi_width = int(
            math.ceil(
                footprint_width_px
                +
                2 * self.roi_margin_px
            )
        )

        roi_height = int(
            math.ceil(
                footprint_height_px
                +
                2 * self.roi_margin_px
            )
        )

        x1 = int(
            math.floor(
                center_col
                -
                roi_width / 2.0
            )
        )

        y1 = int(
            math.floor(
                center_row
                -
                roi_height / 2.0
            )
        )

        x2 = x1 + roi_width
        y2 = y1 + roi_height

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            self.src.width,
            x2
        )

        y2 = min(
            self.src.height,
            y2
        )

        return (
            x1,
            y1,
            x2,
            y2
        )

    # ========================================================
    # READ SATELLITE ROI
    # ========================================================

    def read_roi(
        self,
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

        data = self.src.read(
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

    # ========================================================
    # RESIZE ROI
    # ========================================================

    def resize_for_matching(
        self,
        image
    ):

        height, width = (
            image.shape[:2]
        )

        largest_dimension = max(
            width,
            height
        )

        if (
            largest_dimension
            <=
            self.max_matching_dim
        ):

            return image.copy(), 1.0

        scale = (
            self.max_matching_dim
            /
            largest_dimension
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

    # ========================================================
    # IMAGE -> TORCH
    # ========================================================

    def image_to_tensor(
        self,
        image
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
            /
            255.0
        )

        tensor = tensor[
            None,
            None
        ]

        return tensor.to(
            self.device
        )

    # ========================================================
    # RUN SUPERPOINT + SUPERGLUE
    # ========================================================

    def run_superglue(
        self,
        frame,
        satellite
    ):

        frame_tensor = (
            self.image_to_tensor(
                frame
            )
        )

        satellite_tensor = (
            self.image_to_tensor(
                satellite
            )
        )

        with torch.no_grad():

            pred = self.matching({
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

        valid = (
            matches0 > -1
        )

        matched_kpts0 = (
            keypoints0[valid]
        )

        matched_indices1 = (
            matches0[valid]
            .astype(
                np.int32
            )
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
            matched_confidence
        )

    # ========================================================
    # RANSAC
    # ========================================================

    def run_ransac(
        self,
        kpts0,
        kpts1,
        confidence
    ):

        if len(kpts0) < 4:

            return None

        H, mask = cv2.findHomography(
            kpts0,
            kpts1,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
            None,
            self.ransac_max_iters,
            self.ransac_confidence
        )

        if H is None or mask is None:

            return None

        mask = (
            mask
            .ravel()
            .astype(bool)
        )

        inlier_kpts0 = (
            kpts0[mask]
        )

        inlier_kpts1 = (
            kpts1[mask]
        )

        inlier_confidence = (
            confidence[mask]
        )

        total_matches = len(
            kpts0
        )

        inliers = int(
            np.sum(mask)
        )

        outliers = (
            total_matches
            -
            inliers
        )

        inlier_ratio = (
            inliers
            /
            total_matches
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
            inliers,
            outliers,
            inlier_ratio,
            mean_inlier_confidence
        )

    # ========================================================
    # ESTIMATE POSITION
    # ========================================================

    def estimate_position(
        self,
        H,
        frame_width,
        frame_height,
        resize_scale,
        roi_x1,
        roi_y1
    ):

        center_x = (
            frame_width
            /
            2.0
        )

        center_y = (
            frame_height
            /
            2.0
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

        projected = (
            cv2.perspectiveTransform(
                point,
                H
            )
        )

        matching_x = float(
            projected[0, 0, 0]
        )

        matching_y = float(
            projected[0, 0, 1]
        )

        original_roi_x = (
            matching_x
            /
            resize_scale
        )

        original_roi_y = (
            matching_y
            /
            resize_scale
        )

        map_column = (
            roi_x1
            +
            original_roi_x
        )

        map_row = (
            roi_y1
            +
            original_roi_y
        )

        longitude, latitude = xy(
            self.src.transform,
            map_row,
            map_column
        )

        return (
            float(latitude),
            float(longitude),
            matching_x,
            matching_y,
            map_column,
            map_row
        )

    # ========================================================
    # PUBLISH POSITION
    # ========================================================

    def publish_position(
        self,
        latitude,
        longitude,
        altitude,
        stamp
    ):

        msg = NavSatFix()

        msg.header.stamp = stamp

        msg.header.frame_id = (
            "map"
        )

        msg.latitude = (
            latitude
        )

        msg.longitude = (
            longitude
        )

        msg.altitude = (
            altitude
        )

        self.estimated_position_pub.publish(
            msg
        )

    # ========================================================
    # PUBLISH RVIZ POSE
    # ========================================================

    def publish_rviz_pose(
        self,
        latitude,
        longitude,
        stamp
    ):

        # First valid estimate becomes
        # the local RViz origin.

        if self.origin_latitude is None:

            self.origin_latitude = (
                latitude
            )

            self.origin_longitude = (
                longitude
            )

        meters_per_degree_lat = (
            111320.0
        )

        meters_per_degree_lon = (
            111320.0
            *
            math.cos(
                math.radians(
                    self.origin_latitude
                )
            )
        )

        east = (
            longitude
            -
            self.origin_longitude
        ) * meters_per_degree_lon

        north = (
            latitude
            -
            self.origin_latitude
        ) * meters_per_degree_lat

        pose = PoseStamped()

        pose.header.stamp = stamp

        pose.header.frame_id = (
            "map"
        )

        pose.pose.position.x = (
            east
        )

        pose.pose.position.y = (
            north
        )

        pose.pose.position.z = 0.0

        pose.pose.orientation.w = 1.0

        self.pose_pub.publish(
            pose
        )

    # ========================================================
    # PUBLISH STATUS
    # ========================================================

    def publish_status(
        self,
        text
    ):

        msg = String()

        msg.data = text

        self.status_pub.publish(
            msg
        )

    # ========================================================
    # MAIN PROCESSING
    # ========================================================

    def process_latest_frame(
        self
    ):

        process_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # GET LATEST CAMERA FRAME
        # ----------------------------------------------------

        with self.data_lock:

            if self.latest_frame is None:

                return

            frame = (
                self.latest_frame.copy()
            )

            frame_stamp_sec = (
                self.latest_frame_stamp
            )

        # ----------------------------------------------------
        # FIND NEAREST INS
        # ----------------------------------------------------

        ins = self.get_nearest_ins(
            frame_stamp_sec
        )

        if ins is None:

            self.get_logger().warn(
                "Waiting for INS data..."
            )

            return

        (
            ins_timestamp,
            ins_latitude,
            ins_longitude,
            ins_altitude
        ) = ins

        # ----------------------------------------------------
        # CHECK ALTITUDE
        # ----------------------------------------------------

        if ins_altitude <= 0:

            self.get_logger().warn(
                f"Invalid INS altitude: "
                f"{ins_altitude}"
            )

            return

        # ----------------------------------------------------
        # FRAME SIZE
        # ----------------------------------------------------

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # GPS/INS -> MAP PIXEL
        # ----------------------------------------------------

        center_row, center_col = (
            rowcol(
                self.src.transform,
                ins_longitude,
                ins_latitude
            )
        )

        center_row = float(
            center_row
        )

        center_col = float(
            center_col
        )

        if not (
            0 <= center_col < self.src.width
            and
            0 <= center_row < self.src.height
        ):

            self.publish_status(
                "INS position outside satellite map"
            )

            return

        # ----------------------------------------------------
        # FOV
        # ----------------------------------------------------

        vfov_deg = (
            self.calculate_vfov(
                frame_width,
                frame_height
            )
        )

        # ----------------------------------------------------
        # GROUND FOOTPRINT
        # ----------------------------------------------------

        (
            footprint_width_m,
            footprint_height_m
        ) = self.calculate_ground_footprint(
            ins_altitude,
            vfov_deg
        )

        # ----------------------------------------------------
        # MAP RESOLUTION
        # ----------------------------------------------------

        (
            x_m_per_px,
            y_m_per_px
        ) = (
            self.calculate_map_meters_per_pixel(
                ins_latitude
            )
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        (
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        ) = self.create_roi(
            center_row,
            center_col,
            footprint_width_m,
            footprint_height_m,
            x_m_per_px,
            y_m_per_px
        )

        # ----------------------------------------------------
        # READ SATELLITE ROI
        # ----------------------------------------------------

        original_roi = self.read_roi(
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        )

        # ----------------------------------------------------
        # RESIZE FOR MATCHING
        # ----------------------------------------------------

        satellite_matching, resize_scale = (
            self.resize_for_matching(
                original_roi
            )
        )

        # ----------------------------------------------------
        # SUPERPOINT + SUPERGLUE
        # ----------------------------------------------------

        matching_start = (
            time.perf_counter()
        )

        (
            matched_kpts0,
            matched_kpts1,
            matched_confidence
        ) = self.run_superglue(
            frame,
            satellite_matching
        )

        matching_time = (
            time.perf_counter()
            -
            matching_start
        )

        total_matches = (
            len(matched_kpts0)
        )

        # ----------------------------------------------------
        # RANSAC
        # ----------------------------------------------------

        ransac_result = (
            self.run_ransac(
                matched_kpts0,
                matched_kpts1,
                matched_confidence
            )
        )

        if ransac_result is None:

            self.publish_status(
                "RANSAC failed"
            )

            return

        (
            H,
            ransac_mask,
            inlier_kpts0,
            inlier_kpts1,
            inliers,
            outliers,
            inlier_ratio,
            mean_inlier_confidence
        ) = ransac_result

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        (
            estimated_latitude,
            estimated_longitude,
            matching_x,
            matching_y,
            map_column,
            map_row
        ) = self.estimate_position(
            H,
            frame_width,
            frame_height,
            resize_scale,
            roi_x1,
            roi_y1
        )

        # ----------------------------------------------------
        # TOTAL PROCESSING TIME
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            -
            process_start
        )

        # ----------------------------------------------------
        # PUBLISH POSITION
        # ----------------------------------------------------

        stamp = rclpy.time.Time(
            seconds=int(
                frame_stamp_sec
            ),
            nanoseconds=int(
                (
                    frame_stamp_sec
                    -
                    int(frame_stamp_sec)
                )
                *
                1e9
            )
        ).to_msg()

        self.publish_position(
            estimated_latitude,
            estimated_longitude,
            ins_altitude,
            stamp
        )

        # ----------------------------------------------------
        # RVIZ
        # ----------------------------------------------------

        self.publish_rviz_pose(
            estimated_latitude,
            estimated_longitude,
            stamp
        )

        # ----------------------------------------------------
        # PUBLISH INLIER RATIO
        # ----------------------------------------------------

        ratio_msg = Float32()

        ratio_msg.data = float(
            inlier_ratio
        )

        self.inlier_ratio_pub.publish(
            ratio_msg
        )

        # ----------------------------------------------------
        # PUBLISH PROCESSING TIME
        # ----------------------------------------------------

        time_msg = Float32()

        time_msg.data = float(
            total_time
        )

        self.processing_time_pub.publish(
            time_msg
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = (
            f"frame={self.process_counter} "
            f"INS=({ins_latitude:.7f},"
            f"{ins_longitude:.7f}) "
            f"EST=({estimated_latitude:.7f},"
            f"{estimated_longitude:.7f}) "
            f"matches={total_matches} "
            f"inliers={inliers} "
            f"inlier_ratio="
            f"{inlier_ratio:.3f} "
            f"time={total_time:.3f}s"
        )

        self.publish_status(
            status
        )

        # ----------------------------------------------------
        # DEBUG IMAGE
        # ----------------------------------------------------

        if self.publish_debug_image:

            debug = (
                satellite_matching.copy()
            )

            x = int(
                round(matching_x)
            )

            y = int(
                round(matching_y)
            )

            cv2.circle(
                debug,
                (x, y),
                15,
                (0, 0, 255),
                -1
            )

            text = (
                f"Lat: "
                f"{estimated_latitude:.7f} "
                f"Lon: "
                f"{estimated_longitude:.7f}"
            )

            cv2.putText(
                debug,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            try:

                debug_msg = (
                    self.bridge.cv2_to_imgmsg(
                        debug,
                        encoding="bgr8"
                    )
                )

                debug_msg.header.stamp = (
                    stamp
                )

                debug_msg.header.frame_id = (
                    "map"
                )

                self.debug_image_pub.publish(
                    debug_msg
                )

            except Exception as e:

                self.get_logger().warn(
                    f"Debug image publish failed: {e}"
                )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        self.get_logger().info(
            "MAP MATCH RESULT | "
            f"INS: "
            f"{ins_latitude:.7f}, "
            f"{ins_longitude:.7f} | "
            f"EST: "
            f"{estimated_latitude:.7f}, "
            f"{estimated_longitude:.7f} | "
            f"Matches: {total_matches} | "
            f"Inliers: {inliers} | "
            f"Ratio: "
            f"{inlier_ratio * 100:.1f}% | "
            f"Time: "
            f"{total_time:.3f}s"
        )

        self.process_counter += 1

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(
        self
    ):

        try:

            self.src.close()

        except Exception:
            pass

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(
    args=None
):

    rclpy.init(
        args=args
    )

    node = (
        MapMatchingNode()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
