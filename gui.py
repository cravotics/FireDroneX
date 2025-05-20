#!/usr/bin/env python3

import sys
import threading
import math
import json # For parsing gui_info
import time # For splash screen delay

import rclpy
from rclpy.node import Node as RosNodeBase
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String as RosString
from sensor_msgs.msg import Image as RosImage
from vision_msgs.msg import Detection2DArray, Detection2D
from px4_msgs.msg import VehicleOdometry, VehicleStatus, BatteryStatus

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QFrame, QSplashScreen
)
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject

from cv_bridge import CvBridge
import cv2 # For drawing on images
import numpy as np

# Constants
GUI_UPDATE_RATE = 30 # Hz, for QTimer if we needed one, but signals are event driven
IMG_WIDTH_VID = 640 # Target display width for video feeds
IMG_HEIGHT_VID = 480 # Target display height for video feeds

# Helper function to calculate speed from velocity components
def calculate_speed(vx, vy, vz):
    return math.sqrt(vx**2 + vy**2 + vz**2)

# Helper functions for PX4 states (simplified)
def get_arming_state_str(arming_state_code):
    if arming_state_code == VehicleStatus.ARMING_STATE_ARMED: return "ARMED"
    if arming_state_code == VehicleStatus.ARMING_STATE_STANDBY: return "STANDBY"
    if arming_state_code == VehicleStatus.ARMING_STATE_INIT: return "INIT"
    # Add other common states if known from VehicleStatus.msg
    if arming_state_code == VehicleStatus.ARMING_STATE_ARMED_ERROR: return "ARMED ERROR"
    if arming_state_code == VehicleStatus.ARMING_STATE_STANDBY_ERROR: return "STANDBY ERROR"
    if arming_state_code == VehicleStatus.ARMING_STATE_SHUTDOWN: return "SHUTDOWN"
    if arming_state_code == VehicleStatus.ARMING_STATE_IN_AIR_ERROR: return "IN AIR ERROR"
    return f"ARM_ST_{arming_state_code}"

def get_nav_state_str(nav_state_code):
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_MANUAL: return "MANUAL"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_ALTCTL: return "ALTITUDE CTRL"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_POSCTL: return "POSITION CTRL"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: return "MISSION"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: return "LOITER"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_AUTO_RTL: return "RTL"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_OFFBOARD: return "OFFBOARD"
    # Add other common states
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_AUTO_LAND: return "AUTO LAND"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: return "AUTO TAKEOFF"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_ACRO: return "ACRO"
    if nav_state_code == VehicleStatus.NAVIGATION_STATE_STABILIZED: return "STABILIZED"
    return f"NAV_ST_{nav_state_code}"


class RosNodeWorker(QObject):
    # Signals to update GUI
    yolo_debug_image_ready = Signal(QImage)
    depth_image_ready = Signal(QImage)
    person_focus_image_ready = Signal(QImage)
    odometry_data_ready = Signal(float, float, float, float) # x, y, z, speed
    mission_state_ready = Signal(str)
    vehicle_status_ready = Signal(str, str, bool, bool) # arming_state, nav_state, failsafe, pre_flight_ok
    battery_status_ready = Signal(float, float) # percentage, voltage
    fire_target_data_ready = Signal(int, float, float, float, bool) # id, x, y, z, is_valid_target
    person_safety_alerts_ready = Signal(list) # New signal for person-fire proximity alerts

    def __init__(self):
        super().__init__()
        self.node = None
        self.node_logger = None # For easier access to logger
        self.bridge = CvBridge()
        self.latest_raw_image_for_person_feed = None
        self.latest_person_detections = [] 

        # QoS for general sensor data that can be lossy but we want the latest
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # QoS for image feeds - typically RELIABLE from publishers, GUI wants latest
        # GZ Bridge for /camera is often RELIABLE, VOLATILE.
        # yolo_firedronex.py and firedronex_depth.py debug images are RELIABLE, TRANSIENT_LOCAL.
        # For simplicity, let's try VOLATILE first. If issues with yolo/depth feeds, 
        # we might need a specific QoS for them or change their publishers.
        self.image_feed_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.VOLATILE, # Changed from TRANSIENT_LOCAL on yolo/depth publishers implicitly
            history=HistoryPolicy.KEEP_LAST,
            depth=1 
        )

        # QoS for PX4 status messages (vehicle_status, battery_status)
        # These often come from microRTPS bridge as BEST_EFFORT or sometimes RELIABLE but are VOLATILE.
        self.px4_status_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Matching warning for vehicle_status, battery_status
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5 # A bit of buffer
        )

        # QoS for our custom mission_state topic
        self.mission_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, # Matching warning for mission_state (was TRANSIENT_LOCAL)
            history=HistoryPolicy.KEEP_LAST,
            depth=10 
        )

    @Slot()
    def run(self):
        rclpy.init(args=None)
        self.node = RosNodeBase('firedronex_gui_node')
        self.node_logger = self.node.get_logger() # Get logger instance

        self.node_logger.info("Initializing GUI ROS Node worker subscriptions...")

        self.node.create_subscription(
            RosImage, '/yolo/debug_image',
            self.yolo_debug_callback, self.image_feed_qos
        )
        self.node_logger.info("Subscribed to /yolo/debug_image")

        self.node.create_subscription(
            RosImage, '/fire_depth_localizer/depth_debug_image',
            self.depth_callback, self.image_feed_qos
        )
        self.node_logger.info("Subscribed to /fire_depth_localizer/depth_debug_image")

        self.node.create_subscription(
            RosImage, '/yolo/image_for_depth',
            self.raw_camera_callback, self.image_feed_qos
        )
        self.node_logger.info("Subscribed to /yolo/image_for_depth for person focus feed")

        self.node.create_subscription(
            Detection2DArray, '/detections',
            self.person_detections_callback, self.sensor_qos 
        )
        self.node_logger.info("Subscribed to /detections")

        self.node.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odometry_callback, self.sensor_qos
        )
        self.node_logger.info("Subscribed to /fmu/out/vehicle_odometry")

        self.node.create_subscription(
            RosString, '/firedronex/mission_state',
            self.mission_state_callback, self.mission_state_qos
        )
        self.node_logger.info("Subscribed to /firedronex/mission_state")

        self.node.create_subscription(
            RosString, '/firedronex/gui_info', # GUI Info topic
            self.gui_info_callback, self.mission_state_qos # Assuming similar QoS characteristics
        )
        self.node_logger.info("Subscribed to /firedronex/gui_info")

        self.node.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, self.px4_status_qos
        )
        self.node_logger.info("Subscribed to /fmu/out/vehicle_status")

        self.node.create_subscription(
            BatteryStatus, '/fmu/out/battery_status',
            self.battery_status_callback, self.px4_status_qos
        )
        self.node_logger.info("Subscribed to /fmu/out/battery_status")

        self.node_logger.info("GUI ROS Node worker started and spinning.")
        rclpy.spin(self.node)

        if self.node: # Ensure node exists before trying to destroy
            self.node.destroy_node()
        rclpy.shutdown() # Ensure rclpy is shut down
        print("GUI ROS Node worker shut down.")


    def _convert_ros_image_to_qimage(self, ros_image_msg, target_encoding='bgr8'):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding=target_encoding)
            if target_encoding == 'bgr8':
                h, w, ch = cv_image.shape
                bytes_per_line = ch * w
                return QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()
            elif target_encoding == 'mono8': # Grayscale
                h, w = cv_image.shape
                return QImage(cv_image.data, w, h, w, QImage.Format_Grayscale8).copy()
            # Add other encodings if necessary
            return None # Should not happen if target_encoding is handled
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error converting ROS image (encoding: {ros_image_msg.encoding} to {target_encoding}): {e}") # Removed throttle
            return None

    def yolo_debug_callback(self, msg):
        if self.node_logger: self.node_logger.info("YOLO debug image received.", throttle_duration_sec=5)
        q_img = self._convert_ros_image_to_qimage(msg, target_encoding='bgr8')
        if q_img:
            self.yolo_debug_image_ready.emit(q_img)
        elif self.node_logger: self.node_logger.warn("YOLO debug q_img was None.", throttle_duration_sec=5)

    def depth_callback(self, msg):
        if self.node_logger: self.node_logger.info("Depth debug image received.", throttle_duration_sec=5)
        q_img = self._convert_ros_image_to_qimage(msg, target_encoding='bgr8')
        if q_img:
            self.depth_image_ready.emit(q_img)
        elif self.node_logger: self.node_logger.warn("Depth debug q_img was None.", throttle_duration_sec=5)

    def raw_camera_callback(self, msg):
        if self.node_logger: self.node_logger.debug("Raw camera image received for person focus.", throttle_duration_sec=5)
        self.latest_raw_image_for_person_feed = msg 
        self._process_person_focus_feed()

    def person_detections_callback(self, msg: Detection2DArray):
        if self.node_logger: self.node_logger.info(f"Received {len(msg.detections)} total detections")
        self.latest_person_detections = []
        for detection in msg.detections:
            if detection.results: # Make sure results are not empty
                hyp = detection.results[0].hypothesis
                if hyp.class_id.lower() == 'person': 
                    bbox = detection.bbox
                    x1 = int(bbox.center.position.x - bbox.size_x / 2)
                    y1 = int(bbox.center.position.y - bbox.size_y / 2)
                    x2 = int(bbox.center.position.x + bbox.size_x / 2)
                    y2 = int(bbox.center.position.y + bbox.size_y / 2)
                    self.latest_person_detections.append((x1, y1, x2, y2, hyp.score))
                    if self.node_logger: self.node_logger.info(f"Added person detection: bbox=({x1},{y1},{x2},{y2}), score={hyp.score:.2f}")
        self._process_person_focus_feed() # Process even if no new persons, to clear old boxes

    def _process_person_focus_feed(self):
        if self.latest_raw_image_for_person_feed is None:
            if self.node_logger: self.node_logger.warn("No raw image available for person focus feed")
            return

        try:
            cv_image_bgr = self.bridge.imgmsg_to_cv2(self.latest_raw_image_for_person_feed, desired_encoding='bgr8')
            img_h, img_w = cv_image_bgr.shape[:2]
            
            if self.node_logger: self.node_logger.info(f"Processing person focus feed: image size={img_w}x{img_h}, {len(self.latest_person_detections)} person detections")
            
            q_img_to_emit = None

            if self.latest_person_detections: # If there are any persons
                # Focus on the first detected person.
                first_person = self.latest_person_detections[0]
                px1, py1, px2, py2, score = first_person

                box_w = px2 - px1
                box_h = py2 - py1

                # Define padding: 30% of bbox dimension on each side + 10 pixels fixed
                pad_w = int(box_w * 0.3) + 10
                pad_h = int(box_h * 0.3) + 10

                crop_x1 = max(0, px1 - pad_w)
                crop_y1 = max(0, py1 - pad_h)
                crop_x2 = min(img_w, px2 + pad_w)
                crop_y2 = min(img_h, py2 + pad_h)

                if self.node_logger: self.node_logger.info(f"Crop region: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})")

                if crop_x2 > crop_x1 and crop_y2 > crop_y1 and box_w > 0 and box_h > 0: # Valid crop and valid box
                    focused_cv_image = cv_image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                    # Adjust person's bounding box coordinates for the cropped image
                    new_px1 = px1 - crop_x1
                    new_py1 = py1 - crop_y1
                    new_px2 = px2 - crop_x1
                    new_py2 = py2 - crop_y1
                    
                    # Draw the bounding box and label on the FOCUSED image
                    cv2.rectangle(focused_cv_image, (new_px1, new_py1), (new_px2, new_py2), (255, 0, 0), 2) # Blue box
                    
                    label = f"P: {score:.2f}" # Shorter label for Person

                    # Place text inside the bounding box, near top-left
                    text_x = new_px1 + 5 
                    text_y = new_py1 + 15 # Adjusted for font size 0.5

                    # Basic check to keep text within box if box is tiny, fallback to near top
                    if new_py1 + 15 > new_py2 - 5 : # If default text_y is too close to bottom of box
                         text_y = new_py1 + int((new_py2 - new_py1) * 0.5) # Try to center vertically
                    if new_px1 + 5 > new_px2 - 5 : # If default text_x is too close to right of box
                        text_x = new_px1 + int((new_px2 - new_px1)*0.1)

                    cv2.putText(focused_cv_image, label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # White text, smaller
                    
                    h_f, w_f, ch_f = focused_cv_image.shape
                    if h_f > 0 and w_f > 0: # Ensure cropped image is not empty
                        bytes_per_line_f = ch_f * w_f
                        q_img_to_emit = QImage(focused_cv_image.data, w_f, h_f, bytes_per_line_f, QImage.Format_BGR888).copy()
                        if self.node_logger: self.node_logger.info(f"Created focused image: {w_f}x{h_f}")
                    else:
                        if self.node_logger: self.node_logger.warn("Focused crop resulted in empty image")


            # If no person was detected, or if focusing failed (q_img_to_emit is None),
            # then show the full image, drawing all person detections on it.
            if q_img_to_emit is None:
                cv_image_bgr_display = cv_image_bgr.copy() # Draw on a copy for the full view
                for (x1_full, y1_full, x2_full, y2_full, score_full) in self.latest_person_detections:
                    cv2.rectangle(cv_image_bgr_display, (x1_full, y1_full), (x2_full, y2_full), (255, 0, 0), 2) # Blue
                    label_full = f"Person: {score_full:.2f}"
                    # Standard text placement for full image (above box)
                    cv2.putText(cv_image_bgr_display, label_full, (x1_full, y1_full - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                
                h_full, w_full, ch_full = cv_image_bgr_display.shape
                bytes_per_line_full = ch_full * w_full
                q_img_to_emit = QImage(cv_image_bgr_display.data, w_full, h_full, bytes_per_line_full, QImage.Format_BGR888).copy()

            if q_img_to_emit:
                self.person_focus_image_ready.emit(q_img_to_emit)
            elif self.node_logger:
                self.node_logger.warn("Person focus: q_img_to_emit was None at the end.", throttle_duration_sec=5)
        
        except Exception as e:
            if self.node_logger: # Check if logger exists before using
                self.node_logger.error(f"Error processing person focus feed: {e}", throttle_duration_sec=1)

    def odometry_callback(self, msg: VehicleOdometry):
        if self.node_logger: self.node_logger.debug("Odometry data received.", throttle_duration_sec=10)
        x, y, z = msg.position[0], msg.position[1], msg.position[2]
        speed = calculate_speed(msg.velocity[0], msg.velocity[1], msg.velocity[2])
        self.odometry_data_ready.emit(x, y, z, speed)

    def mission_state_callback(self, msg: RosString):
        if self.node_logger: self.node_logger.info(f"Mission state received: {msg.data}", throttle_duration_sec=2)
        self.mission_state_ready.emit(msg.data)

    def gui_info_callback(self, msg: RosString):
        if self.node_logger: self.node_logger.debug("GUI info received.", throttle_duration_sec=5)
        try:
            data = json.loads(msg.data)
            target_fire_id = -1
            target_x, target_y, target_z = 0.0, 0.0, 0.0
            is_valid_target = False
            person_alerts = [] # Initialize list for person safety alerts

            if "detections" in data:
                for det in data["detections"]:
                    if det.get("object_type") == "fire" and det.get("is_current_target") == True:
                        target_fire_id = det.get("object_id", -1)
                        world_pos = det.get("world_position", {})
                        target_x = world_pos.get("x", 0.0)
                        target_y = world_pos.get("y", 0.0)
                        target_z = world_pos.get("z", 0.0)
                        is_valid_target = True
                        # Do not break here if we want to process other detections like persons
                
                # If no primary target, take the first fire found (if any) for basic info (already existing logic)
                if not is_valid_target:
                    for det in data["detections"]:
                        if det.get("object_type") == "fire":
                            target_fire_id = det.get("object_id", -1)
                            world_pos = det.get("world_position", {})
                            target_x = world_pos.get("x", 0.0)
                            target_y = world_pos.get("y", 0.0)
                            target_z = world_pos.get("z", 0.0)
                            break 
                
                # Process persons for safety alerts
                for det in data["detections"]:
                    if det.get("object_type") == "person":
                        person_world_pos = det.get("world_position", {})
                        alert_data = {
                            "px": person_world_pos.get("x", 0.0),
                            "py": person_world_pos.get("y", 0.0),
                            "pz": person_world_pos.get("z", 0.0),
                            "dist": det.get("person_fire_distance", -1.0),
                            "fire_id": det.get("associated_fire_id", -1),
                            "conf": det.get("confidence", 0.0)
                        }
                        # Add to alerts if distance is valid and potentially dangerous
                        if alert_data["dist"] != -1.0: # Only add if distance is calculated
                            person_alerts.append(alert_data)

            self.fire_target_data_ready.emit(target_fire_id, target_x, target_y, target_z, is_valid_target)
            if person_alerts: # Emit only if there are alerts
                self.person_safety_alerts_ready.emit(person_alerts)
            else: # Emit an empty list if no relevant person detections
                self.person_safety_alerts_ready.emit([])

        except json.JSONDecodeError as e:
            if self.node_logger: self.node_logger.error(f"Error decoding gui_info JSON: {e}", throttle_duration_sec=5)
        except Exception as e:
            if self.node_logger: self.node_logger.error(f"Error processing gui_info_callback: {e}", throttle_duration_sec=5)

    def vehicle_status_callback(self, msg: VehicleStatus):
        if self.node_logger: self.node_logger.info("Vehicle status received.", throttle_duration_sec=2)
        arming_str = get_arming_state_str(msg.arming_state)
        nav_str = get_nav_state_str(msg.nav_state)
        self.vehicle_status_ready.emit(arming_str, nav_str, msg.failsafe, msg.pre_flight_checks_pass)

    def battery_status_callback(self, msg: BatteryStatus):
        if self.node_logger: self.node_logger.info(f"Battery status received: R={msg.remaining:.2f}, V={msg.voltage_v:.2f}", throttle_duration_sec=2)
        percentage = msg.remaining * 100.0 if msg.remaining >= 0 else 0.0 # ensure positive
        voltage = msg.voltage_v
        self.battery_status_ready.emit(percentage, voltage)

    def stop(self):
        if self.node and rclpy.ok():
            self.node.get_logger().info("Requesting GUI ROS Node worker to stop...")
            # A more graceful shutdown might involve creating a future and waiting in spin
            # For now, setting a flag or relying on main thread to call rclpy.shutdown() is typical
            # If rclpy.spin is blocking, this stop might not be effective unless spin is in a loop
            # with rclpy.ok() check. The current design should be okay as spin is called once.
            # The main window closeEvent will handle rclpy.shutdown


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FireDroneX Monitoring GUI")
        self.setGeometry(50, 50, 1800, 1000) 
        self.setStyleSheet("background-color: #2E2E2E; color: white;")


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10,10,10,10)

        # --- Video Feeds Section ---
        video_feeds_container = QWidget()
        video_feeds_layout = QHBoxLayout(video_feeds_container)
        video_feeds_layout.setSpacing(10)
        self.main_layout.addWidget(video_feeds_container, stretch=3) 

        self.yolo_feed_label_container = self._create_video_feed_label_container("YOLO Debug")
        video_feeds_layout.addWidget(self.yolo_feed_label_container)

        self.depth_feed_label_container = self._create_video_feed_label_container("Depth View (Fires)")
        video_feeds_layout.addWidget(self.depth_feed_label_container)

        self.person_focus_feed_label_container = self._create_video_feed_label_container("Camera (Person Focus)")
        video_feeds_layout.addWidget(self.person_focus_feed_label_container)

        # --- Data and Status Section ---
        data_status_container = QWidget()
        data_status_layout = QHBoxLayout(data_status_container)
        data_status_layout.setSpacing(15)
        self.main_layout.addWidget(data_status_container, stretch=1)

        # Left Panel: Odometry and Mission State
        left_data_panel = QVBoxLayout()
        data_status_layout.addLayout(left_data_panel, stretch=1)

        odom_group = QGroupBox("Odometry")
        odom_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        odom_layout = QGridLayout(odom_group)
        self.pos_x_label = self._create_data_label("X: -- m")
        self.pos_y_label = self._create_data_label("Y: -- m")
        self.pos_z_label = self._create_data_label("Z: -- m")
        self.speed_label = self._create_data_label("Speed: -- m/s")
        odom_layout.addWidget(QLabel("Pos X:"), 0, 0); odom_layout.addWidget(self.pos_x_label, 0, 1)
        odom_layout.addWidget(QLabel("Pos Y:"), 1, 0); odom_layout.addWidget(self.pos_y_label, 1, 1)
        odom_layout.addWidget(QLabel("Pos Z:"), 2, 0); odom_layout.addWidget(self.pos_z_label, 2, 1)
        odom_layout.addWidget(QLabel("Speed:"), 3, 0); odom_layout.addWidget(self.speed_label, 3, 1)
        left_data_panel.addWidget(odom_group)

        mission_group = QGroupBox("Mission Status")
        mission_group.setStyleSheet(odom_group.styleSheet()) # Reuse style
        mission_layout = QVBoxLayout(mission_group)
        self.mission_state_label = QLabel("UNKNOWN")
        self.mission_state_label.setAlignment(Qt.AlignCenter)
        self.mission_state_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.mission_state_label.setStyleSheet("padding: 10px; border: 1px solid #444; border-radius: 4px; background-color: #3a3a3a;")
        mission_layout.addWidget(self.mission_state_label)
        left_data_panel.addWidget(mission_group)
        left_data_panel.addStretch()

        # Center Panel: Fire Target Info (New)
        center_data_panel = QVBoxLayout()
        data_status_layout.addLayout(center_data_panel, stretch=1)

        fire_target_group = QGroupBox("Fire Target Details")
        fire_target_group.setStyleSheet(odom_group.styleSheet()) # Reuse style
        fire_target_layout = QGridLayout(fire_target_group)
        self.fire_target_id_label = self._create_data_label("ID: --")
        self.fire_target_x_label = self._create_data_label("X: -- m")
        self.fire_target_y_label = self._create_data_label("Y: -- m")
        self.fire_target_z_label = self._create_data_label("Z: -- m")
        self.fire_target_status_label = self._create_status_indicator_label("NO TARGET") # For visual cue
        self.fire_target_status_label.setStyleSheet("background-color: #555; color: white; padding: 5px; border-radius: 4px;font-weight:bold;")


        fire_target_layout.addWidget(QLabel("Status:"), 0, 0); fire_target_layout.addWidget(self.fire_target_status_label, 0, 1)
        fire_target_layout.addWidget(QLabel("Target ID:"), 1, 0); fire_target_layout.addWidget(self.fire_target_id_label, 1, 1)
        fire_target_layout.addWidget(QLabel("Target X:"), 2, 0); fire_target_layout.addWidget(self.fire_target_x_label, 2, 1)
        fire_target_layout.addWidget(QLabel("Target Y:"), 3, 0); fire_target_layout.addWidget(self.fire_target_y_label, 3, 1)
        fire_target_layout.addWidget(QLabel("Target Z:"), 4, 0); fire_target_layout.addWidget(self.fire_target_z_label, 4, 1)
        center_data_panel.addWidget(fire_target_group)
        
        # New GroupBox for Person Safety Alerts
        self.person_alerts_group = QGroupBox("Person Safety Alerts")
        self.person_alerts_group.setStyleSheet(odom_group.styleSheet())
        self.person_alerts_layout = QVBoxLayout(self.person_alerts_group)
        initial_alert_label = QLabel("Waiting for person/fire data...")
        initial_alert_label.setAlignment(Qt.AlignCenter)
        self.person_alerts_layout.addWidget(initial_alert_label)
        center_data_panel.addWidget(self.person_alerts_group) # Add to the center panel

        center_data_panel.addStretch()


        # Right Panel: Vehicle and Battery Status
        right_status_panel = QVBoxLayout()
        data_status_layout.addLayout(right_status_panel, stretch=1)

        status_indicators_group = QGroupBox("Drone Critical Status")
        status_indicators_group.setStyleSheet(odom_group.styleSheet())
        status_indicators_layout = QGridLayout(status_indicators_group)
        self.arming_status_label = self._create_status_indicator_label("ARMING: --")
        self.nav_status_label = self._create_status_indicator_label("NAV MODE: --")
        self.failsafe_status_label = self._create_status_indicator_label("FAILSAFE: --")
        self.preflight_status_label = self._create_status_indicator_label("PREFLIGHT: --")
        status_indicators_layout.addWidget(QLabel("Arming:"), 0, 0, Qt.AlignRight); status_indicators_layout.addWidget(self.arming_status_label, 0, 1)
        status_indicators_layout.addWidget(QLabel("Nav State:"), 1, 0, Qt.AlignRight); status_indicators_layout.addWidget(self.nav_status_label, 1, 1)
        status_indicators_layout.addWidget(QLabel("Failsafe:"), 2, 0, Qt.AlignRight); status_indicators_layout.addWidget(self.failsafe_status_label, 2, 1)
        status_indicators_layout.addWidget(QLabel("Preflight OK:"), 3, 0, Qt.AlignRight); status_indicators_layout.addWidget(self.preflight_status_label, 3, 1)
        right_status_panel.addWidget(status_indicators_group)

        battery_group = QGroupBox("Battery")
        battery_group.setStyleSheet(odom_group.styleSheet())
        battery_layout = QGridLayout(battery_group)
        self.battery_percentage_label = self._create_data_label("-- %")
        self.battery_voltage_label = self._create_data_label("-- V")
        battery_layout.addWidget(QLabel("Percentage:"),0,0); battery_layout.addWidget(self.battery_percentage_label,0,1)
        battery_layout.addWidget(QLabel("Voltage:"),1,0); battery_layout.addWidget(self.battery_voltage_label,1,1)
        right_status_panel.addWidget(battery_group)
        right_status_panel.addStretch()

        self.ros_thread = QThread(self)
        self.ros_worker = RosNodeWorker()
        self.ros_worker.moveToThread(self.ros_thread)

        self.ros_worker.yolo_debug_image_ready.connect(self.update_yolo_feed)
        self.ros_worker.depth_image_ready.connect(self.update_depth_feed)
        self.ros_worker.person_focus_image_ready.connect(self.update_person_focus_feed)
        self.ros_worker.odometry_data_ready.connect(self.update_odometry_data)
        self.ros_worker.mission_state_ready.connect(self.update_mission_state)
        self.ros_worker.vehicle_status_ready.connect(self.update_vehicle_status)
        self.ros_worker.battery_status_ready.connect(self.update_battery_status)
        self.ros_worker.fire_target_data_ready.connect(self.update_fire_target_info)
        self.ros_worker.person_safety_alerts_ready.connect(self.update_person_safety_display) # Connect new signal

        self.ros_thread.started.connect(self.ros_worker.run)
        self.ros_thread.finished.connect(self.ros_worker.deleteLater) # Cleanup worker
        self.ros_thread.start()

    def _create_video_feed_label_container(self, title="Video Feed"):
        container = QGroupBox(title)
        container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        layout = QVBoxLayout(container)
        label = QLabel("Waiting for feed...")
        label.setAlignment(Qt.AlignCenter)
        label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken) # Changed from Panel|Sunken for better theme adapt
        label.setMinimumSize(IMG_WIDTH_VID // 2, IMG_HEIGHT_VID // 2) 
        label.setStyleSheet("background-color: #1C1C1C; color: #777; border-radius: 3px;")
        layout.addWidget(label)
        return container

    def _create_data_label(self, initial_text="--"):
        label = QLabel(initial_text)
        label.setFont(QFont("Arial", 10))
        label.setStyleSheet("padding: 2px; color: #DDD;")
        return label

    def _create_status_indicator_label(self, initial_text="--"):
        label = QLabel(initial_text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setMinimumWidth(110) # Ensure enough space for "STANDBY ERROR"
        label.setMinimumHeight(30)
        label.setStyleSheet("padding: 5px; border-radius: 4px; color: white; background-color: #555; qproperty-alignment: AlignCenter;")
        return label
        
    def _update_image_on_label(self, container_groupbox: QGroupBox, q_img: QImage):
        label_widget = container_groupbox.findChild(QLabel)
        if label_widget:
            if q_img and not q_img.isNull():
                pixmap = QPixmap.fromImage(q_img)
                label_widget.setPixmap(pixmap.scaled(
                    label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                label_widget.setText("Feed Error")
                label_widget.setStyleSheet("background-color: #1C1C1C; color: red; border-radius: 3px;")


    @Slot(QImage)
    def update_yolo_feed(self, q_img):
        self._update_image_on_label(self.yolo_feed_label_container, q_img)

    @Slot(QImage)
    def update_depth_feed(self, q_img):
        self._update_image_on_label(self.depth_feed_label_container, q_img)

    @Slot(QImage)
    def update_person_focus_feed(self, q_img):
        self._update_image_on_label(self.person_focus_feed_label_container, q_img)

    @Slot(float, float, float, float)
    def update_odometry_data(self, x, y, z, speed):
        self.pos_x_label.setText(f"{x:.2f} m")
        self.pos_y_label.setText(f"{y:.2f} m")
        self.pos_z_label.setText(f"{z:.2f} m")
        self.speed_label.setText(f"{speed:.2f} m/s")

    @Slot(str)
    def update_mission_state(self, state_str):
        self.mission_state_label.setText(state_str.upper())
        color = "#AAA" # Default
        if state_str == "SEARCH": color = "orange"
        elif state_str == "HOLD": color = "yellow"
        elif state_str == "APPROACH": color = "lightgreen"
        elif state_str == "CIRCLE": color = "lightblue" # Though CIRCLE might be less used now
        self.mission_state_label.setStyleSheet(f"padding: 10px; border: 1px solid #444; border-radius: 4px; background-color: #3a3a3a; color: {color}; font-weight: bold; font-size: 18pt;")

    @Slot(str, str, bool, bool)
    def update_vehicle_status(self, arming_str, nav_str, is_failsafe, preflight_ok):
        self.arming_status_label.setText(f"ARMING: {arming_str}")
        armed_bg_color = "#2E7D32" # Green
        if "ERROR" in arming_str or arming_str != "ARMED":
            armed_bg_color = "#C62828" # Red
        elif arming_str == "STANDBY":
            armed_bg_color = "#FF8F00" # Amber
        self.arming_status_label.setStyleSheet(f"background-color: {armed_bg_color}; color: white; padding: 5px; border-radius: 4px; font-weight:bold;")

        self.nav_status_label.setText(f"NAV MODE: {nav_str}")
        nav_bg_color = "#0277BD" # Light Blue
        if nav_str == "OFFBOARD": nav_bg_color = "#558B2F" # Darker Green for offboard
        elif nav_str == "MANUAL": nav_bg_color = "#EF6C00" # Orange
        self.nav_status_label.setStyleSheet(f"background-color: {nav_bg_color}; color: white; padding: 5px; border-radius: 4px;font-weight:bold;")

        self.failsafe_status_label.setText(f"FAILSAFE: {'ACTIVE!' if is_failsafe else 'OK'}")
        self.failsafe_status_label.setStyleSheet(f"background-color: {'#C62828' if is_failsafe else '#2E7D32'}; color: white; padding: 5px; border-radius: 4px;font-weight:bold;")
        
        self.preflight_status_label.setText(f"PREFLIGHT: {'PASS' if preflight_ok else 'FAIL'}")
        self.preflight_status_label.setStyleSheet(f"background-color: {'#2E7D32' if preflight_ok else '#C62828'}; color: white; padding: 5px; border-radius: 4px;font-weight:bold;")

    @Slot(float, float)
    def update_battery_status(self, percentage, voltage):
        self.battery_percentage_label.setText(f"{percentage:.1f} %")
        self.battery_voltage_label.setText(f"{voltage:.2f} V")
        
        color = "#4CAF50" # Green
        if percentage < 20: color = "#D32F2F" # Red
        elif percentage < 50: color = "#FFA000" # Amber
        self.battery_percentage_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.battery_voltage_label.setStyleSheet(f"color: #DDD;")

    @Slot(int, float, float, float, bool)
    def update_fire_target_info(self, fire_id, x, y, z, is_valid_target):
        if is_valid_target:
            self.fire_target_id_label.setText(f"{fire_id}")
            self.fire_target_x_label.setText(f"{x:.2f} m")
            self.fire_target_y_label.setText(f"{y:.2f} m")
            self.fire_target_z_label.setText(f"{z:.2f} m")
            self.fire_target_status_label.setText("TARGET LOCK")
            self.fire_target_status_label.setStyleSheet("background-color: #2E7D32; color: white; padding: 5px; border-radius: 4px;font-weight:bold;") # Green
        elif fire_id != -1 : # A fire is detected but not explicitly the "current_target"
            self.fire_target_id_label.setText(f"{fire_id} (Not Primary)")
            self.fire_target_x_label.setText(f"{x:.2f} m")
            self.fire_target_y_label.setText(f"{y:.2f} m")
            self.fire_target_z_label.setText(f"{z:.2f} m")
            self.fire_target_status_label.setText("FIRE DETECTED")
            self.fire_target_status_label.setStyleSheet("background-color: #FF8F00; color: white; padding: 5px; border-radius: 4px;font-weight:bold;") # Amber
        else: # No fire target or no fire detected
            self.fire_target_id_label.setText("--")
            self.fire_target_x_label.setText("-- m")
            self.fire_target_y_label.setText("-- m")
            self.fire_target_z_label.setText("-- m")
            self.fire_target_status_label.setText("NO TARGET")
            self.fire_target_status_label.setStyleSheet("background-color: #555; color: white; padding: 5px; border-radius: 4px;font-weight:bold;") # Grey

    @Slot(list)
    def update_person_safety_display(self, alerts_data):
        # Clear previous alerts
        for i in reversed(range(self.person_alerts_layout.count())):
            widget = self.person_alerts_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        if not alerts_data:
            no_alerts_label = QLabel("No critical person proximity alerts.")
            no_alerts_label.setAlignment(Qt.AlignCenter)
            no_alerts_label.setStyleSheet("color: #AAA;")
            self.person_alerts_layout.addWidget(no_alerts_label)
            return

        for alert in alerts_data:
            alert_text = f"Person (Conf: {alert['conf']:.2f}) at ({alert['px']:.1f}, {alert['py']:.1f})"
            alert_text += f" is {alert['dist']:.1f}m from Fire ID {alert['fire_id']}."
            
            label = QLabel(alert_text)
            label.setFont(QFont("Arial", 9))
            label.setWordWrap(True)
            
            # Style based on proximity
            if alert['dist'] < 1.5 and alert['dist'] != -1.0:
                label.setStyleSheet("color: red; font-weight: bold; padding: 3px; border: 1px solid #500; background-color: #400;")
            elif alert['dist'] < 3.0 and alert['dist'] != -1.0:
                label.setStyleSheet("color: orange; font-weight: bold; padding: 3px;")
            else:
                label.setStyleSheet("color: #DDD; padding: 3px;") # Default for less critical or N/A distance
            self.person_alerts_layout.addWidget(label)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Force re-scaling of pixmaps in video labels
        # This ensures that when the window (and thus labels) are resized,
        # the image scales correctly according to KeepAspectRatio.
        # We call _update_image_on_label with its current QImage if we stored them,
        # or rely on the next incoming image to trigger the rescale.
        # For simplicity, we let the next frame handle it.
        # If immediate rescale is needed, QImages for feeds would need to be stored in MainWindow.

    def closeEvent(self, event):
        print("Closing GUI application...")
        if self.ros_thread.isRunning():
            # Request the worker to stop its operations if possible (not strictly implemented in worker.stop)
            # Then quit the thread.
            self.ros_thread.quit() 
            if not self.ros_thread.wait(3000): # Wait up to 3s
                print("ROS thread did not quit gracefully, terminating...")
                self.ros_thread.terminate() # Force terminate if necessary
                self.ros_thread.wait() # Wait for termination
        print("ROS thread stopped.")
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- Splash Screen ---
    logo_path = "/home/varun/ws_offboard_control/src/px4_ros_com/src/examples/scripts/X.png"
    splash_pix = QPixmap(logo_path)
    
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask()) # For transparency if logo has it

    # Add text to splash screen
    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    
    # Main Title (already in image, but can be drawn if needed or for consistency)
    # splash.setFont(font) # Set font before drawing text

    # Owners text
    owner_text = "FireDroneX Control Station\\nOwners: Sai and Varun"
    
    # Get screen geometry to center splash screen
    screen_geometry = QApplication.primaryScreen().geometry()
    splash.move(int((screen_geometry.width() - splash_pix.width()) / 2),
                int((screen_geometry.height() - splash_pix.height()) / 2))

    splash.show()
    
    # Process events to make sure splash screen is displayed
    app.processEvents()

    # Simulate some loading time (optional)
    # For a real app, this would be replaced by actual initialization time.
    # If initialization is very fast, the splash might just flicker.
    # We'll show it for a minimum duration.
    
    start_time = time.time()

    window = MainWindow() # Create main window

    # Ensure splash is shown for at least a few seconds
    while time.time() < start_time + 3.0: # Show for at least 3 seconds
        app.processEvents()
        time.sleep(0.05) # Brief sleep to yield CPU

    window.show()
    splash.finish(window) # Close splash screen when main window is ready

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("GUI interrupted by user (Ctrl+C)")
    finally:
        # Ensure ROS cleanup if app.exec() was bypassed by interrupt
        if rclpy.ok():
            print("Ensuring rclpy shutdown from main.")
            # rclpy.shutdown() # This might be handled by worker's shutdown now.
            # If worker's spin is blocking, this won't be reached until thread joins.
            # The closeEvent should be the primary place for this.
        print("Exiting GUI application.") 