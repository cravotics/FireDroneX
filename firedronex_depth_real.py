#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import math
import cv2
from depth_anything_v2_estimator import DepthAnythingV2Estimator
from vision_msgs.msg import Detection2DArray
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint, OffboardControlMode, VehicleCommand, VehicleControlMode
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory
import os
import torch 
from std_msgs.msg import String
import json # Added for JSON serialization
from geometry_msgs.msg import Point # Added for constructing world_position for GUI msg

from fire_data_logger import FireDataLogger

class FireDepthLocalizer(Node):
    def __init__(self):
        super().__init__('fire_depth_localizer')

        # Log PyTorch and CUDA status
        # self.get_logger().info(f"PyTorch version: {torch.__version__}")
        # self.get_logger().info(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
        # if torch.cuda.is_available():
            # self.get_logger().info(f"Torch CUDA version: {torch.version.cuda}")
        # else:
            # self.get_logger().warn("CUDA is NOT available to PyTorch. Depth estimator might run on CPU.")

        # INTEGRATION: Initialize Data Logger
        self.log_main_dir = "/home/varun/ws_offboard_control/firedronex_plots" 
        self.data_logger = FireDataLogger(log_directory=self.log_main_dir)
        # self.get_logger().info(f"Initialized data logger. Log files will be in: {self.log_main_dir}")

        self.bridge = CvBridge()
        self.latest_image = None
        
        # Camera intrinsics
        self.fx = 298.43184847474606 # Was 1397.2235870361328 (scaled from 1920x1080)
        self.fy = 297.4815940045744  # Was 1397.2235298156738 (scaled from 1920x1080)
        self.cx = 318.41391448716506 # Was 960.0 (center of 1920x1080), now scaled from hires_small_color calib
        self.cy = 241.95370726748156 # Was 540.0 (center of 1920x1080), now scaled from hires_small_color calib
        
        # Orientation and position of the camera and drone
        self.camera_pitch = -0.785  
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = -5.0
        self.current_yaw = 0.0 
        self.orientation_q_scipy = np.array([0.0, 0.0, 0.0, 1.0]) 
        
        # Mission parameters from firedronex_depth.py & safe_firedronex_sim.py
        self.state = 'SEARCH'
        self.circle_radius = 2.0 
        self.circle_theta = 0.0
        
        self.search_center = None
        self.initial_search = True 
        self.last_theta = 0.0
        
        self.max_circle_velocity = 0.1
        self.radius_increment = 0.25
        self.fire_id_map = {} 
        self.fire_id = 1

        # Constants for fire mission
        self.FIRE_APPROACH_ALTITUDE = -2.5 # Was -1.5, reduced by 1m
        self.SEARCH_ALTITUDE = -2.5      # Was -1.5, reduced by 1m
        self.MAX_APPROACH_DURATION_SEC = 20.0 # Max time to spend in APPROACH before giving up
        
        # Visited fires & detected fires
        self.visited_fires = []
        self.detected_fires = []

        # Parameters from safe_firedronex_sim.py (STOP_AND_RECALCULATE related ones removed)
        self.ROI_WIDTH = 640   # Set to processed image width (was 1920)
        self.ROI_HEIGHT = 480  # Set to processed image height (was 1080)
        self.target_face_yaw = 0.0 # Yaw for HOLD state to face fire
        self.approach_start_time = None # For HOLD state duration
        
        # State variables
        self.fire_target = None # This will store (x,y,z) of the active fire target in OFFBOARD mode
        self.circling = False # Note: SEARCH state handles circular patterns. This might be redundant or for specific CIRCLE state.
        self.circle_center = None # Center for CIRCLE state, if used distinctly from SEARCH.
        
        # QoS profile for general subscriptions (e.g., odometry, control mode)
        # BEST_EFFORT is often suitable for high-frequency sensor data where occasional drops are acceptable.
        subscriber_qos = QoSProfile(
            depth=10, 
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # QoS profile for debug image publishers
        self.debug_image_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, # Changed from TRANSIENT_LOCAL
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # For image streams, often only latest is needed
        )
        
        # Visualization
        # For markers, TRANSIENT_LOCAL is good so Rviz can see them even if started late.
        # Marker publisher QoS should remain TRANSIENT_LOCAL
        marker_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10 # Markers are low frequency, allow some buffer and ensure visibility
        )
        self.fire_marker_pub = self.create_publisher(MarkerArray, '/fire_markers', qos_profile=marker_publisher_qos) 
        
        # Add publisher for debug image
        self.debug_image_pub = self.create_publisher(Image, '/fire_depth_localizer/debug_image', qos_profile=self.debug_image_publisher_qos)
        # Add publisher for depth debug image
        self.depth_debug_image_pub = self.create_publisher(Image, '/fire_depth_localizer/depth_debug_image', qos_profile=self.debug_image_publisher_qos)
        
        # PID Parameters
        self.kp = 0.3
        self.ki = 0.0
        self.kd = 0.15
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.dt = 0.1 
        
        self.control_mode = None
        
        # Safety Boundaries for indoor lab
        self.X_BOUND_MIN = -1.0
        self.X_BOUND_MAX = 1.0
        self.Y_BOUND_MIN = -1.0
        self.Y_BOUND_MAX = 1.0
        self.OPERATIONAL_ALTITUDE_REAL = -2.5 # Was -1.5, explicit constant for real drone altitude

        # model_path = '/home/varun/ws_offboard_control/src/px4_ros_com/src/examples/midas/midas_v21_small_256.pt'
        # self.depth_estimator = MidasDepthEstimator(model_path)
        package_share_dir = get_package_share_directory('px4_ros_com')
        depth_model_path = os.path.join(package_share_dir, 'depth_anything_models', 'depth_anything_v2_vits.pth')
        # self.get_logger().info(f"Loading Depth Anything V2 model from: {depth_model_path}")

        self.depth_estimator = DepthAnythingV2Estimator(model_path=depth_model_path, encoder='vits')
        
        self.create_subscription(
            Image,
            '/yolo/image_for_depth',
            self.image_callback,
            qos_profile = subscriber_qos
        )

        self.create_subscription(
            Detection2DArray,
            '/detections', 
            self.detection_callback,
            qos_profile = subscriber_qos
        )
        
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile = subscriber_qos
        )
        # self.get_logger().info("Subscription to /fmu/out/vehicle_control_mode created.")
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.vehicle_control_mode_callback,
            qos_profile = subscriber_qos
        )
        
        # For command topics to PX4, RELIABLE is generally safer.
        # The TrajectorySetpoint and OffboardControlMode publishers might benefit from RELIABLE.
        # However, the original code used 'qos_profile=qos' which was BEST_EFFORT.
        # Let's stick to the subscriber_qos (BEST_EFFORT) for consistency with previous setup for now for these,
        # unless issues arise.
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile = subscriber_qos 
        )
        
        self.offboard_control_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile = subscriber_qos 
        )
        
        # Add publisher for mission state
        self.mission_state_pub = self.create_publisher(String, '/firedronex/mission_state', 10)
        
        # Publisher for detailed GUI information
        self.gui_info_pub = self.create_publisher(String, '/firedronex/gui_info', 10)
        
        self.create_timer(0.1, self.mission_loop)
        
        # self.get_logger().info("Fire Depth Localizer Node Initialized (STOP_AND_RECALCULATE logic removed).")
        
    def detection_callback(self, msg):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet for detection callback.")
            return

        current_ros_time_sec = self.get_clock().now().nanoseconds / 1e9
        debug_image_display = None
        if self.latest_image is not None:
            debug_image_display = self.latest_image.copy()
            
        depth_map = self.depth_estimator.predict_depth(self.latest_image)
        if depth_map is None:
            self.get_logger().warn("Failed to get depth map for current frame.")
            if debug_image_display is not None:
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image_display, encoding="bgr8")
                    debug_msg.header = msg.header 
                    self.get_logger().info(f"Publishing /fire_depth_localizer/debug_image (no depth map) (shape: {debug_image_display.shape}, encoding: bgr8)")
                    self.debug_image_pub.publish(debug_msg)
                except Exception as e:
                    self.get_logger().error(f"Error publishing debug image (no depth map): {e}")
            return

        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_display_image = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        person_detections_in_frame_coords = [] 
        fire_target_was_refined_this_callback = False

        # For GUI Info message
        gui_detection_details = []
        added_fire_ids_to_gui = set() # Initialize set to track fire IDs added to GUI message in this callback

        for detection_idx, detection in enumerate(msg.detections):
            label = detection.results[0].hypothesis.class_id.lower() 
            score = detection.results[0].hypothesis.score
            
            bbox = detection.bbox 
            cx_bbox = int(bbox.center.position.x)
            cy_bbox = int(bbox.center.position.y)
            w = int(bbox.size_x)
            h = int(bbox.size_y)
            x1 = cx_bbox - w // 2
            y1 = cy_bbox - h // 2
            x2 = cx_bbox + w // 2
            y2 = cy_bbox + h // 2

            # NEW: Check if cx_bbox, cy_bbox are valid for depth_map array access
            if not (0 <= cy_bbox < depth_map.shape[0] and 0 <= cx_bbox < depth_map.shape[1]):
                if debug_image_display is not None: 
                    color_label_oob = (128,128,128)
                    if label == "fire": color_label_oob = (0,0,128)
                    elif label == "person": color_label_oob = (128,0,0)
                    cv2.rectangle(debug_image_display, (x1, y1), (x2, y2), color_label_oob, 1)
                    cv2.putText(debug_image_display, f"{label} ({score:.2f}) OOB_depth_px", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label_oob, 1)
                self.get_logger().warn(f"Pixel ({cx_bbox}, {cy_bbox}) for {label} out of bounds in depth map array. Skipping this detection.")
                continue # Skip this detection

            Z = float(depth_map[cy_bbox, cx_bbox]) # Z is now defined for valid coords
            
            # NEW: Universal drawing on depth_display_image for all valid detections with depth
            if depth_display_image is not None:
                if label == "fire" and score >= 0.4:
                    cv2.circle(depth_display_image, (cx_bbox, cy_bbox), 10, (0, 0, 255), -1)
                    fire_info_text = f"Fire Z:{Z:.1f}m"
                    cv2.putText(depth_display_image, fire_info_text, (cx_bbox + 12, cy_bbox + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                elif label == "person" and score >= 0.7:
                    cv2.circle(depth_display_image, (cx_bbox, cy_bbox), 10, (255, 0, 0), -1)
                    person_info_text = f"Person Z:{Z:.1f}m"
                    cv2.putText(depth_display_image, person_info_text, (cx_bbox + 12, cy_bbox + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

            roi_status_for_log = "NA_NON_RELEVANT" # Will be updated based on mission logic

            # Target refinement logic for "fire" if in HOLD or APPROACH state
            # This should only happen if we are in offboard mode and actively managing a target
            if self.control_mode and self.control_mode.flag_control_offboard_enabled:
                if label == "fire" and (self.state == 'HOLD' or self.state == 'APPROACH') and self.fire_target is not None:
                    # Z_refine is the same Z calculated above
                    x_cam_refine = (cx_bbox - self.cx) / self.fx * Z
                    y_cam_refine = (cy_bbox - self.cy) / self.fy * Z
                    P_cam_optical_refine = np.array([x_cam_refine, y_cam_refine, Z])
                    
                    R_body_to_world_NED_refine = R.from_quat(self.orientation_q_scipy).as_matrix()
                    R_optical_to_bodyFRD_refine = np.array([[0,0,1],[1,0,0],[0,1,0]])
                    R_pitch_around_body_Y_refine = R.from_euler('y', self.camera_pitch).as_matrix()
                    world_vector_refine = R_body_to_world_NED_refine @ R_pitch_around_body_Y_refine @ R_optical_to_bodyFRD_refine @ P_cam_optical_refine
                    detected_object_world_coords_refine = np.array([self.odom_x, self.odom_y, self.odom_z]) + world_vector_refine

                    dist_to_current_target = np.linalg.norm(detected_object_world_coords_refine[:2] - np.array(self.fire_target[:2]))
                    
                    refinement_threshold = 0.75 # meters; tune this
                    if dist_to_current_target < refinement_threshold:
                        alpha = 0.3 # Smoothing factor for EMA
                        new_target_x = alpha * detected_object_world_coords_refine[0] + (1 - alpha) * self.fire_target[0]
                        new_target_y = alpha * detected_object_world_coords_refine[1] + (1 - alpha) * self.fire_target[1]
                        
                        self.get_logger().info(f"Refining target {self.fire_target[:2]} to ({new_target_x:.2f}, {new_target_y:.2f}) based on new detection.")
                        self.fire_target = (new_target_x, new_target_y, self.FIRE_APPROACH_ALTITUDE)
                        fire_target_was_refined_this_callback = True # Mark that target was refined
                        
                        # Update face yaw if in HOLD
                        if self.state == 'HOLD':
                            dx_refine = self.fire_target[0] - self.odom_x
                            dy_refine = self.fire_target[1] - self.odom_y
                            self.target_face_yaw = math.atan2(dy_refine, dx_refine)
                        
                        # Logging and visualization for refined target
                        self.data_logger.log_detection_event(
                            current_ros_time_sec,
                            label, score, cx_bbox, cy_bbox, Z,
                            detected_object_world_coords_refine, 
                            [self.odom_x, self.odom_y, self.odom_z],
                            self.orientation_q_scipy,
                            "IN_ROI_FIRE_REFINED_TARGET" 
                        )
                        if debug_image_display is not None:
                            cv2.rectangle(debug_image_display, (x1, y1), (x2, y2), (0,255,255), 2) 
                            cv2.circle(debug_image_display, (cx_bbox, cy_bbox), 5, (0,255,255), -1)
                            cv2.putText(debug_image_display, f"Refining Target ({score:.2f})", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                        
                        # This detection has served its purpose (refinement), continue to next detection.
                        # We also need to add this refined target to the gui_detection_details.
                        # Handled later by iterating all_known_fires_for_gui.
                        continue 
            
            # General detection processing (fire for new target, person, etc.)
            # Process fire for new target acquisition only if in SEARCH or CIRCLE states
            # AND if the target wasn't just refined (to avoid race conditions on self.fire_target)
            if label == "fire" and (self.state == 'SEARCH' or self.state == 'CIRCLE') and not fire_target_was_refined_this_callback:
                roi_x_min = self.cx - self.ROI_WIDTH / 2
                roi_x_max = self.cx + self.ROI_WIDTH / 2
                roi_y_min = self.cy - self.ROI_HEIGHT / 2
                roi_y_max = self.cy + self.ROI_HEIGHT / 2

                if not (roi_x_min <= cx_bbox <= roi_x_max and roi_y_min <= cy_bbox <= roi_y_max):
                    self.get_logger().info(f"Fire detection at ({cx_bbox:.1f},{cy_bbox:.1f}) is outside central ROI. Skipping.")
                    roi_status_for_log = "OOR_FIRE_SKIPPED"
                    if debug_image_display is not None: 
                         cv2.rectangle(debug_image_display, (x1,y1), (x2,y2), (100,100,100) , 1)
                         cv2.putText(debug_image_display, f"fire (OOR)", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100),1)
                    continue
                else:
                    roi_status_for_log = "IN_ROI_FIRE"
            elif label == "person":
                 roi_status_for_log = "PROCESSED_OTHER"
            # Add other label categorizations for roi_status_for_log if needed
        
            if 0 <= cy_bbox < depth_map.shape[0] and 0 <= cx_bbox < depth_map.shape[1]:
                Z = float(depth_map[cy_bbox, cx_bbox]) 
                
                x_cam = (cx_bbox - self.cx) / self.fx * Z 
                y_cam = (cy_bbox - self.cy) / self.fy * Z
                
                P_cam_optical = np.array([x_cam, y_cam, Z])
                
                R_body_to_world_NED = R.from_quat(self.orientation_q_scipy).as_matrix()
                R_optical_to_bodyFRD = np.array([[0,0,1],[1,0,0],[0,1,0]])
                R_pitch_around_body_Y = R.from_euler('y', self.camera_pitch).as_matrix()
                world_vector = R_body_to_world_NED @ R_pitch_around_body_Y @ R_optical_to_bodyFRD @ P_cam_optical
                detected_object_world_coords = np.array([self.odom_x, self.odom_y, self.odom_z]) + world_vector
       
                # INTEGRATION: Log detection event for processed objects
                if roi_status_for_log != "OOR_FIRE_SKIPPED":
                    if label == "fire": # Only log world coords if it's a fire and not skipped
                        self.get_logger().info(f"Calculated world coordinates for FIRE (Score: {score:.2f}): NED ({detected_object_world_coords[0]:.2f}, {detected_object_world_coords[1]:.2f}, {detected_object_world_coords[2]:.2f}), Depth Est: {Z:.2f}m")
                    
                    self.data_logger.log_detection_event(
                        current_ros_time_sec,
                        label, score, cx_bbox, cy_bbox, Z,
                        detected_object_world_coords,
                        [self.odom_x, self.odom_y, self.odom_z],
                        self.orientation_q_scipy,
                        roi_status_for_log
                    )

                # Standard drawing for all ROI detections
                if debug_image_display is not None:
                    color_label = (0,255,0) # Default green
                    if label == "fire": color_label = (0,0,255) # Red for fire (if IN ROI and processed)
                    elif label == "person": color_label = (255,0,0) # Blue for person
                    # No special color for 'cone' as per updated request
                    cv2.rectangle(debug_image_display, (x1, y1), (x2, y2), color_label, 2)
                    cv2.circle(debug_image_display, (cx_bbox, cy_bbox), 5, color_label, -1)
                    cv2.putText(debug_image_display, f"{label} ({score:.2f}) Z:{Z:.2f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label, 2)

                if label == "fire" and score >= 0.4: # This fire is IN ROI and has valid depth
                    # Specific drawing on depth_display_image was here; removed.

                    gx, gy, gz_world_fire = detected_object_world_coords[0], detected_object_world_coords[1], self.FIRE_APPROACH_ALTITUDE
                    clustered_fire_pos_3D = self.cluster_fire_positions(gx, gy, gz_world_fire)

                    if clustered_fire_pos_3D is None:
                        # self.get_logger().debug(f"Fire detection at world ({gx:.1f},{gy:.1f}) discarded by clustering.")
                        continue
                    
                    # Regardless of mode, assign an ID and publish marker if it's a new cluster
                    current_fire_tuple = tuple(clustered_fire_pos_3D)
                    fire_id_for_this_fire = self.fire_id_map.get(current_fire_tuple)
                    if fire_id_for_this_fire is None:
                        fire_id_for_this_fire = self.fire_id
                        self.fire_id_map[current_fire_tuple] = fire_id_for_this_fire
                        self.publish_fire_marker(current_fire_tuple, fire_id_for_this_fire) # Publish marker on new ID
                        self.fire_id += 1
                    
                    is_near_visited = False
                    for visited_fx, visited_fy, _ in self.visited_fires:
                        if np.linalg.norm(np.array(clustered_fire_pos_3D[:2]) - np.array([visited_fx, visited_fy])) < 1.0: # Visited threshold
                            is_near_visited = True
                            break
                    if is_near_visited:
                        # self.get_logger().info(f"Fire at ({clustered_fire_pos_3D[0]:.1f}, {clustered_fire_pos_3D[1]:.1f}) ignored: too close to a visited fire.")
                        # Still add to GUI as a detected (and visited) fire
                        if not any(np.array_equal(clustered_fire_pos_3D, df) for df in self.detected_fires): # Keep track for GUI
                           self.detected_fires.append(clustered_fire_pos_3D)
                        continue

                    # If in OFFBOARD mode, this fire might become the new self.fire_target
                    if self.control_mode and self.control_mode.flag_control_offboard_enabled:
                        if self.state == 'SEARCH' or \
                           (self.state == 'CIRCLE' and self.fire_target is not None and \
                            np.linalg.norm(np.array(clustered_fire_pos_3D[:2]) - np.array(self.fire_target[:2])) > 1.0):
                            
                            # Logic for handling transition from CIRCLE to a new target
                            if self.state == 'CIRCLE' and self.fire_target is not None:
                                 self.get_logger().info(f"CIRCLE: New distinct fire {clustered_fire_pos_3D[:2]} found while circling {self.fire_target[:2]}. Marking current target visited.")
                                 if self.fire_target not in self.visited_fires:
                                     self.visited_fires.append(self.fire_target)
                                     # Marker for old target published when it was first ID'd or when visited.
                                 self.circling = False 
                                 self.circle_center = None

                            self.fire_target = clustered_fire_pos_3D 
                            self.get_logger().info(f"OFFBOARD: New fire target acquired: {self.fire_target[:2]}. Transitioning to HOLD.")
                            
                            dx = self.fire_target[0] - self.odom_x
                            dy = self.fire_target[1] - self.odom_y
                            self.target_face_yaw = math.atan2(dy, dx)
                            
                            self.state = 'HOLD' # Transition to HOLD
                            self.approach_start_time = self.get_clock().now().nanoseconds / 1e9 # Reset approach timer
                            
                            # Reset PID errors for the new target
                            self.prev_error_x = 0.0
                            self.prev_error_y = 0.0
                            self.integral_x = 0.0
                            self.integral_y = 0.0
                            self.initial_search = False # No longer in initial search if a target is found
                            
                            if not any(np.array_equal(self.fire_target, df) for df in self.detected_fires):
                                 self.detected_fires.append(self.fire_target) # Add to general detected list
                            
                            # GUI info for this newly acquired active target will be handled by the general GUI update logic below.
                            # No need to add to gui_detection_details here specifically for this case.
                            break # Found a new fire target to act upon, exit detection loop for this message

                    # If not in offboard, or if in offboard but not in SEARCH/suitable CIRCLE state:
                    # Still track it as a detected fire for visualization if it's not already known.
                    if not any(np.array_equal(clustered_fire_pos_3D, df) for df in self.detected_fires):
                        self.detected_fires.append(clustered_fire_pos_3D)
                        self.get_logger().info(f"VISUALIZATION: New fire cluster detected at {clustered_fire_pos_3D[:2]} (ID: {fire_id_for_this_fire}). Not currently in Offboard action state.")


                elif label == "person" and score >= 0.7: # This will execute regardless of fire processing path
                    self.get_logger().info(f"PERSON detected: Score: {score:.2f} at pix({cx_bbox},{cy_bbox}), World Z: {Z:.2f}m, World Coords: {detected_object_world_coords[:2]}")
                    person_detections_in_frame_coords.append({
                        "world_coords": detected_object_world_coords, 
                        "score": score, 
                        "temp_id": len(person_detections_in_frame_coords) # simple temp ID
                    })
                    # Specific drawing on depth_display_image was here; removed.

            # The 'else' block that previously handled OOB for depth_map is now removed,
            # as this condition is checked and handled at the beginning of the loop for each detection.

        if person_detections_in_frame_coords:
            all_fire_locations_for_person_check = []
            # Add current target if it exists
            if self.fire_target:
                fire_id = self.fire_id_map.get(tuple(self.fire_target), -1) # Use -1 if somehow not mapped yet
                all_fire_locations_for_person_check.append({"coords": self.fire_target, "id": fire_id})

            # Add visited fires
            for visited_fire_coords in self.visited_fires:
                fire_id = self.fire_id_map.get(tuple(visited_fire_coords), -1)
                # Avoid duplicates if a visited fire is also the current target (though unlikely with current logic)
                if not any(np.array_equal(visited_fire_coords, f["coords"]) for f in all_fire_locations_for_person_check):
                    all_fire_locations_for_person_check.append({"coords": visited_fire_coords, "id": fire_id})

            # Add other detected (but not target/visited) fires from self.detected_fires
            for detected_fire_coords in self.detected_fires:
                fire_id = self.fire_id_map.get(tuple(detected_fire_coords), -1)
                if not any(np.array_equal(detected_fire_coords, f["coords"]) for f in all_fire_locations_for_person_check):
                     all_fire_locations_for_person_check.append({"coords": detected_fire_coords, "id": fire_id})
            
            for p_info in person_detections_in_frame_coords:
                person_coords = p_info["world_coords"]
                person_score = p_info["score"]
                person_temp_id = p_info["temp_id"]
                
                min_dist_to_fire = float('inf')
                closest_fire_id = -1
                closest_fire_coords_for_log = None

                if not all_fire_locations_for_person_check:
                    self.get_logger().info(f"Person at {person_coords[:2]} detected, but no known fire locations to check distance against.")
                    pass # Ensure this 'if' block has a body
                else:
                    for fire_info in all_fire_locations_for_person_check:
                        fire_loc = fire_info["coords"]
                        current_fire_id = fire_info["id"]
                        if fire_loc is None: continue 
                        distance = np.linalg.norm(np.array(person_coords[:2]) - np.array(fire_loc[:2]))
                        if distance < min_dist_to_fire:
                            min_dist_to_fire = distance
                            closest_fire_id = current_fire_id
                            closest_fire_coords_for_log = fire_loc[:2]
                        
                        if distance < 3.0: 
                            self.get_logger().warn(f"SAFETY ALERT: Person at {person_coords[:2]} (Score: {person_score:.2f}) is {distance:.2f}m close to fire ID {current_fire_id} at {fire_loc[:2]}.")
                            pass # Ensure this 'if' block has a body
                            # This alert might fire multiple times if person is close to multiple fires. The GUI will show the closest.
                
                # Add person to GUI details
                gui_detection_details.append({
                    "object_id": person_temp_id, # Temporary ID for this frame
                    "object_type": "person",
                    "world_position": {"x": person_coords[0], "y": person_coords[1], "z": person_coords[2]},
                    "confidence": person_score,
                    "is_current_target": False,
                    "is_visited": False,
                    "person_fire_distance": min_dist_to_fire if min_dist_to_fire != float('inf') else -1.0,
                    "associated_fire_id": closest_fire_id
                })
                if closest_fire_id != -1:
                    self.get_logger().info(f"Person (TempID {person_temp_id}, Score {person_score:.2f}) at ({person_coords[0]:.1f},{person_coords[1]:.1f}) is closest to Fire ID {closest_fire_id} at ({closest_fire_coords_for_log[0]:.1f},{closest_fire_coords_for_log[1]:.1f}), dist: {min_dist_to_fire:.2f}m.")
                    pass


        # Add other fires (not new target, but detected/visited) to gui_detection_details
        # Current target already added if it was newly acquired or refined (refinement needs to add to gui_detection_details too)
        # Let's ensure all relevant fires are in gui_detection_details:
        
        # Create a set of already added fire IDs to gui_detection_details to avoid duplicates
        # Reinitialize added_fire_ids_to_gui for this specific GUI message construction pass
        added_fire_ids_to_gui.clear() 
        # Add persons first to gui_detection_details from person_detections_in_frame_coords (done above loop)

        all_known_fires_for_gui = []
        # 1. Current self.fire_target (if any, and if in offboard pursuit)
        is_offboard_and_pursuing = self.control_mode and self.control_mode.flag_control_offboard_enabled and self.fire_target is not None
        if self.fire_target: # self.fire_target is the one actively pursued in offboard
            all_known_fires_for_gui.append({
                "coords": self.fire_target, 
                "is_actively_targeted_in_offboard": is_offboard_and_pursuing, 
                "is_visited": self.fire_target in self.visited_fires,
                "score_at_detection": -1 # May need to store score when target is chosen
            })

        # 2. Visited fires
        for vf_coords_tuple in self.visited_fires:
            vf_coords = np.array(vf_coords_tuple)
            # Avoid adding if it's currently the self.fire_target (already handled)
            if not (self.fire_target is not None and np.array_equal(vf_coords, self.fire_target)):
                all_known_fires_for_gui.append({
                    "coords": vf_coords_tuple, 
                    "is_actively_targeted_in_offboard": False, 
                    "is_visited": True,
                    "score_at_detection": -1
                })

        # 3. Other detected fires (in self.detected_fires but not target and not visited)
        for df_coords_tuple in self.detected_fires:
            df_coords = np.array(df_coords_tuple)
            # Avoid adding if it's current self.fire_target or already visited
            is_target = self.fire_target is not None and np.array_equal(df_coords, self.fire_target)
            is_visited = df_coords_tuple in self.visited_fires
            if not is_target and not is_visited:
                 all_known_fires_for_gui.append({
                    "coords": df_coords_tuple, 
                    "is_actively_targeted_in_offboard": False, 
                    "is_visited": False,
                    "score_at_detection": -1 # Need a way to get original score if required by GUI
                })


        for fire_info in all_known_fires_for_gui:
            f_coords_tuple = fire_info["coords"]
            f_id = self.fire_id_map.get(f_coords_tuple, -1) 
            
            # Check if this fire (by ID if valid, or by coords if ID is -1) is already in gui_detection_details
            # This is to prevent adding the same fire multiple times from different lists if logic overlaps.
            already_added = False
            if f_id != -1:
                if f_id in added_fire_ids_to_gui:
                    # If already added by ID, update its status if necessary
                    for item in gui_detection_details:
                        if item["object_id"] == f_id and item["object_type"] == "fire":
                            item["is_current_target"] = item["is_current_target"] or fire_info["is_actively_targeted_in_offboard"]
                            item["is_visited"] = item["is_visited"] or fire_info["is_visited"]
                            already_added = True
                            break
                    if already_added: continue
            else: # No valid ID, check by coordinates (less reliable if precision issues)
                for item in gui_detection_details:
                    if item["object_type"] == "fire" and \
                       abs(item["world_position"]["x"] - f_coords_tuple[0]) < 0.01 and \
                       abs(item["world_position"]["y"] - f_coords_tuple[1]) < 0.01:
                        item["is_current_target"] = item["is_current_target"] or fire_info["is_actively_targeted_in_offboard"]
                        item["is_visited"] = item["is_visited"] or fire_info["is_visited"]
                        already_added = True
                        break
                if already_added: continue


            gui_detection_details.append({
                "object_id": f_id,
                "object_type": "fire",
                "world_position": {"x": f_coords_tuple[0], "y": f_coords_tuple[1], "z": f_coords_tuple[2]},
                "confidence": fire_info["score_at_detection"], 
                "is_current_target": fire_info["is_actively_targeted_in_offboard"],
                "is_visited": fire_info["is_visited"],
                "person_fire_distance": -1.0, # Calculated when persons are processed
                "associated_fire_id": f_id 
            })
            if f_id != -1:
                added_fire_ids_to_gui.add(f_id)
        
        # Now, iterate through persons again to update their associated fire distances,
        # using the final list of fires in gui_detection_details
        for p_item in gui_detection_details:
            if p_item["object_type"] == "person":
                person_world_pos_obj = p_item["world_position"]
                person_coords_np = np.array([person_world_pos_obj["x"], person_world_pos_obj["y"]])
                min_dist_to_fire_for_person = float('inf')
                associated_fire_id_for_person = -1

                for fire_item_for_person_check in gui_detection_details:
                    if fire_item_for_person_check["object_type"] == "fire":
                        fire_world_pos_obj = fire_item_for_person_check["world_position"]
                        fire_coords_np = np.array([fire_world_pos_obj["x"], fire_world_pos_obj["y"]])
                        distance = np.linalg.norm(person_coords_np - fire_coords_np)
                        if distance < min_dist_to_fire_for_person:
                            min_dist_to_fire_for_person = distance
                            associated_fire_id_for_person = fire_item_for_person_check["object_id"]
                
                p_item["person_fire_distance"] = min_dist_to_fire_for_person if min_dist_to_fire_for_person != float('inf') else -1.0
                p_item["associated_fire_id"] = associated_fire_id_for_person


        # Publish GUI Info
        if gui_detection_details:
            gui_info_msg = String()
            gui_info_msg.data = json.dumps({
                "header": { # Minimal header for context, actual timestamping is by ROS
                    "stamp_sec": current_ros_time_sec, 
                    "frame_id": "map" # Assuming world coordinates are in map frame
                },
                "detections": gui_detection_details
            })
            self.gui_info_pub.publish(gui_info_msg)
            self.get_logger().info(f"Published {len(gui_detection_details)} items to /firedronex/gui_info")

        # Corrected try-except block for debug_image_pub
        if debug_image_display is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image_display, encoding="bgr8")
                debug_msg.header = msg.header 
                self.get_logger().info(f"Publishing /fire_depth_localizer/debug_image (shape: {debug_image_display.shape}, encoding: bgr8)")
                self.debug_image_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing debug image: {e}")

        try:
            depth_debug_msg = self.bridge.cv2_to_imgmsg(depth_display_image, encoding="bgr8")
            depth_debug_msg.header = msg.header
            self.get_logger().info(f"Publishing /fire_depth_localizer/depth_debug_image (shape: {depth_display_image.shape}, encoding: bgr8)")
            self.depth_debug_image_pub.publish(depth_debug_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing depth debug image: {e}")
            
    def cluster_fire_positions(self, x, y, z, threshold=1.0, min_separation=0.8):
        for i, (fx, fy, fz) in enumerate(self.detected_fires):
            distance = math.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if distance < threshold:
                new_x = round((fx + x) / 2, 1)
                new_y = round((fy + y) / 2, 1)
                new_fire_pos = (new_x, new_y, z)
                self.detected_fires[i] = new_fire_pos
                return new_fire_pos
            
        for fx, fy, _ in self.detected_fires:
            distance = math.sqrt((x - fx) ** 2 + (y - fy) ** 2) 
            if distance < min_separation:
                return None
            
        return (round(x, 1), round(y, 1), z)
                
    def mission_loop(self):
        self.publish_offboard_control() # Keep this heartbeat publishing
        
        effective_state_for_gui = self.state # Default to current self.state, will be overridden if not offboard

        if self.control_mode and self.control_mode.flag_control_offboard_enabled:
            # ---- BEGIN OFFBOARD LOGIC ----
            self.get_logger().debug(f"Offboard Active: Current State: {self.state}, Target: {self.fire_target[:2] if self.fire_target else 'None'}")

            # Check if current target has been visited
            if self.fire_target and self.fire_target in self.visited_fires:
                self.get_logger().info(f"Offboard: Target {self.fire_target[:2]} already visited. Clearing target, returning to SEARCH.")
                self.fire_target = None
                self.state = 'SEARCH' 
                # Reset search parameters for a new search from current location
                self.initial_search = True 
                self.search_center = (self.odom_x, self.odom_y)
                self.circle_radius = 2.0 
                self.circle_theta = 0.0
                self.last_theta = 0.0
                self.prev_error_x = 0.0; self.prev_error_y = 0.0
                self.integral_x = 0.0; self.integral_y = 0.0

            # Mission State Machine (only if offboard)
            if self.state == 'HOLD' and self.fire_target:
                if self.approach_start_time is None: # Should have been set when transitioning to HOLD
                    self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
                
                hold_duration = (self.get_clock().now().nanoseconds / 1e9) - self.approach_start_time
                self.get_logger().info(f"Offboard: State HOLD, Target: {self.fire_target[:2]}, Hold Duration: {hold_duration:.2f}s")
                if hold_duration < 2.0: # Hold duration
                    self.publish_setpoint(self.odom_x, self.odom_y, self.FIRE_APPROACH_ALTITUDE, yaw=self.target_face_yaw)
                else:
                    self.get_logger().info("Offboard: Hold complete. Transitioning to APPROACH.")
                    self.state = 'APPROACH'
                effective_state_for_gui = self.state

            elif self.state == 'APPROACH' and self.fire_target:
                self.get_logger().info(f"Offboard: State APPROACH, Target: {self.fire_target[:2]}")
                target_x_approach, target_y_approach, target_z_approach = self.fire_target
                current_time_sec_approach = self.get_clock().now().nanoseconds / 1e9
                time_in_approach_state = current_time_sec_approach - (self.approach_start_time if self.approach_start_time else current_time_sec_approach)

                if self.approach_start_time and time_in_approach_state > self.MAX_APPROACH_DURATION_SEC:
                    self.get_logger().warn(f"Offboard: APPROACH timeout for {self.fire_target[:2]}. Marking visited, to SEARCH.")
                    if self.fire_target not in self.visited_fires: self.visited_fires.append(self.fire_target)
                    # Marker published when ID was assigned.
                    
                    self.fire_target = None; self.state = 'SEARCH'
                    self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                    self.circle_radius = 2.0; self.circle_theta = 0.0; self.last_theta = 0.0
                    self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
                else:
                    dx_approach = target_x_approach - self.odom_x; dy_approach = target_y_approach - self.odom_y
                    distance_to_target = math.sqrt(dx_approach**2 + dy_approach**2)
                    if distance_to_target > 0.1: # Approach threshold
                        step_size_approach = 0.2 
                        target_step_x_approach = self.odom_x + step_size_approach * dx_approach / distance_to_target
                        target_step_y_approach = self.odom_y + step_size_approach * dy_approach / distance_to_target
                        yaw_approach = math.atan2(dy_approach, dx_approach)
                        self.publish_setpoint(target_step_x_approach, target_step_y_approach, target_z_approach, yaw=yaw_approach)
                    else: # Reached target
                        self.get_logger().info(f"Offboard: Reached fire target {self.fire_target[:2]}. Marking visited, to SEARCH.")
                        if self.fire_target not in self.visited_fires: self.visited_fires.append(self.fire_target)
                        # Marker published when ID was assigned.

                        self.fire_target = None; self.state = 'SEARCH'
                        self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                        self.circle_radius = 2.0; self.circle_theta = 0.0; self.last_theta = 0.0
                        self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
                effective_state_for_gui = self.state
            
            elif self.state == 'SEARCH': 
                self.get_logger().info(f"Offboard: State SEARCH. Center: {self.search_center}, Radius: {self.circle_radius:.2f}, Theta: {self.circle_theta:.2f}")
                if self.search_center is None: self.search_center = (self.odom_x, self.odom_y)
                
                center_x_search, center_y_search = self.search_center
                # Ensure dt is reasonable, e.g., from a timer or fixed value
                # self.dt is 0.1 from __init__
                
                # Max velocity and radius increment are class members
                arc_step_search = self.max_circle_velocity * self.dt / max(self.circle_radius, 0.1) # Avoid division by zero if radius is tiny
                self.circle_theta += arc_step_search
                
                target_x_search = center_x_search + self.circle_radius * math.cos(self.circle_theta)
                target_y_search = center_y_search + self.circle_radius * math.sin(self.circle_theta)
                target_z_search = self.SEARCH_ALTITUDE

                # Radius expansion logic (after completing a circle)
                if self.circle_theta >= 2 * math.pi: # Completed a full circle
                    self.circle_theta -= 2 * math.pi # Reset theta for the new circle
                    self.last_theta = self.circle_theta # Reset last_theta as well
                    self.circle_radius += self.radius_increment 
                    self.get_logger().info(f"Offboard SEARCH: Expanded radius to {self.circle_radius:.2f} m")
                    if self.circle_radius > 6.0: # Max search radius
                         self.get_logger().warn("Offboard SEARCH: Max search radius reached. Consider next action (e.g., RTL, land, or hold).")
                         # For now, it will continue trying to expand or search at max radius.
                         # Could add logic here to change state or behavior.
                
                # self.last_theta comparison for initial_search seems to be for a different kind of expansion.
                # The current logic is: complete a circle (theta > 2pi), then expand.
                # self.initial_search flag might not be needed with this clearer expansion logic.

                yaw_search = math.atan2(target_y_search - center_y_search, target_x_search - center_x_search) # Point outwards
                self.publish_setpoint(target_x_search, target_y_search, target_z_search, yaw=yaw_search)
                effective_state_for_gui = self.state
            
            else:
                self.get_logger().warn(f"Offboard: Unhandled state '{self.state}' or inconsistent condition (e.g. no fire_target for HOLD/APPROACH). Defaulting to SEARCH.")
                self.state = 'SEARCH'
                self.initial_search = True; self.search_center = (self.odom_x, self.odom_y) # Reset search
                self.circle_radius = 2.0; self.circle_theta = 0.0; self.last_theta = 0.0
                effective_state_for_gui = self.state # Will show SEARCH

            # ---- END OFFBOARD LOGIC ----
        else: # Not in Offboard mode or control_mode is None (indentation for this 'else' block starts here, 8 spaces)
            effective_state_for_gui = "VISUALIZING_DETECTIONS" # 12 spaces
            if self.fire_target is not None: # 12 spaces
                self.get_logger().info(f"Non-Offboard: Clearing active fire target {self.fire_target[:2]}.") # 16 spaces
                self.fire_target = None # 16 spaces
            
            if self.state != 'SEARCH': # 12 spaces
                self.get_logger().info(f"Non-Offboard: Resetting internal state from {self.state} to SEARCH.") # 16 spaces
                self.state = 'SEARCH' # 16 spaces
                # Reset search parameters
                self.initial_search = True # 16 spaces
                self.search_center = (self.odom_x, self.odom_y) if self.odom_x is not None else (0.0,0.0) # 16 spaces
                self.circle_radius = 2.0 # 16 spaces
                self.circle_theta = 0.0 # 16 spaces
                self.last_theta = 0.0 # 16 spaces
                self.prev_error_x = 0.0; self.prev_error_y = 0.0 # 16 spaces
                self.integral_x = 0.0; self.integral_y = 0.0 # 16 spaces

        # Publish the determined mission state for GUI (indentation for this block starts here, 8 spaces)
        state_msg = String()
        state_msg.data = effective_state_for_gui
        self.mission_state_pub.publish(state_msg)
        
    def publish_fire_marker(self, fire_target, fire_id):
        # fire_target is expected to be a tuple (x,y,z)
        if not isinstance(fire_target, tuple) or len(fire_target) != 3:
            self.get_logger().error(f"Cannot publish fire marker: fire_target is not a 3-tuple: {fire_target}")
            return

        x, y, z_actual_detection_alt = fire_target # z_actual_detection_alt might be different from marker Z
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fire"
        marker.id = fire_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0 # Marker at ground level for visualization
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  
        marker.scale.y = 0.5  
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        array = MarkerArray()
        array.markers.append(marker)
        self.fire_marker_pub.publish(array)
        
        self.get_logger().info(f"Published marker for Fire ID {fire_id} at X: {x:.2f}, Y: {y:.2f} (detection alt {z_actual_detection_alt:.2f})")
            

    def publish_setpoint(self, x, y, z, yaw):
        setpoint_msg = TrajectorySetpoint()
        current_ros_time_sec = self.get_clock().now().nanoseconds / 1e9
        setpoint_msg.timestamp = int(current_ros_time_sec * 1000000)
        
        # Apply geofence/safety limits
        clamped_x = max(self.X_BOUND_MIN, min(x, self.X_BOUND_MAX))
        clamped_y = max(self.Y_BOUND_MIN, min(y, self.Y_BOUND_MAX))
        clamped_z = self.OPERATIONAL_ALTITUDE_REAL # Force Z to fixed operational altitude for all setpoints

        if abs(x - clamped_x) > 0.01 or abs(y - clamped_y) > 0.01 or abs(z - clamped_z) > 0.01 : # Added tolerance for float comparison
            self.get_logger().warn(f"Setpoint ({x:.2f},{y:.2f},{z:.2f}, yaw:{yaw:.2f}) out of bounds or not at op alt. Clamped to ({clamped_x:.2f},{clamped_y:.2f},{clamped_z:.2f})")
        
        setpoint_pos_list = [clamped_x, clamped_y, clamped_z]
        setpoint_msg.position = setpoint_pos_list
        setpoint_msg.yaw = float(yaw)
        self.trajectory_pub.publish(setpoint_msg)
        
        # INTEGRATION: Gather additional data for trajectory logging
        # Log the clamped values for accuracy in what was commanded
        target_fire_pos_log = self.fire_target
        target_fire_id_log = -1
        if target_fire_pos_log is not None and self.fire_id_map is not None:
            target_fire_id_log = self.fire_id_map.get(tuple(target_fire_pos_log), -1)

        circling_fire_pos_log = None 
        circling_fire_id_log = -1
        if self.circling and self.fire_target is not None: 
            circling_fire_pos_log = self.fire_target 
            if self.fire_id_map is not None:
                 circling_fire_id_log = self.fire_id_map.get(tuple(circling_fire_pos_log), -1)
        elif self.circling and self.circle_center is not None:
            pass
        
        num_visited_log = len(self.visited_fires) if self.visited_fires is not None else 0
        hold_buffer_size_log = len(self.detected_fires) if self.detected_fires is not None else 0 

        # INTEGRATION: Call logger
        self.data_logger.log_trajectory_event(
            current_ros_time_sec,
            setpoint_pos_list, # Log the actual (clamped) setpoint sent
            float(yaw),
            [self.odom_x, self.odom_y, self.odom_z],
            self.current_yaw,
            self.state,
            target_fire_pos=target_fire_pos_log,
            target_fire_id=target_fire_id_log,
            circling_fire_pos=circling_fire_pos_log,
            circling_fire_id=circling_fire_id_log,
            num_visited_fires=num_visited_log,
            hold_buffer_size=hold_buffer_size_log
        )
        
    def odom_callback(self, msg):
        self.odom_x = msg.position[0]
        self.odom_y = msg.position[1]
        self.odom_z = msg.position[2]
        
        # PX4's VehicleOdometry msg.q is typically [w, x, y, z]
        # SciPy's Rotation.from_quat expects [x, y, z, w]
        q_w = msg.q[0]
        q_x = msg.q[1]
        q_y = msg.q[2]
        q_z = msg.q[3]

        # Update self.orientation_q_scipy for use in detection_callback's coordinate transforms
        self.orientation_q_scipy = np.array([q_x, q_y, q_z, q_w])
        
        # Update self.current_yaw using the correctly ordered quaternion
        # R.from_quat uses the [x, y, z, w] order
        body_to_world_rotation = R.from_quat(self.orientation_q_scipy)
        # Get Euler angles: 'zyx' sequence means yaw, pitch, roll. We need the yaw (first element).
        self.current_yaw = body_to_world_rotation.as_euler('zyx', degrees=False)[0]
    
    def vehicle_control_mode_callback(self,msg):
        self.control_mode = msg

    def publish_offboard_control(self):
        ctrl_msg = OffboardControlMode()
        ctrl_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        ctrl_msg.position = True
        ctrl_msg.velocity = False 
        ctrl_msg.acceleration = False
        ctrl_msg.attitude = False
        ctrl_msg.body_rate = False
        self.offboard_control_pub.publish(ctrl_msg)

    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.get_logger().info("Image received") # Too verbose for every frame
        except Exception as e:
            # self.get_logger().error(f"Error converting image: {e}")
            self.latest_image = None # Ensure latest_image is None if conversion fails
    
def main(args=None):
    rclpy.init(args=args)
    node = FireDepthLocalizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down FireDepthLocalizer node...")
    finally:
        # Add cleanup similar to yolo_firedronex_real.py
        # Example: if hasattr(node, 'data_logger') and node.data_logger is not None: node.data_logger.close_files()
        if rclpy.ok() and hasattr(node, 'destroy_node') and callable(node.destroy_node): 
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
if __name__ == '__main__':
    main() 