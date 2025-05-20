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
import json 
from geometry_msgs.msg import Point 

from fire_data_logger import FireDataLogger

class FireDepthLocalizer(Node):
    def __init__(self):
        super().__init__('fire_depth_localizer')

        # Initializing Data Logger
        self.log_main_dir = "/home/varun/ws_offboard_control/firedronex_plots" 
        self.data_logger = FireDataLogger(log_directory=self.log_main_dir)

        self.bridge = CvBridge()
        self.latest_image = None
        
        # Camera intrinsics
        self.fx = 1397.2235870361328
        self.fy = 1397.2235298156738
        self.cx = 960.0
        self.cy = 540.0
        
        # Orientation and position of the camera and drone
        self.camera_pitch = -0.785  
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = -5.0        
        self.current_yaw = 0.0 
        self.orientation_q_scipy = np.array([0.0, 0.0, 0.0, 1.0]) 
        
        # Mission parameters
        self.state = 'SEARCH'
        self.circle_radius = 2.0 
        self.circle_theta = 0.0
        
        self.search_center = None
        self.initial_search = True 
        self.last_theta = 0.0
        
        self.max_circle_velocity = 0.2 
        self.radius_increment = 0.5
        self.fire_id_map = {} 
        self.fire_id = 1
        self.MAX_SEARCH_RADIUS_METERS = 6.0 

        # Constants for fire mission
        self.FIRE_APPROACH_ALTITUDE = -5.90
        self.SEARCH_ALTITUDE = -5.90
        self.MAX_APPROACH_DURATION_SEC = 20.0
        
        # Empty list for Visited fires & detected fires
        self.visited_fires = []
        self.detected_fires = []

        # Camera ROI Parameters
        self.ROI_WIDTH = 1920   
        self.ROI_HEIGHT = 1080 

        self.target_face_yaw = 0.0 
        self.approach_start_time = None 
        
        # State variables
        self.fire_target = None
        self.circling = False 
        self.circle_center = None
        
        # QoS settings for subscribers and publishers
        subscriber_qos = QoSProfile(
            depth=10, 
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST 
        )

        self.debug_image_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publisher for fire markers
        marker_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10 
        )
        self.fire_marker_pub = self.create_publisher(MarkerArray, '/fire_markers', qos_profile=marker_publisher_qos) 
        
        # Adding publisher for debug image
        self.debug_image_pub = self.create_publisher(Image, '/fire_depth_localizer/debug_image', qos_profile=self.debug_image_publisher_qos)
        # Adding publisher for depth debug image
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
        
        # Initializing the Depth Anything V2 model
        package_share_dir = get_package_share_directory('px4_ros_com')
        depth_model_path = os.path.join(package_share_dir, 'depth_anything_models', 'depth_anything_v2_vits.pth')

        self.depth_estimator = DepthAnythingV2Estimator(model_path=depth_model_path, encoder='vits')
        
        # Subscriptions for camera feed, detections, odometry and control mode
        self.create_subscription(
            Image,
            '/camera',
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

        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.vehicle_control_mode_callback,
            qos_profile = subscriber_qos
        )
        
        # Setting up publishers for trajectory setpoint and offboard control mode
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
        
        # Adding publisher for mission state
        self.mission_state_pub = self.create_publisher(String, '/firedronex/mission_state', 10)
        
        # Adding publisher for detailed GUI information
        self.gui_info_pub = self.create_publisher(String, '/firedronex/gui_info', 10)
        
        self.create_timer(0.1, self.mission_loop)
        
    # Function of processing the camera feed and publishing the image    
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
        
        gui_detection_details = []
        added_fire_ids_to_gui = set()
        
        # Iterating through the detections
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

            # Checking if cx_bbox, cy_bbox are valid for depth_map array access
            if not (0 <= cy_bbox < depth_map.shape[0] and 0 <= cx_bbox < depth_map.shape[1]):
                if debug_image_display is not None: 
                    color_label_oob = (128,128,128)
                    if label == "fire": color_label_oob = (0,0,128)
                    elif label == "person": color_label_oob = (128,0,0)
                    cv2.rectangle(debug_image_display, (x1, y1), (x2, y2), color_label_oob, 1)
                    cv2.putText(debug_image_display, f"{label} ({score:.2f}) OOB_depth_px", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label_oob, 1)
                self.get_logger().warn(f"Pixel ({cx_bbox}, {cy_bbox}) for {label} out of bounds in depth map array. Skipping this detection.")
                continue 

            Z = float(depth_map[cy_bbox, cx_bbox]) 
            
            # Drawing on depth_display_image for all valid detections with depth
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

            roi_status_for_log = "NA_NON_RELEVANT" 

            # Checking if the label is "fire" and if the state is either HOLD or APPROACH
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
                
                refinement_threshold = 0.75 # meters
                if dist_to_current_target < refinement_threshold:
                    alpha = 0.3 # Smoothing factor for target refinement
                    new_target_x = alpha * detected_object_world_coords_refine[0] + (1 - alpha) * self.fire_target[0]
                    new_target_y = alpha * detected_object_world_coords_refine[1] + (1 - alpha) * self.fire_target[1]
                    
                    self.get_logger().info(f"Refining target {self.fire_target[:2]} to ({new_target_x:.2f}, {new_target_y:.2f}) based on new detection.")
                    self.fire_target = (new_target_x, new_target_y, self.FIRE_APPROACH_ALTITUDE)
                    fire_target_was_refined_this_callback = True
                    
                    if self.state == 'HOLD' or self.state == 'APPROACH': 
                        if self.state == 'HOLD': 
                            dx_refine = self.fire_target[0] - self.odom_x
                            dy_refine = self.fire_target[1] - self.odom_y
                            self.target_face_yaw = math.atan2(dy_refine, dx_refine)
                        
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

                continue
            
            # General detection processing for fire and other objects
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
       
                # Logging the detection event for processed objects
                if roi_status_for_log != "OOR_FIRE_SKIPPED":
                    if label == "fire": 
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
                    if label == "fire": color_label = (0,0,255) # Red for fire
                    elif label == "person": color_label = (255,0,0) # Blue for person
                    cv2.rectangle(debug_image_display, (x1, y1), (x2, y2), color_label, 2)
                    cv2.circle(debug_image_display, (cx_bbox, cy_bbox), 5, color_label, -1)
                    cv2.putText(debug_image_display, f"{label} ({score:.2f}) Z:{Z:.2f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label, 2)

                if label == "fire" and score >= 0.4: 

                    gx, gy, gz_world_fire = detected_object_world_coords[0], detected_object_world_coords[1],self.FIRE_APPROACH_ALTITUDE
                    clustered_fire_pos_3D = self.cluster_fire_positions(gx, gy, gz_world_fire)

                    if clustered_fire_pos_3D is None:
                        continue
                    
                    # Improved Visited Fire Check 
                    is_already_visited_and_close = False
                    for visited_fire_coords_tuple in self.visited_fires:
                        dist_to_visited = np.linalg.norm(np.array(clustered_fire_pos_3D[:2]) - np.array(visited_fire_coords_tuple[:2]))
                        if dist_to_visited < 1.0: # meters
                            is_already_visited_and_close = True
                            visited_fire_id = self.fire_id_map.get(visited_fire_coords_tuple, -1) 
                            self.get_logger().info(f"Same visited fire ID {visited_fire_id} re-detected at ({clustered_fire_pos_3D[0]:.2f}, {clustered_fire_pos_3D[1]:.2f}) within 1m. Not re-targeting.")
                            if visited_fire_id not in added_fire_ids_to_gui:
                                gui_detection_details.append({
                                    "object_id": visited_fire_id,
                                    "object_type": "fire",
                                    "world_position": {"x": clustered_fire_pos_3D[0], "y": clustered_fire_pos_3D[1], "z": clustered_fire_pos_3D[2]},
                                    "confidence": score, 
                                    "is_current_target": False, 
                                    "is_visited": True,
                                    "person_fire_distance": -1.0, 
                                    "associated_fire_id": visited_fire_id
                                })
                                added_fire_ids_to_gui.add(visited_fire_id)
                            break # 
                    
                    if is_already_visited_and_close:
                        if not any(np.array_equal(clustered_fire_pos_3D, df_coords) for df_coords in self.detected_fires):
                             self.detected_fires.append(clustered_fire_pos_3D)
                        continue 

                    current_potential_target_tuple = tuple(clustered_fire_pos_3D)
                    fire_id_for_potential_target = self.fire_id_map.get(current_potential_target_tuple)
                    if fire_id_for_potential_target is None:
                        fire_id_for_potential_target = self.fire_id
                        self.fire_id_map[current_potential_target_tuple] = fire_id_for_potential_target
                        self.fire_id +=1

                    # Checking if the state is SEARCH or CIRCLE along with distance to the current target
                    if self.state == 'SEARCH' or \
                       (self.state == 'CIRCLE' and self.fire_target is not None and \
                        np.linalg.norm(np.array(clustered_fire_pos_3D[:2]) - np.array(self.fire_target[:2])) > 1.0): 
                        
                        if self.state == 'CIRCLE':
                            if self.fire_target not in self.visited_fires:
                                self.visited_fires.append(self.fire_target)
                                fire_id_to_publish = self.fire_id_map.get(self.fire_target)
                                if fire_id_to_publish is None:
                                    fire_id_to_publish = self.fire_id
                                    self.fire_id_map[self.fire_target] = fire_id_to_publish
                                    self.fire_id +=1
                                self.publish_fire_marker(self.fire_target, fire_id_to_publish)
                            self.circling = False 
                            self.circle_center = None

                        self.fire_target = clustered_fire_pos_3D 
                        
                        dx = self.fire_target[0] - self.odom_x
                        dy = self.fire_target[1] - self.odom_y
                        self.target_face_yaw = math.atan2(dy, dx)
                        
                        self.state = 'HOLD'
                        self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
                        
                        self.prev_error_x = 0.0
                        self.prev_error_y = 0.0
                        self.integral_x = 0.0
                        self.integral_y = 0.0
                        self.initial_search = False 
                        
                        self.get_logger().info(f"New fire target ID {fire_id_for_potential_target}: {self.fire_target[:2]}. Transitioning to HOLD.")
                        if not any(np.array_equal(self.fire_target, df) for df in self.detected_fires):
                             self.detected_fires.append(self.fire_target)
                        self.publish_fire_marker(self.fire_target, fire_id_for_potential_target) 
                        
                        if fire_id_for_potential_target not in added_fire_ids_to_gui:
                            gui_detection_details.append({
                                "object_id": fire_id_for_potential_target,
                                "object_type": "fire",
                                "world_position": {"x": self.fire_target[0], "y": self.fire_target[1], "z": self.fire_target[2]},
                                "confidence": score, 
                                "is_current_target": True, 
                                "is_visited": False, 
                                "person_fire_distance": -1.0,
                                "associated_fire_id": fire_id_for_potential_target 
                            })
                            added_fire_ids_to_gui.add(fire_id_for_potential_target)
                        break 
                    else:
                        if not any(np.array_equal(clustered_fire_pos_3D, df_coords) for df_coords in self.detected_fires):
                            self.detected_fires.append(clustered_fire_pos_3D)
                            self.get_logger().info(f"Fire ID {fire_id_for_potential_target} detected at {clustered_fire_pos_3D[:2]} added to general detections list.")

                        if fire_id_for_potential_target not in added_fire_ids_to_gui:
                             gui_detection_details.append({
                                "object_id": fire_id_for_potential_target,
                                "object_type": "fire",
                                "world_position": {"x": clustered_fire_pos_3D[0], "y": clustered_fire_pos_3D[1], "z": clustered_fire_pos_3D[2]},
                                "confidence": score,
                                "is_current_target": False, 
                                "is_visited": False, 
                                "person_fire_distance": -1.0,
                                "associated_fire_id": fire_id_for_potential_target
                            })
                             added_fire_ids_to_gui.add(fire_id_for_potential_target)
                
                # Checking if the label is "person" and if the score is above a certain threshold
                elif label == "person" and score >= 0.7: 
                    self.get_logger().info(f"PERSON detected: Score: {score:.2f} at pix({cx_bbox},{cy_bbox}), World Z: {Z:.2f}m, World Coords: {detected_object_world_coords[:2]}")
                    person_detections_in_frame_coords.append({
                        "world_coords": detected_object_world_coords, 
                        "score": score, 
                        "temp_id": len(person_detections_in_frame_coords)
                    })

        if person_detections_in_frame_coords:
            all_fire_locations_for_person_check = []
            if self.fire_target:
                fire_id = self.fire_id_map.get(tuple(self.fire_target), -1) 
                all_fire_locations_for_person_check.append({"coords": self.fire_target, "id": fire_id})

            # Adding visited fires
            for visited_fire_coords in self.visited_fires:
                fire_id = self.fire_id_map.get(tuple(visited_fire_coords), -1)
                # Avoiding duplicates if a visited fire is also the current target
                if not any(np.array_equal(visited_fire_coords, f["coords"]) for f in all_fire_locations_for_person_check):
                    all_fire_locations_for_person_check.append({"coords": visited_fire_coords, "id": fire_id})

            # Adding other detected fires 
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
                    pass 
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
                            pass 
                
                # Adding person details to GUI
                gui_detection_details.append({
                    "object_id": person_temp_id, 
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

        added_fire_ids_to_gui = {
            item["object_id"] for item in gui_detection_details if item["object_type"] == "fire"
        }

        all_known_fires_for_gui = []
        if self.fire_target:
            all_known_fires_for_gui.append({"coords": self.fire_target, "is_target": True, "is_visited": False}) 
        for vf_coords in self.visited_fires:
            is_target = self.fire_target is not None and np.array_equal(vf_coords, self.fire_target)
            if not any(np.array_equal(vf_coords, f["coords"]) for f in all_known_fires_for_gui if f["is_target"] == is_target): 
                 all_known_fires_for_gui.append({"coords": vf_coords, "is_target": is_target, "is_visited": True})
        for df_coords in self.detected_fires:
            is_target = self.fire_target is not None and np.array_equal(df_coords, self.fire_target)
            is_visited_already = any(np.array_equal(df_coords, vf_coord_tuple) for vf_coord_tuple in self.visited_fires)
            if not any(np.array_equal(df_coords, f["coords"]) for f in all_known_fires_for_gui if f["is_target"] == is_target and f["is_visited"] == is_visited_already):
                all_known_fires_for_gui.append({"coords": df_coords, "is_target": is_target, "is_visited": is_visited_already})

        for fire_info in all_known_fires_for_gui:
            f_coords = fire_info["coords"]
            f_id = self.fire_id_map.get(tuple(f_coords), -1) 
            
            if f_id != -1 and f_id in added_fire_ids_to_gui: 
                for item in gui_detection_details:
                    if item["object_id"] == f_id and item["object_type"] == "fire":
                        item["is_current_target"] = item["is_current_target"] or fire_info["is_target"]
                        item["is_visited"] = item["is_visited"] or fire_info["is_visited"]
                        break
                continue 
            
            gui_detection_details.append({
                "object_id": f_id,
                "object_type": "fire",
                "world_position": {"x": f_coords[0], "y": f_coords[1], "z": f_coords[2]},
                "confidence": -1.0, 
                "is_current_target": fire_info["is_target"],
                "is_visited": fire_info["is_visited"],
                "person_fire_distance": -1.0,
                "associated_fire_id": f_id
            })
            added_fire_ids_to_gui.add(f_id)


        # Publishing GUI Info
        if gui_detection_details:
            gui_info_msg = String()
            gui_info_msg.data = json.dumps({
                "header": { 
                    "stamp_sec": current_ros_time_sec, 
                    "frame_id": "map"
                },
                "detections": gui_detection_details
            })
            self.gui_info_pub.publish(gui_info_msg)
            self.get_logger().info(f"Published {len(gui_detection_details)} items to /firedronex/gui_info")

        # Publishing debug image
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

    # Function to cluster fire positions that are close to each other            
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

    # Main Function for mission loop            
    def mission_loop(self):
        self.publish_offboard_control()
        
        current_mission_state_for_gui = self.state 

        if self.control_mode and self.control_mode.flag_control_offboard_enabled:
            self.get_logger().debug(f"Offboard Active: State: {self.state}, Target: {self.fire_target[:2] if self.fire_target else 'None'}, Visited: {len(self.visited_fires)}")

            # Checking if the fire target is already visited
            if self.fire_target and self.fire_target in self.visited_fires:
                self.get_logger().info(f"Offboard: Current target {self.fire_target[:2]} is already visited. Clearing target, returning to SEARCH.")
                self.fire_target = None
                self.state = 'SEARCH'
                self.initial_search = True 
                self.search_center = (self.odom_x, self.odom_y)
                self.circle_radius = 2.0 
                self.circle_theta = self.current_yaw
                self.last_theta = self.current_yaw
                self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
            
            # Handling HOLD state
            if self.state == 'HOLD' and self.fire_target:
                if self.approach_start_time is None: 
                    self.get_logger().warn("HOLD state entered without approach_start_time set. Setting now.")
                    self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
                
                hold_duration = (self.get_clock().now().nanoseconds / 1e9) - self.approach_start_time
                self.get_logger().info(f"Offboard: HOLD Target: {self.fire_target[:2]}, Duration: {hold_duration:.2f}s / 2.0s")
                if hold_duration < 2.0: 
                    self.publish_setpoint(self.odom_x, self.odom_y, self.FIRE_APPROACH_ALTITUDE, yaw=self.target_face_yaw)
                else:
                    self.get_logger().info("Offboard: Hold complete. Transitioning to APPROACH.")
                    self.state = 'APPROACH'
            
            # Handling APPROACH state
            elif self.state == 'APPROACH' and self.fire_target:
                target_x_app, target_y_app, target_z_app = self.fire_target
                self.get_logger().info(f"Offboard: APPROACH Target: {self.fire_target[:2]}")
                current_time_sec_app = self.get_clock().now().nanoseconds / 1e9
                time_in_approach_state = current_time_sec_app - (self.approach_start_time if self.approach_start_time else current_time_sec_app)

                if self.approach_start_time and time_in_approach_state > self.MAX_APPROACH_DURATION_SEC:
                    self.get_logger().warn(f"Offboard: APPROACH timeout for {self.fire_target[:2]}. Marking visited, to SEARCH.")
                    if self.fire_target not in self.visited_fires: self.visited_fires.append(self.fire_target)
                    
                    self.fire_target = None; self.state = 'SEARCH'
                    self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                    self.circle_radius = 2.0
                    self.circle_theta = self.current_yaw
                    self.last_theta = self.current_yaw
                    self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
                else:
                    dx_app = target_x_app - self.odom_x; dy_app = target_y_app - self.odom_y
                    distance_to_target = math.sqrt(dx_app**2 + dy_app**2)
                    if distance_to_target > 0.2: 
                        step_size_app = 0.2 
                        target_step_x = self.odom_x + step_size_app * dx_app / distance_to_target
                        target_step_y = self.odom_y + step_size_app * dy_app / distance_to_target
                        yaw_app = math.atan2(dy_app, dx_app)
                        self.publish_setpoint(target_step_x, target_step_y, target_z_app, yaw=yaw_app)
                    else:
                        self.get_logger().info(f"Offboard: Reached fire target {self.fire_target[:2]}. Marking visited, to SEARCH.")
                        if self.fire_target not in self.visited_fires: self.visited_fires.append(self.fire_target)
                        self.fire_target = None; self.state = 'SEARCH'
                        self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                        self.circle_radius = 2.0
                        self.circle_theta = self.current_yaw
                        self.last_theta = self.current_yaw
                        self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
            
            # Handling SEARCH state
            elif self.state == 'SEARCH':
                if self.fire_target is not None:
                    self.get_logger().warn(f"Offboard: In SEARCH state but fire_target is {self.fire_target[:2]}. Clearing target.")
                    self.fire_target = None 
                
                self.get_logger().info(f"Offboard: SEARCH. Center: {self.search_center}, R: {self.circle_radius:.2f}, Theta: {self.circle_theta:.2f}")
                if self.search_center is None: self.search_center = (self.odom_x, self.odom_y)
                
                center_x_s, center_y_s = self.search_center
                arc_step_s = self.max_circle_velocity * self.dt / max(self.circle_radius, 0.1)
                self.circle_theta += arc_step_s
                
                target_x_s = center_x_s + self.circle_radius * math.cos(self.circle_theta)
                target_y_s = center_y_s + self.circle_radius * math.sin(self.circle_theta)
                target_z_s = self.SEARCH_ALTITUDE

                if self.circle_theta >= (2 * math.pi + self.last_theta): 
                    if self.circle_radius < self.MAX_SEARCH_RADIUS_METERS:
                        self.circle_radius += self.radius_increment 
                        self.get_logger().info(f"Offboard SEARCH: Expanded radius to {self.circle_radius:.2f} m")
                    else:
                        self.get_logger().warn(f"Offboard SEARCH: Max search radius {self.MAX_SEARCH_RADIUS_METERS:.2f}m reached. Continuing search at this radius.")
                    self.last_theta = self.circle_theta % (2*math.pi) 
                
                yaw_s = math.atan2(target_y_s - center_y_s, target_x_s - center_x_s) 
                self.publish_setpoint(target_x_s, target_y_s, target_z_s, yaw=yaw_s)
            
            # Handling CIRCLE state
            elif self.state == 'CIRCLE' and self.fire_target: 
                self.get_logger().info(f"Offboard: CIRCLE around target {self.fire_target[:2]}. R: {self.circle_radius:.2f}, Theta: {self.circle_theta:.2f}")
                if self.circle_center is None: self.circle_center = self.fire_target[:2] 

                center_x_c, center_y_c = self.circle_center
                arc_step_c = self.max_circle_velocity * self.dt / max(self.circle_radius, 0.1)
                self.circle_theta += arc_step_c

                target_x_c = center_x_c + self.circle_radius * math.cos(self.circle_theta)
                target_y_c = center_y_c + self.circle_radius * math.sin(self.circle_theta)
                target_z_c = self.FIRE_APPROACH_ALTITUDE 

                if self.circle_theta >= (2 * math.pi + self.last_theta):
                    self.get_logger().info(f"Offboard CIRCLE: Completed a lap around {self.fire_target[:2]}. Marking visited, to SEARCH.")
                    if self.fire_target not in self.visited_fires: self.visited_fires.append(self.fire_target)
                    self.fire_target = None; self.state = 'SEARCH'; self.circling = False; self.circle_center = None
                    self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                    self.circle_radius = 2.0
                    self.circle_theta = self.current_yaw
                    self.last_theta = self.current_yaw
                    self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0
                else:
                    yaw_c = math.atan2(target_y_c - center_y_c, target_x_c - center_x_c) 
                    self.publish_setpoint(target_x_c, target_y_c, target_z_c, yaw=yaw_c)
            
            # If not in any known state go to SEARCH
            else: 
                if self.fire_target: 
                    self.get_logger().warn(f"Offboard: Unknown state '{self.state}' with target. Re-evaluating: HOLD target {self.fire_target[:2]}.")
                    self.state = 'HOLD' 
                    self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
                else: 
                    self.get_logger().warn(f"Offboard: Unknown state '{self.state}' and no target. Defaulting to SEARCH.")
                    self.state = 'SEARCH'
                    self.initial_search = True; self.search_center = (self.odom_x, self.odom_y)
                    self.circle_radius = 2.0
                    self.circle_theta = self.current_yaw
                    self.last_theta = self.current_yaw
            
            current_mission_state_for_gui = self.state 
        
        else: 
            current_mission_state_for_gui = "VISUALIZING_DETECTIONS"
            if self.fire_target is not None:
                self.fire_target = None 
            
            if self.state != 'SEARCH': 
                self.state = 'SEARCH'
                # Resetting search parameters for consistency
                self.initial_search = True 
                self.search_center = (self.odom_x, self.odom_y) if self.odom_x is not None else (0.0,0.0)
                self.circle_radius = 2.0 
                self.circle_theta = 0.0; self.last_theta = 0.0
                self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.integral_x = 0.0; self.integral_y = 0.0

        # Publishing the determined mission state for GUI
        state_msg = String()
        state_msg.data = current_mission_state_for_gui
        self.mission_state_pub.publish(state_msg)
        
    # Function to publish fire marker
    def publish_fire_marker(self, fire_target, fire_id):
        x, y, z = fire_target
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fire"
        marker.id = fire_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
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
        
    # Function to publish setpoint
    def publish_setpoint(self, x, y, z, yaw):
        setpoint_msg = TrajectorySetpoint()
        current_ros_time_sec = self.get_clock().now().nanoseconds / 1e9
        setpoint_msg.timestamp = int(current_ros_time_sec * 1000000)
        
        setpoint_pos_list = [float(x), float(y), float(z)]
        setpoint_msg.position = setpoint_pos_list
        setpoint_msg.yaw = float(yaw)
        self.trajectory_pub.publish(setpoint_msg)
        
        # Gathering additional data for trajectory logging
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

        # Calling the logger function
        self.data_logger.log_trajectory_event(
            current_ros_time_sec,
            setpoint_pos_list,
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

    # Function to handle odometry callback    
    def odom_callback(self, msg):
        self.odom_x = msg.position[0]
        self.odom_y = msg.position[1]
        self.odom_z = msg.position[2]
        
        q_w = msg.q[0]
        q_x = msg.q[1]
        q_y = msg.q[2]
        q_z = msg.q[3]

        # Updating Quaternion in the correct order
        self.orientation_q_scipy = np.array([q_x, q_y, q_z, q_w])
        # Rotating the quaternion to match the body frame
        body_to_world_rotation = R.from_quat(self.orientation_q_scipy)
        # Getting Euler angles
        self.current_yaw = body_to_world_rotation.as_euler('zyx', degrees=False)[0]
    
    # Function to handle control mode callback
    def vehicle_control_mode_callback(self,msg):
        self.control_mode = msg
    
    # Function to handle camera info callback
    def publish_offboard_control(self):
        ctrl_msg = OffboardControlMode()
        ctrl_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        ctrl_msg.position = True
        ctrl_msg.velocity = False
        ctrl_msg.acceleration = False
        ctrl_msg.attitude = False
        ctrl_msg.body_rate = False
        self.offboard_control_pub.publish(ctrl_msg)

    # Function to handle image callback
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.get_logger().info("Image received") # Too verbose for every frame
        except Exception as e:
            # self.get_logger().error(f"Error converting image: {e}")
            self.latest_image = None # Ensure latest_image is None if conversion fails

# Main function to run the node
def main(args=None):
    rclpy.init(args=args)
    node = FireDepthLocalizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()