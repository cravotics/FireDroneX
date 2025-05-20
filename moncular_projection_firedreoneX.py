#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, TrajectorySetpoint, VehicleControlMode
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

class SafeFireDroneX(Node):
    def __init__(self):
        super().__init__('safe_firedronex_sim')
        self.fire_detected = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = -7.0
        self.circle_radius = 1.0
        self.circle_theta = 0.0
        self.circle_step = 0.01
        self.control_mode = None
        self.fire_target = None
        self.detected_fires = []
        self.visited_fires = []
        self.circling = False
        self.circle_center = None
        self.circling_started = False
        self.initial_search = True
        self.last_theta = 0.0
        self.search_center = None
        self.cx = 960.0
        self.cy = 540.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        self.fx = 1397.2235870361328
        self.fy = 1397.2235298156738
        self.camera_pitch = -0.785
        self.fire_id = 1
        self.fire_id_map = {}
        self.state = 'SEARCH'
        self.ROI_WIDTH = 500   
        self.ROI_HEIGHT = 300  

        ## PID PARAMETERS
        self.kp = 0.3
        self.ki = 0.0
        self.kd = 0.15
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.dt = 0.1
        # --------------------- #
        self.tolerance = 0.2
        self.approach_start_time = None
        self.fire_projection_buffer = []
        self.fire_projection_buffer_size = 2
        self.max_circle_velocity = 0.3
        self.drone_marker_pub = self.create_publisher(Marker, '/drone_marker', 10)
        self.fire_marker_pub = self.create_publisher(MarkerArray, '/fire_marker', 10)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            qos_profile=qos
        )

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile=qos
        )

        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        )

        self.create_timer(0.1, self.mission_loop)

        self.get_logger().info("FireDroneX node started")
        
    def detection_callback(self, msg):
        
        if self.state not in ['SEARCH', 'CIRCLE']:
            return
        
        for detection in msg.detections:
            label = detection.results[0].hypothesis.class_id
            score = float(detection.results[0].hypothesis.score)
            
            if label not in ['fire', 'person'] or score < 0.7:
                continue
            
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y

            x_min = self.cx - self.ROI_WIDTH / 2
            x_max = self.cx + self.ROI_WIDTH / 2
            y_min = self.cy - self.ROI_HEIGHT / 2
            y_max = self.cy + self.ROI_HEIGHT / 2

            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                self.get_logger().info(f"Detection at ({cx:.1f},{cy:.1f}) is outside central ROI.")
                continue

            self.get_logger().info(f"Detected fire at ({cx}, {cy}) with confidence {score:.2f}")            
            pos = self.pixel_to_ground(cx, cy)
            
            if pos is None:
                continue
            
            gx, gy, _ = pos
            
            dx = gx - self.odom_x
            dy = gy - self.odom_y
            target_yaw = math.atan2(dy, dx)
            self.target_face_yaw = target_yaw
            
            if label == 'person':
                for fx, fy, _ in self.detected_fires + self.visited_fires:
                    distance = math.sqrt((gx - fx) ** 2 + (gy - fy) ** 2)
                    if distance < 3.0:
                        self.get_logger().info(f"Person detected at ({gx:.1f}, {gy:.1f}) too close to fire at ({fx:.1f}, {fy:.1f}).")
                        break
            
            target_x, target_y, target_z = gx, gy, -7.0
            
            for fx, fy, fz in self.visited_fires:
                if (abs(target_x - fx) < 3.0 and
                    abs(target_y - fy) < 3.0 ):
                    self.get_logger().info(f" Fire at ({target_x:.1f}, {target_y:.1f}) ignored: too close to visited fire at ({fx:.1f}, {fy:.1f})")
                    return

            fire_pos = self.cluster_fire_positions(target_x, target_y, target_z)
            if fire_pos in self.visited_fires:
                self.get_logger().info(f"Ignoring fire target {fire_pos} (already visited).")
                continue
            
            already_detected = any(
                math.sqrt((fire_pos[0] - fx) ** 2 + (fire_pos[1] - fy) ** 2) < 1.0
                for fx, fy, _ in self.detected_fires
            )
            
            if not already_detected:
                self.detected_fires.append(fire_pos)
            
            self.fire_target = fire_pos
            previous_state = self.state
            self.state = 'STOP_AND_RECALCULATE'
            self.recalculate_start_time = self.get_clock().now().nanoseconds / 1e9
            self.fire_projection_buffer = []
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
            self.integral_x = 0.0
            self.integral_y = 0.0
            self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
            self.initial_search = False
        
            self.hover_start_time = None
            self.get_logger().info(f"New location for fire target: {fire_pos} - Moving Towards it")
    
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
    
    def pixel_to_ground(self, cx, cy):
        x_cam = (cx - self.cx) / self.fx
        y_cam = (cy - self.cy) / self.fy
        direction = np.array([x_cam, y_cam, 1.0])
        direction = direction / np.linalg.norm(direction)
        
        R_cam_body = R.from_euler(
            'xyz', [0.0, self.camera_pitch, 0.0]).as_matrix()
        R_body_world = R.from_quat([self.qx, self.qy, self.qz, self.qw]).as_matrix()
        dir_world = R_body_world @ R_cam_body @ direction
        dir_z = dir_world[2]
        if dir_z >= -1e-6:
            return None
        
        # Calculate the intersection with the ground plane
        ground_z = 0.4
        s =  (ground_z- self.odom_z) / dir_z
        ground_xy = np.array([self.odom_x, self.odom_y]) + s * dir_world[:2]
        
        if self.state == 'STOP_AND_RECALCULATE':
            self.fire_projection_buffer.append((ground_xy[0], ground_xy[1]))
            if len(self.fire_projection_buffer) > self.fire_projection_buffer_size:
                self.fire_projection_buffer.pop(0)
            
            if len(self.fire_projection_buffer) >= self.fire_projection_buffer_size:
                avg_x = sum(p[0] for p in self.fire_projection_buffer) / len(self.fire_projection_buffer)
                avg_y = sum(p[1] for p in self.fire_projection_buffer) / len(self.fire_projection_buffer)
                return (float(avg_x), float(avg_y), ground_z)
            else:
                return None        
        else:
            return (float(ground_xy[0]), float(ground_xy[1]), ground_z)
        
    def mission_loop(self):        
        self.publish_offboard_control()
        
        if self.control_mode is None:
            self.get_logger().warn("Control mode not set")
            return
        
        if not self.control_mode.flag_control_offboard_enabled:
            self.get_logger().warn("Offboard control not enabled. Waiting for offboard mode.")
            return
        
        if self.fire_target is not None:
            if self.fire_target in self.visited_fires:
                self.get_logger().info(f"Fire target {self.fire_target} already visited. Searching for new fire.")
                self.fire_target = None
                return
        
        if self.fire_target:
            target_x, target_y, target_z = self.fire_target
            distance = math.sqrt((target_x - self.odom_x) ** 2 + (target_y - self.odom_y) ** 2)
            
            if self.state == 'STOP_AND_RECALCULATE':
                
                self.publish_setpoint(self.odom_x, self.odom_y, -7.0, yaw=self.target_face_yaw)

                # Recalculate for 1.5 seconds
                time_elapsed = self.get_clock().now().nanoseconds / 1e9 - self.recalculate_start_time
                if time_elapsed < 10.0:
                    if not hasattr(self, 'last_log_time'):
                        self.last_log_time = 0.0

                    now = self.get_clock().now().nanoseconds / 1e9
                    if now - self.last_log_time > 0.5:
                        self.get_logger().info("Holding to face fire and recalculate projection...")
                        self.last_log_time = now
                    return

                if len(self.fire_projection_buffer) > 0:
                    avg_x = sum(p[0] for p in self.fire_projection_buffer) / len(self.fire_projection_buffer)
                    avg_y = sum(p[1] for p in self.fire_projection_buffer) / len(self.fire_projection_buffer)
                    self.recalculated_projection = (avg_x, avg_y, -7.0)
                    self.fire_target = self.recalculated_projection
                    self.get_logger().info(f"Accurate projection complete: New fire target at {self.fire_target}")
                else:
                    self.get_logger().warn("No projection collected while stopped. Resuming search.")
                
                self.state = 'HOLD'
                self.approach_start_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.state == 'HOLD':
                hold_duration = self.get_clock().now().nanoseconds / 1e9 - self.approach_start_time
                if hold_duration < 2.0:
                    self.publish_setpoint(self.odom_x, self.odom_y, -7.0)  # Hold
                    return
                else:
                    self.get_logger().info("Hold complete. Starting slow approach to fire.")
                    self.state = 'APPROACH'

            if self.state == 'APPROACH' and self.fire_target:
                target_x, target_y, target_z = self.fire_target
                dx = target_x - self.odom_x
                dy = target_y - self.odom_y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                
                # PID control
                error_x = dx
                error_y = dy

                self.integral_x += error_x * self.dt
                self.integral_y += error_y * self.dt

                derivative_x = (error_x - self.prev_error_x) / max(self.dt, 1e-6)
                derivative_y = (error_y - self.prev_error_y) / max(self.dt, 1e-6)

                vx = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
                vy = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y

                # Save error for next iteration
                self.prev_error_x = error_x
                self.prev_error_y = error_y

                # Clamp velocity
                max_vel = 0.3
                vx = max(min(vx, max_vel), -max_vel)
                vy = max(min(vy, max_vel), -max_vel)

                if distance > 0.1:
                    step_size = 0.4
                    target_step_x = self.odom_x + step_size * (dx) / distance
                    target_step_y = self.odom_y + step_size * (dy) / distance
                    yaw = math.atan2(dy, dx)
                    self.publish_setpoint(target_step_x, target_step_y, target_z, yaw=yaw)                    
                    return
                else:
                    self.state = 'CIRCLE'
                    fire_id = self.fire_id
                    self.fire_id_map[self.fire_target] = fire_id
                    self.get_logger().info("Reached fire target. Starting circling.")
                    self.visited_fires.append(self.fire_target)
                    self.publish_vizualization()
                    self.get_logger().info(f"[VISITED] Fire ID {fire_id} at {self.fire_target}")
                    self.fire_id += 1
                    self.fire_target = None
                    self.circling = True
                    self.circle_center = (target_x, target_y)
                    self.circle_radius = 1.0
                    self.circle_theta = 0.0
                    return
        
        if self.circling and self.circle_center:
            center_x, center_y = self.circle_center
            
            arc_step = self.max_circle_velocity * self.dt / self.circle_radius
            self.circle_theta += arc_step
            
            target_x = center_x + self.circle_radius * math.cos(self.circle_theta)
            target_y = center_y + self.circle_radius * math.sin(self.circle_theta)
            target_z = -7.0
            
            dx = target_x - self.odom_x
            dy = target_y - self.odom_y
            
            error_x = dx
            error_y = dy
            
            self.integral_x += error_x * self.dt
            self.integral_y += error_y * self.dt
            
            derivative_x = (error_x - self.prev_error_x) / max(self.dt, 1e-6)
            derivative_y = (error_y - self.prev_error_y) / max(self.dt, 1e-6)
            
            vx = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
            vy = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y
            
            self.prev_error_x = error_x
            self.prev_error_y = error_y
            
            max_vel = 0.3
            vx = max(min(vx, max_vel), -max_vel)
            vy = max(min(vy, max_vel), -max_vel)
            
            if self.circle_theta > 2 * math.pi:
                self.circle_theta = 0.0
                self.circle_radius += 0.5
                
            if self.circle_radius > 6.0:
                self.state = 'SEARCH'
                self.circling = False
                self.initial_search = True
                self.circle_center = None
                self.circle_radius = 2.0
                self.circle_theta = 0.0
                self.prev_error_x = 0.0
                self.prev_error_y = 0.0
                self.integral_x = 0.0
                self.integral_y = 0.0
                self.search_center = (self.odom_x, self.odom_y)

                self.get_logger().info("Resuming SEARCH after full circle around fire.")
                return

            yaw = math.atan2(target_y - center_y, target_x - center_x)
            self.publish_setpoint(target_x, target_y, target_z, yaw = yaw)
            return
        
        if not self.fire_target and not self.circling:

            if self.search_center is None:
                self.search_center = (self.odom_x, self.odom_y)
                self.get_logger().info("Starting initial search for fire...")
            
            center_x, center_y = self.search_center
            
            arc_step = self.max_circle_velocity * self.dt / self.circle_radius
            self.circle_theta += arc_step
            
            target_x = center_x + self.circle_radius * math.cos(self.circle_theta)
            target_y = center_y + self.circle_radius * math.sin(self.circle_theta)
            target_z = -7.0

            if self.initial_search and self.last_theta > self.circle_theta:
                    self.circle_radius += 0.5
                    self.get_logger().info(f"Expanding search radius to {self.circle_radius:.1f} m")
        
            self.last_theta = self.circle_theta
            if self.circle_theta > 2 * math.pi:
                self.circle_theta = 0.0
            
            yaw = math.atan2(target_y - center_y, target_x - center_x)
            self.publish_setpoint(target_x, target_y, target_z, yaw = yaw)
            
            
    def publish_velocity_setpoint(self, vx, vy, vz):
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        setpoint.velocity = [float(vx), float(vy), float(vz)]
        setpoint.yaw = float(self.current_yaw) 
        self.trajectory_pub.publish(setpoint)
        
    def publish_vizualization(self):
        drone_marker = Marker()
        drone_marker.header.frame_id = "map"
        drone_marker.header.stamp = self.get_clock().now().to_msg()
        drone_marker.ns = "drone"
        drone_marker.id = 0
        drone_marker.type = Marker.SPHERE
        drone_marker.action = Marker.ADD
        drone_marker.pose.position.x = float(self.qx)
        drone_marker.pose.position.y = float(self.qy)
        drone_marker.pose.position.z = float(self.qz)
        drone_marker.pose.orientation.w = float(self.qw)
        drone_marker.scale.x = 0.4
        drone_marker.scale.y = 0.4
        drone_marker.scale.z = 0.2
        drone_marker.color.r = 0.0
        drone_marker.color.g = 1.0
        drone_marker.color.b = 1.0
        self.drone_marker_pub.publish(drone_marker)
        
        fire_array = MarkerArray()
        for i, (x, y, z) in enumerate(self.visited_fires):
            fire_marker = Marker()
            fire_marker.header.frame_id = "map"
            fire_marker.header.stamp = self.get_clock().now().to_msg()
            fire_marker.ns = "fire"
            fire_marker.id = i
            fire_marker.type = Marker.SPHERE
            fire_marker.action = Marker.ADD
            fire_marker.pose.position.x = float(x)
            fire_marker.pose.position.y = float(y)
            fire_marker.pose.position.z = 0.0
            fire_marker.pose.orientation.w = 1.0
            fire_marker.scale.x = 0.5
            fire_marker.scale.y = 0.5
            fire_marker.scale.z = 1.0
            fire_marker.color.r = 1.0
            fire_marker.color.g = 0.2
            fire_marker.color.b = 0.0
            fire_marker.color.a = 1.0
            fire_array.markers.append(fire_marker)
            
        self.fire_marker_pub.publish(fire_array)
    
    def odom_callback(self, msg):
        self.odom_x = msg.position[0]
        self.odom_y = msg.position[1]
        self.odom_z = msg.position[2]
        
        self.qx = msg.q[0]
        self.qy = msg.q[1]
        self.qz = msg.q[2]
        self.qw = msg.q[3]
        
        q = msg.q
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        self.current_yaw = r.as_euler('zyx')[0]
    
    def publish_setpoint(self, x, y, z, yaw = 0.0):
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        setpoint.position = [float(x), float(y), float(z)]
        setpoint.yaw = float(yaw)         
        self.trajectory_pub.publish(setpoint)

    def control_mode_callback(self, msg):
        self.control_mode = msg

    def publish_offboard_control(self):
        offboard_mode = OffboardControlMode()
        offboard_mode.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_mode.position = True
        offboard_mode.velocity = False
        offboard_mode.acceleration = False
        offboard_mode.attitude = False
        offboard_mode.body_rate = False
        self.offboard_pub.publish(offboard_mode)
    
def main(args=None):
    rclpy.init(args=args)
    node = SafeFireDroneX()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()