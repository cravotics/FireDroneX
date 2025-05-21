#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import time 
import os
import torch 

class YoloDetectionNodeReal(Node):
    def __init__(self):
        super().__init__('yolo_detection_node_real')

        # Logging PyTorch and CUDA status
        self.get_logger().info(f"PyTorch version: {torch.__version__}")
        self.get_logger().info(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.get_logger().info(f"Torch CUDA version: {torch.version.cuda}")
            self.get_logger().info(f"Number of CUDA devices: {torch.cuda.device_count()}")
            current_device_id = torch.cuda.current_device()
            self.get_logger().info(f"Current PyTorch CUDA device ID: {current_device_id}")
            self.get_logger().info(f"Device name: {torch.cuda.get_device_name(current_device_id)}")
        else:
            self.get_logger().warn("CUDA is NOT available to PyTorch. YOLO will run on CPU if 'cpu' is selected.")

        self.declare_parameter('model_path', '/home/varun/ws_offboard_control/src/px4_ros_com/src/examples/yolo_models/best.pt')
        self.declare_parameter('rtsp_uri', 'rtsp://192.168.8.1:8900/live?intra=1')
        self.declare_parameter('yolo_device', 'cpu')
        self.declare_parameter('inference_width', 640)
        self.declare_parameter('inference_height', 480)
        self.declare_parameter('loop_rate_hz', 20.0)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.rtsp_uri = self.get_parameter('rtsp_uri').get_parameter_value().string_value
        self.yolo_device = self.get_parameter('yolo_device').get_parameter_value().string_value
        self.inference_width = self.get_parameter('inference_width').get_parameter_value().integer_value
        self.inference_height = self.get_parameter('inference_height').get_parameter_value().integer_value
        self.loop_rate = self.get_parameter('loop_rate_hz').get_parameter_value().double_value

        if "cuda" in self.yolo_device and not torch.cuda.is_available():
            self.get_logger().warn(f"'{self.yolo_device}' was requested for YOLO, but CUDA is not available. Falling back to 'cpu'.")
            self.yolo_device = 'cpu'

        self.get_logger().info(f"Attempting to load YOLO model from: {self.model_path}")
        self.get_logger().info(f"YOLO model will attempt to use device: '{self.yolo_device}' for inference.")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        self.cap = None
        self.attempt_rtsp_connect()
        
        # QoS settings for ROS 2
        self.custom_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher for the detection topic
        self.detections_pub = self.create_publisher(
            Detection2DArray, 
            '/detections', 
            qos_profile=self.custom_qos_profile
        )
        # Publishing debug image as uncompressed image stream
        self.debug_image_pub = self.create_publisher(
            Image,  
            '/yolo/debug_image',
            qos_profile=self.custom_qos_profile 
        )
        # Publishing the raw image used for inference
        self.raw_image_pub = self.create_publisher(
            Image,
            '/yolo/image_for_depth', 
            qos_profile=self.custom_qos_profile
        )

        self.class_names = self.model.names
        self.get_logger().info(f"Model classes: {self.class_names}")
        
        # Setting Confidence thresholds
        self.publish_confidence_threshold = 0.4  # General threshold for fire
        self.person_detection_confidence_threshold = 0.7 # For high-confidence person logging

        self.timer = self.create_timer(1.0 / self.loop_rate, self.loop)
        self.get_logger().info(f"YOLO Detection Node (Real Drone) started. Attempting to stream from {self.rtsp_uri}")
    
    # Attempt to connect to the RTSP stream using OpenCV
    def attempt_rtsp_connect(self):
        self.get_logger().info(f"Attempting to open RTSP stream: {self.rtsp_uri} using FFmpeg backend")
        if self.cap is not None:
            self.cap.release()
        
        original_ffmpeg_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        self.cap = cv2.VideoCapture(self.rtsp_uri, cv2.CAP_FFMPEG)
        
        if original_ffmpeg_options is not None:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = original_ffmpeg_options
        else:
            del os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open RTSP stream: {self.rtsp_uri}. Will retry.")
            return False
        self.get_logger().info("Successfully opened RTSP stream.")
        return True

    def loop(self):
        current_timestamp = self.get_clock().now().to_msg()
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn('RTSP stream is not open. Attempting to reconnect...')
            if not self.attempt_rtsp_connect():
                time.sleep(1.0) 
                return
            
        ok, frame_full = self.cap.read()
        if not ok or frame_full is None:
            self.get_logger().warn('RTSP dropout - frame not ok or None from cap.read(). Attempting to reconnect.')
            self.attempt_rtsp_connect() 
            return

        try:
            frame_resized = cv2.resize(frame_full, (self.inference_width, self.inference_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.get_logger().error(f"Failed to resize frame: {e}")
            return

        # Publishing the resized frame that will be used for YOLO and depth
        try:
            raw_image_msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding="bgr8")
            raw_image_msg.header.stamp = current_timestamp
            raw_image_msg.header.frame_id = "camera_link_resized" 
            self.raw_image_pub.publish(raw_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw image: {e}")


        # Performimg inference
        results = self.model(frame_resized, verbose=False, device=self.yolo_device)[0] 

        det_array = Detection2DArray()
        det_array.header.stamp = current_timestamp 
        det_array.header.frame_id = "camera_link_resized" 

        processed_detections_for_publishing = []
        detections_for_drawing = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names.get(class_id, str(class_id))

                passes_threshold = False
                if class_name.lower() == 'fire' and confidence >= self.publish_confidence_threshold:
                    passes_threshold = True
                elif class_name.lower() == 'person' and confidence >= self.person_detection_confidence_threshold:
                     passes_threshold = True 
                elif class_name.lower() == 'person' and confidence >= self.publish_confidence_threshold: 
                    passes_threshold = True
                
                if not passes_threshold:
                    continue
                
                # Creating and drawing bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                w = x2 - x1
                h = y2 - y1

                bbox_msg = BoundingBox2D()
                bbox_msg.center.position.x = float(x1 + w / 2)
                bbox_msg.center.position.y = float(y1 + h / 2)
                bbox_msg.center.theta = 0.0 
                bbox_msg.size_x = float(w)
                bbox_msg.size_y = float(h)

                det_msg = Detection2D()
                det_msg.header = det_array.header 
                det_msg.bbox = bbox_msg

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = class_name 
                hyp.hypothesis.score = confidence
                det_msg.results.append(hyp)
                
                processed_detections_for_publishing.append(det_msg)
                detections_for_drawing.append({'box': (x1, y1, x2, y2), 'label': class_name, 'conf': confidence, 'id': class_id})
        
        if processed_detections_for_publishing:
            det_array.detections = processed_detections_for_publishing
            self.detections_pub.publish(det_array)

        # Creating and publishing debug image
        if self.debug_image_pub.get_subscription_count() > 0:
            vis_frame = frame_resized.copy() # Drawing on a new copy
            for det_info in detections_for_drawing:
                x1, y1, x2, y2 = det_info['box']
                label = det_info['label']
                conf = det_info['conf']

                color = (0, 255, 0) 
                if label.lower() == 'fire': 
                    color = (0, 0, 255) 
                elif label.lower() == 'person':
                    color = (255, 0, 0) 
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
                
                text = f"{label} {conf:.2f}"
                cv2.putText(vis_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8") 
                debug_msg.header.stamp = current_timestamp 
                debug_msg.header.frame_id = "camera_link_debug" 
                self.debug_image_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNodeReal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down YOLO node...")
    except RuntimeError as e:
        if "RTSP" in str(e) or "VideoCapture" in str(e): 
             node.get_logger().fatal(f"Critical media I/O error: {e}")
        else:
             node.get_logger().fatal(f"Unhandled runtime error: {e}")
    finally:
        if hasattr(node, 'cap') and node.cap is not None:
            node.cap.release()
        if rclpy.ok() and hasattr(node, 'destroy_node') and callable(node.destroy_node): 
            node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__':
    main() 
