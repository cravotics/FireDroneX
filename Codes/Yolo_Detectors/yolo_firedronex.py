#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import torch 

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # Logging PyTorch and CUDA information
        self.get_logger().info(f"PyTorch version: {torch.__version__}")
        self.get_logger().info(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.get_logger().info(f"Torch CUDA version: {torch.version.cuda}")
            self.get_logger().info(f"Current PyTorch CUDA device: {torch.cuda.current_device()}")
            self.get_logger().info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            self.get_logger().warn("CUDA is NOT available to PyTorch. YOLO will run on CPU.")

        self.declare_parameter('model_path', '/home/varun/ws_offboard_control/src/px4_ros_com/src/examples/yolo_models/best.pt')
        self.declare_parameter('camera_topic', '/camera')

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # Defining the desired QoS profile for the image publisher
        self.image_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1 
        )

        # Define a specific QoS profile for the camera subscription
        self.camera_subscription_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.VOLATILE,    
            history=HistoryPolicy.KEEP_LAST,
            depth=1 
        )
        
        # Creating a new publisher for the detections
        self.detections_pub = self.create_publisher(
            Detection2DArray, 
            '/detections', 
            qos_profile=self.image_publisher_qos 
        )

        # Creating a new publisher for the debug image
        self.debug_image_pub = self.create_publisher(
            Image, 
            '/yolo/debug_image', 
            qos_profile=self.image_publisher_qos 
        )
        
        # Creating a new publisher for the raw image 
        self.image_for_depth_pub = self.create_publisher(
            Image,
            '/yolo/image_for_depth',
            qos_profile=self.image_publisher_qos
        )
        
        # Subscribing to the camera topic with the defined QoS profile
        self.create_subscription(
            Image, 
            self.camera_topic, 
            self.image_callback, 
            qos_profile=self.camera_subscription_qos_profile
        )

        self.class_names = self.model.names
        self.get_logger().info(f"Model classes: {self.class_names}")
        self.get_logger().info(f"YOLO Detection Node started. Subscribed to {self.camera_topic}")

        # Confidence thresholds
        self.publish_confidence_threshold = 0.4  # General threshold 
        self.person_detection_confidence_threshold = 0.7 # Specific threshold for person detection
    

    def image_callback(self, msg: Image):
        self.get_logger().info("image_callback TRIGGERED!")
        self.get_logger().info(f"Incoming image encoding: {msg.encoding}")

        # Publishing the raw image for depth node and GUI person focus
        try:
            self.image_for_depth_pub.publish(msg)
            self.get_logger().info(f"Published raw image to /yolo/image_for_depth (encoding: {msg.encoding})")
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw image to /yolo/image_for_depth: {e}")

        # Converting image with proper encoding handling
        try:
            if msg.encoding == 'mono8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        results = self.model(cv_image, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            self.get_logger().info("No raw detections from YOLO model.")
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.get_logger().info(f"No dets: Publishing /yolo/debug_image (shape: {cv_image.shape}, encoding: bgr8)")
            self.debug_image_pub.publish(debug_msg)
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        # Processing detections
        processed_detections_for_publishing = []
        detections_for_drawing = [] # Storing all detections that pass the publish_confidence_threshold for drawing

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names.get(class_id, str(class_id))

            # Logging every raw detection from YOLO
            self.get_logger().info(f"RAW YOLO Detection: Class: '{class_name}' (ID: {class_id}), Confidence: {confidence:.4f}")

            # Checking against general publishing threshold
            if confidence < self.publish_confidence_threshold:
                continue

            # Person detection with specific logging
            if class_name.lower() == 'person' and confidence >= self.person_detection_confidence_threshold:
                self.get_logger().info(f"Detected PERSON with high confidence: {confidence:.2f} at bbox=({box.xyxy[0][0]},{box.xyxy[0][1]},{box.xyxy[0][2]},{box.xyxy[0][3]})")
            elif class_name.lower() == 'person':
                self.get_logger().info(f"Detected PERSON with moderate confidence: {confidence:.2f} at bbox=({box.xyxy[0][0]},{box.xyxy[0][1]},{box.xyxy[0][2]},{box.xyxy[0][3]})")
            elif class_name.lower() == 'fire':
                 self.get_logger().info(f"Detected FIRE with confidence: {confidence:.2f}")

            # Preparing bounding boxes for detection
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
            det_msg.header = msg.header
            det_msg.bbox = bbox_msg

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = confidence
            det_msg.results.append(hyp)

            processed_detections_for_publishing.append(det_msg)
            detections_for_drawing.append({'box': (x1, y1, x2, y2), 'label': class_name, 'conf': confidence})


        if not processed_detections_for_publishing:
            self.get_logger().info("No detections passed the confidence threshold for publishing.")
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.get_logger().info(f"No dets: Publishing /yolo/debug_image (shape: {cv_image.shape}, encoding: bgr8)")
            self.debug_image_pub.publish(debug_msg)
            return

        det_array.detections = processed_detections_for_publishing
        self.detections_pub.publish(det_array)
        self.get_logger().info(f"Published {len(det_array.detections)} detections passing threshold {self.publish_confidence_threshold}.")

        # Drawing bounding boxes for all published detections
        for det_info in detections_for_drawing:
            x1, y1, x2, y2 = det_info['box']
            label = det_info['label']
            conf = det_info['conf']

            # Assigning color based on class for better visualization
            color = (0, 255, 0) # Default Green
            if label.lower() == 'fire':
                color = (0, 0, 255) # Red for fire
            elif label.lower() == 'person':
                color = (255, 0, 0) # Blue for person

            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            
            # Calculating and drawing centroid
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(cv_image, (center_x, center_y), 5, color, -1) 
            
            # Adding label and confidence text
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Ensuring the rectangle for text is within image bounds, especially at the top
            text_bg_y1 = y1 - th - 4
            text_bg_y2 = y1
            if text_bg_y1 < 0:
                text_bg_y1 = y1 + 2 
                text_bg_y2 = y1 + th + 4
            
            cv2.rectangle(cv_image, (x1, text_bg_y1), (x1 + tw, text_bg_y2), color, -1)
            # Adjusting text position accordingly
            text_y = y1 - 2 if text_bg_y2 == y1 else y1 + th 
            cv2.putText(cv_image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


        # Converting and publishing debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.get_logger().info(f"Publishing /yolo/debug_image with overlays (shape: {cv_image.shape}, encoding: bgr8)")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Error converting or publishing debug image: {e}")
        
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
