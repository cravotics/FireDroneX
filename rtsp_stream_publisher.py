#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import os
import subprocess

RTSP_URL_DEFAULT = 'rtsp://192.168.8.1:8900/live'
CAMERA_TOPIC_DEFAULT = '/camera_feed'
TIMER_RATE_DEFAULT = 30  # Hz
RECONNECTION_DELAY = 5  # Seconds
MAX_RECONNECT_ATTEMPTS = 3
CONSECUTIVE_FAILURES_THRESHOLD = 5
STREAM_TIMEOUT = 10  # Seconds before considering stream dead
KEEPALIVE_INTERVAL = 5  # Seconds between keepalive packets

class RTSPStreamPublisher(Node):
    def __init__(self):
        super().__init__('rtsp_stream_publisher')
        
        self.declare_parameter('rtsp_url', RTSP_URL_DEFAULT)
        self.declare_parameter('camera_topic', CAMERA_TOPIC_DEFAULT)
        self.declare_parameter('timer_rate', TIMER_RATE_DEFAULT)
        
        self.rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.timer_rate_val = self.get_parameter('timer_rate').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.cap = None
        self.consecutive_failures = 0
        self.reconnect_attempts = 0
        self.last_frame_time = 0
        self.last_keepalive_time = 0
        self.ffmpeg_process = None
        
        # Configure OpenCV RTSP settings with more robust options
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'  # Use TCP for more reliable connection
            'buffer_size;102400|'  # Larger buffer for network jitter
            'stimeout;10000000|'   # Socket timeout in microseconds (10 seconds)
            'rtsp_flags;prefer_tcp|'  # Prefer TCP over UDP
            'rtsp_transport;tcp|'  # Force TCP transport
            'max_delay;500000|'     # Maximum demux-decode delay in microseconds
            'flags;low_delay|'      # Low latency mode
            'fflags;nobuffer|'      # Disable buffering
            'flags2;fast'           # Fast decoding
        )
        
        self.image_publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1 
        )
        
        self.image_pub = self.create_publisher(
            Image, 
            self.camera_topic, 
            self.image_publisher_qos
        )
        
        self._connect_to_stream()
        self.timer = self.create_timer(1.0 / max(1, self.timer_rate_val), self.publish_frame)
        self.get_logger().info(f"RTSP Publisher started. Attempting to stream from {self.rtsp_url} to {self.camera_topic}")

    def _check_rtsp_server(self):
        """Check if RTSP server is responding using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-rtsp_transport', 'tcp',
                '-show_entries', 'stream=codec_type,codec_name',
                '-of', 'json',
                self.rtsp_url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.get_logger().info(f"RTSP server info: {result.stdout}")
                return True
            else:
                self.get_logger().error(f"RTSP server check failed: {result.stderr}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error checking RTSP server: {e}")
            return False

    def _connect_to_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # First check if RTSP server is responding
        if not self._check_rtsp_server():
            self.get_logger().error("RTSP server not responding. Will retry.")
            return False

        self.get_logger().info(f"Attempting to open RTSP stream: {self.rtsp_url} using FFmpeg backend")
        
        # Try different codec configurations
        codec_configs = [
            (cv2.CAP_FFMPEG, None),  # Default
            (cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'H264')),  # Force H264
            (cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'HEVC'))   # Try HEVC
        ]
        
        for backend, codec in codec_configs:
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url, backend)
                if not self.cap.isOpened():
                    continue
                
                if codec:
                    self.cap.set(cv2.CAP_PROP_FOURCC, codec)
                
                # Configure OpenCV capture properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                
                # Set additional properties for better stream handling
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set expected FPS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  # Set expected width
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)  # Set expected height
                
                # Verify we can actually read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.get_logger().info(f"Successfully opened RTSP stream with codec: {codec if codec else 'default'}")
                    self.consecutive_failures = 0
                    self.reconnect_attempts = 0
                    self.last_frame_time = time.time()
                    self.last_keepalive_time = time.time()
                    return True
                
                self.cap.release()
                self.cap = None
                
            except Exception as e:
                self.get_logger().error(f"Error trying codec configuration: {e}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        
        self.get_logger().error("Failed to open RTSP stream with any codec configuration")
        return False

    def _send_keepalive(self):
        """Send a keepalive packet to maintain the RTSP connection"""
        if self.cap is not None and self.cap.isOpened():
            try:
                # Read a frame but don't process it
                self.cap.grab()
                self.last_keepalive_time = time.time()
            except Exception as e:
                self.get_logger().warn(f"Keepalive failed: {e}")

    def publish_frame(self):
        current_time = time.time()
        
        # Check if we need to send a keepalive
        if current_time - self.last_keepalive_time >= KEEPALIVE_INTERVAL:
            self._send_keepalive()
        
        # Check for stream timeout
        if current_time - self.last_frame_time >= STREAM_TIMEOUT:
            self.get_logger().warn(f"Stream timeout detected. Last frame received {current_time - self.last_frame_time:.1f} seconds ago.")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            if not self._connect_to_stream():
                return
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn("Stream not connected. Attempting to reconnect...", 
                                 throttle_duration_sec=RECONNECTION_DELAY)
            if not self._connect_to_stream():
                return

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.consecutive_failures += 1
                self.get_logger().warn(f"RTSP dropout - frame not ok or None from cap.read(). Attempting to reconnect.")
                
                if self.consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
                    self.get_logger().error("Too many consecutive failures. Releasing capture and reconnecting.")
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = None
                    self.consecutive_failures = 0
                    self.reconnect_attempts += 1
                    
                    if self.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        self.get_logger().error("Max reconnection attempts reached. Waiting before next attempt.")
                        time.sleep(RECONNECTION_DELAY)
                        self.reconnect_attempts = 0
                return

            # Reset failure counters on successful frame
            self.consecutive_failures = 0
            self.reconnect_attempts = 0
            self.last_frame_time = current_time

            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)
            
            # Log frame rate periodically
            if current_time - self.last_frame_time >= 5.0:  # Log every 5 seconds
                self.get_logger().info(f"Publishing frame: {frame.shape}")
                
        except Exception as e:
            self.get_logger().error(f"Error converting or publishing frame: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down RTSP publisher.")
        if self.cap is not None:
            self.cap.release()
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RTSPStreamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("RTSP publisher interrupted.")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in spin: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 