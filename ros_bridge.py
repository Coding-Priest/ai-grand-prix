import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import zmq
import cv2
import numpy as np
import struct


class ZMQtoROSBridge(Node):
    def __init__(self):
        super().__init__("zmq_ros_bridge")

        # 1. Setup ROS Publisher
        self.publisher_ = self.create_publisher(Image, "/cam0/image_raw", 10)
        self.bridge = CvBridge()

        # 2. Setup ZMQ Subscriber
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://127.0.0.1:5555")
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Listen to everything

        # 3. Check for new frames at 100Hz (non-blocking)
        self.timer = self.create_timer(0.01, self.receive_frame)
        self.get_logger().info("ROS Bridge listening on port 5555...")

    def receive_frame(self):

        try:

            parts = self.zmq_socket.recv_multipart(flags=zmq.NOBLOCK)
            time_bytes, image_bytes = parts[0], parts[1]

            # 1. Decode Timestamp
            sim_time = struct.unpack("d", time_bytes)[0]

            # Decode the bytes back into an OpenCV image
            np_arr = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert to ROS message and publish
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
            msg.header.frame_id = "camera_optical_frame"  # Crucial for SLAM transforms

            self.publisher_.publish(msg)

        except zmq.Again:
            # No new frame arrived yet, just pass
            pass


def main(args=None):

    rclpy.init(args=args)
    node = ZMQtoROSBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
