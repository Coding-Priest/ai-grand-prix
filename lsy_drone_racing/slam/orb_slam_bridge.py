import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv


class OrbSlamBridge(Node):

    def __init__(self):
        super().__init__("drone_sim_bridge")

        self.publisher_ = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

    def publish_frame(self, frame):

        if frame is None:
            self.get_logger().warn("Blank Frame Passed")
            return

        msg = self.bridge.cv_to_imgmsg(cv_image, encoding="bgr8")

        self.publisher_.publish(msg)
