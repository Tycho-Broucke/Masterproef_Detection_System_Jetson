import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from masterproef_interfaces.srv import GetStitchedImage
import cv2


class ImageProviderNode(Node):
    def __init__(self):
        super().__init__('image_provider')

        # Bridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Open the video stream from one camera
        self.cap = cv2.VideoCapture("/dev/video0")

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera at /dev/video0")

        # Create service for providing camera images
        self.srv = self.create_service(
    		GetStitchedImage,
    		'get_stitched_image',  # same as YOLO node expects
    		self.handle_image_request
	)

        self.get_logger().info("Image provider service ready at 'get_single_image'")

    def handle_image_request(self, request, response):
        # Capture a single frame from the camera
        ret, frame = self.cap.read()

        if not ret:
            response.success = False
            response.message = "Camera read failed"
            return response

        # Optional: resize for consistency (same as in your stitcher code)
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert OpenCV image to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding="bgr8")

        # Fill the response
        response.image = img_msg
        response.success = True
        response.message = "Image capture succeeded"

        return response

    def destroy_node(self):
        # Release camera resources when node is destroyed
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageProviderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

