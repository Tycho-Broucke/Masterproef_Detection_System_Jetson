import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from masterproef_interfaces.srv import GetStitchedImage
import cv2
import os
import pickle


class CalibratedImageProviderNode(Node):
    def __init__(self):
        super().__init__('calibrated_image_provider')

        # Bridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Open the video stream from one camera
        self.cap = cv2.VideoCapture("/dev/video0")
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera at /dev/video0")

        # Path to calibration data (workspace-level data/output folder)
        self.calibration_file = "/home/tabloo/jetson_ws/data/output/calibration_data.pkl"

        if not os.path.exists(self.calibration_file):
            self.get_logger().error(f"Calibration file not found: {self.calibration_file}")
            self.mtx = None
            self.dist = None
            self.mapx = None
            self.mapy = None
            self.roi = None
        else:
            with open(self.calibration_file, "rb") as f:
                calibration_data = pickle.load(f)
            self.mtx = calibration_data["camera_matrix"]
            self.dist = calibration_data["dist_coeff"]

            # Get frame size from the camera
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Use cropped FOV (alpha=0) for undistortion
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (width, height), 0, (width, height)
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, newcameramtx, (width, height), cv2.CV_32FC1
            )
            self.roi = roi  # (x, y, w, h)

            self.get_logger().info(f"Camera calibration loaded successfully from {self.calibration_file}")

        # Create service for providing camera images
        self.srv = self.create_service(
            GetStitchedImage,
            "get_stitched_image",  # must match YOLO node expectation
            self.handle_image_request,
        )

        self.get_logger().info(
            "Calibrated image provider service ready at 'get_stitched_image'"
        )

    def handle_image_request(self, request, response):
        # Capture a single frame
        ret, frame = self.cap.read()
        if not ret:
            response.success = False
            response.message = "Camera read failed"
            return response

        # Apply undistortion if calibration data is available
        if self.mapx is not None and self.mapy is not None:
            undistorted = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

            # Crop to valid ROI (remove black borders)
            if self.roi is not None:
                x, y, w, h = self.roi
                undistorted = undistorted[y:y + h, x:x + w]

        else:
            undistorted = frame

        # Resize for consistency
        frame_resized = cv2.resize(undistorted, (640, 480))

        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding="bgr8")

        # Fill response
        response.image = img_msg
        response.success = True
        response.message = "Image capture succeeded"
        return response

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CalibratedImageProviderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

