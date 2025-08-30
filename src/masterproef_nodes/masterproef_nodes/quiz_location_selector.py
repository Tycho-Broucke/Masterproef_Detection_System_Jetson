# JETSON NODE
#
# this code runs a yolo node that requests a stitched image from the image server with the custom GetStitchedImage service
# once the requested image is received, the user can click a location on the screen to point out the quiz location for the robot
# this location is saved to a csv file for later use (to navigate the robot to the location when needed)
# after selecting the first point, the user should select a second point to provide the orientation in which the robot should be
# this second coordinate is also saved to the csv file for later use
# the visual tool shows an arrow to point out the orientation

# --- IMPORTS ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from masterproef_interfaces.srv import GetStitchedImage
import cv2
import csv

# --- QUIZ LOCATION SELECTOR NODE CLASS ---
class QuizLocationSelector(Node):
    def __init__(self):
        # Initialize node
        super().__init__('quiz_location_selector')

        # Create client for the GetStitchedImage service
        self.cli = self.create_client(GetStitchedImage, 'get_stitched_image')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # CSV file path to save quiz location and orientation points
        self.csv_path = "/home/tabloo/jetson_ws/data/quiz_location.csv"  # CHANGE THIS PATH TO THE CORRECT PATH ON THE JETSON

        # Variables to store selected points
        self.first_point = None
        self.second_point = None
        self.temp_image = None

        # Wait for the stitched image service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for stitched image service...')

        self.get_logger().info('Quiz Location Selector ready.')
        self.send_request()

    # --- SEND SERVICE REQUEST ---
    def send_request(self):
        request = GetStitchedImage.Request()
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response.success:
            self.get_logger().info("Received stitched image.")
            self.display_image(response.image)
        else:
            self.get_logger().error(f"Failed to get stitched image: {response.message}")

    # --- MOUSE CALLBACK TO MARK POINTS ---
    def mark_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.first_point is None:
                self.first_point = (x, y)
                self.get_logger().info(f"First point selected at: {self.first_point}")
            elif self.second_point is None:
                self.second_point = (x, y)
                self.get_logger().info(f"Second point selected at: {self.second_point}")
                self.draw_arrow(self.image, self.first_point, self.second_point)
                cv2.imshow('Select Quiz Location', self.image)
                self.save_points_to_csv()
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        elif event == cv2.EVENT_MOUSEMOVE and self.first_point is not None and self.second_point is None:
            self.temp_image = self.image.copy()
            self.draw_arrow(self.temp_image, self.first_point, (x, y))
            cv2.imshow('Select Quiz Location', self.temp_image)

    # --- DRAW ORIENTATION ARROW ---
    def draw_arrow(self, image, pt1, pt2):
        cv2.arrowedLine(image, pt1, pt2, (0, 255, 0), 2, tipLength=0.2)

    # --- SAVE POINTS TO CSV ---
    def save_points_to_csv(self):
        try:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['X1', 'Y1', 'X2', 'Y2'])  # Location and Orientation point headers
                writer.writerow([self.first_point[0], self.first_point[1],
                                 self.second_point[0], self.second_point[1]])
            self.get_logger().info(f'Points saved to CSV: {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save points to CSV: {str(e)}')

    # --- DISPLAY THE STITCHED IMAGE AND SET CALLBACK ---
    def display_image(self, ros_image_msg):
        """Converts ROS image message to OpenCV image and displays it."""
        self.image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')
        height, width = self.image.shape[:2]
        # self.get_logger().info(f"Stitched image dimensions: {width}x{height} (width x height)")
        cv2.imshow('Select Quiz Location', self.image)
        cv2.setMouseCallback('Select Quiz Location', self.mark_point)
        cv2.waitKey(0)  # Wait for the user interaction
        cv2.destroyAllWindows()

# --- MAIN FUNCTION ---
def main(args=None):
    rclpy.init(args=args)
    node = QuizLocationSelector()
    try:
        rclpy.spin(node)  # Keeps node alive to process callbacks
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

# --- ENTRY POINT ---
if __name__ == '__main__':
    main()
