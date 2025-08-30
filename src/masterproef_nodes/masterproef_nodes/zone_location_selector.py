# this code runs a node which requests a stitched image from the getstitchedimage service
# once the image is received, it is displayed and the user can click 4 points on the image to determine the keepout zone
# after clicking, the coordinates of the zone are automatically saved to the csv file
# it also subscribes to an acknowledgement topic which lets the user know if the updated zone is correctly updated in the yolo node

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from rclpy.time import Time
from masterproef_interfaces.srv import GetStitchedImage
import cv2
import csv

class ZoneLocationSelector(Node):
    def __init__(self):
        super().__init__('zone_location_selector')

        # Initialize client for the get_stitched_image service
        self.cli = self.create_client(GetStitchedImage, 'get_stitched_image')
        
        # Bridge for converting between ROS and OpenCV image formats
        self.bridge = CvBridge()

        # Path to CSV file where selected zone points will be saved
        self.csv_path = "/home/tabloo/jetson_ws/data/zone_coordinates.csv"  # CHANGE THIS PATH TO THE CORRECT PATH ON THE JETSON
        
        # Store clicked points
        self.points = []

        # Subscribe to acknowledgment topic for zone update confirmation
        self.ack_subscriber = self.create_subscription(
            String,
            'ack_zone',
            self.ack_callback,
            10
        )

        # Wait until the image service is available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the getstitchedimage service...')

        self.get_logger().info('ZoneLocationSelector is ready.')
        
        # Send request to retrieve stitched image
        self.send_request()

    def send_request(self):
        """Sends request to getstitchedimage service and handles the image."""
        request = GetStitchedImage.Request()
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if not response:
            self.get_logger().error('Service call failed.')
            return

        self.get_logger().info('Received stitched image from service.')

        # Convert ROS image message to OpenCV format
        self.current_image = self.bridge.imgmsg_to_cv2(response.image, desired_encoding='bgr8')

        # Display image and allow user to click points
        cv2.imshow('Select Zone Location', self.current_image)
        cv2.setMouseCallback('Select Zone Location', self.mark_point)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mark_point(self, event, x, y, flags, param):
        # Handle mouse click events to collect four corner points
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.current_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select Zone Location', self.current_image)
            cv2.waitKey(1)
            self.get_logger().info(f'Point marked at: ({x}, {y})')

            # Once 4 points are selected, organize and save them
            if len(self.points) == 4:
                self.arrange_points()
                self.save_points_to_csv()

    def arrange_points(self):
        # Sorts and arranges clicked points into [UpperLeft, UpperRight, LowerLeft, LowerRight]
        self.points.sort(key=lambda p: (p[0], p[1]))
        left_points = self.points[:2]
        right_points = self.points[2:]

        upper_left = min(left_points, key=lambda p: p[1])
        lower_left = max(left_points, key=lambda p: p[1])
        upper_right = min(right_points, key=lambda p: p[1])
        lower_right = max(right_points, key=lambda p: p[1])

        self.points = [upper_left, upper_right, lower_left, lower_right]

    def save_points_to_csv(self):
        # Saves the arranged points to a CSV file
        try:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['UpperLeft', 'UpperRight', 'LowerLeft', 'LowerRight'])
                writer.writerow([str(p) for p in self.points])
            self.get_logger().info(f'Points saved to CSV: {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save points to CSV: {str(e)}')

    def ack_callback(self, msg):
        """Handles acknowledgment from another node."""
        self.get_logger().info(f'Acknowledgment received on "ack_zone": {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = ZoneLocationSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
