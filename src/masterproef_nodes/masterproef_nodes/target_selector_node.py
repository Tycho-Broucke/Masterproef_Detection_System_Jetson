# RASPI4 NODE
#
# this code runs a node which subscribes to messages from the yolo node. these messages contain the pixel coordinates of the robot
# and 0–9 people detected inside the zone. the target selector chooses the person closest to the robot in bird's eye view and transforms
# that person's pixel coordinates to real-world coordinates on the SLAM map. these coordinates are then published to the 'goal_pose' topic
# so the robot can move to the location using nav2.
# the transformation from pixel to SLAM coordinates is done by retrieving the transform matrix from the csv_transform_watcher node.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import math
import time
import pandas as pd
import numpy as np
from geometry_msgs.msg import PoseStamped
import threading

# Global transform matrix and readiness flag
GLOBAL_TRANSFORM_MATRIX = np.identity(3)
TRANSFORM_READY = threading.Event()


class TargetSelector:
    def __init__(self):
        # Initializes a list to hold detected targets
        self.targets = []

    def add_target(self, x, y, target_id):
        # Adds a detected target to the list
        self.targets.append({'x': x, 'y': y, 'id': target_id})

    def process_coordinates(self, coordinates, robot_id, person_id, null_id):
        # Filters and categorizes targets into robot and person lists
        self.targets.clear()
        for coord in coordinates:
            x, y, target_id = coord
            if target_id != null_id:
                self.add_target(x, y, target_id)

        robot_points = [t for t in self.targets if t['id'] == robot_id]
        person_points = [t for t in self.targets if t['id'] == person_id]

        # Calculate distances between each robot and each person
        distances = []
        for robot in robot_points:
            for person in person_points:
                dist = math.hypot(person['x'] - robot['x'], person['y'] - robot['y'])
                distances.append((dist, person))

        return distances, robot_points

    def get_shortest_distance(self, coordinates, robot_id, person_id, null_id):
        # Returns the person closest to the robot, or fallback if no robot present
        distances, robot_points = self.process_coordinates(coordinates, robot_id, person_id, null_id)

        if not robot_points:
            return None, None, coordinates[0] if coordinates else None

        if not distances:
            return None, None, None

        shortest_distance, closest_person = min(distances, key=lambda x: x[0])
        return shortest_distance, closest_person, None


class TargetSelectorNode(Node):
    def __init__(self):
        super().__init__('target_selector_node')

        # Publisher for PoseStamped goals
        self.goal_publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Publisher for transformation acknowledgments
        self.ack_publisher_ = self.create_publisher(String, 'ack_transform', 10)

        # Subscribe to coordinate data from YOLO node
        self.subscription = self.create_subscription(String, 'coordinates_topic', self.callback, 10)

        # Subscribe to transformation updates from transform watcher node
        self.create_subscription(String, 'csv_transform_data', self.csv_transform_callback, 10)

        # Client to trigger transform file reload
        self.client = self.create_client(Trigger, 'csv_transform_trigger')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for csv_transform_trigger service...')

        self.get_logger().info("Transform trigger service available, requesting update...")
        self.request_transform_csv()
        self.get_logger().info("Target Selector Node started, waiting for coordinates...")

    def request_transform_csv(self):
        # Requests transform matrix update from the csv_transform_watcher node
        request = Trigger.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Transform trigger response: {future.result().message}")
        else:
            self.get_logger().error("Failed to trigger transform CSV update.")

    def csv_transform_callback(self, msg):
        try:
            # Parses and updates the global transform matrix
            self.get_logger().info(f"Received transform CSV data: {msg.data}")
            df = pd.read_json(msg.data)
            if df.empty:
                self.get_logger().warn("Transform CSV was parsed but empty.")
                return

            row = df.iloc[0].to_numpy().astype(float)
            if len(row) != 9:
                self.get_logger().error("Transform CSV does not contain 9 elements.")
                return

            TRANSFORM_READY.clear()

            global GLOBAL_TRANSFORM_MATRIX
            GLOBAL_TRANSFORM_MATRIX = np.array(row).reshape((3, 3))
            TRANSFORM_READY.set()

            self.get_logger().info(f"Updated global transform matrix:\n{GLOBAL_TRANSFORM_MATRIX}")

            # Publish acknowledgment to confirm transform update
            ack_msg = String()
            ack_msg.data = "Transformation updated successfully in the target selector node"
            self.ack_publisher_.publish(ack_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to parse transform CSV: {e}")

    def apply_homography(self, x, y):
        # Applies homography transformation to convert pixel to map coordinates
        global GLOBAL_TRANSFORM_MATRIX
        point = np.array([x, y, 1])
        transformed = GLOBAL_TRANSFORM_MATRIX @ point
        x_out = transformed[0] / transformed[2]
        y_out = transformed[1] / transformed[2]
        return x_out, y_out

    def create_pose_stamped(self, x, y):
        # Creates a PoseStamped message with SLAM coordinates
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def callback(self, data):
        # Callback for incoming coordinate data
        if not TRANSFORM_READY.is_set():
            self.get_logger().warn("Global transform matrix not ready yet. Dropping coordinate message.")
            return

        coordinates = self.parse_coordinates(data.data)
        selector = TargetSelector()
        shortest_distance, closest_person, fallback_coord = selector.get_shortest_distance(
            coordinates, 'R', 'P', None
        )

        # Fallback: Use the first coordinate if no robot is found
        if fallback_coord:
            x_trans, y_trans = self.apply_homography(fallback_coord[0], fallback_coord[1])
            pose = self.create_pose_stamped(x_trans, y_trans)
            self.goal_publisher_.publish(pose)

        # Normal case: Publish closest person’s transformed coordinates
        elif shortest_distance is not None and closest_person is not None:
            x_trans, y_trans = self.apply_homography(closest_person['x'], closest_person['y'])
            pose = self.create_pose_stamped(x_trans, y_trans)
            self.goal_publisher_.publish(pose)

        else:
            self.get_logger().info("No valid target found.")

    def parse_coordinates(self, data):
        # WARNING: eval should only be used with trusted data sources
        # Converts string representation of coordinates into a list of tuples
        return eval(data)


def main(args=None):
    rclpy.init(args=args)
    node = TargetSelectorNode()
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
