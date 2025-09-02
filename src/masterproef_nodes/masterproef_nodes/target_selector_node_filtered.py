import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_srvs.srv import Trigger
from std_msgs.msg import String
import math
import threading
import pandas as pd
import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.task import Future

# Global homography transform matrix and readiness event
GLOBAL_TRANSFORM_MATRIX = np.identity(3)
TRANSFORM_READY = threading.Event()


class TargetSelector:
    """Class for selecting the closest person to the robot."""
    def __init__(self):
        self.targets = []

    def add_target(self, x, y, target_id):
        """Add a detected target to the internal list."""
        self.targets.append({'x': x, 'y': y, 'id': target_id})

    def process_coordinates(self, coordinates, robot_id, person_id, null_id=None):
        """
        Separate robot and person points, calculate distances.
        
        Returns:
            distances: list of (distance, person) tuples
            robot_points: list of robot points
        """
        self.targets.clear()
        for coord in coordinates:
            x, y, target_id = coord
            if target_id != null_id:
                self.add_target(x, y, target_id)

        robot_points = [t for t in self.targets if t['id'] == robot_id]
        person_points = [t for t in self.targets if t['id'] == person_id]

        distances = []
        for robot in robot_points:
            for person in person_points:
                dist = math.hypot(person['x'] - robot['x'], person['y'] - robot['y'])
                distances.append((dist, person))

        return distances, robot_points

    def get_shortest_distance(self, coordinates, robot_id, person_id, null_id=None):
        """
        Return the closest person to the robot, with fallbacks:
        - If robot not found: return first person as fallback
        - If no people: return None
        """
        distances, robot_points = self.process_coordinates(coordinates, robot_id, person_id, null_id)

        if not robot_points:
            return None, None, coordinates[0] if coordinates else None

        if not distances:
            return None, None, None

        shortest_distance, closest_person = min(distances, key=lambda x: x[0])
        return shortest_distance, closest_person, None


class TargetSelectorNodeFiltered(Node):
    """
    Node that selects the closest person from incoming coordinates and sends
    goals to Nav2 asynchronously. Previous goals are cancelled before sending
    new ones to ensure only the latest target is used.

    Enhancement: all coordinates are transformed into SLAM coordinates first,
    and if at least one person remains within a configurable distance threshold
    of the current goal, the goal is maintained. Otherwise, the goal is cancelled
    and renewed.
    """
    def __init__(self):
        super().__init__('target_selector_node_filtered')

        # Declare parameter for proximity threshold (default: 1.0 meter)
        self.declare_parameter('proximity_threshold', 1.0)
        self.proximity_threshold = self.get_parameter('proximity_threshold').value

        # Action client to send NavigateToPose goals to Nav2
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publisher for acknowledging transform updates
        self.ack_publisher_ = self.create_publisher(String, 'ack_transform', 10)

        # Subscription to YOLO coordinates
        self.create_subscription(String, 'coordinates_topic', self.coordinates_callback, 10)

        # Subscription to CSV transform data updates
        self.create_subscription(String, 'csv_transform_data', self.csv_transform_callback, 10)

        # Store the latest target to always send only the newest goal
        self.latest_target = None

        # Current active goal handle for potential cancellation
        self.current_goal_handle = None

        # Flag to track if a goal cancellation is in progress
        self.cancelling = False

        # Store last goal coordinates for distance checking
        self.last_goal = None

        # Client to request transform CSV updates
        self.client = self.create_client(Trigger, 'csv_transform_trigger')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for csv_transform_trigger service...')
        self.request_transform_csv()

        self.get_logger().info("TargetSelectorNodeFiltered started")

    def request_transform_csv(self):
        """Request the latest transform CSV from the csv_transform_watcher node."""
        request = Trigger.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Transform trigger response: {future.result().message}")
        else:
            self.get_logger().error("Failed to trigger transform CSV update.")

    def csv_transform_callback(self, msg):
        """
        Parse incoming CSV transform message and update the global homography matrix.
        Also publishes an acknowledgment message.
        """
        try:
            # Parses and updates the global transform matrix
            self.get_logger().info(f"Received transform CSV data: {msg.data}")
            df = pd.read_json(msg.data)
            if df.empty:
                self.get_logger().warn("Received empty transform CSV.")
                return

            row = df.iloc[0].to_numpy().astype(float)
            if len(row) != 9:
                self.get_logger().error("Transform CSV does not contain 9 elements.")
                return

            TRANSFORM_READY.clear()
            global GLOBAL_TRANSFORM_MATRIX
            GLOBAL_TRANSFORM_MATRIX = np.array(row).reshape((3, 3))
            TRANSFORM_READY.set()
            self.get_logger().info(f"Updated homography matrix:\n{GLOBAL_TRANSFORM_MATRIX}")

            # Publish acknowledgment
            ack_msg = String()
            ack_msg.data = "Transformation updated successfully in the target selector node"
            self.ack_publisher_.publish(ack_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to parse transform CSV: {e}")

    def apply_homography(self, x, y):
        """Apply homography to transform pixel coordinates to SLAM map coordinates."""
        global GLOBAL_TRANSFORM_MATRIX
        point = np.array([x, y, 1])
        transformed = GLOBAL_TRANSFORM_MATRIX @ point
        x_out = transformed[0] / transformed[2]
        y_out = transformed[1] / transformed[2]
        return x_out, y_out

    def create_pose_stamped(self, x, y):
        """Create a PoseStamped message for Nav2 with hardcoded orientation."""
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = 1.0  # Facing up
        msg.pose.orientation.w = 0.0
        return msg

    def coordinates_callback(self, data):
        """
        Handle incoming YOLO coordinates:
        - Transform all coordinates into SLAM coordinates
        - Determine closest person
        - If someone is within threshold of current goal, keep current goal
        - Otherwise, cancel and send new goal
        """
        if not TRANSFORM_READY.is_set():
            self.get_logger().warn("Transform not ready. Dropping message.")
            return

        raw_coords = self.parse_coordinates(data.data)

        # Transform all coordinates to SLAM
        transformed_coords = [(self.apply_homography(x, y)[0], self.apply_homography(x, y)[1], t_id) for x, y, t_id in raw_coords]

        selector = TargetSelector()
        _, closest_person, fallback_coord = selector.get_shortest_distance(transformed_coords, 'R', 'P', None)

        # Determine the latest target coordinates
        if closest_person:
            self.latest_target = (closest_person['x'], closest_person['y'])
        elif fallback_coord:
            self.latest_target = (fallback_coord[0], fallback_coord[1])
        else:
            self.latest_target = None

        # Check if current goal should be kept
        if self.last_goal and self.latest_target:
            within_threshold = any(
                math.hypot(px - self.last_goal[0], py - self.last_goal[1]) < self.proximity_threshold
                for (px, py, pid) in transformed_coords if pid == 'P'
            )
            if within_threshold:
                self.get_logger().info(f"A person is still within {self.proximity_threshold}m of the current goal. Keeping goal.")
                return

        # Process the goal (send or cancel as needed)
        self.process_goal()

    def process_goal(self):
        """
        Decide whether to send a new goal or cancel the previous one.
        Ensures only the latest target is sent.
        """
        # Wait for action server
        if not self._client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("NavigateToPose action server not available yet")
            return

        if self.cancelling:
            # Already cancelling previous goal; wait for callback
            return

        if self.current_goal_handle:
            # Cancel previous goal asynchronously
            self.get_logger().info("Cancelling previous goal")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            self.cancelling = True
            cancel_future.add_done_callback(self.cancel_done_callback)
        elif self.latest_target:
            # No previous goal to cancel; send immediately
            self.send_goal(self.latest_target)

    def cancel_done_callback(self, future: Future):
        """
        Callback when goal cancellation completes:
        - Reset state
        - Send the latest target if available
        """
        self.cancelling = False
        self.current_goal_handle = None
        if self.latest_target:
            self.send_goal(self.latest_target)

    def send_goal(self, target):
        """Send a NavigateToPose goal to Nav2."""
        x, y = target
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(x, y)  # assign the PoseStamped directly
        self.get_logger().info(f"Sending new goal: x={x}, y={y}")
        send_future = self._client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
        self.last_goal = (x, y)

    def goal_response_callback(self, future: Future):
        """Handle goal acceptance from Nav2."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            self.current_goal_handle = None
            return
        self.get_logger().info("Goal accepted")
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future: Future):
        """Handle the result when the goal is completed."""
        result = future.result().result
        self.get_logger().info(f"Goal completed with result: {result}")
        self.current_goal_handle = None

    def parse_coordinates(self, data):
        """
        Parse incoming coordinate string into a list of tuples.
        WARNING: eval should only be used on trusted sources!
        """
        return eval(data)


def main(args=None):
    rclpy.init(args=args)
    node = TargetSelectorNodeFiltered()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

