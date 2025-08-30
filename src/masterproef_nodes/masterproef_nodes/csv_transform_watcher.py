# JETSON NODE
#
# this code runs a node which publishes the transform matrix needed to transform an image pixel to a real world coordinate
# it publishes the transform when a trigger request is sent by the target selector node that needs the transform at startup
# it also publishes the transform once an update to the csv file containing the transform, is detected
#
# the log statements that will be called in each iteration are commented out to keep the memory from filling up

# Import required libraries: ROS 2, pandas for CSV, filesystem tools, and watchdog for monitoring file changes
import rclpy
import pandas as pd
import os
import json
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Main ROS 2 node that watches for changes to the transformation CSV and publishes updates
class CSVTransformWatcher(Node):
    def __init__(self):
        # Initialize the ROS node
        super().__init__('csv_transform_watcher')

        # Path to the CSV file that contains the 3x3 transformation matrix
        self.csv_path = "/home/tabloo/jetson_ws/data/transform.csv"  # CHANGE THIS PATH TO THE CORRECT PATH ON THE JETSON

        # Publisher that sends transformation data as a JSON string over a ROS topic
        self.publisher_ = self.create_publisher(String, 'csv_transform_data', 10)

        # Subscription to listen for acknowledgments from nodes using the transform
        self.create_subscription(String, 'ack_transform', self.ack_callback, 10)

        # Trigger service to allow other nodes (e.g., Target Selector) to request the latest transformation data on demand
        self.csv_trigger_service = self.create_service(Trigger, 'csv_transform_trigger', self.trigger_callback)

        # Set up a watchdog file observer to detect changes to the CSV file automatically
        self.event_handler = TransformCSVHandler(self.csv_path, self.publisher_, self.get_logger())
        self.observer = Observer()
        self.observer.schedule(self.event_handler, os.path.dirname(self.csv_path), recursive=False)
        self.observer.start()

        self.get_logger().info(f"Watching transform CSV file: {self.csv_path}")

    # Called when an acknowledgment message is received (for logging/debugging)
    def ack_callback(self, msg):
        """Callback for acknowledgment messages."""
        self.get_logger().info(f"Received acknowledgment: {msg.data}")

    # Called when another node triggers the service to publish the CSV manually
    def trigger_callback(self, request, response):
        """Service callback to manually trigger CSV data publishing."""
        self.get_logger().info("Transform CSV trigger service called. Publishing data.")
        self.publish_csv()
        response.success = True
        response.message = "Transform CSV data published successfully"
        return response

    # Reads the CSV and publishes its content as a JSON-formatted ROS String message
    def publish_csv(self):
        """Reads and publishes the transform CSV."""
        if not os.path.exists(self.csv_path):
            self.get_logger().error(f"CSV file not found: {self.csv_path}")
            return

        try:
            df = pd.read_csv(self.csv_path)
            csv_data = df.to_json()

            msg = String()
            msg.data = csv_data
            self.publisher_.publish(msg)

            self.get_logger().info(f"Published transform CSV data: {csv_data}")

        except Exception as e:
            self.get_logger().error(f"Error reading transform CSV: {e}")

    # Ensure observer is stopped gracefully when the node shuts down
    def destroy_node(self):
        """Stops the observer on shutdown."""
        self.observer.stop()
        self.observer.join()
        super().destroy_node()

# Watchdog event handler that detects modifications to the CSV file
class TransformCSVHandler(FileSystemEventHandler):
    def __init__(self, csv_path, publisher, logger):
        self.csv_path = csv_path
        self.publisher = publisher
        self.logger = logger

    # Called when the CSV file is modified
    def on_modified(self, event):
        if event.src_path == self.csv_path:
            self.publish_csv()

    # Re-reads and publishes the CSV content
    def publish_csv(self):
        if not os.path.exists(self.csv_path):
            self.logger.error(f"CSV file not found: {self.csv_path}")
            return

        try:
            df = pd.read_csv(self.csv_path)
            csv_data = df.to_json()

            msg = String()
            msg.data = csv_data
            self.publisher.publish(msg)

            self.logger.info(f"Published transform CSV data: {csv_data}")

        except Exception as e:
            self.logger.error(f"Error reading transform CSV: {e}")

# Standard ROS 2 entry point
def main(args=None):
    rclpy.init(args=args)
    node = CSVTransformWatcher()
    try:
        rclpy.spin(node)  # Keeps the node alive to process callbacks
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Start the main function when the script is run directly
if __name__ == '__main__':
    main()
