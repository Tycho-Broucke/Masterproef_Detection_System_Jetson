# JETSON NODE
#
# this code runs a node which publishes the zone coordinates needed to check if tracked people are inside of the zone
# it publishes the coordinates when a trigger request is sent by the yolo node that needs the coordinates at startup
# it also publishes the coordinates once an update to the csv file containing the coordinates, is detected
#
# the log statements that will be called in each iteration are commented out to keep the memory from filling up

# Required imports for ROS 2, file handling, and CSV monitoring
import rclpy
import pandas as pd
import os
import json
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Main ROS 2 node class that watches the zone coordinate CSV file and publishes its content
class CSVZoneWatcher(Node):
    def __init__(self):
        # Initialize the ROS node
        super().__init__('csv_zone_watcher')

        # Path to the CSV file containing zone coordinates
        self.csv_path = "/home/tabloo/jetson_ws/data/zone_coordinates.csv"  # CHANGE THIS PATH TO THE CORRECT PATH ON THE JETSON

        # Publisher to send the CSV data over a topic as JSON
        self.publisher_ = self.create_publisher(String, 'csv_zone_data', 10)

        # Subscriber to listen for acknowledgment messages after data is received
        self.create_subscription(String, 'ack_zone', self.ack_zone_callback, 10)

        # ROS 2 service to allow external nodes (like YOLO) to trigger manual publishing
        self.csv_zone_trigger_service = self.create_service(Trigger, 'csv_zone_trigger', self.csv_zone_trigger_callback)

        # Setup for monitoring file changes using Watchdog
        self.event_handler = CSVFileHandler(self.csv_path, self.publisher_, self.get_logger())
        self.observer = Observer()
        self.observer.schedule(self.event_handler, os.path.dirname(self.csv_path), recursive=False)
        self.observer.start()

        self.get_logger().info(f"Watching CSV file: {self.csv_path}")

    # Callback for receiving acknowledgment messages
    def ack_zone_callback(self, msg):
        """Callback function to handle the acknowledgment messages."""
        self.get_logger().info(f"Received acknowledgment: {msg.data}")

    # Callback that handles the Trigger service request to publish CSV data
    def csv_zone_trigger_callback(self, request, response):
        """Callback function for the csv_zone_trigger service."""
        self.get_logger().info("CSV zone trigger service called. Publishing CSV data.")
        
        # Publish the CSV content
        self.publish_csv()

        # Respond with a success message
        response.success = True
        response.message = "Zone CSV data published successfully"
        return response

    # Function to read the CSV file and publish its contents
    def publish_csv(self):
        """Reads CSV and publishes its contents."""
        if not os.path.exists(self.csv_path):
            self.get_logger().error(f"CSV file not found: {self.csv_path}")
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            csv_data = df.to_json()  # Convert to JSON string format
            
            msg = String()
            msg.data = csv_data
            self.publisher_.publish(msg)

            self.get_logger().info(f"CSV file updated! Published new data: {csv_data}")

        except Exception as e:
            self.get_logger().error(f"Error reading CSV: {e}")

    # Ensure file monitoring stops properly when the node shuts down
    def destroy_node(self):
        """Stop file observer when shutting down."""
        self.observer.stop()
        self.observer.join()
        super().destroy_node()

# FileSystemEventHandler subclass to watch for CSV modifications
class CSVFileHandler(FileSystemEventHandler):
    """Handles file change events for the CSV file."""
    def __init__(self, csv_path, publisher, logger):
        self.csv_path = csv_path
        self.publisher = publisher
        self.logger = logger

    # Called automatically by watchdog when the file is modified
    def on_modified(self, event):
        """Triggered when the CSV file is modified."""
        if event.src_path == self.csv_path:
            self.publish_csv()

    # Reads and publishes the modified CSV file
    def publish_csv(self):
        """Reads CSV and publishes its contents."""
        if not os.path.exists(self.csv_path):
            self.logger.error(f"CSV file not found: {self.csv_path}")
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            csv_data = df.to_json()  # Convert to JSON string format
            
            msg = String()
            msg.data = csv_data
            self.publisher.publish(msg)

            self.logger.info(f"CSV file updated! Published new data: {csv_data}")

        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")

# Standard ROS 2 node startup logic
def main(args=None):
    rclpy.init(args=args)
    node = CSVZoneWatcher()
    try:
        rclpy.spin(node)  # Keeps the node alive to process callbacks
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Entry point when the script is executed directly
if __name__ == '__main__':
    main()
