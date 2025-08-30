# RASPI4 NODE
#
# this code runs a node that publishes a message containing the node name and timestamp at a predefined interval rate
# a monitor node should detect these messages and use it for error handling once it does not receive the messages anymore
#
# the log statements that will be called in each iteration are commented out to keep the memory from filling up

# --- IMPORTS ---
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# --- HEARTBEAT PUBLISHER NODE CLASS ---
class HeartbeatPublisher(Node):
    def __init__(self):
        # Initialize node
        super().__init__('heartbeat_publisher')

        # Create publisher on '/heartbeat' topic
        self.publisher_ = self.create_publisher(String, '/heartbeat', 10)

        # Timer to publish heartbeat at 1Hz
        self.timer = self.create_timer(1.0, self.publish_heartbeat)

        # Store node name
        self.node_name = self.get_name()

    # --- PUBLISH HEARTBEAT MESSAGE ---
    def publish_heartbeat(self):
        msg = String()
        # Compose message: "node_name,timestamp"
        msg.data = f"{self.node_name},{self.get_clock().now().seconds_nanoseconds()[0]}"
        self.publisher_.publish(msg)
        # self.get_logger().info(f"Sent heartbeat from {self.node_name}")

# --- MAIN FUNCTION ---
def main(args=None):
    rclpy.init(args=args)
    node = HeartbeatPublisher()
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
