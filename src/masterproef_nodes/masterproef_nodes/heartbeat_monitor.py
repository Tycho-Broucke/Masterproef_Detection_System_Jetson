# JETSON NODE
#
# this code runs a node that keeps track of the messages received by the heartbeat publisher
# once it does not receive anymore messages, the connection is considered as lost
# it then gives an error, but this error still has to be handled (emergency stop should be performed)
#
# the log statements that will be called in each iteration are commented out to keep the memory from filling up

# --- IMPORTS ---
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

# --- CONFIGURATION CONSTANTS ---
TIMEOUT = 3  # Seconds before considering a node as "lost"

# --- HEARTBEAT MONITOR NODE CLASS ---
class HeartbeatMonitor(Node):
    def __init__(self):
        # Initialize the node
        super().__init__('heartbeat_monitor')

        # Subscribe to the heartbeat topic
        self.subscription = self.create_subscription(String, '/heartbeat', self.heartbeat_callback, 10)

        # Dictionary to track the last timestamp received per node
        self.last_heartbeat = {}

        # Set of nodes currently considered disconnected
        self.lost_nodes = set()

        # Timer to periodically check for lost connections
        self.timer = self.create_timer(1.0, self.check_heartbeats)  # Every 1 second

    # --- CALLBACK: Process incoming heartbeat messages ---
    def heartbeat_callback(self, msg):
        node_name, timestamp = msg.data.split(",")
        timestamp = float(timestamp)

        # Node has reconnected
        if node_name in self.lost_nodes:
            self.get_logger().info(f"✅ Reconnected: {node_name}")
            self.lost_nodes.remove(node_name)

        # New node first seen
        if node_name not in self.last_heartbeat:
            self.get_logger().info(f"🔵 New node detected: {node_name}")

        # Update the timestamp
        self.last_heartbeat[node_name] = timestamp

    # --- CHECK FOR TIMEOUTS ---
    def check_heartbeats(self):
        current_time = time.time()
        for node, last_time in list(self.last_heartbeat.items()):
            if last_time is not None and current_time - last_time > TIMEOUT:
                if node not in self.lost_nodes:
                    self.get_logger().warn(f"⚠️ Lost connection to {node}")
                    self.lost_nodes.add(node)

# --- MAIN FUNCTION ---
def main(args=None):
    rclpy.init(args=args)
    node = HeartbeatMonitor()
    try:
        rclpy.spin(node)  # Keep the node alive
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
