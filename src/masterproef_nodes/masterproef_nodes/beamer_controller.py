# RASPI4 NODE
#
# this code runs a node that subscribes to the quiz topic to receive messages that contain the state of the quiz
# based on the state it receives, it either turns on or off the beamer
# the beamer is connected to the pi4 with an ethernet cable, and is controller with via a socket
# both the beamer and pi4 have static eth ip addresses in the same subnet
#
# the log statements that will be called in each iteration are commented out to keep the memory from filling up

# Import necessary ROS 2 and system libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket

# Define the IP and port for communicating with the projector over PJLink protocol
PROJECTOR_IP = "192.168.1.20"  # Replace with your projector's IP
PJLINK_PORT = 4352  # PJLink uses port 4352

# Define a ROS 2 Node that controls the projector (beamer)
class BeamerController(Node):
    def __init__(self):
        # Initialize the ROS 2 Node with a name
        super().__init__('beamer_controller')

        # Create a subscription to the 'quiz' topic with String messages
        # When a message is received, call the listener_callback function
        self.subscription = self.create_subscription(
            String,
            'quiz',
            self.listener_callback,
            10)

        # Log that the controller has started
        self.get_logger().info('Beamer Controller has been started.')

    # Sends a PJLink command to the projector via a TCP socket
    def send_command(self, command):
        try:
            # Create a TCP socket and connect to the projector
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((PROJECTOR_IP, PJLINK_PORT))

                # Receive the initial handshake or response from the projector
                initial_response = s.recv(1024).decode()

                # Check if authentication is required (PJLINK 0 means no auth)
                if "PJLINK 0" in initial_response:
                    # Send the command (e.g., turn on or off)
                    s.sendall(command.encode('utf-8'))
                    response = s.recv(1024).decode()
                    # self.get_logger().info(f'Response: {response}')  # Disabled for memory safety
                else:
                    # Warn if the projector expects a password (not handled here)
                    self.get_logger().warning('Authentication required, but no password provided.')
        except Exception as e:
            # Log any errors encountered during socket communication
            self.get_logger().error(f'Error sending command: {str(e)}')

    # Callback that gets triggered when a message is received on the 'quiz' topic
    def listener_callback(self, msg):
        message = msg.data
        # self.get_logger().info(f'Received message: {message}')  # Disabled for memory safety

        # Interpret message content and control projector accordingly
        if message == 'quiz_finished':
            self.send_command('%1POWR 0\r')  # Turn off projector
        elif message == 'drive_to_quiz_location':
            self.send_command('%1POWR 1\r')  # Turn on projector

# Main function that starts the ROS 2 node and handles shutdown
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2 communication
    node = BeamerController()  # Create an instance of the controller node
    try:
        rclpy.spin(node)  # Keep the node alive to handle callbacks
    except KeyboardInterrupt:
        # Gracefully handle shutdown via Ctrl+C
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    except Exception as e:
        # Log any unexpected runtime errors
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        # Ensure node resources are properly released
        node.destroy_node()
        rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()
