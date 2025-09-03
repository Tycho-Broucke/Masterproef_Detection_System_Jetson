import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from masterproef_interfaces.srv import GetStitchedImage
from ultralytics import YOLO
from cv_bridge import CvBridge
import numpy as np
import pandas as pd
import time
import cv2
import os

class YoloNodeDualModels(Node):
    def __init__(self):
        super().__init__('yolo_node_dual_models')

        # Paths to YOLO TensorRT engine files
        person_model_path = '/home/tabloo/jetson_ws/src/masterproef_nodes/masterproef_nodes/yolo11n_float32.engine'
        robot_model_path = '/home/tabloo/jetson_ws/src/masterproef_nodes/masterproef_nodes/yolo11n_robot_detection_float32.engine'

        # Check model files exist
        if not os.path.isfile(person_model_path):
            self.get_logger().error(f"Person detection model not found at {person_model_path}")
            exit(1)
        if not os.path.isfile(robot_model_path):
            self.get_logger().error(f"Robot detection model not found at {robot_model_path}")
            exit(1)

        # Load both YOLO models
        self.person_model = YOLO(person_model_path)
        self.robot_model = YOLO(robot_model_path)
        self.get_logger().info("Loaded YOLO models for person and robot detection.")

        self.bridge = CvBridge()

        # Publishers
        self.publisher_ = self.create_publisher(String, 'coordinates_topic', 10)
        self.ack_publisher = self.create_publisher(String, 'ack_zone', 10)

        # Service clients
        self.zone_client = self.create_client(Trigger, 'csv_zone_trigger')
        while not self.zone_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for CSV zone service...')
        self.get_logger().info('CSV zone service is available.')

        self.image_client = self.create_client(GetStitchedImage, 'get_stitched_image')
        while not self.image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for stitched image service...')
        self.get_logger().info('Image service is ready.')

        # Subscriptions
        self.create_subscription(String, 'csv_zone_data', self.csv_data_callback, 10)

        # Default zone
        self.upperleft = (0, 0)
        self.upperright = (640, 0)
        self.lowerright = (640, 480)
        self.lowerleft = (0, 480)
        self.zone_contour = self.get_zone_contour()

        self.update_zone_from_service()
        self.request_image()

    def get_zone_contour(self):
        return np.array([self.upperleft, self.upperright, self.lowerright, self.lowerleft], dtype=np.int32)

    def update_zone_from_service(self):
        request = Trigger.Request()
        future = self.zone_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"Trigger response: {future.result().message}")
        else:
            self.get_logger().error("Zone service call failed.")

    def csv_data_callback(self, msg):
        try:
            if not msg.data:
                self.get_logger().warn("Empty CSV zone data received.")
                return
            df = pd.read_json(msg.data)
            if df.empty:
                self.get_logger().warn("Parsed CSV data is empty.")
                return

            row = df.iloc[0]
            self.upperleft = eval(row['UpperLeft'])
            self.upperright = eval(row['UpperRight'])
            self.lowerleft = eval(row['LowerLeft'])
            self.lowerright = eval(row['LowerRight'])
            self.zone_contour = self.get_zone_contour()

            self.get_logger().info(f"Updated zone: {self.zone_contour.tolist()}")
            ack_msg = String()
            ack_msg.data = "Zone updated successfully in yolo node"
            self.ack_publisher.publish(ack_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to update zone from CSV: {e}")

    def is_inside_zone(self, x, y):
        return cv2.pointPolygonTest(self.zone_contour, (int(x), int(y)), False) >= 0

    def request_image(self):
        self.image_request_start = time.time()
        request = GetStitchedImage.Request()
        future = self.image_client.call_async(request)
        future.add_done_callback(self.image_response_callback)

    def image_response_callback(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error(f"Failed to get stitched image: {response.message}")
            else:
                frame = self.bridge.imgmsg_to_cv2(response.image, desired_encoding='bgr8')
                self.detect_objects_dual_models(frame)
        except Exception as e:
            self.get_logger().error(f"Error in image response callback: {e}")
        self.request_image()

    def detect_objects_dual_models(self, frame):
        start = time.time()

        # Run robot detection model
        robot_results = self.robot_model.predict(source=frame, show=False, stream=False, verbose=False)
        robot_boxes = robot_results[0].boxes

        # Run person detection model
        person_results = self.person_model.predict(source=frame, show=False, stream=False, verbose=False)
        person_boxes = person_results[0].boxes

        coordinates = []
        robot_detected = False
        people_count = 0

        # Process robot detections
        for box in robot_boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.robot_model.names[cls_id]
            if class_name != 'robot':
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_bottom = xyxy[3]
            if not robot_detected and self.is_inside_zone(x_center, y_bottom):
                coordinates.append((x_center, y_bottom, 'R'))
                robot_detected = True

        # Process person detections
        for box in person_boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.person_model.names[cls_id]
            if class_name != 'person':
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_bottom = xyxy[3]
            if people_count < 9 and self.is_inside_zone(x_center, y_bottom):
                coordinates.append((x_center, y_bottom, 'P'))
                people_count += 1

        # Fill up to 10 entries
        while len(coordinates) < 10:
            coordinates.append((0, 0, None))

        # Publish formatted coordinates
        msg = String()
        msg.data = str(coordinates)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published coordinates: {msg.data}")

        end = time.time()
        self.get_logger().info(f"Inference took {end - start:.3f}s | Full cycle {end - self.image_request_start:.3f}s")

        self.display_frame(frame, coordinates)

    def display_frame(self, frame, coordinates):
        cv2.polylines(frame, [self.zone_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        for x, y, label in coordinates:
            if label == 'R':
                color, text = (0, 0, 255), 'Robot'
            elif label == 'P':
                color, text = (255, 0, 0), 'Person'
            else:
                continue
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.putText(frame, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('YOLO Dual Model Detection', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloNodeDualModels()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO dual model node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

