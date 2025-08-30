# this code runs a node which infinitely requests stitched images from the image service
# it runs yolo on each image to detect persons and robots and it publishes their pixel coordinates in a certain format so that the target selector can use it to choose a target person
# the format exists of 1 robot coordinate and 0-9 person coordinates, if less than 9 persons are detected inside of the zone, it is filled up with NULL values
# it also reads in the zone in which people should be detected on startup, and it also updates the zone when the csv_zone_watcher detects an update
# when the zone is succesfully updated, is sends back an acknowledgement to let the user know of the succes

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

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Path to pre-optimized TensorRT engine file
        model_path = '/home/tabloo/jetson_ws/src/masterproef_nodes/masterproef_nodes/yolo11n_float32.engine'  # Change to your engine file
        if not os.path.isfile(model_path):
            self.get_logger().error(f"TensorRT engine not found at {model_path}")
            exit(1)

        # Load TensorRT-optimized YOLO model
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded YOLO TensorRT model: {model_path}")

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(String, 'coordinates_topic', 10)
        self.ack_publisher = self.create_publisher(String, 'ack_zone', 10)

        self.zone_client = self.create_client(Trigger, 'csv_zone_trigger')
        while not self.zone_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for CSV zone service...')
        self.get_logger().info('CSV zone service is available.')

        self.image_client = self.create_client(GetStitchedImage, 'get_stitched_image')
        while not self.image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for stitched image service...')
        self.get_logger().info('Image service is ready.')

        self.create_subscription(String, 'csv_zone_data', self.csv_data_callback, 10)

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
                self.detect_objects(frame)
        except Exception as e:
            self.get_logger().error(f"Error in image response callback: {e}")
        self.request_image()

    def detect_objects(self, frame):
        start = time.time()

        results = self.model.predict(source=frame, show=False, stream=False, verbose=False)
        detections = results[0].boxes

        coordinates = []
        robot_detected = False
        people_count = 0

        for box in detections:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            if class_name not in ['person', 'robot']:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_bottom = xyxy[3]
            inside = self.is_inside_zone(x_center, y_bottom)

            if class_name == 'robot' and not robot_detected and inside:
                coordinates.append((x_center, y_bottom, 'R'))
                robot_detected = True
            elif class_name == 'person' and people_count < 9 and inside:
                coordinates.append((x_center, y_bottom, 'P'))
                people_count += 1

        while len(coordinates) < 10:
            coordinates.append((0, 0, None))

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
        cv2.imshow('YOLO TensorRT Detection', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

