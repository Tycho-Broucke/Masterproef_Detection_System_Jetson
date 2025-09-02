#!/bin/bash

# Source ROS 2 and your workspace
source /opt/ros/humble/setup.bash  # Change to your ROS 2 distro if different
source ~/jetson_ws/install/setup.bash  # Or your actual workspace path

# Run each ROS 2 node in the background
ros2 run masterproef_nodes beamer_controller &
#ros2 run masterproef_nodes fast_image_stitcher &
ros2 run masterproef_nodes csv_zone_watcher &
ros2 run masterproef_nodes csv_transform_watcher &
ros2 run masterproef_nodes target_selector_node &
wait  # Keeps the script alive while all background processes run
