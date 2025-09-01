from setuptools import setup

package_name = 'masterproef_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pandas',
        'watchdog',
        'opencv-python'
    ],
    zip_safe=True,
    maintainer='tycho',
    maintainer_email='tycho@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # Entry points for CSV subscriber and watcher nodes  
            'csv_zone_watcher = masterproef_nodes.csv_zone_watcher:main',
            'target_selector_node = masterproef_nodes.target_selector_node:main',
            'yolo_node = masterproef_nodes.yolo_node:main',
            'yolo_node_speed_test = masterproef_nodes.yolo_node_speed_test:main',
            'heartbeat_publisher = masterproef_nodes.heartbeat_publisher:main',  
            'heartbeat_monitor = masterproef_nodes.heartbeat_monitor:main',
            'beamer_controller = masterproef_nodes.beamer_controller:main',
            'compute_transformation = masterproef_nodes.compute_transformation:main',
            'slow_image_stitcher = masterproef_nodes.slow_image_stitcher:main',
            'fast_image_stitcher = masterproef_nodes.fast_image_stitcher:main',
            'quiz_location_selector = masterproef_nodes.quiz_location_selector:main',
            'charge_location_selector = masterproef_nodes.charge_location_selector:main',
            'zone_location_selector = masterproef_nodes.zone_location_selector:main',
            'image_provider = masterproef_nodes.image_provider:main',
            'calibrated_image_provider = masterproef_nodes.calibrated_image_provider:main',
            'csv_transform_watcher = masterproef_nodes.csv_transform_watcher:main',
            'goal_sender = masterproef_nodes.goal_sender:main',
            'target_selector_node_unfiltered = masterproef_nodes.target_selector_node_unfiltered:main',
        ],
    },
)
