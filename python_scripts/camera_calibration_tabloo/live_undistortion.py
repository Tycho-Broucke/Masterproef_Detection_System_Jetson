import cv2
import numpy as np
import os
import pickle

# Live undistortion parameters
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CALIBRATION_FILE = 'output/calibration_data.pkl'  # Path to calibration data

def live_undistortion():
    """
    Demonstrate live camera undistortion using calibration results.
    """
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
        return
    
    # Load calibration data
    with open(CALIBRATION_FILE, 'rb') as f:
        calibration_data = pickle.load(f)
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeff']
    
    print("Loaded camera calibration data:")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients: {dist.ravel()}")
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Two versions of new camera matrix: full FOV (alpha=1) and cropped (alpha=0)
    newcameramtx_full, roi_full = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    newcameramtx_crop, roi_crop = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
    
    # Create undistortion maps for both
    mapx_full, mapy_full = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx_full, (width, height), 5)
    mapx_crop, mapy_crop = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx_crop, (width, height), 5)
    
    print("Press 'q' to quit, 'd' to toggle distortion correction, 'c' to toggle crop mode")
    
    correct_distortion = True
    crop_mode = False  # False = full FOV (black borders), True = cropped
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        if correct_distortion:
            if crop_mode:
                undistorted = cv2.remap(frame, mapx_crop, mapy_crop, cv2.INTER_LINEAR)
                #x, y, w, h = roi_crop
                #undistorted = undistorted[y:y+h, x:x+w]
                #undistorted = cv2.resize(undistorted, (width, height))
                label = "Undistorted (cropped)"
            else:
                undistorted = cv2.remap(frame, mapx_full, mapy_full, cv2.INTER_LINEAR)
                label = "Undistorted (full FOV)"
            
            cv2.putText(undistorted, label, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Feed', undistorted)
        else:
            cv2.putText(frame, "Original", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            correct_distortion = not correct_distortion
            print(f"Distortion correction {'ON' if correct_distortion else 'OFF'}")
        elif key == ord('c'):
            crop_mode = not crop_mode
            print(f"Crop mode {'ON' if crop_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_undistortion()

