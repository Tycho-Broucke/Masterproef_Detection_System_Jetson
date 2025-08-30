import cv2
import numpy as np
import os
import glob
import pickle

# Parameters
CAMERA_ID = 0
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 2.65
OUTPUT_DIR = 'calibration_images'
CALIB_OUTPUT_DIR = 'output'

def capture_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_ID}")
        return 0
    
    img_counter = 0

    print("Press 'c' to capture an image when chessboard is detected.")
    print("Press 'q' or ESC to finish capturing and run calibration.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        
        if found:
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)
            cv2.putText(frame, "Chessboard detected! Press 'c' to capture.", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "Chessboard NOT detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Captured images: {img_counter}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('Capture Calibration Images', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Quit
            break
        elif key == ord('c') and found:
            img_name = os.path.join(OUTPUT_DIR, f"calibration_{img_counter:02d}.jpg")
            cv2.imwrite(img_name, gray)
            print(f"Saved {img_name}")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return img_counter

def calibrate_camera():
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg'))
    if not images:
        print("No calibration images found!")
        return None, None, None

    if not os.path.exists(CALIB_OUTPUT_DIR):
        os.makedirs(CALIB_OUTPUT_DIR)

    print(f"Calibrating using {len(images)} images...")
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imwrite(os.path.join(CALIB_OUTPUT_DIR, f'corners_{os.path.basename(fname)}'), img)
            print(f"Image {i+1}/{len(images)}: chessboard detected.")
        else:
            print(f"Image {i+1}/{len(images)}: chessboard NOT detected.")

    if not objpoints:
        print("No chessboard corners found in any image. Calibration failed.")
        return None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration done. RMS error: {ret}")

    # Save calibration
    calib_data = {'camera_matrix': mtx, 'dist_coeff': dist,
                  'rvecs': rvecs, 'tvecs': tvecs}
    with open(os.path.join(CALIB_OUTPUT_DIR, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calib_data, f)

    # Save matrix and dist coeff as text files
    np.savetxt(os.path.join(CALIB_OUTPUT_DIR, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(CALIB_OUTPUT_DIR, 'dist_coeff.txt'), dist)

    return mtx, dist

def main():
    count = capture_images()
    if count > 0:
        calibrate_camera()
    else:
        print("No images captured, calibration skipped.")

if __name__ == "__main__":
    main()
