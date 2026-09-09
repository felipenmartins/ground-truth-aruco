import cv2 as cv
from cv2 import aruco
import numpy as np
import csv
import time

# Parameters
save_csv = True  # Set to True to save the output to a CSV file

# Camera settings
# Camera calibration - tested with my laptop camera - maybe replace with the real calibration
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs   = np.zeros((5, 1))  # No distortion

# Marker real-world size in meters
marker_length = 0.05 # meters

# Select dictionaries
aruco_dicts = [
    aruco.DICT_4X4_50, aruco.DICT_5X5_100, aruco.DICT_6X6_250
]
parameters = aruco.DetectorParameters()
detectors  = [aruco.ArucoDetector(aruco.getPredefinedDictionary(d), parameters) for d in aruco_dicts]

# Define the 3D coordinates of the marker corners (assuming flat on Z=0)
object_points = np.array([
    [-0.5, 0.5, 0],
    [0.5, 0.5, 0],
    [0.5, -0.5, 0],
    [-0.5, -0.5, 0]
]) * marker_length

if save_csv:
    # Open CSV file for logging
    csv_file   = open('data/rotation_output.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'marker_id', 'yaw_deg', 'pitch_deg', 'roll_deg', 'x', 'y', 'z'])

# Native camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for detector in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if corners:
            for i, corner in enumerate(corners):
                image_points = corner[0].astype(np.float32)

                # Calculate the 3D rotation and translation vectors
                # solvePnP estimates the position and orientation of the camera in three-dimensional space
                success, rvec, tvec = cv.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )

                # Check if solvePnP was successful
                if success:
                    # Draw axis
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                    # Convert rvec to rotation matrix then to Euler angles (yaw, pitch, roll)
                    rotation_matrix, _ = cv.Rodrigues(rvec)
                    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                    singular = sy < 1e-6
                    if not singular:
                        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                        z_angle = 0

                    # Convert to degrees
                    euler_angles = np.degrees([x_angle, y_angle, z_angle])
                    text = f"Yaw: {euler_angles[2]:.1f} deg.\nPitch: {euler_angles[1]:.1f} deg.\nRoll: {euler_angles[0]:.1f} deg."

                    pos = tuple(image_points[0].astype(int))
                    for idx, line in enumerate(text.split('\n')):
                        cv.putText(frame, line, (pos[0], pos[1] + 20 * idx),
                                   cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

                    if save_csv:
                        # Save to CSV
                        timestamp = time.time()
                        marker_id = int(ids[i][0]) if ids is not None else -1

                        row = [timestamp, marker_id,
                            euler_angles[2],  # yaw
                            euler_angles[1],  # pitch
                            euler_angles[0],  # roll
                            tvec[0][0], tvec[1][0], tvec[2][0]]
                        csv_writer.writerow(row)

    cv.imshow("ArUco Rotation Detection", frame)
    if cv.waitKey(1) in [ord('q'), 27]:
        break

cap.release()
if save_csv:
    cv_file_closed = csv_file.close()
cv.destroyAllWindows()