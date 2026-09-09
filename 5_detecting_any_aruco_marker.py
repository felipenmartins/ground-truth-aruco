import cv2 as cv
from cv2 import aruco
import numpy as np

# Show detected corners
show_coordinates = True

# List of all available predefined ArUco dictionaries
aruco_dicts = [
    aruco.DICT_4X4_50, aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
    aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
    aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
    aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
    aruco.DICT_ARUCO_ORIGINAL, aruco.DICT_APRILTAG_16h5, aruco.DICT_APRILTAG_25h9,
    aruco.DICT_APRILTAG_36h10, aruco.DICT_APRILTAG_36h11
]

# Initialize webcam
cap = cv.VideoCapture(0)

# ArUco detector parameters
parameters = aruco.DetectorParameters()
detectors = [aruco.ArucoDetector(aruco.getPredefinedDictionary(d), parameters) for d in aruco_dicts]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Increase resolution for better detection
    frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for detector in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if corners:
            if show_coordinates:
                for corner in corners:
                    corner = corner.reshape(4, 2)
                    for point in corner:
                        cv.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)

            for id_, corner in zip(ids, corners):
                cv.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
                corner = corner.reshape(4, 2).astype(int)
                top_right = corner[0].ravel()
                cv.putText(
                    frame,
                    f"id: {id_[0]}",
                    tuple(top_right),
                    cv.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,
                    cv.LINE_AA
                )

    # Show result
    cv.imshow("ArUco Marker Detection", frame)

    key = cv.waitKey(1)
    if key == ord("q") or key == 27:
        break

cap.release()
cv.destroyAllWindows()
