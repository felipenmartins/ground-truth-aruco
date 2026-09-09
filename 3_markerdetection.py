# Detect ArUco markers in a live camera feed.

import os

import cv2 as cv
from cv2 import aruco
import numpy as np

camera_index = 1  # 0 for the default camera, 1 for the external camera.
isShowCoordinatesActive = True

def main():
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    param_markers = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(marker_dict, param_markers)

    print(f"Opening camera {camera_index}...", flush=True)
    # Prefer DirectShow on Windows to avoid slow automatic backend selection.
    backend = cv.CAP_DSHOW if os.name == "nt" else cv.CAP_ANY
    cap = cv.VideoCapture(camera_index, backend)
    if not cap.isOpened() and backend != cv.CAP_ANY:
        cap.release()
        print("DirectShow could not open the camera; trying the default backend.", flush=True)
        cap = cv.VideoCapture(camera_index)

    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_index}. Check its connection, "
                "camera_index, and whether another application is using it."
            )

        print("Camera opened. Press Q or Esc to quit.", flush=True)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Camera {camera_index} could not deliver a frame.")

            # Detect at the captured resolution; enlarging adds processing cost.
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, _ = detector.detectMarkers(gray_frame)

            if marker_IDs is not None and len(marker_corners) > 0:
                # Support both flat IDs and the (N, 1) arrays returned by OpenCV.
                for marker_id, corners in zip(np.asarray(marker_IDs).reshape(-1), marker_corners):
                    points = np.asarray(corners).reshape(4, 2).astype(np.int32)
                    cv.polylines(frame, [points], True, (0, 255, 255), 4, cv.LINE_AA)

                    if isShowCoordinatesActive:
                        for point in points:
                            cv.circle(frame, tuple(map(int, point)), 5, (0, 0, 255), -1)

                    cv.putText(
                        frame,
                        f"id: {int(marker_id)}",
                        tuple(map(int, points[0])),
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (200, 100, 0),
                        2,
                        cv.LINE_AA,
                    )

            cv.imshow("frame", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
