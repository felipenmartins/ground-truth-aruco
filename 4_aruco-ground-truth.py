# ----------------- Description -----------------
import numpy as np
import cv2
from cv2 import aruco
import datetime
import csv
import time
import os
import platform

# ----------------- Constants -----------------
# Flags to control the display of the Cartesian plane and the route
isDrawCartesianPlaneActive = True
isShowRouteActive = True
isShowYPRActive   = False  # True: mostra yaw/pitch/roll; False: oculta o texto.

camera_index = 1  # 0 for the default camera, 1 for the external camera.

# New parameters for saving CSV and video
isSaveCSVActive   = True   # Flag to control CSV saving
isSaveVideoActive = True  # Flag to control video saving
datarate = 25
minSamplingPeriod = 1/datarate # 0.25   # Time interval in seconds for saving CSV data
marker_length = 0.06  # All markers: 6 cm per side, in meters (excluding white margin).
save_csv = False
tracked_marker_id = 3  # Marcador central: coordenadas, orientacao e trajeto.

# Distancias reais ENTRE OS CENTROS dos marcadores, em centimetros.
# Disposicao: M6 --- M7 (em cima), M8 --- M9 (embaixo).
reference_width_cm = 54.0   # Horizontal: M6-M7 ou M8-M9.
reference_height_cm = 41.0  # Vertical: M6-M8 ou M7-M9.

# Set the camera index based on the operating system
# os_name = platform.system()
# print(f"Operating System: {os_name}")
# if os_name == "Windows":
#     cam_index = 0  # Index of the camera (0 for the default camera - 1 for the external camera)
# else:
#     cam_index = 1

# ----------------- Main code -----------------
def main():
    cap = None
    out = None
    try:
        for name, value in (("reference_width_cm", reference_width_cm),
                            ("reference_height_cm", reference_height_cm)):
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Preencha {name} no inicio do codigo com a distancia "
                    "entre centros em centimetros (numero maior que zero)."
                )
        # Print the OpenCV version
        print(cv2.__version__)  # Print the OpenCV version

        # Load calibration data
        with np.load('calib_data/MultiMatrix.npz') as data:
            mtx = data['mtx']
            dist = data['dist']
        print("Camera calibration data loaded.")

        # Video input
        print(f"Opening camera {camera_index}...", flush=True)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            print("DirectShow could not open the camera; trying the default backend.", flush=True)
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_index}. Check its connection, camera_index, "
                "and whether another application is using it."
            )
        print("Camera opened. Press Q or Esc to quit.", flush=True)

        # Define ArUco marker dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)

        # Plane coordinates in cm: origin M6, +X toward M7, +Y toward M8.
        plane_unit = "cm"
        x = reference_width_cm
        y = reference_height_cm


        pts_known = np.array([
            [0,   0],    # M6 (top-left)
            [x,   0],    # M7 (top-right)
            [x,   y],    # M9 (bottom-right)
            [0,   y]     # M8 (bottom-left)
        ], dtype='float32')

        object_points = np.array([
            [-0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [-0.5, -0.5, 0]
        ]) * marker_length

        camera_matrix = np.array([[800, 0, 320],
                                  [0, 800, 240],
                                  [0, 0, 1]], dtype=np.float64)

        dist_coeffs   = np.zeros((5, 1))  # No distortion

        # Get video frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer, saving as output.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or use another codec
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'output_{timestamp}.avi'
        # Create the writer from the first actual frame's dimensions.
        out = None

        # Initialize CSV file
        if isSaveCSVActive:
            csv_filename = 'aruco_log.csv'
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Virtual_X', 'Virtual_Y'])
            last_saved_time = time.time()

        while True:
            # Capture camera frame
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Camera {camera_index} could not deliver a frame.")

            # Undistort the image using calibration data
            frame = cv2.undistort(frame, mtx, dist)

            # Convert the image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = detector.detectMarkers(gray)
            
            """ for i, corner in enumerate(corners):
                        image_points = corner[0].astype(np.float32)

                        # Calculate the 3D rotation and translation vectors
                        # solvePnP estimates the position and orientation of the camera in three-dimensional space
                        success, rvec, tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            camera_matrix,
                            dist_coeffs
                        )

                        # Check if solvePnP was successful
                        if success:
                            # Draw axis
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                            # Convert rvec to rotation matrix then to Euler angles (yaw, pitch, roll)
                            rotation_matrix, _ = cv2.Rodrigues(rvec)
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
                                cv2.putText(frame, line, (pos[0], pos[1] + 20 * idx),
                                           cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

                            if save_csv:
                                # Save to CSV
                                timestamp = time.time()
                                marker_id = int(ids[i]) if ids is not None else -1

                                row = [timestamp, marker_id,
                                    euler_angles[2],  # yaw
                                    euler_angles[1],  # pitch
                                    euler_angles[0],  # roll
                                    tvec[0][0], tvec[1][0], tvec[2][0]]"""

            if ids is not None:
                ids = np.asarray(ids).reshape(-1)  # Support scalar, flat and column IDs.
                corners = [np.asarray(c).reshape(1, 4, 2) for c in corners]

                # Check if all four known ArUco markers are detected
                # known_ids = [0, 1, 2, 3]  # Assume known marker IDs are 0, 1, 2, and 3
                known_ids = [6, 7, 9, 8]  # Perimeter order, matching pts_known.

                detected_known = [i for i in known_ids if i in ids]
                pose_marker_id = tracked_marker_id

                if len(detected_known) == len(known_ids):
                    for i, ids_corner in enumerate(zip(ids, corners)):
                        
                        ids_, corner = ids_corner
                        
                        if pose_marker_id != ids_:
                            continue
                        
                        
                        image_points = corner[0].astype(np.float32)

                        # Calculate the 3D rotation and translation vectors
                        # solvePnP estimates the position and orientation of the camera in three-dimensional space
                        success, rvec, tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            camera_matrix,
                            dist_coeffs
                        )

                        # Check if solvePnP was successful
                        if success:
                            # Draw axis
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                            # Convert rvec to rotation matrix then to Euler angles (yaw, pitch, roll)
                            rotation_matrix, _ = cv2.Rodrigues(rvec)
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

                            if isShowYPRActive:
                                pos = tuple(image_points[0].astype(int))
                                for idx, line in enumerate(text.split('\n')):
                                    cv2.putText(frame, line, (pos[0], pos[1] + 20 * idx),
                                               cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

                            if save_csv:
                                # Save to CSV
                                timestamp = time.time()
                                marker_id = int(ids[i]) if ids is not None else -1

                                row = [timestamp, marker_id,
                                    euler_angles[2],  # yaw
                                    euler_angles[1],  # pitch
                                    euler_angles[0],  # roll
                                    tvec[0][0], tvec[1][0], tvec[2][0]]
                    
                    # Find corner points of the four known markers
                    pts_detected = np.zeros((4, 2), dtype='float32')
                    for i in range(4):
                        idx = np.where(ids == known_ids[i])[0][0]
                        c = corners[idx][0]
                        pts_detected[i] = np.mean(c, axis=0)  # Compute the center of the corner points

                    # Compute the homography matrix from world coordinates to image coordinates
                    H, _ = cv2.findHomography(pts_known, pts_detected)

                    # Draw green lines connecting the markers of the borders
                    for i in range(4):
                        start_point = tuple(pts_detected[i].astype(int))
                        end_point = tuple(pts_detected[(i + 1) % 4].astype(int))
                        cv2.line(frame, start_point, end_point, (144, 238, 144), 2)

                        # Reference length comes from plane coordinates, not pixels.
                        length = np.linalg.norm(pts_known[i] - pts_known[(i + 1) % 4])
                        mid_point = (pts_detected[i] + pts_detected[(i + 1) % 4]) / 2
                        cv2.putText(frame, f"ref: {length:.2f} {plane_unit}", tuple(mid_point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)

                    # Detect the configured central marker.
                    if tracked_marker_id in ids:
                        
                        
                        
                        idx_5 = np.where(ids == tracked_marker_id)[0][0]
                        c_5 = corners[idx_5][0]
                        center_5 = np.mean(c_5, axis=0)  # Compute the center of the fifth marker

                        # Transform the fifth marker’s image coordinates back to the virtual square plane
                        pts_5_img = np.array([[center_5[0], center_5[1]]], dtype='float32')
                        pts_5_virtual = cv2.perspectiveTransform(np.array([pts_5_img]), np.linalg.inv(H))

                        if isDrawCartesianPlaneActive:
                            # Draw Cartesian plane at the virtual coordinates origin (0,0)
                            origin = np.array([[0, 0]], dtype='float32')
                            origin_img = cv2.perspectiveTransform(np.array([origin]), H)[0][0]

                            # Define axis length in virtual coordinates
                            axis_length = 20

                            # Define points for the x and y axes in virtual coordinates
                            x_axis = np.array([[axis_length, 0]], dtype='float32')
                            y_axis = np.array([[0, axis_length]], dtype='float32')

                            # Transform axis points to image coordinates
                            x_axis_img = cv2.perspectiveTransform(np.array([x_axis]), H)[0][0]
                            y_axis_img = cv2.perspectiveTransform(np.array([y_axis]), H)[0][0]

                            # Draw the x and y axes
                            grayColor = 128
                            cv2.arrowedLine(frame, tuple(origin_img.astype(int)), tuple(x_axis_img.astype(int)), (grayColor, grayColor, grayColor), 2)
                            cv2.arrowedLine(frame, tuple(origin_img.astype(int)), tuple(y_axis_img.astype(int)), (grayColor, grayColor, grayColor), 2)

                            # # Draw x and y labels
                            # cv2.putText(frame, 'X', tuple(x_axis_img.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # cv2.putText(frame, 'Y', tuple(y_axis_img.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                        if isSaveCSVActive:
                            # Save virtual coordinates to CSV every half second
                            now = time.time()
                            if now - last_saved_time >= minSamplingPeriod:
                                # Output the virtual coordinates of the fifth ArUco marker
                                current_time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                print(f"\033[94mCoordinates of the tag and marker at {current_time_str}: ({pts_5_virtual[0][0][0]:.3f}, {pts_5_virtual[0][0][1]:.3f})\033[0m")
                                with open(csv_filename, mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pts_5_virtual[0][0][0], pts_5_virtual[0][0][1]])
                                last_saved_time = now

                    for ids1, corners1 in zip(ids, corners):
                        color = (0, 0, 255) if ids1 == tracked_marker_id else (0, 255, 255)
                        cv2.polylines(
                            frame, [corners1.astype(np.int32)], True, color, 4, cv2.LINE_AA
                        )

                        corners1 = corners1.reshape(4, 2)
                        corners1 = corners1.astype(int)
                        top_right = corners1[0].ravel()
                        top_left = corners1[1].ravel()
                        bottom_right = corners1[2].ravel()
                        bottom_left = corners1[3].ravel()
                        # Calculate the center top point of the marker
                        top_center = np.mean([top_left, top_right], axis=0).astype(int)
                        
                        # Draw the marker ID at the top center
                        cv2.putText(
                            frame,
                            f"M{ids1}",
                            (top_center[0]-10, top_center[1] - 10),  # Adjust the height by subtracting 10 pixels
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 64, 255),  # Gray color
                            2,  # Thickness increased to make it bold
                            cv2.LINE_AA,
                        )

                    # Show plane coordinates for every detected marker.
                    for marker_id in ids:
                        idx = np.where(ids == marker_id)[0][0]
                        c = corners[idx][0]
                        center = np.mean(c, axis=0)  # Compute the center of the marker

                        # Transform the marker’s image coordinates back to the virtual square plane
                        pts_img = np.array([[center[0], center[1]]], dtype='float32')
                        pts_virtual = cv2.perspectiveTransform(np.array([pts_img]), np.linalg.inv(H))

                        # Draw the virtual coordinates of the marker at the bottom of the marker
                        bottom_center = np.mean(c[2:], axis=0).astype(int)
                        virtual_coords_text = f"X: {pts_virtual[0][0][0]:.2f} cm  Y: {pts_virtual[0][0][1]:.2f} cm"
                        text_size = cv2.getTextSize(virtual_coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        text_x = bottom_center[0] - text_size[0] // 2
                        text_y = bottom_center[1] + 20
                        cv2.putText(
                            frame,
                            virtual_coords_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),  # Color in BGR format (red)
                            1,
                            cv2.LINE_AA,
                        )

                    if tracked_marker_id in ids:
                        if isShowRouteActive:
                            # Draw the route that the robot went through
                            if 'route' not in locals():
                                route = []

                            # Append the current virtual coordinates of the fifth marker to the route
                            route.append((pts_5_virtual[0][0][0], pts_5_virtual[0][0][1]))

                            # Transform the route points to image coordinates and draw them
                            for i in range(1, len(route)):
                                pt1_virtual = np.array([[route[i-1]]], dtype='float32')
                                pt2_virtual = np.array([[route[i]]], dtype='float32')
                                pt1_img = cv2.perspectiveTransform(pt1_virtual, H)[0][0]
                                pt2_img = cv2.perspectiveTransform(pt2_virtual, H)[0][0]
                                cv2.line(frame, tuple(pt1_img.astype(int)), tuple(pt2_img.astype(int)), (0, 0, 255), 2)
                else:
                    known_ids += [tracked_marker_id]  # Add the fifth marker ID to the known IDs
                    # print(f"Known IDs: {known_ids}")
                    missing_ids = [i for i in known_ids if i not in ids]
                    print(f"\033[91mMissing markers: {missing_ids}\033[0m")
                    # Draw detected markers
                    for ids1, corners1 in zip(ids, corners):
                        color = (0, 255, 0) if ids1 in detected_known else (0, 0, 255)
                        cv2.polylines(frame, [corners1.astype(np.int32)], True, color, 4, cv2.LINE_AA)
                        corners1 = corners1.reshape(4, 2)
                        corners1 = corners1.astype(int)
                        top_right = corners1[0].ravel()
                        top_left = corners1[1].ravel()
                        bottom_right = corners1[2].ravel()
                        bottom_left = corners1[3].ravel()
                        # Calculate the center top point of the marker
                        top_center = np.mean([top_left, top_right], axis=0).astype(int)
                        # Draw the marker ID at the top center
                        cv2.putText(
                            frame,
                            f"M{ids1}",
                            (top_center[0] - 10, top_center[1] - 10),  # Adjust the height by subtracting 10 pixels
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 64, 255),  # Gray color
                            2,  # Thickness increased to make it bold
                            cv2.LINE_AA,
                        )

            # Write the current frame to the video file if video saving is active
            if isSaveVideoActive:
                if out is None:
                    frame_height, frame_width = frame.shape[:2]
                    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))
                    if not out.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output_filename}.")
                out.write(frame)

            # Display the frame
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('frame', frame)

            # Press 'q' or ESC to exit the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
