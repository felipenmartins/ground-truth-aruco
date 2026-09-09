import cv2 as cv
import os

Chess_Board_Dimensions = (7, 11)  # 8 x 12 squares have 7 x 11 inner corners.
camera_index = 1  # 0 for the default camera, 1 for the external camera.
image_path = "images_multiple_planes_test"
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(
        grayImage, boardDimension,
        cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK,
    )
    if ret:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret


def main():
    os.makedirs(image_path, exist_ok=True)
    n = 0
    image_number = 0
    print(f"Images directory: {os.path.abspath(image_path)}", flush=True)
    print(f"Opening camera {camera_index}...", flush=True)
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
        print("Camera opened. Press S to save a detected board; Q or Esc to quit.", flush=True)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Camera {camera_index} could not deliver a frame.")

            copyFrame = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            image, board_detected = detect_checker_board(
                frame, gray, criteria, Chess_Board_Dimensions
            )
            cv.putText(
                image, f"saved_img : {n}", (30, 40), cv.FONT_HERSHEY_PLAIN,
                1.4, (0, 255, 0), 2, cv.LINE_AA,
            )
            cv.imshow("frame", image)
            cv.imshow("copyFrame", copyFrame)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if not board_detected:
                    print("Image not saved: the full checkerboard must be detected.")
                    continue
                # Keep images from previous capture sessions.
                filename = os.path.join(image_path, f"image{image_number}.png")
                while os.path.exists(filename):
                    image_number += 1
                    filename = os.path.join(image_path, f"image{image_number}.png")
                if not cv.imwrite(filename, copyFrame):
                    raise RuntimeError(f"Could not save image to {filename}.")
                print(f"Saved {filename}")
                image_number += 1
                n += 1
    finally:
        cap.release()
        cv.destroyAllWindows()
        print("Total saved images this session:", n)


if __name__ == "__main__":
    main()
