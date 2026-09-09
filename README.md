# Ground-truth ArUco

Computer-vision scripts for generating ArUco markers, calibrating a camera, and estimating marker position/orientation from a live video feed. Can be used as ground truth for indoors robot localization.

## What this repository does

This project provides a practical workflow for ArUco-based tracking:

1. Generate marker images.
2. Capture checkerboard images for camera calibration.
3. Compute calibration parameters.
4. Detect ArUco markers in real time.
5. Estimate pose (yaw/pitch/roll) and map a tracked marker into a 2D reference plane.

The repository is script-based (not a packaged library), focused on experimentation and data collection.

## Features

- Generate multiple `DICT_5X5_250` marker images (`0_generate_markers.py`)
- Capture checkerboard images from webcam (`1_capture_images.py`)
- Calibrate camera and save intrinsic parameters (`2_calibration_script.py`)
- Detect markers from live feed, including detection across multiple dictionaries (`3_markerdetection.py`, `5_detecting_any_aruco_marker.py`)
- Estimate marker rotation and log results to CSV (`6_detecting_rotation.py`)
- Compute homography-based plane coordinates and route tracking (`4_aruco-ground-truth.py`)

## Requirements

- Python 3.9+ (recommended)
- Webcam (internal or USB)
- Printed checkerboard and ArUco markers

Python dependencies:

- `opencv-contrib-python` (needed for `cv2.aruco`)
- `numpy`

## Installation

You need OpenCV:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install opencv-contrib-python numpy
```

Other than that, just copy the scripts (or clone this repository) to a folder in your computer.

## Usage

> Most scripts use configurable constants near the top (for example `camera_index`, marker sizes, IDs, and paths). Adjust those before running.

### 1) Generate markers

You will need to place ArUco markers on the corners of the area used for localization. 
First, adjust the size and number of markers to be generated in the script before running it. Then, run:

```bash
python 0_generate_markers.py
```

Expected output:
- Marker PNG files written to `docs/markers/`.

Print the markers and put 4 of them in the corners of the area.

### 2) Capture calibration images

The camera calibration is done with a checkerboard. You can generate a calibration checkerboard [here](https://markhedleyjones.com/projects/calibration-checkerboard-collection).

Adjust the parameters of your checkerboard in the script and run it:

```bash
python 1_capture_images.py
```

Place the checkerboard in different positions in front of the camera (it must be fully visible). Save about 15-20 images of the checkerboard in different positions and orientations.

Controls:
- `s`: save frame (only when checkerboard is detected)
- `q` or `Esc`: quit

Expected output:
- Captured images in the configured folder (`image_path` in the script).

### 3) Calibrate camera

The calibration scripts will use the images saved in the previous step and generate the camera calibration matrix.

```bash
python 2_calibration_script.py
```

Expected output:
- Calibration file: `calib_data/MultiMatrix.npz`

### 4) Run detection / tracking scripts

```bash
python 3_markerdetection.py
python 5_detecting_any_aruco_marker.py
python 6_detecting_rotation.py
python 4_aruco-ground-truth.py
```

Expected output (varies by script):
- Live visualization windows
- Optional CSV logs (for scripts with CSV enabled)
- Optional video output (`output_YYYYMMDD_HHMMSS.avi` in `4_aruco-ground-truth.py`)

## Notes and placeholders

- Directory names for captured images differ between scripts by default. Align these paths in your local setup.
- Some defaults (camera index, marker IDs, physical dimensions) are environment-specific and should be treated as project placeholders to tune for your hardware/layout.
