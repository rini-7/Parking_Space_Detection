# Parking Lot Detector

A real-time parking lot occupancy detection system using OpenCV and a pre-trained machine learning classifier.
![](parkingoutput.gif)

## What It Does

Reads a parking lot video feed, identifies individual parking spots using a mask image, and classifies each spot as **empty** or **occupied** in real time, drawing green boxes for empty spots and red boxes for occupied ones.

## How It Works

1. A grayscale **mask image** defines the parking spots — each spot is a white blob
2. OpenCV's `connectedComponentsWithStats` extracts bounding boxes from the mask
3. For each video frame, every spot is cropped and resized to 15×15×3
4. The cropped patch is flattened and passed into a pre-trained classifier (`model.p`)
5. Bounding boxes are drawn on the frame based on the prediction

## Project Structure
```bash
├── parking_main.py   # Main detection loop
├── util.py           # Bounding box extraction + classifier inference
├── model.p           # Pre-trained ML model (imported through pickle)
├── mask_crop.png     # Grayscale mask defining parking spot locations
└── parking_crop_loop.mp4  # Input parking lot video
```

## Requirements

```bash
pip install opencv-python scikit-image numpy scikit-learn
```

## Usage

1. Update the paths in `parking_main.py`:
```python
   mask_path = 'path/to/mask_crop.png'
   video_path = 'path/to/parking_video.mp4'
```
2. Run:
```bash
   python parking_main.py
```
3. Press `q` to quit.

## Model

`model.p` is a pre-trained scikit-learn classifier trained on labeled images of empty vs. occupied parking spots. 

Credit to the original model: https://www.youtube.com/watch?v=F-884J2mnOY&t=1691s

## Output
Green -> Empty

Red -> Occupied
