# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video object detection application using modern YOLO models (v8-v11) with the supervision library for annotations and tracking.

## Setup

```bash
pip install -r requirements.txt
```

The first run will automatically download the YOLO model weights.

## Run

```bash
# Webcam (default)
python video_detector.py

# Video file
python video_detector.py --source video.mp4

# With output saved
python video_detector.py --source video.mp4 --output resultado.mp4

# Different model
python video_detector.py --source video.mp4 --model yolov8n.pt
```

## Architecture

**Pipeline**: Video frame → YOLO inference → supervision Detections → ByteTrack → Annotators → Output frame

### Components

1. **YOLO Model** (Ultralytics): Object detection inference
   - Models: `yolo11n.pt`, `yolov8n.pt`, `yolov9n.pt`, `yolov10n.pt`
   - Variants: n (nano), s (small), m (medium), l (large), x (extra-large)

2. **ByteTrack** (supervision): Multi-object tracking across frames
   - Assigns persistent IDs to detected objects
   - Handles occlusions and re-identification

3. **Annotators** (supervision):
   - `BoxAnnotator`: Bounding boxes
   - `LabelAnnotator`: Class names, confidence, tracker IDs
   - `TraceAnnotator`: Motion trails

## Key Functions

- `run_detection()`: Main entry point - handles webcam and video file modes
- `process_frame()`: Single frame processing pipeline
- `create_annotators()`: Initializes supervision annotators

## Video Processing Modes

**Webcam**: Uses OpenCV VideoCapture with real-time display
**File**: Uses `sv.process_video()` for efficient batch processing with optional `sv.get_video_frames_generator()` for display

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video path or webcam index |
| `--model` | `yolo11n.pt` | YOLO model to use |
| `--output` | None | Save processed video |
| `--confidence` | `0.5` | Detection threshold |
| `--no-track` | False | Disable object tracking |
| `--no-show` | False | Don't display video |
