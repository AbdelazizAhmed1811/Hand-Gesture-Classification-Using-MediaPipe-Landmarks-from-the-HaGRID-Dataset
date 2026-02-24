"""
Hand Gesture Video Processing Script
=====================================
Uses MediaPipe HandLandmarker (Tasks API) to detect hand landmarks and
KNN / SVM / Random Forest models to classify gestures.

Generates **one output video per model** with predictions stabilized
using a rolling-window mode filter.

All output videos use H.264 codec for browser compatibility.

Usage:
    ./hand_gesture/bin/python process_video.py
"""

import os
import sys
import cv2
import joblib
import numpy as np
import pandas as pd
import subprocess
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    _BaseOptions,
    _RunningMode,
)

from utility import normalize_landmarks_row


FEATURE_NAMES = [f"{axis}{i}" for i in range(1, 22) for axis in ('x', 'y', 'z')]

HAND_CONNECTIONS = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]

MODEL_FILES = {
    # "KNN": "models/knn_best_model.joblib",
    # "SVM": "models/svm_best_model.joblib",
    "RF":  "models/random_forest_best_model.joblib",
}

STABILIZATION_WINDOW = 15


def convert_to_h264(input_path: str) -> None:
    """Re-encode a video to H.264/AAC using ffmpeg for browser compatibility.
    Replaces the original file in-place."""
    tmp_path = input_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",   
        "-movflags", "+faststart",  
        "-an", 
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.isfile(tmp_path):
        os.replace(tmp_path, input_path)
        print(f"  ✓ Converted to H.264: {input_path}")
    else:
        print(f"  ⚠ ffmpeg conversion failed for {input_path}")
        if result.stderr:
            print(f"    {result.stderr[-200:]}")
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


def draw_landmarks_on_frame(
    frame: np.ndarray,
    landmarks: list,
    width: int,
    height: int,
) -> None:
    """Draw landmarks and connections on *frame* (in-place)."""
    pts = []
    for lm in landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        pts.append((cx, cy))

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(pts) and end_idx < len(pts):
            cv2.line(frame, pts[start_idx], pts[end_idx], (250, 44, 250), 2)

    for cx, cy in pts:
        cv2.circle(frame, (cx, cy), 4, (121, 22, 76), -1)
        cv2.circle(frame, (cx, cy), 2, (250, 44, 250), -1)


def get_stabilized_prediction(window: deque) -> str:
    """Return the mode (most frequent) prediction from the window."""
    if not window:
        return "..."
    counter = Counter(window)
    return counter.most_common(1)[0][0]


def process_single_model(
    input_path: str,
    output_path: str,
    model_name: str,
    model,
    landmarker_model: str,
) -> None:
    """Process a video with a single model and write the output."""

    options = HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=landmarker_model),
        running_mode=_RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  Error: could not open video '{input_path}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write with mp4v first, then re-encode to H.264 via ffmpeg
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  Error: could not create output video '{output_path}'")
        cap.release()
        return

    prediction_window: deque = deque(maxlen=STABILIZATION_WINDOW)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx * (1000.0 / fps))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_landmarks_on_frame(frame, hand_lms, width, height)

                landmark_row: list[float] = []
                for lm in hand_lms:
                    landmark_row.extend([lm.x, lm.y, lm.z])

                X_pred = normalize_landmarks_row(landmark_row).reshape(1, -1)
                
                X_pred_df = pd.DataFrame(X_pred, columns=FEATURE_NAMES)

                try:
                    pred = model.predict(X_pred_df)[0]
                    prediction_window.append(str(pred))
                except Exception as exc:
                    prediction_window.append("Error")
                    print(f"  [frame {frame_idx}] {model_name} error: {exc}")

        stabilized_label = get_stabilized_prediction(prediction_window)
        display_text = f"{model_name}: {stabilized_label}"

        cv2.putText(
            frame, display_text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            frame, display_text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_idx % 200 == 0 or frame_idx == total_frames:
            print(f"    {model_name}: {frame_idx}/{total_frames} frames …")

    cap.release()
    writer.release()
    landmarker.close()
    print(f"  ✓ {model_name} rendered → {output_path}")

    # Re-encode to H.264 for browser compatibility
    convert_to_h264(output_path)


def process_video(
    input_path: str = "videos/input/input_video.mp4",
    landmarker_model: str = "models/hand_landmarker.task",
) -> None:
    """Process the input video with each model separately."""

    print("=" * 60)
    print("  Hand Gesture Video Processing (per-model)")
    print("=" * 60)

    if not os.path.isfile(input_path):
        print(f"Error: input video not found at '{input_path}'")
        sys.exit(1)

    if not os.path.isfile(landmarker_model):
        print(f"Error: hand landmarker model not found at '{landmarker_model}'")
        sys.exit(1)

    models: dict = {}
    for name, path in MODEL_FILES.items():
        if not os.path.isfile(path):
            print(f"  ⚠ Model file not found: '{path}' — skipping {name}")
            continue
        models[name] = joblib.load(path)
        print(f"  Loaded {name} from '{path}'")

    if not models:
        print("Error: no models found!")
        sys.exit(1)

    output_files = {
        # "KNN": "videos/output/output_knn.mp4",
        # "SVM": "videos/output/output_svm.mp4",
        "RF":  "videos/output/output_rf.mp4",
    }

    for model_name, model in models.items():
        output_path = output_files[model_name]
        print(f"\n  Processing {model_name} → {output_path}")
        process_single_model(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            model=model,
            landmarker_model=landmarker_model,
        )

    print(f"\n{'=' * 60}")
    print(f"  All done! Browser-compatible output videos:")
    for name in models:
        print(f"    • {output_files[name]}")
    print(f"{'=' * 60}")
