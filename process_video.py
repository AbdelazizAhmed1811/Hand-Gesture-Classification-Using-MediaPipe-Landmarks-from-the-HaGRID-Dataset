"""
Live Hand Gesture Recognition & Recording Script
==============================================
Detects hand landmarks from a live webcam feed, classifies gestures 
in real-time using a Random Forest model, and saves the output.

The final video is automatically re-encoded to H.264/AAC using FFmpeg 
to ensure maximum compatibility across all devices and browsers.

Usage:
    Press and hold 'q' for a split second in the video window to stop and quit.
"""

import os
import sys
import cv2
import time
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

# Assumes utility.py is in the same directory
from utility import normalize_landmarks_row

FEATURE_NAMES = [f"{axis}{i}" for i in range(1, 22) for axis in ('x', 'y', 'z')]

HAND_CONNECTIONS = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]

MODEL_FILES = {
    "RF": "models/random_forest_best_model.joblib"
}

STABILIZATION_WINDOW = 15


def convert_to_h264(input_path: str) -> None:
    """Re-encode a video to H.264 using ffmpeg for maximum compatibility."""
    print(f"\n[Processing] Converting '{input_path}' to H.264 (this may take a few seconds)...")
    tmp_path = input_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",   
        "-movflags", "+faststart",  
        "-an", # No audio 
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.isfile(tmp_path):
        os.replace(tmp_path, input_path)
        print(f"  ✓ Successfully converted to H.264: {input_path}")
    else:
        print(f"  ⚠ ffmpeg conversion failed for {input_path}")
        if result.stderr:
            print(f"    {result.stderr[-200:]}")
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


def draw_landmarks_on_frame(frame: np.ndarray, landmarks: list, width: int, height: int) -> None:
    """Draw landmarks and connections on *frame* (in-place)."""
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

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
    return Counter(window).most_common(1)[0][0]


def run_live_prediction(
    landmarker_model: str = "models/hand_landmarker.task",
    output_path: str = "videos/output/output_rf.mp4"
) -> None:
    """Run real-time gesture recognition and save the output video."""

    print("=" * 60)
    print("  Live Hand Gesture Recognition & Recording")
    print("=" * 60)

    # 1. Verify and Load Files
    if not os.path.isfile(landmarker_model):
        print(f"Error: hand landmarker model not found at '{landmarker_model}'")
        sys.exit(1)

    rf_path = MODEL_FILES["RF"]
    if not os.path.isfile(rf_path):
        print(f"Error: ML model not found at '{rf_path}'")
        sys.exit(1)

    print("Loading ML model...")
    model = joblib.load(rf_path)
    model_name = "RF"

    # 2. Initialize MediaPipe HandLandmarker
    options = HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=landmarker_model),
        running_mode=_RunningMode.VIDEO, 
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. Initialize Webcam & VideoWriter
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        sys.exit(1)

    # Grab camera properties for the video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0 # Default fallback for webcams

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n  ✓ Webcam ready. Recording to '{output_path}'.")
    print("  ✓ Press and hold 'q' in the video window to stop and save.")
    
    prediction_window: deque = deque(maxlen=STABILIZATION_WINDOW)

    # 4. Main Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip horizontally for a natural mirror-like experience
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        # Detect landmarks
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

        # Stabilize and display prediction
        stabilized_label = get_stabilized_prediction(prediction_window)
        display_text = f"{model_name}: {stabilized_label}"

        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Write frame to video file
        writer.write(frame)

        # Show frame on screen
        cv2.imshow("Live Gesture Recognition", frame)

        # Check for 'q' press - waitKey(1) waits for 1ms. 
        # If 'q' is pressed, break immediately.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[Stopped] 'q' pressed. Shutting down camera...")
            break

    # 5. Cleanup 
    # Release the camera and destroy windows BEFORE FFmpeg starts
    # so the user knows the recording has officially stopped.
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    landmarker.close()
    
    # Run the FFmpeg conversion to ensure it plays everywhere
    convert_to_h264(output_path)
    print("\nExited gracefully.")
