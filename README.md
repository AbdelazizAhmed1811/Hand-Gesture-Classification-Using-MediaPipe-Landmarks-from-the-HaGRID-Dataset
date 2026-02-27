# Hand Gesture Classification Using MediaPipe Landmarks

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-teal.svg)](https://google.github.io/mediapipe/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/AbdelazizAhmed1811/Hand-Gesture-Classification-Using-MediaPipe-Landmarks-from-the-HaGRID-Dataset/graphs/commit-activity)

![Hand Landmarks](hand_landmarks.png)

## Overview
This repository contains a comprehensive pipeline for **Hand Gesture Classification** using localized hand landmarks. By utilizing **MediaPipe** to extract 21 key coordinate points from the **HaGRID (Hand Gesture Recognition Image Dataset)**, we transform raw images into high-dimensional geometric data, enabling efficient and accurate gesture recognition.

## Key Features
- **Landmark Extraction**: Uses MediaPipe to identify 21 3D hand landmarks.
- **Robust Normalization**: All coordinates are centered on the wrist and scaled relative to the hand size to ensure rotation and scale invariance.
- **Model Training**: Detailed experimentation with multiple classifiers (KNN, SVM, Random Forest).
- **MLflow Integration**: Full experiment tracking, including hyperparameter tuning and model versioning.
- **Real-time Inference**: Processing pipeline for classifying gestures in video streams.

## Dataset: HaGRID
The project utilizes the **HaGRID** dataset, a large-scale collection of hand gestures. Specifically, this implementation focuses on **18 distinct gesture classes**, including:
*   `call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`.

Each sample consists of **63 features** (21 landmarks × 3 coordinates: x, y, z).

## Model Selection: Random Forest
After evaluating several architectural approaches, **Random Forest** was selected as the primary classifier for this project.

### Why Random Forest?
The choice of Random Forest is justified by its exceptional **robustness** and **reliability** in handling landmark-based geometric data:
1.  **Robustness to Outliers**: Hand landmark data can occasionally contain noise due to occlusions or MediaPipe tracking errors. Random Forest's ensemble nature (Bagging) prevents individual noisy trees from significantly impacting the final prediction.
2.  **Reliability & Generalization**: By averaging multiple decision trees, the model significantly reduces the risk of overfitting, ensuring it generalizes well to new users and different camera angles.

## Installation & Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/AbdelazizAhmed1811/Hand-Gesture-Classification-Using-MediaPipe-Landmarks-from-the-HaGRID-Dataset.git
    cd Hand-Gesture-Classification-Using-MediaPipe-Landmarks-from-the-HaGRID-Dataset
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Exploration**:
    Open `Hand-Gesture-Classification.ipynb` to view the data analysis and training pipeline.

## Project Structure
- `Hand-Gesture-Classification.ipynb`: Main research and development notebook.
- `utility.py`: Shared helper functions for normalization and visualization.
- `mlflow_utils.py`: Utilities for hyperparameter tuning and model logging.
- `process_video.py`: Script for gesture classification on video files.
- `hand_landmarks.png`: Reference diagram for the 21 MediaPipe landmarks.

---
*Developed as part of the ITI ML Project Suite.*
