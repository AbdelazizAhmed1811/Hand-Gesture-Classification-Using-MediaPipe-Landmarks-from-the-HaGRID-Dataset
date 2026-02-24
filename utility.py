"""
utility.py – Shared normalization for Hand Gesture Classification
=================================================================
Provides two public helpers that implement the **same** algorithm
(center on wrist → scale to [-1, 1]):

* ``normalize_landmarks_df``  – operates on a pandas DataFrame  (training / notebook)
* ``normalize_landmarks_row`` – operates on a single flat list   (video inference)
"""

import numpy as np


# ── DataFrame-level normalization (used during training) ────────────────


def normalize_landmarks_df(df):
    """
    Normalize hand landmark coordinates to match LandmarkNormalizer logic.
    - Centers (x, y) relative to the wrist (landmark 1).
    - Scales (x, y) by the distance to the middle finger tip (landmark 13).
    - Z coordinates are left untouched.
    """
    df_norm = df.copy()

    x_cols = [f"x{i}" for i in range(1, 22)]
    y_cols = [f"y{i}" for i in range(1, 22)]

    df_norm[x_cols] = df_norm[x_cols].subtract(df_norm["x1"], axis=0)
    df_norm[y_cols] = df_norm[y_cols].subtract(df_norm["y1"], axis=0)


    scale_factor = np.sqrt(df_norm["x13"]**2 + df_norm["y13"]**2)
    
    scale_factor = scale_factor.replace(0, 1) 

    df_norm[x_cols] = df_norm[x_cols].div(scale_factor, axis=0)
    df_norm[y_cols] = df_norm[y_cols].div(scale_factor, axis=0)

    return df_norm




def normalize_landmarks_row(landmark_row: list[float]) -> np.ndarray:
    """
    Normalize a single flat list of 63 landmark values.
    Matches LandmarkNormalizer logic: Centers and scales ONLY X and Y.
    """
    arr = np.array(landmark_row, dtype=np.float64)

    xs = arr[0::3]
    ys = arr[1::3]

    wrist_x = xs[0]
    wrist_y = ys[0]

    xs -= wrist_x
    ys -= wrist_y


    scale_factor = np.linalg.norm([xs[12], ys[12]])
    
    if scale_factor > 0:
        xs /= scale_factor
        ys /= scale_factor


    arr[0::3] = xs
    arr[1::3] = ys
    
    return arr





HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]


FINGER_COLORS = {
    'thumb':  '#FF6B6B',   
    'index':  '#4ECDC4',   
    'middle': '#45B7D1',   
    'ring':   '#96CEB4',   
    'pinky':  '#FFEAA7',   
    'palm':   '#DDA0DD'    
}


def plot_hand_landmarks(row, ax, title=''):
    xs = []
    ys = []

    for i in range(1, 22):
        xs.append(row[f'x{i}'])
        ys.append(row[f'y{i}'])

    # Draw connections
    for (i, j) in HAND_CONNECTIONS:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], linewidth=2, alpha=0.8)

    # Draw landmark points
    ax.scatter(xs, ys, c='white', edgecolors='black',
               s=40, zorder=5, linewidths=1)

    # Highlight wrist and fingertips
    tips = [0, 4, 8, 12, 16, 20]
    ax.scatter([xs[t] for t in tips], [ys[t] for t in tips],
               c='#FF4757', edgecolors='black', s=70, zorder=6, linewidths=1)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()  
    ax.set_aspect('equal')
    ax.axis('off')
