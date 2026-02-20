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
    Normalize hand landmark coordinates to be translation- and scale-invariant.

    Steps
    -----
    1. Center all landmarks relative to the wrist (landmark 1).
    2. Scale by the max absolute coordinate value so all values ∈ [-1, 1].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns x1..x21, y1..y21, z1..z21, and 'label'.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with the same structure.
    """
    df_norm = df.copy()

    x_cols = [f"x{i}" for i in range(1, 22)]
    y_cols = [f"y{i}" for i in range(1, 22)]
    z_cols = [f"z{i}" for i in range(1, 22)]

    # Step 1: Center on the wrist (landmark 1)
    df_norm[x_cols] = df_norm[x_cols].subtract(df_norm["x1"], axis=0)
    df_norm[y_cols] = df_norm[y_cols].subtract(df_norm["y1"], axis=0)
    df_norm[z_cols] = df_norm[z_cols].subtract(df_norm["z1"], axis=0)

    # Step 2: Scale to [-1, 1] per sample
    all_landmark_cols = x_cols + y_cols + z_cols
    max_abs = df_norm[all_landmark_cols].abs().max(axis=1)
    max_abs = max_abs.replace(0, 1)  # avoid division by zero
    df_norm[all_landmark_cols] = df_norm[all_landmark_cols].div(max_abs, axis=0)

    return df_norm


# ── Single-row normalization (used during video inference) ──────────────

def normalize_landmarks_row(landmark_row: list[float]) -> np.ndarray:
    """
    Normalize a single flat list of 63 landmark values (x1,y1,z1, … ,x21,y21,z21).

    Uses the exact same algorithm as :func:`normalize_landmarks_df`:
    center on wrist (indices 0,1,2) then scale to [-1, 1].

    Parameters
    ----------
    landmark_row : list[float]
        Flat list of 63 floats in [x1, y1, z1, x2, y2, z2, …] order.

    Returns
    -------
    np.ndarray
        Shape (63,) with normalized values.
    """
    arr = np.array(landmark_row, dtype=np.float64)
    wrist_x, wrist_y, wrist_z = arr[0], arr[1], arr[2]

    xs = arr[0::3]
    ys = arr[1::3]
    zs = arr[2::3]

    xs -= wrist_x
    ys -= wrist_y
    zs -= wrist_z

    all_coords = np.concatenate([xs, ys, zs])
    max_abs = np.max(np.abs(all_coords))
    if max_abs > 0:
        xs /= max_abs
        ys /= max_abs
        zs /= max_abs

    arr[0::3] = xs
    arr[1::3] = ys
    arr[2::3] = zs
    return arr
