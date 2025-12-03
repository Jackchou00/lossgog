"""
Train GOG model for XGIMI projector characterization.

Usage:
    uv run src/xgimi_train.py

This script:
 1. Loads XGIMI measurement data (GOG_XGIMI.csv + Rgb96.mat)
 2. Trains a GOG model using 72 samples, validates on 24 samples
 3. Saves results to xgimi_results/

Author: Jack Chou
Date: Dec 1, 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_rgb96(mat_path: str) -> np.ndarray:
    """Load RGB values from .mat file.

    Returns:
        (96, 3) array with RGB values in [0, 1].
    """
    mat = loadmat(mat_path)
    rgb = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            if v.size == 0:
                continue
            if v.shape == (96, 3):
                rgb = v.astype(float)
                break
            if v.shape == (3, 96):
                rgb = v.T.astype(float)
                break
            if v.shape[0] == 96 and v.ndim == 2 and v.shape[1] >= 3:
                rgb = v[:, :3].astype(float)
                break

    if rgb is None:
        raise ValueError("Failed to find a 96x3 RGB array in mat file")

    # Normalize if values are in 0-255 range
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    return rgb


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "xgimi_data")
    mat_path = os.path.join(data_dir, "Rgb96.mat")

    out_dir = os.path.join(repo_root, "xgimi_results")

    rgb96 = load_rgb96(mat_path)

    # Prepare data
    n = 96
    rgb_all = np.clip(rgb96[:n, :], 0.0, 1.0)

    # Split: 72 train, 24 test
    train_n = 72
    rgb_train = rgb_all[:train_n, :]
    rgb_test = rgb_all[train_n:n, :]

    # Plot RGB values in two figures, one for train, one for test
    # train: 18 cols, 4 rows, test: 6 cols, 4 rows
    fig_train, axes_train = plt.subplots(4, 18, figsize=(18, 4))
    fig_test, axes_test = plt.subplots(4, 6, figsize=(6, 4))
    for i in range(train_n):
        r, c = divmod(i, 18)
        axes_train[r, c].imshow(np.ones((10, 10, 3)) * rgb_train[i, :].reshape(1, 1, 3))
        axes_train[r, c].axis("off")
    
    for i in range(n - train_n):
        r, c = divmod(i, 6)
        axes_test[r, c].imshow(np.ones((10, 10, 3)) * rgb_test[i, :].reshape(1, 1, 3))
        axes_test[r, c].axis("off")
    
    fig_train.suptitle("Training RGB Samples (72)")
    fig_test.suptitle("Testing RGB Samples (24)")
    plt.tight_layout()
    fig_train.savefig(os.path.join(out_dir, "rgb_samples_train.jpg"), dpi=300)
    fig_test.savefig(os.path.join(out_dir, "rgb_samples_test.jpg"), dpi=300)
    plt.close(fig_train)
    plt.close(fig_test)

if __name__ == "__main__":
    main()
