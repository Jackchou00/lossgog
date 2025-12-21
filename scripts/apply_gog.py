"""
Apply GOG model to XYZ to predict display RGB and back to XYZ.

This script demonstrates the GOG model pipeline:
1. XYZ → Display RGB (via inverse GOG model)

Usage:
    uv run scripts/apply_gog.py

Authors: Jack Chou
Date: Dec 16, 2025
"""

import os
import numpy as np
from scipy.io import loadmat
from loss_gog import xyz_to_rgb_gog


def load_gog_model(model_path: str) -> dict:
    """Load GOG model parameters from .npz file.

    Parameters:
        model_path: Path to the .npz file containing GOG model parameters.

    Returns:
        Dict with keys: 'gain', 'offset', 'gamma', 'matrix'.
    """
    data = np.load(model_path)
    return {
        "gain": data["gain"],
        "offset": data["offset"],
        "gamma": data["gamma"],
        "matrix": data["matrix"],
    }


def main():
    # Load GOG model
    model_path = "mate80pm_results/gog_optimized.npz"

    if not os.path.exists(model_path):
        print(f"GOG model not found: {model_path}")
        return

    gog_model = load_gog_model(model_path)

    # input should be in shape (N, 3)
    mat_path = "XYZ_2020_100nit_1931.mat"
    xyz_input = loadmat(mat_path)["XYZ_m"]

    rgb_display = xyz_to_rgb_gog(xyz_input, gog_model)

    combined = np.hstack((xyz_input, rgb_display))
    np.savetxt(
        "gog_mate80pm_xyzm_100.csv",
        delimiter=",",
        X=combined,
        fmt="%.2f,%.2f,%.2f,%.3f,%.3f,%.3f",
        header="X,Y,Z,R,G,B",
        comments="",
    )


if __name__ == "__main__":
    main()
