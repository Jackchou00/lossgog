"""
Train a SPGOG model and evaluate its performance.

Usage:
    python scripts/train_sp_gog.py

Authors: Jack Chou
Date: Dec 4, 2025
"""

import numpy as np

from loss_gog import read_cs2000_csv
from loss_gog.exp.sp_gog import make_sp_gog, evaluate_sp_gog, rgb_to_xyz_sp_gog


# White point from measurements (D65 illuminant on display)
WHITE_POINT = np.array([474.94, 503.27, 579.53])


def main():
    # --- Load large dataset (3375 points, 15x15x15) for building model ---
    large_file = "measured_data/cs2000_lut_measurements_20251124_163713.csv"
    large_data = read_cs2000_csv(large_file)
    large_rgb = large_data["RGB"] / 255.0  # Normalize to [0, 1]
    large_xyz = large_data["XYZ"] / WHITE_POINT[1]
    print(f"Large dataset: {large_rgb.shape[0]} points (15x15x15 grid)")

    # --- Load small dataset (216 points, 6x6x6) for validation ---
    small_file = "measured_data/cs2000_lut_measurements_20251124_194618.csv"
    small_data = read_cs2000_csv(small_file)
    small_rgb = small_data["RGB"] / 255.0  # Normalize to [0, 1]
    small_xyz = small_data["XYZ"] / WHITE_POINT[1]
    print(f"Small dataset: {small_rgb.shape[0]} points (6x6x6 grid)")

    # --- Build GOG Model ---
    print("\n" + "=" * 60)
    print("Building SP GOG Model (RGB -> XYZ)")
    print("=" * 60)

    mode = "de2000"  # Options: "xyz", "de2000", "sucs"
    D0 = 0.045
    sp_gog_model = make_sp_gog(large_rgb, large_xyz, mode=mode, d0=D0, train_d0=True, verbose=True)

    # Print model parameters
    print("\nSP GOG Model Parameters:")
    print(sp_gog_model)

    # --- Evaluate on training set ---
    train_metrics = evaluate_sp_gog(sp_gog_model, large_rgb, large_xyz)
    print(f"\nTraining set (large, {large_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {train_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {train_metrics['max_delta_e']:.2f}")
    print(f"  95th percentile Delta E: {np.percentile(train_metrics['delta_e'], 95):.2f}")
    
    # Find worst points
    worst_idx = np.argsort(train_metrics['delta_e'])[-5:]
    print("\n  Worst 5 points:")
    for idx in worst_idx[::-1]:
        print(f"    RGB: {large_rgb[idx]}, ΔE: {train_metrics['delta_e'][idx]:.2f}")

    # --- Evaluate on validation set ---
    val_metrics = evaluate_sp_gog(sp_gog_model, small_rgb, small_xyz)
    print(f"\nValidation set (small, {small_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {val_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {val_metrics['max_delta_e']:.2f}")

    # test black and white
    D0 = sp_gog_model["d0"]  # Use trained D0
    test_rgb = np.array([[0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0]])
    test_xyz = rgb_to_xyz_sp_gog(test_rgb, sp_gog_model)
    test_xyz *= WHITE_POINT[1]
    print("\nTest RGB to XYZ:")
    for i in range(test_rgb.shape[0]):
        print(f"  RGB: {test_rgb[i]} -> XYZ: {test_xyz[i]}")


if __name__ == "__main__":
    main()
