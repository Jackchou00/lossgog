"""
Train a GOG model and evaluate its performance.

Usage:
    python scripts/train_gog.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np

from lossgog import read_cs2000_csv, make_gog, evaluate_gog, plot_delta_e_histogram


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
    print("Building GOG Model (RGB -> XYZ)")
    print("=" * 60)

    mode = "xyz"  # Options: "xyz", "de2000", "sucs"
    gog_model = make_gog(large_rgb, large_xyz, mode=mode, verbose=True)

    # Print model parameters
    print("\nGOG Model Parameters:")
    print(f"  Gain:   {gog_model['gain']}")
    print(f"  Offset: {gog_model['offset']}")
    print(f"  Gamma:  {gog_model['gamma']}")
    print(f"  Matrix:\n{gog_model['matrix']}")

    # --- Evaluate on training set ---
    train_metrics = evaluate_gog(gog_model, large_rgb, large_xyz)
    print(f"\nTraining set (large, {large_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {train_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {train_metrics['max_delta_e']:.2f}")

    # --- Evaluate on validation set ---
    val_metrics = evaluate_gog(gog_model, small_rgb, small_xyz)
    print(f"\nValidation set (small, {small_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {val_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {val_metrics['max_delta_e']:.2f}")

    # --- Save histogram ---
    plot_delta_e_histogram(
        val_metrics["delta_e"],
        f"results/gog_delta_e_histogram_{mode}.svg",
    )


if __name__ == "__main__":
    main()
