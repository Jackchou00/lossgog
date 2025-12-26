"""
Train a GOG model and evaluate its performance.

Usage:
    python scripts/train_gog.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

import rich
from lossgog import read_cs2000_csv, make_gog, evaluate_gog


def main():
    measured_data_path = "xgimi/xgimi_96_gog.csv"

    # --- Load measured data ---
    print("Loading Measured Data")
    data = read_cs2000_csv(measured_data_path, spectral_length=771)
    rgb_values = data["RGB"] / 255.0  # Normalize RGB to [0, 1]
    xyz_values = data["XYZ"]

    # find (255, 255, 255) in rgb_values, save its XYZ
    white_index = None
    for i, rgb in enumerate(rgb_values):
        if all(rgb == [1.0, 1.0, 1.0]):
            white_index = i
            break
    if white_index is not None:
        white_xyz = xyz_values[white_index]
        print(f"Found white point at index {white_index}, XYZ: {white_xyz}")
    else:
        print("White point (255, 255, 255) not found in measured data.")
    xyz_values = xyz_values / white_xyz[1]

    train_rgb = rgb_values[:72]
    train_xyz = xyz_values[:72]
    test_rgb = rgb_values[72:]
    test_xyz = xyz_values[72:]

    # --- Build GOG Model ---
    print("Building GOG Model (RGB -> XYZ)")

    mode = "de2000"  # Options: "xyz", "de2000", "sucs"
    gog_model = make_gog(
        train_rgb,
        train_xyz,
        mode=mode,
        verbose=True,
    )

    # Print model parameters
    print("\nGOG Model Parameters:")
    rich.print(gog_model)

    # --- Evaluate on training set ---
    train_metrics = evaluate_gog(gog_model, train_rgb, train_xyz)
    print(f"\nTraining set (large, {train_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {train_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {train_metrics['max_delta_e']:.2f}")

    # --- Evaluate on validation set ---
    val_metrics = evaluate_gog(gog_model, test_rgb, test_xyz)
    print(f"\nValidation set (small, {test_rgb.shape[0]} points):")
    print(f"  Mean Delta E: {val_metrics['mean_delta_e']:.2f}")
    print(f"  Max Delta E: {val_metrics['max_delta_e']:.2f}")


if __name__ == "__main__":
    main()
