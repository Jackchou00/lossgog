"""
Train a GOG model and evaluate its performance.

Usage:
    python scripts/train_gog.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

import rich

from lossgog import classic_gog, evaluate_gog, make_gog, read_cs2000_csv


def main():
    measured_data_path = "/Users/jackchou/Desktop/data/raw/lossgog_data/xdr/cs2000_lut_measurements_20251124_163713.csv"
    test_data_path = "/Users/jackchou/Desktop/data/raw/lossgog_data/xdr/cs2000_lut_measurements_20251124_194618.csv"
    # --- Load measured data ---
    print("Loading Measured Data")
    train_data = read_cs2000_csv(measured_data_path)
    rgb_values = train_data["RGB"] / 255.0  # Normalize RGB to [0, 1]
    xyz_values = train_data["XYZ"]

    # --- load test data ---
    test_data = read_cs2000_csv(test_data_path)
    test_rgb_values = test_data["RGB"] / 255.0
    test_xyz_values = test_data["XYZ"]

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

    train_rgb = rgb_values
    train_xyz = xyz_values
    test_rgb = test_rgb_values
    test_xyz = test_xyz_values / white_xyz[1]

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

    # --- classic GOG ---
    print("\nBuilding Classic GOG Model (RGB -> XYZ)")
    classic_gog_model = classic_gog(rgb_values, xyz_values)
    classic_metrics = evaluate_gog(classic_gog_model, test_rgb, test_xyz)
    print("\nClassic GOG Validation set:")
    print(f"  Mean Delta E: {classic_metrics['mean_delta_e']:.3f}")
    print(f"  Max Delta E: {classic_metrics['max_delta_e']:.3f}")


if __name__ == "__main__":
    main()
