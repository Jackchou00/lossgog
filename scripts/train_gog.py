"""
Train a GOG model and evaluate its performance.

Usage:
    python scripts/train_gog.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np

from data_io import read_cs2000_csv
from lossgog import classic_gog, evaluate_gog, make_gog, rgb_to_xyz_gog


def main():
    measured_data_path = "measured_data/xdr/cs2000_lut_measurements_20251124_163713.csv"
    test_data_path = "measured_data/xdr/cs2000_lut_measurements_20251124_194618.csv"
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

    # --- Build GOG Models ---
    mode = "de2000"  # Options: "xyz", "de2000", "sucs"

    print("\n" + "=" * 60)
    print("1. Building GOG Model (Unconstrained)")
    print("=" * 60)
    gog_unconstrained = make_gog(
        train_rgb,
        train_xyz,
        mode=mode,
        constrain_white=False,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("2. Building GOG Model (Strict White Point Constrained)")
    print("=" * 60)
    gog_constrained = make_gog(
        train_rgb,
        train_xyz,
        mode=mode,
        constrain_white=True,
        verbose=True,
    )

    # --- Classic GOG ---
    print("\n" + "=" * 60)
    print("3. Building Classic GOG Model")
    print("=" * 60)
    gog_classic = classic_gog(rgb_values, xyz_values, verbose=True)

    # --- Evaluations ---
    target_white = white_xyz / white_xyz[1]

    def evaluate_and_print(name, model):
        train_metrics = evaluate_gog(model, train_rgb, train_xyz)
        val_metrics = evaluate_gog(model, test_rgb, test_xyz)

        # Grayscale points (R == G == B)
        train_gray_mask = np.isclose(train_rgb[:, 0], train_rgb[:, 1]) & np.isclose(
            train_rgb[:, 1], train_rgb[:, 2]
        )
        val_gray_mask = np.isclose(test_rgb[:, 0], test_rgb[:, 1]) & np.isclose(
            test_rgb[:, 1], test_rgb[:, 2]
        )

        train_gray_metrics = evaluate_gog(
            model, train_rgb[train_gray_mask], train_xyz[train_gray_mask]
        )
        val_gray_metrics = evaluate_gog(
            model, test_rgb[val_gray_mask], test_xyz[val_gray_mask]
        )

        # Predict white point at RGB = [1, 1, 1]
        pred_white = rgb_to_xyz_gog(np.array([[1.0, 1.0, 1.0]]), model)[0]
        white_err = pred_white - target_white
        white_err_pct = (white_err / target_white) * 100.0

        print(f"\n[{name}] Evaluation:")
        print(
            f"  Training Set (3375 pts):  Mean ΔE = {train_metrics['mean_delta_e']:.3f}, Max ΔE = {train_metrics['max_delta_e']:.3f}"
        )
        print(
            f"  Validation Set (216 pts): Mean ΔE = {val_metrics['mean_delta_e']:.3f}, Max ΔE = {val_metrics['max_delta_e']:.3f}"
        )
        print(
            f"  Training Gray (15 pts):   Mean ΔE = {train_gray_metrics['mean_delta_e']:.3f}, Max ΔE = {train_gray_metrics['max_delta_e']:.3f}"
        )
        print(
            f"  Validation Gray (6 pts):  Mean ΔE = {val_gray_metrics['mean_delta_e']:.3f}, Max ΔE = {val_gray_metrics['max_delta_e']:.3f}"
        )
        print(f"  White Point Prediction:  {pred_white}")
        print(f"  White Point Target:      {target_white}")
        print(f"  White Point Error:       {white_err} (Pct Err: {white_err_pct}%)")

    evaluate_and_print("Unconstrained GOG", gog_unconstrained)
    evaluate_and_print("Constrained GOG", gog_constrained)
    evaluate_and_print("Classic GOG", gog_classic)


if __name__ == "__main__":
    main()
