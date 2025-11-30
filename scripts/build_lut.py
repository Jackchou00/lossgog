"""
Build 3D LUTs from display measurements.

This script creates both forward (RGB → XYZ) and inverse (Mid Space → RGB)
LUTs from CS2000 color measurement data.

Usage:
    python scripts/build_lut.py

Authors: Jack Chou
Date: Dec 1, 2025
"""

import numpy as np
import colour

from data_io import read_cs2000_csv
from lut import build_forward_lut, build_inverse_lut, evaluate_lut
from plotting import plot_delta_e_histogram


# Display white point from measurements (RGB = 255, 255, 255)
WHITE_POINT = np.array([474.94, 503.27, 579.53])


def build_and_save_forward_luts(
    train_rgb: np.ndarray,
    train_xyz: np.ndarray,
    val_rgb: np.ndarray,
    val_xyz: np.ndarray,
    source_grid_size: int,
    output_sizes: list[int],
) -> None:
    """Build and evaluate forward LUTs (RGB → XYZ) of various sizes."""
    for size in output_sizes:
        print(f"\n{'=' * 60}")
        print(f"Building {size}×{size}×{size} RGB → XYZ LUT")
        print("=" * 60)

        # Build LUT
        lut_data = build_forward_lut(
            train_rgb, train_xyz, source_grid_size, output_size=size
        )

        # Create colour.LUT3D object
        lut = colour.LUT3D(
            table=lut_data,
            name=f"CS2000_RGB_to_XYZ_{size}",
            domain=np.array([[0, 0, 0], [1, 1, 1]]),
            size=size,
        )

        # Save LUT
        lut_path = f"results/cs2000_rgb_to_xyz_{size}.cube"
        colour.write_LUT(lut, lut_path, decimals=6)
        print(f"LUT saved to: {lut_path}")

        # Evaluate on training set
        train_metrics = evaluate_lut(lut, train_rgb, train_xyz, WHITE_POINT)
        print(f"\nTraining set ({len(train_rgb)} points):")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  Mean ΔE: {train_metrics['mean_delta_e']:.2f}")
        print(f"  Max ΔE: {train_metrics['max_delta_e']:.2f}")

        # Evaluate on validation set
        val_metrics = evaluate_lut(lut, val_rgb, val_xyz, WHITE_POINT)
        print(f"\nValidation set ({len(val_rgb)} points):")
        print(f"  MSE: {val_metrics['mse']:.6f}")
        print(f"  Mean ΔE: {val_metrics['mean_delta_e']:.2f}")
        print(f"  Max ΔE: {val_metrics['max_delta_e']:.2f}")

        # Save histogram
        plot_delta_e_histogram(
            val_metrics["delta_e"],
            f"results/lut_delta_e_histogram_{size}.png",
        )


def build_and_save_inverse_luts(
    train_rgb: np.ndarray,
    train_xyz: np.ndarray,
    output_sizes: list[int],
    methods: list[dict],
) -> None:
    """Build and save inverse LUTs (Mid Space → RGB) using various methods."""
    white_Y = WHITE_POINT[1]

    for config in methods:
        method_name = config["name"]
        print(f"\n{'#' * 70}")
        print(f"# Building Inverse LUTs with method: {method_name}")
        print("#" * 70)

        for size in output_sizes:
            print(f"\n{'=' * 60}")
            print(f"Building {size}³ Mid Space → RGB LUT ({method_name})")
            print("=" * 60)

            lut_data = build_inverse_lut(
                measured_xyz=train_xyz,
                measured_rgb=train_rgb,
                white_point_luminance=white_Y,
                output_size=size,
                method=config["method"],
                rbf_kernel=config.get("kernel", "thin_plate_spline"),
                rbf_smoothing=config.get("smoothing", 0.0),
            )

            # Create and save LUT
            lut = colour.LUT3D(
                table=lut_data,
                name=f"CS2000_Mid_to_RGB_{size}_{method_name}",
                domain=np.array([[0, 0, 0], [1, 1, 1]]),
                size=size,
            )

            lut_path = f"results/cs2000_mid_to_rgb_{size}_{method_name.lower()}.cube"
            colour.write_LUT(lut, lut_path, decimals=6)
            print(f"LUT saved to: {lut_path}")


def main():
    # --- Load datasets ---
    large_file = "measured_data/cs2000_lut_measurements_20251124_163713.csv"
    small_file = "measured_data/cs2000_lut_measurements_20251124_194618.csv"

    large_data = read_cs2000_csv(large_file)
    small_data = read_cs2000_csv(small_file)

    # Normalize RGB to [0, 1], keep XYZ in absolute units
    large_rgb = large_data["RGB"] / 255.0
    large_xyz = large_data["XYZ"]
    small_rgb = small_data["RGB"] / 255.0
    small_xyz = small_data["XYZ"]

    print(f"Training set: {len(large_rgb)} points (15×15×15 grid)")
    print(f"Validation set: {len(small_rgb)} points (6×6×6 grid)")

    # --- Build Forward LUTs (RGB → XYZ) ---
    print("\n" + "#" * 70)
    print("# FORWARD LUTs (RGB → XYZ)")
    print("#" * 70)

    build_and_save_forward_luts(
        train_rgb=large_rgb,
        train_xyz=large_xyz,
        val_rgb=small_rgb,
        val_xyz=small_xyz,
        source_grid_size=15,
        output_sizes=[17, 33, 65],
    )

    # --- Build Inverse LUTs (Mid Space → RGB) ---
    print("\n" + "#" * 70)
    print("# INVERSE LUTs (Mid Space → RGB)")
    print("#" * 70)

    interpolation_methods = [
        {"name": "Linear", "method": "linear"},
        {"name": "RBF-TPS", "method": "rbf", "kernel": "thin_plate_spline", "smoothing": 0.0},
    ]

    build_and_save_inverse_luts(
        train_rgb=large_rgb,
        train_xyz=large_xyz,
        output_sizes=[17, 33, 65],
        methods=interpolation_methods,
    )

    print("\n" + "=" * 70)
    print("All LUTs built successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
