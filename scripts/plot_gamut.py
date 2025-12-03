"""
Plot display gamut in CIE 1976 UCS chromaticity diagram.

Usage:
    python scripts/plot_gamut.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

from loss_gog import read_cs2000_csv, plot_chromaticity_diagram


def main():
    # --- Load dataset ---
    large_file = "measured_data/cs2000_lut_measurements_20251124_163713.csv"
    large_data = read_cs2000_csv(large_file)
    large_xyz = large_data["XYZ"]
    print(f"Dataset: {large_xyz.shape[0]} points")

    # --- Plot chromaticity diagram ---
    plot_chromaticity_diagram(
        large_xyz,
        output_path="results/display_gamut.svg",
        data_label="Training Set",
        show_gamuts=True,
    )


if __name__ == "__main__":
    main()
