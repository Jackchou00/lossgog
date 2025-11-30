"""
Plotting utilities for color science visualization.

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import colour


def plot_delta_e_histogram(
    delta_e: np.ndarray,
    output_path: str,
    title: str | None = None,
) -> None:
    """Plot and save Delta E histogram.

    Parameters:
        delta_e: Array of Delta E values.
        output_path: Path to save the histogram image.
        title: Optional title for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Delta E histogram
    ax.hist(delta_e, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        float(np.mean(delta_e)),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(delta_e):.2f}",
    )

    ax.set_xlabel("CIEDE2000")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to: {output_path}")


def plot_chromaticity_diagram(
    xyz_data: np.ndarray,
    output_path: str,
    data_label: str = "Data",
    show_gamuts: bool = True,
) -> None:
    """Plot CIE 1976 UCS (u'v') chromaticity diagram with data points.

    Parameters:
        xyz_data: Array of shape (n, 3), XYZ values to plot.
        output_path: Path to save the plot.
        data_label: Label for the data points in legend.
        show_gamuts: Whether to show sRGB, DCI-P3, BT.2020 gamut triangles.
    """
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_facecolor("white")

    # Plot spectral locus in u'v' coordinates
    wavelengths = np.arange(360, 831, 1)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    xyz_spectral = cmfs[wavelengths]
    uv_spectral = colour.xy_to_Luv_uv(colour.XYZ_to_xy(xyz_spectral))

    # Spectral locus and purple line
    ax.plot(uv_spectral[:, 0], uv_spectral[:, 1], color="black", linewidth=1)
    ax.plot(
        [uv_spectral[0, 0], uv_spectral[-1, 0]],
        [uv_spectral[0, 1], uv_spectral[-1, 1]],
        color="black",
        linewidth=1,
        linestyle="--",
    )

    # Set axis limits and labels
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("u'", fontsize=16)
    ax.set_ylabel("v'", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot data points (convert XYZ to u'v')
    data_Luv = colour.XYZ_to_Luv(xyz_data)
    data_uv = colour.Luv_to_uv(data_Luv)

    ax.scatter(
        data_uv[:, 0],
        data_uv[:, 1],
        color="blue",
        s=1,
        label=data_label,
        alpha=0.5,
    )

    # Plot standard gamut triangles
    if show_gamuts:
        colourspaces = {
            "sRGB": colour.RGB_COLOURSPACES["sRGB"].primaries,
            "DCI-P3": colour.RGB_COLOURSPACES["Display P3"].primaries,
            "BT.2020": colour.RGB_COLOURSPACES["ITU-R BT.2020"].primaries,
        }

        for name, primaries in colourspaces.items():
            # Convert xy to u'v' for each primary
            primaries_uv = colour.xy_to_Luv_uv(primaries)
            ax.plot(
                [
                    primaries_uv[0, 0],
                    primaries_uv[1, 0],
                    primaries_uv[2, 0],
                    primaries_uv[0, 0],
                ],
                [
                    primaries_uv[0, 1],
                    primaries_uv[1, 1],
                    primaries_uv[2, 1],
                    primaries_uv[0, 1],
                ],
                label=name,
                linewidth=1.5,
            )

    # Add legend
    ax.legend(loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Chromaticity diagram saved to: {output_path}")


def plot_training_size_comparison(
    sample_sizes: np.ndarray,
    results_dict: dict,
    output_path: str,
    ylabel: str = "Mean CIEDE2000 on Test Set",
) -> None:
    """Plot training size experiment results comparing multiple modes.

    Parameters:
        sample_sizes: Array of training sample sizes.
        results_dict: Dict mapping mode names to (n_sizes, n_repeats) arrays.
        output_path: Path to save the plot.
        ylabel: Label for y-axis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (mode_name, values) in enumerate(results_dict.items()):
        mean_values = np.mean(values, axis=1)
        ax.plot(
            sample_sizes,
            mean_values,
            marker=markers[i % len(markers)],
            label=mode_name,
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Training size comparison plot saved to: {output_path}")
