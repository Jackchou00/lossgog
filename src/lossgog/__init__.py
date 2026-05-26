"""
lossgog: Display Color Characterization Toolkit

This package provides tools for display color characterization, including:
- GOG (Gain-Offset-Gamma) model training and evaluation
- 3D LUT (Look-Up Table) construction and manipulation
- Color difference metrics (CIEDE2000, sUCS)
- Data I/O utilities for measurement data
- Plotting and visualization tools

Authors: Jack Chou
Date: Dec 3, 2025
"""

# Plotting utilities
from .plotting import (
    plot_delta_e_histogram,
    plot_chromaticity_diagram,
    plot_training_size_comparison,
)


# GOG model
from .gog import (
    make_gog,
    rgb_to_xyz_gog,
    xyz_to_rgb_gog,
    classic_gog,
    evaluate_gog,
)

# LUT utilities
from .lut import (
    build_forward_lut,
    build_inverse_lut,
    xyz_to_mid_space,
    mid_space_to_xyz,
    evaluate_lut,
)

# Color difference metrics
from .ucs import (
    calculate_de2000,
    calculate_de_sucs,
    xyz_to_sucs,
)

__all__ = [
    # Plotting
    "plot_delta_e_histogram",
    "plot_chromaticity_diagram",
    "plot_training_size_comparison",
    "plot_rgb_cube",
    # GOG model
    "make_gog",
    "rgb_to_xyz_gog",
    "xyz_to_rgb_gog",
    "classic_gog",
    "evaluate_gog",
    # LUT utilities
    "build_forward_lut",
    "build_inverse_lut",
    "xyz_to_mid_space",
    "mid_space_to_xyz",
    "evaluate_lut",
    # Color difference metrics
    "calculate_de2000",
    "calculate_de_sucs",
    "xyz_to_sucs",
]
