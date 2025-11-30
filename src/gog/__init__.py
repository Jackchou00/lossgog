"""
GOG (Gain-Offset-Gamma) display characterization model.

This subpackage provides tools for building and using GOG models
to characterize display color response.

Authors: Jack Chou
Date: Nov 30, 2025
"""

from .model import make_gog, rgb_to_xyz_gog, xyz_to_rgb_gog
from .evaluation import evaluate_gog

__all__ = [
    "make_gog",
    "rgb_to_xyz_gog",
    "xyz_to_rgb_gog",
    "evaluate_gog",
]
