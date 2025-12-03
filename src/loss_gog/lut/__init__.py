"""
3D LUT (Look-Up Table) construction and manipulation.

This subpackage provides tools for building 3D LUTs from color measurements,
including both forward (RGB → XYZ) and inverse (XYZ → RGB) transformations.

Authors: Jack Chou
Date: Dec 1, 2025
"""

from .model import (
    build_forward_lut,
    build_inverse_lut,
    xyz_to_mid_space,
    mid_space_to_xyz,
)
from .evaluation import evaluate_lut

__all__ = [
    "build_forward_lut",
    "build_inverse_lut",
    "xyz_to_mid_space",
    "mid_space_to_xyz",
    "evaluate_lut",
]
