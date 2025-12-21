"""
Uniform Color Space (UCS) utilities.

This subpackage provides fast implementations of color difference metrics:
- CIEDE2000: Standard color difference formula
- sUCS: Simplified Uniform Color Space

Authors: Jack Chou
Date: Nov 30, 2025
"""

from .de2000 import calculate_de2000
from .sucs import calculate_de_sucs, xyz_to_sucs

__all__ = [
    "calculate_de2000",
    "calculate_de_sucs",
    "xyz_to_sucs",
]
