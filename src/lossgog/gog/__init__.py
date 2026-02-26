"""
GOG (Gain-Offset-Gamma) display characterization model.

This subpackage provides tools for building and using GOG models
to characterize display color response.

Standard GOG model formula:
    L = (gain * RGB + offset)^gamma   (per-channel tone response)
    XYZ = M @ L                        (3x3 matrix transform)
"""

from .model import make_gog, rgb_to_xyz_gog, xyz_to_rgb_gog, classic_gog
from .evaluation import evaluate_gog

__all__ = [
    "make_gog",
    "rgb_to_xyz_gog",
    "xyz_to_rgb_gog",
    "classic_gog",
    "evaluate_gog",
]
