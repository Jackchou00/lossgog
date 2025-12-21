"""
GOG model evaluation functions.

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np
import colour

from lossgog.model import rgb_to_xyz_gog


def evaluate_gog(
    gog_model: dict,
    rgb: np.ndarray,
    xyz: np.ndarray,
) -> dict:
    """Evaluate GOG model accuracy using Delta E (CIE 2000).

    Parameters:
        gog_model: GOG model dict with gain, offset, gamma, matrix.
        rgb: Array of shape (N, 3), RGB values in [0, 1].
        xyz: Array of shape (N, 3), measured XYZ values (normalized to [0, 1]).

    Returns:
        Dict with keys:
            - "mean_delta_e": Mean Delta E value.
            - "max_delta_e": Maximum Delta E value.
            - "delta_e": Array of all Delta E values.
    """
    xyz_pred = rgb_to_xyz_gog(rgb, gog_model)

    # Convert to Lab for Delta E calculation
    # Note: Input XYZ should already be normalized (e.g., divided by white_Y)
    # colour.XYZ_to_Lab expects XYZ in [0, 1] range
    lab_measured = colour.XYZ_to_Lab(xyz)
    lab_pred = colour.XYZ_to_Lab(xyz_pred)
    delta_e = colour.delta_E(lab_measured, lab_pred, method="CIE 2000")

    return {
        "mean_delta_e": np.mean(delta_e),
        "max_delta_e": np.max(delta_e),
        "delta_e": delta_e,
    }
