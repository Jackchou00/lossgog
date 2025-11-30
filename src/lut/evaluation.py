"""
3D LUT evaluation functions.

Authors: Jack Chou
Date: Dec 1, 2025
"""

import numpy as np
import colour


def evaluate_lut(
    lut: colour.LUT3D,
    test_rgb: np.ndarray,
    test_xyz: np.ndarray,
    white_point: np.ndarray,
) -> dict:
    """Evaluate a forward LUT (RGB → XYZ) against test measurements.

    Computes the prediction error between LUT output and measured XYZ values,
    using both MSE and CIEDE2000 color difference metrics.

    Parameters:
        lut: A colour.LUT3D object mapping RGB → XYZ.
        test_rgb: Array of shape (N, 3), test RGB values in [0, 1].
        test_xyz: Array of shape (N, 3), measured XYZ values.
        white_point: Array of shape (3,), display white point XYZ values.
                     Used for XYZ → Lab conversion.

    Returns:
        Dict with keys:
            - "mse": Mean squared error in XYZ space.
            - "mean_delta_e": Mean CIEDE2000 color difference.
            - "max_delta_e": Maximum CIEDE2000 color difference.
            - "delta_e": Array of all CIEDE2000 values.
    """
    # Apply LUT to get predicted XYZ
    xyz_predicted = lut.apply(test_rgb)

    # Compute MSE in XYZ space
    mse = float(np.mean((test_xyz - xyz_predicted) ** 2))

    # Compute CIEDE2000 in Lab space
    # Normalize by white point Y (luminance) for Lab conversion
    white_Y = white_point[1]
    lab_measured = colour.XYZ_to_Lab(test_xyz / white_Y)
    lab_predicted = colour.XYZ_to_Lab(xyz_predicted / white_Y)
    delta_e = colour.delta_E(lab_measured, lab_predicted, method="CIE 2000")

    return {
        "mse": mse,
        "mean_delta_e": float(np.mean(delta_e)),
        "max_delta_e": float(np.max(delta_e)),
        "delta_e": delta_e,
    }
