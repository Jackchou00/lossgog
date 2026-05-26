"""
Simplified Uniform Color Space (sUCS) implementation.

This module provides a fast vectorized implementation of the sUCS color space
transformation and color difference calculation.

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np


# sUCS transformation matrices
M_XYZ_TO_LMS = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0, 0.0, 0.9184],
    ]
)

M_LMS_TO_IAB = np.array(
    [
        [200 / 3.05, 100 / 3.05, 5 / 3.05],
        [430, -470, 40],
        [49, 49, -98],
    ]
)

# Pre-computed constants and transposed matrices for optimization
_M_XYZ_TO_LMS_T = M_XYZ_TO_LMS.T
_M_LMS_TO_IAB_T = M_LMS_TO_IAB.T
_INV_0_0252 = 1.0 / 0.0252
_CONST_0_0447 = 0.0447


def xyz_to_sucs(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert XYZ to sUCS color space.

    Parameters:
        xyz: Array of shape (n, 3), XYZ values.

    Returns:
        Tuple of (lightness, a_prime, b_prime) arrays, each of shape (n,).
    """
    # 1. Linear transform XYZ -> LMS
    # Optimization: use xyz @ M.T instead of (M @ xyz.T).T
    lms = xyz @ _M_XYZ_TO_LMS_T

    # 2. Nonlinear transform
    # Calculate sign * (|x|^0.43)
    lms_prime = np.sign(lms) * np.power(np.abs(lms), 0.43)

    # 3. Linear transform LMS -> IAB
    iab = lms_prime @ _M_LMS_TO_IAB_T

    # Extract lightness (I), a, b components
    lightness = iab[:, 0]
    a = iab[:, 1]
    b = iab[:, 2]

    # 4. Calculate chromaticity coordinates a_prime, b_prime
    # Optimization: use algebraic identity instead of trigonometric functions
    # Original: h = atan2(b, a); a' = C * cos(h); b' = C * sin(h)
    # Optimized: r = sqrt(a^2+b^2); factor = C / r; a' = a * factor; b' = b * factor

    r = np.sqrt(a * a + b * b)

    # C = (1 / 0.0252) * ln(1 + 0.0447 * r)
    chroma = _INV_0_0252 * np.log(1 + _CONST_0_0447 * r)

    # Calculate scale factor = C / r
    # Use np.divide with where parameter to safely handle r=0 case
    factor = np.divide(chroma, r, out=np.zeros_like(chroma), where=r != 0)

    a_prime = factor * a
    b_prime = factor * b

    return lightness, a_prime, b_prime


def calculate_de_sucs(xyz_pred: np.ndarray, xyz_target: np.ndarray) -> np.ndarray:
    """Calculate sUCS color difference between two sets of XYZ colors.

    Parameters:
        xyz_pred: Predicted XYZ values, shape (n, 3).
        xyz_target: Target XYZ values, shape (n, 3).

    Returns:
        sUCS color difference values, shape (n,).
    """
    lightness_pred, a_prime_pred, b_prime_pred = xyz_to_sucs(xyz_pred)
    lightness_target, a_prime_target, b_prime_target = xyz_to_sucs(xyz_target)

    delta_lightness = lightness_pred - lightness_target
    delta_a_prime = a_prime_pred - a_prime_target
    delta_b_prime = b_prime_pred - b_prime_target

    return np.sqrt(delta_lightness**2 + delta_a_prime**2 + delta_b_prime**2)


if __name__ == "__main__":
    xyz_pred = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])
    xyz_target = np.array([[0.48, 0.38, 0.28], [0.58, 0.48, 0.38]])

    de_sucs = calculate_de_sucs(xyz_pred, xyz_target)

    print("sUCS color difference:", de_sucs)
