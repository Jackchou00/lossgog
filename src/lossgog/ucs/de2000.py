"""
Fast vectorized CIEDE2000 color difference calculation.

This module provides a NumPy-based implementation of the CIEDE2000 formula
for computing color differences between XYZ color values.

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np


# D65 white point reference values (CIE Standard Illuminant D65, 2° Observer)
_Xn, _Yn, _Zn = 95.047, 100.000, 108.883


def _f(t: np.ndarray) -> np.ndarray:
    """Nonlinear transformation function for XYZ to Lab conversion."""
    delta = 6 / 29
    limit = delta**3
    return np.where(t > limit, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))


def _xyz_to_lab(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert XYZ (0-1 range) to CIELAB.
    
    Parameters:
        xyz: Array of shape (n, 3), XYZ values in range 0-1.
        
    Returns:
        Tuple of (L, a, b) arrays, each of shape (n,).
    """
    # Scale to 0-100 range
    xyz_scaled = xyz * 100.0

    # Normalize to white point
    x = xyz_scaled[:, 0] / _Xn
    y = xyz_scaled[:, 1] / _Yn
    z = xyz_scaled[:, 2] / _Zn

    fx, fy, fz = _f(x), _f(y), _f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return L, a, b


def calculate_de2000(
    xyz_pred: np.ndarray,
    xyz_target: np.ndarray,
) -> np.ndarray:
    """Calculate CIEDE2000 color difference between two sets of XYZ colors.

    Parameters:
        xyz_pred: Predicted XYZ values, shape (n, 3), range 0-1.
        xyz_target: Target XYZ values, shape (n, 3), range 0-1.

    Returns:
        CIEDE2000 color difference values, shape (n,).
    """
    # Step 1: XYZ (D65) -> CIELAB
    L1, a1, b1 = _xyz_to_lab(xyz_target)  # Reference/target
    L2, a2, b2 = _xyz_to_lab(xyz_pred)    # Sample/predicted

    # Step 2: Calculate CIEDE2000 (Vectorized)

    # 1. Calculate C_prime and h_prime
    C1_ab = np.sqrt(a1**2 + b1**2)
    C2_ab = np.sqrt(a2**2 + b2**2)
    C_ab_mean = (C1_ab + C2_ab) / 2.0

    G = 0.5 * (1 - np.sqrt(C_ab_mean**7 / (C_ab_mean**7 + 25**7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    # Calculate hue angle h_prime (np.arctan2 returns radians, convert to degrees)
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # 2. Calculate Delta L', Delta C', Delta H'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Calculate delta_h_prime (handle angle wraparound)
    diff_h = h2_prime - h1_prime
    delta_h_prime = np.where(
        np.abs(diff_h) <= 180,
        diff_h,
        np.where(diff_h > 180, diff_h - 360, diff_h + 360),
    )

    # Calculate Delta H' (Residual)
    delta_H_prime = (
        2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))
    )

    # 3. Calculate weighted means
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    # Calculate h_bar_prime (hue mean, handle angle wraparound)
    sum_h = h1_prime + h2_prime
    abs_diff_h = np.abs(h1_prime - h2_prime)

    h_bar_prime = np.where(
        abs_diff_h <= 180,
        sum_h / 2.0,
        np.where(sum_h < 360, (sum_h + 360) / 2.0, (sum_h - 360) / 2.0),
    )

    # 4. Calculate T
    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    )

    # 5. Calculate weighting functions S_L, S_C, S_H
    S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / np.sqrt(20 + (L_bar_prime - 50) ** 2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    # 6. Calculate rotation term R_T
    delta_theta = 30 * np.exp(-(((h_bar_prime - 275) / 25) ** 2))
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    # 7. Combine final result (Parametric factors kL=kC=kH=1)
    k_L, k_C, k_H = 1.0, 1.0, 1.0

    de2000_sq = (
        (delta_L_prime / (k_L * S_L)) ** 2
        + (delta_C_prime / (k_C * S_C)) ** 2
        + (delta_H_prime / (k_H * S_H)) ** 2
        + R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    de2000 = np.sqrt(de2000_sq)

    return de2000


if __name__ == "__main__":
    # Generate random test data (n=5, 3)
    n = 5
    xyz_p = np.random.rand(n, 3)
    xyz_t = np.random.rand(n, 3)

    result = calculate_de2000(xyz_p, xyz_t)

    print(f"Input shape: {xyz_p.shape}")
    print(f"Output shape: {result.shape}")
    print(f"DE2000 results: {result}")
