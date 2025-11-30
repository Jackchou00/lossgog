"""
GOG (Gain-Offset-Gamma) model core implementation.

The GOG model formula:
    L = gain * (RGB + offset)^gamma  (per-channel tone response)
    XYZ = M @ L                       (3x3 matrix transform)

Total 18 parameters:
    - 3 gain values (one per RGB channel)
    - 3 offset values (one per RGB channel)
    - 3 gamma values (one per RGB channel)
    - 9 matrix values (3x3 transformation matrix)

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np
from scipy.optimize import minimize

from ucs import calculate_de2000, calculate_de_sucs


def rgb_to_xyz_gog(rgb: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert RGB to XYZ using the GOG model.

    GOG model formula:
        L = gain * (RGB + offset)^gamma  (per-channel)
        XYZ = M @ L

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        gog_model: Dict containing GOG model parameters:
            - "gain": (3,) array
            - "offset": (3,) array
            - "gamma": (3,) array
            - "matrix": (3, 3) array

    Returns:
        XYZ values, array of shape (n_samples, 3).
    """
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]

    # Apply GOG tone response curve per channel
    # L = gain * (RGB + offset)^gamma
    # Clamp (RGB + offset) to avoid negative values before power
    rgb_offset = np.maximum(rgb + offset, 0.0)
    L = gain * np.power(rgb_offset, gamma)

    # Apply 3x3 matrix transform: XYZ = L @ M.T (for row vectors)
    xyz = L @ matrix.T
    return xyz


def xyz_to_rgb_gog(xyz: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert XYZ to RGB using the inverse GOG model.

    Inverse GOG model formula:
        L = M^(-1) @ XYZ
        RGB = (L / gain)^(1/gamma) - offset

    Parameters:
        xyz: Array of shape (n_samples, 3), XYZ values.
        gog_model: Dict containing GOG model parameters:
            - "gain": (3,) array
            - "offset": (3,) array
            - "gamma": (3,) array
            - "matrix": (3, 3) array

    Returns:
        RGB values in [0, 1], array of shape (n_samples, 3).
    """
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]

    # Inverse matrix transform: L = XYZ @ M^(-1).T = XYZ @ (M.T)^(-1)
    matrix_inv = np.linalg.inv(matrix)
    L = xyz @ matrix_inv.T
    # Inverse tone response: RGB = (L / gain)^(1/gamma) - offset
    # Clamp L/gain to avoid negative values before power
    L_normalized = np.maximum(L / gain, 0.0)
    rgb = np.power(L_normalized, 1.0 / gamma) - offset
    # Clamp to [0, 1] range
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _pack_params(
    gain: np.ndarray, offset: np.ndarray, gamma: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Pack GOG parameters into a flat array for optimization."""
    return np.concatenate([gain, offset, gamma, matrix.flatten()])


def _unpack_params(params: np.ndarray) -> tuple:
    """Unpack flat parameter array into GOG components."""
    gain = params[0:3]
    offset = params[3:6]
    gamma = params[6:9]
    matrix = params[9:18].reshape(3, 3)
    return gain, offset, gamma, matrix


def _loss_function(
    params: np.ndarray, rgb: np.ndarray, xyz_target: np.ndarray, mode: str = "xyz"
) -> float:
    """Compute loss between predicted and target XYZ values.

    Parameters:
        params: Flat array of GOG parameters (18 values).
        rgb: Training RGB values, shape (n, 3).
        xyz_target: Target XYZ values, shape (n, 3).
        mode: Loss mode - "xyz", "de2000", or "sucs".

    Returns:
        Loss value (MSE or mean squared Delta E).
    """
    gain, offset, gamma, matrix = _unpack_params(params)

    gog_model = {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "matrix": matrix,
    }

    xyz_pred = rgb_to_xyz_gog(rgb, gog_model)

    if mode == "xyz":
        # Compute MSE in XYZ space
        mse = np.mean((xyz_pred - xyz_target) ** 2)
    elif mode == "de2000":
        # Compute MSE to minimize Delta E (CIE 2000)
        delta_e = calculate_de2000(xyz_pred, xyz_target)
        mse = np.mean(delta_e**2)
    elif mode == "sucs":
        # Compute MSE to minimize sUCS Delta E
        delta_e = calculate_de_sucs(xyz_pred, xyz_target)
        mse = np.mean(delta_e**2)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")

    return float(mse)


def make_gog(
    rgb: np.ndarray,
    xyz: np.ndarray,
    mode: str = "xyz",
    verbose: bool = True,
) -> dict:
    """Create a GOG model from RGB and XYZ measurements using optimization.

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        xyz: Array of shape (n_samples, 3), corresponding XYZ values.
        mode: Loss mode for optimization - "xyz", "de2000", or "sucs".
        verbose: Whether to print optimization progress.

    Returns:
        Dict with keys:
            - "gain": array of shape (3,), scaling factors.
            - "offset": array of shape (3,), offset values.
            - "gamma": array of shape (3,), gamma correction values.
            - "matrix": array of shape (3, 3), transformation matrix.
    """
    # Initial parameter estimates
    # Gain: start with max XYZ values to scale output appropriately
    init_gain = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])

    # Offset: small positive value to handle black level
    init_offset = np.array([0.001, 0.001, 0.001])

    # Gamma: typical display gamma around 2.2-2.4
    init_gamma = np.array([2.2, 2.2, 2.2])

    # Matrix: start with identity matrix
    init_matrix = np.eye(3)

    # Pack initial parameters
    init_params = _pack_params(init_gain, init_offset, init_gamma, init_matrix)

    # Parameter bounds
    max_xyz = xyz.max()
    bounds = (
        # gain bounds (3)
        [(0.0, 2 * max_xyz)] * 3
        +
        # offset bounds (3)
        [(-0.1, 0.1)] * 3
        +
        # gamma bounds (3)
        [(1.0, 4.0)] * 3
        +
        # matrix bounds (9)
        [(-2 * max_xyz, 2 * max_xyz)] * 9
    )

    if verbose:
        print("Optimizing GOG model parameters (18 parameters)...")
        print(f"  Initial MSE: {_loss_function(init_params, rgb, xyz, mode=mode):.4f}")

    # Run optimization
    result = minimize(
        _loss_function,
        init_params,
        args=(rgb, xyz, mode),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )

    if verbose:
        print(f"  Final MSE: {result.fun:.4f}")
        print(f"  Optimization success: {result.success}")

    # Unpack optimized parameters
    gain, offset, gamma, matrix = _unpack_params(result.x)

    gog_model = {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "matrix": matrix,
    }

    return gog_model
