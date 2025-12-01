"""
GOG (Gain-Offset-Gamma) model core implementation.

Standard GOG model formula:
    L = (gain * RGB + offset)^gamma   (per-channel tone response)
    XYZ = M @ L                        (3x3 matrix transform)

Total 18 parameters:
    - 3 gain values (input linear scaling, one per RGB channel)
    - 3 offset values (black level offset, one per RGB channel)
    - 3 gamma values (non-linear power, one per RGB channel)
    - 9 matrix values (3x3 color transformation matrix)

Authors: Jack Chou
Date: Nov 30, 2025
"""

import numpy as np
from scipy.optimize import minimize

from ucs import calculate_de2000, calculate_de_sucs


# ==============================================================================
# Parameter packing/unpacking utilities
# ==============================================================================

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


# ==============================================================================
# Standard GOG Model: L = (gain * RGB + offset)^gamma
# ==============================================================================

def rgb_to_xyz_gog(rgb: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert RGB to XYZ using the GOG model.

    Standard GOG model formula:
        L = (gain * RGB + offset)^gamma  (per-channel)
        XYZ = M @ L

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        gog_model: Dict containing GOG model parameters:
            - "gain": (3,) array - input linear scaling
            - "offset": (3,) array - black level offset
            - "gamma": (3,) array - non-linear power
            - "matrix": (3, 3) array - color transform

    Returns:
        XYZ values, array of shape (n_samples, 3).
    """
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]

    # Standard GOG: L = (gain * RGB + offset)^gamma
    linear = np.maximum(gain * rgb + offset, 0.0)
    L = np.power(linear, gamma)

    # Apply 3x3 matrix transform: XYZ = L @ M.T (for row vectors)
    xyz = L @ matrix.T
    return xyz


def xyz_to_rgb_gog(xyz: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert XYZ to RGB using the inverse GOG model.

    Inverse GOG model formula:
        L = M^(-1) @ XYZ
        RGB = (L^(1/gamma) - offset) / gain

    Parameters:
        xyz: Array of shape (n_samples, 3), XYZ values.
        gog_model: Dict containing GOG model parameters.

    Returns:
        RGB values in [0, 1], array of shape (n_samples, 3).
    """
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]

    # Inverse matrix transform
    matrix_inv = np.linalg.inv(matrix)
    L = xyz @ matrix_inv.T

    # Inverse tone response: RGB = (L^(1/gamma) - offset) / gain
    L_positive = np.maximum(L, 0.0)
    rgb = (np.power(L_positive, 1.0 / gamma) - offset) / gain

    # Clamp to [0, 1] range
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


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
        mse = np.mean((xyz_pred - xyz_target) ** 2)
    elif mode == "de2000":
        delta_e = calculate_de2000(xyz_pred, xyz_target)
        mse = np.mean(delta_e**2)
    elif mode == "sucs":
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

    Standard GOG formula: L = (gain * RGB + offset)^gamma

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        xyz: Array of shape (n_samples, 3), corresponding XYZ values.
        mode: Loss mode for optimization - "xyz", "de2000", or "sucs".
        verbose: Whether to print optimization progress.

    Returns:
        Dict with keys:
            - "gain": array of shape (3,), input linear scaling.
            - "offset": array of shape (3,), black level offset.
            - "gamma": array of shape (3,), non-linear power.
            - "matrix": array of shape (3, 3), color transformation matrix.
    """
    # Initial parameter estimates
    init_gain = np.array([1.0, 1.0, 1.0])
    init_offset = np.array([0.001, 0.001, 0.001])
    init_gamma = np.array([2.2, 2.2, 2.2])

    # Matrix: start with scaled identity to match XYZ range
    max_xyz = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])
    init_matrix = np.diag(max_xyz)

    # Pack initial parameters
    init_params = _pack_params(init_gain, init_offset, init_gamma, init_matrix)

    # Parameter bounds
    max_val = xyz.max()
    bounds = (
        # gain bounds (3)
        [(0.1, 10.0)] * 3
        +
        # offset bounds (3)
        [(-0.1, 0.1)] * 3
        +
        # gamma bounds (3)
        [(1.0, 4.0)] * 3
        +
        # matrix bounds (9)
        [(-2 * max_val, 2 * max_val)] * 9
    )

    if verbose:
        print("Optimizing GOG model: L = (gain*RGB + offset)^gamma")
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

    return {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "matrix": matrix,
    }
