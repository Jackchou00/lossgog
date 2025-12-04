"""
Segmented Parametric GOG model core implementation.

SP GOG model formula:
    L = (gain * RGB + offset)^gamma   (per-channel tone response in Light Area, if RGB > D0)
    L = (slope * RGB + leakage)       (per-channel tone response in Dark Area, if RGB <= D0)
    XYZ = M @ L                        (3x3 matrix transform)

Continuity constraints at D0 (per channel):
    Value continuity:  (gain * D0 + offset)^gamma = slope * D0 + leakage
    Slope continuity:  gamma * gain * (gain * D0 + offset)^(gamma - 1) = slope

These constraints mean slope and leakage are derived from (gain, offset, gamma, d0):
    slope = gamma * gain * (gain * d0 + offset)^(gamma - 1)
    leakage = (gain * d0 + offset)^gamma - slope * d0

Reduced parameter set (15 free parameters + d0):
    - 3 gain values (input linear scaling, one per RGB channel)
    - 3 offset values (black level offset, one per RGB channel)  
    - 3 gamma values (non-linear power, one per RGB channel)
    - 9 matrix values (3x3 color transformation matrix)
    - d0: segmentation threshold (can be per-channel or shared)

slope and leakage are computed from continuity constraints.

Authors: Jack Chou
Date: Dec 4, 2025
"""

import numpy as np
from scipy.optimize import minimize
from loss_gog.ucs import calculate_de2000, calculate_de_sucs
import colour


# ==============================================================================
# Parameter packing/unpacking utilities  
# ==============================================================================

def _compute_slope_leakage(gain: np.ndarray, offset: np.ndarray, gamma: np.ndarray, d0: float) -> tuple:
    """Compute slope and leakage from continuity constraints.
    
    Given gain, offset, gamma and d0, compute slope and leakage such that:
    1. Value continuity: (gain * d0 + offset)^gamma = slope * d0 + leakage
    2. Slope continuity: gamma * gain * (gain * d0 + offset)^(gamma - 1) = slope
    
    Parameters:
        gain: (3,) array
        offset: (3,) array  
        gamma: (3,) array
        d0: float, segmentation threshold
        
    Returns:
        slope: (3,) array
        leakage: (3,) array
    """
    # Ensure numerical stability
    base = np.maximum(gain * d0 + offset, 1e-10)
    
    # slope = gamma * gain * base^(gamma - 1)
    slope = gamma * gain * np.power(base, gamma - 1)
    
    # leakage = base^gamma - slope * d0
    leakage = np.power(base, gamma) - slope * d0
    
    return slope, leakage


def _pack_params_reduced(
    gain: np.ndarray, offset: np.ndarray, gamma: np.ndarray, matrix: np.ndarray, d0: float
) -> np.ndarray:
    """Pack reduced SPGOG parameters into a flat array for optimization.
    
    Total 16 parameters: 3 gain + 3 offset + 3 gamma + 9 matrix + 1 d0
    """
    return np.concatenate([gain, offset, gamma, matrix.flatten(), [d0]])


def _unpack_params_reduced(params: np.ndarray) -> tuple:
    """Unpack flat parameter array into reduced SPGOG components."""
    gain = params[0:3]
    offset = params[3:6]
    gamma = params[6:9]
    matrix = params[9:18].reshape(3, 3)
    d0 = params[18]
    return gain, offset, gamma, matrix, d0


def rgb_to_xyz_sp_gog(rgb: np.ndarray, sp_gog_model: dict) -> np.ndarray:
    """Convert RGB to XYZ using the SPGOG model.

    Segmented Parametric GOG model formula:
        Light Area (RGB > D0): L = (gain * RGB + offset)^gamma
        Dark Area (RGB <= D0): L = slope * RGB + leakage
        XYZ = M @ L

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        sp_gog_model: Dict containing SPGOG model parameters:
            - "gain": (3,) array - input linear scaling
            - "offset": (3,) array - black level offset
            - "gamma": (3,) array - non-linear power
            - "matrix": (3, 3) array - color transform
            - "d0": float - segmentation threshold
            Optional (computed if not present):
            - "slope": (3,) array - linear scaling in Dark Area
            - "leakage": (3,) array - black level in Dark Area

    Returns:
        XYZ values, array of shape (n_samples, 3).
    """
    gain = sp_gog_model["gain"]
    offset = sp_gog_model["offset"]
    gamma = sp_gog_model["gamma"]
    matrix = sp_gog_model["matrix"]
    d0 = sp_gog_model.get("d0", 0.1)

    
    # Compute slope and leakage from continuity constraints if not provided
    if "slope" in sp_gog_model and "leakage" in sp_gog_model:
        slope = sp_gog_model["slope"]
        leakage = sp_gog_model["leakage"]
    else:
        slope, leakage = _compute_slope_leakage(gain, offset, gamma, d0)

    # Initialize L array
    L = np.zeros_like(rgb, dtype=np.float64)

    # Process each channel independently
    for ch in range(3):
        rgb_ch = rgb[:, ch]
        
        # Dark Area: RGB <= D0
        dark_mask = rgb_ch <= d0
        L[dark_mask, ch] = slope[ch] * rgb_ch[dark_mask] + leakage[ch]
        
        # Light Area: RGB > D0
        light_mask = ~dark_mask
        linear = gain[ch] * rgb_ch[light_mask] + offset[ch]
        linear = np.maximum(linear, 1e-10)  # Avoid zero for power
        L[light_mask, ch] = np.power(linear, gamma[ch])

    # Apply 3x3 matrix transform: XYZ = L @ M.T (for row vectors)
    xyz = L @ matrix.T
    return xyz


def xyz_to_rgb_sp_gog(xyz: np.ndarray, sp_gog_model: dict) -> np.ndarray:
    """Convert XYZ to RGB using the inverse SPGOG model.

    Inverse Segmented Parametric GOG model formula:
        L = M^(-1) @ XYZ
        For each channel:
            if L <= L_threshold: RGB = (L - leakage) / slope
            if L > L_threshold: RGB = ((L^(1/gamma) - offset) / gain
        where L_threshold = slope * D0 + leakage

    Parameters:
        xyz: Array of shape (n_samples, 3), XYZ values.
        sp_gog_model: Dict containing SPGOG model parameters.

    Returns:
        RGB values in [0, 1], array of shape (n_samples, 3).
    """
    gain = sp_gog_model["gain"]
    offset = sp_gog_model["offset"]
    gamma = sp_gog_model["gamma"]
    matrix = sp_gog_model["matrix"]
    d0 = sp_gog_model.get("d0", 0.1)
    
    # Compute slope and leakage from continuity constraints if not provided
    if "slope" in sp_gog_model and "leakage" in sp_gog_model:
        slope = sp_gog_model["slope"]
        leakage = sp_gog_model["leakage"]
    else:
        slope, leakage = _compute_slope_leakage(gain, offset, gamma, d0)

    # Inverse matrix transform
    matrix_inv = np.linalg.inv(matrix)
    L = xyz @ matrix_inv.T

    # Initialize RGB array
    rgb = np.zeros_like(L)

    # Process each channel independently
    for ch in range(3):
        L_ch = L[:, ch]
        
        # Calculate L threshold at D0
        L_threshold = slope[ch] * d0 + leakage[ch]
        
        # Dark Area: L <= L_threshold
        dark_mask = L_ch <= L_threshold
        # Avoid division by zero
        if slope[ch] > 1e-10:
            rgb[dark_mask, ch] = (L_ch[dark_mask] - leakage[ch]) / slope[ch]
        else:
            rgb[dark_mask, ch] = 0.0
        
        # Light Area: L > L_threshold
        light_mask = ~dark_mask
        L_positive = np.maximum(L_ch[light_mask], 1e-10)
        rgb[light_mask, ch] = (np.power(L_positive, 1.0 / gamma[ch]) - offset[ch]) / gain[ch]

    # Clamp to [0, 1] range
    # rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _loss_function_reduced(
    params: np.ndarray, rgb: np.ndarray, xyz_target: np.ndarray, mode: str = "xyz",
    robust: bool = False
) -> float:
    """Compute loss between predicted and target XYZ values.

    Parameters:
        params: Flat array of reduced SPGOG parameters (16 values).
        rgb: Training RGB values, shape (n, 3).
        xyz_target: Target XYZ values, shape (n, 3).
        mode: Loss mode - "xyz", "de2000", or "sucs".
        robust: If True, use robust loss (Huber-like) to reduce outlier impact.

    Returns:
        Loss value (MSE or mean squared Delta E).
    """
    gain, offset, gamma, matrix, d0 = _unpack_params_reduced(params)

    sp_gog_model = {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "matrix": matrix,
        "d0": d0,
    }

    xyz_pred = rgb_to_xyz_sp_gog(rgb, sp_gog_model)

    if mode == "xyz":
        errors = (xyz_pred - xyz_target) ** 2
        if robust:
            # Huber-like: clip large errors
            errors = np.minimum(errors, 0.01)
        mse = np.mean(errors)
    elif mode == "de2000":
        delta_e = calculate_de2000(xyz_pred, xyz_target)
        if robust:
            # Cap delta_e to reduce outlier influence
            delta_e = np.minimum(delta_e, 3.0)
        mse = np.mean(delta_e**2)
    elif mode == "sucs":
        delta_e = calculate_de_sucs(xyz_pred, xyz_target)
        if robust:
            delta_e = np.minimum(delta_e, 3.0)
        mse = np.mean(delta_e**2)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")

    return float(mse)


def _black_point_constraint(params: np.ndarray, target_black_L: np.ndarray) -> np.ndarray:
    """Constraint to ensure L(RGB=0) matches target black level.
    
    At RGB=0: L = leakage (from dark area formula)
    We want L(0) >= 0 or L(0) = target_black_L
    
    Parameters:
        params: Flat array of reduced SPGOG parameters (16 values).
        target_black_L: Target L values at RGB=0, shape (3,).
        
    Returns:
        Constraint values (should be >= 0 for inequality constraint).
    """
    gain, offset, gamma, matrix, d0 = _unpack_params_reduced(params)
    slope, leakage = _compute_slope_leakage(gain, offset, gamma, d0)
    
    # L at RGB=0 is just leakage
    # We want leakage >= 0 (or close to target_black_L)
    return leakage - target_black_L


def make_sp_gog(
    rgb: np.ndarray,
    xyz: np.ndarray,
    d0: float = 0.1,
    train_d0: bool = True,
    mode: str = "xyz",
    robust: bool = False,
    verbose: bool = True,
) -> dict:
    """Create a Segmented Parametric GOG model from RGB and XYZ measurements.

    SPGOG formula with automatic continuity:
        Light Area (RGB > D0): L = (gain * RGB + offset)^gamma
        Dark Area (RGB <= D0): L = slope * RGB + leakage
    
    Continuity is enforced by computing slope and leakage from (gain, offset, gamma, d0):
        slope = gamma * gain * (gain * d0 + offset)^(gamma - 1)
        leakage = (gain * d0 + offset)^gamma - slope * d0
    
    This reduces the free parameters from 24 to 16 (or 15 if d0 is fixed).
    
    Key insight: At RGB=0, L = leakage = offset^gamma (when d0 is small)
    So offset >= 0 ensures L(0) >= 0.

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        xyz: Array of shape (n_samples, 3), corresponding XYZ values.
        d0: Initial segmentation threshold, default 0.1.
        train_d0: Whether to train d0 as a parameter, default True.
        mode: Loss mode for optimization - "xyz", "de2000", or "sucs".
        robust: If True, use robust loss to reduce outlier impact.
        verbose: Whether to print optimization progress.

    Returns:
        Dict with keys:
            - "gain": array of shape (3,), input linear scaling.
            - "offset": array of shape (3,), black level offset (>= 0).
            - "gamma": array of shape (3,), non-linear power.
            - "slope": array of shape (3,), linear scaling in Dark Area (derived).
            - "leakage": array of shape (3,), black level in Dark Area (derived).
            - "matrix": array of shape (3, 3), color transformation matrix.
            - "d0": float, segmentation threshold.
    """
    # Find black point in training data (RGB closest to 0)
    rgb_norm = np.linalg.norm(rgb, axis=1)
    black_idx = np.argmin(rgb_norm)
    black_xyz = xyz[black_idx]
    
    if verbose:
        print(f"  Black point RGB: {rgb[black_idx]}")
        print(f"  Black point XYZ: {black_xyz}")
    
    # Initial parameter estimates
    init_gain = np.array([1.0, 1.0, 1.0])
    # offset must be >= 0 to ensure L(0) = offset^gamma >= 0
    # Start with small positive offset
    init_offset = np.array([0.01, 0.01, 0.01])
    init_gamma = np.array([2.2, 2.2, 2.2])

    # Matrix: start with scaled identity to match XYZ range
    max_xyz = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])
    init_matrix = np.diag(max_xyz)
    
    # Initial d0
    init_d0 = d0

    # Pack initial parameters
    init_params = _pack_params_reduced(init_gain, init_offset, init_gamma, init_matrix, init_d0)

    # Parameter bounds
    max_val = xyz.max()
    
    if train_d0:
        d0_bounds = [(0.0, 0.2)]  # d0 in reasonable range
    else:
        d0_bounds = [(d0, d0)]  # Fixed d0
    
    bounds = (
        # gain bounds (3) - must be positive
        [(0.5, 3.0)] * 3
        +
        # offset bounds (3) - must be >= 0 to ensure non-negative black level
        # When RGB=0, L = leakage. For small d0, leakage ≈ offset^gamma
        [(0.0, 0.15)] * 3
        +
        # gamma bounds (3)
        [(1.8, 3.0)] * 3
        +
        # matrix bounds (9)
        [(-2 * max_val, 2 * max_val)] * 9
        +
        # d0 bounds (1)
        d0_bounds
    )

    if verbose:
        print("Optimizing SPGOG model (reduced parameterization)")
        print(f"  Initial D0: {init_d0:.4f}, trainable: {train_d0}")
        print(f"  Number of parameters: {len(init_params)}")
        print(f"  Robust loss: {robust}")
        initial_loss = _loss_function_reduced(init_params, rgb, xyz, mode=mode, robust=robust)
        print(f"  Initial loss: {initial_loss:.6f}")
        
        # Check initial black point
        init_model = {
            "gain": init_gain,
            "offset": init_offset,
            "gamma": init_gamma,
            "matrix": init_matrix,
            "d0": init_d0,
        }
        init_black_xyz = rgb_to_xyz_sp_gog(np.array([[0.0, 0.0, 0.0]]), init_model)[0]
        print(f"  Initial XYZ at RGB=0: {init_black_xyz}")

    # Run optimization
    result = minimize(
        _loss_function_reduced,
        init_params,
        args=(rgb, xyz, mode, robust),
        method="L-BFGS-B",  # L-BFGS-B is efficient and supports bounds
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10},
    )

    if verbose:
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Optimization success: {result.success}")
        print(f"  Iterations: {result.nit}")
        if not result.success:
            print(f"  Warning: {result.message}")

    # Unpack optimized parameters
    gain, offset, gamma, matrix, d0_opt = _unpack_params_reduced(result.x)
    
    # Compute derived parameters
    slope, leakage = _compute_slope_leakage(gain, offset, gamma, d0_opt)
    
    model = {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "slope": slope,
        "leakage": leakage,
        "matrix": matrix,
        "d0": d0_opt,
    }
    
    if verbose:
        # Verify black point
        final_black_xyz = rgb_to_xyz_sp_gog(np.array([[0.0, 0.0, 0.0]]), model)[0]
        print(f"  Final D0: {d0_opt:.4f}")
        print(f"  Final XYZ at RGB=0: {final_black_xyz}")
        print(f"  Final leakage: {leakage}")

    return model


def evaluate_sp_gog(
    sp_gog_model: dict,
    rgb: np.ndarray,
    xyz: np.ndarray,
) -> dict:
    """Evaluate SPGOG model accuracy using Delta E (CIE 2000).

    Parameters:
        sp_gog_model: SPGOG model dict with gain, offset, gamma, slope, leakage, matrix, d0.
        rgb: Array of shape (N, 3), RGB values in [0, 1].
        xyz: Array of shape (N, 3), measured XYZ values (normalized to [0, 1]).

    Returns:
        Dict with keys:
            - "mean_delta_e": Mean Delta E value.
            - "max_delta_e": Maximum Delta E value.
            - "delta_e": Array of all Delta E values.
    """
    xyz_pred = rgb_to_xyz_sp_gog(rgb, sp_gog_model)

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
