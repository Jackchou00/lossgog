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
from scipy.optimize import minimize, curve_fit

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


def classic_gog(
    rgb: np.ndarray,
    xyz: np.ndarray,
    ramp_indices: list = None,
    verbose: bool = True,
) -> dict:
    """Build a classic per-channel GOG from primary ramps.

    This function expects that the dataset contains per-channel ramps.
    For each channel it fits a model:

        L = (gain * input + offset)^gamma

    using the measured luminance (Y component of XYZ). After fitting the
    per-channel tone response (gain/offset/gamma) it constructs the 3x3
    color transform matrix by taking the measured XYZ at the maximum input
    for each primary and dividing by the corresponding L value.

    Parameters:
        rgb: (n,3) array of input RGB (normalized 0..1).
        xyz: (n,3) array of measured XYZ (same scale as used elsewhere).
        ramp_indices: Optional list of 3 tuples, each (start, end) specifying
                      the index range for R, G, B ramps respectively.
                      E.g., [(0, 18), (18, 36), (36, 54)] for 18 samples each.
                      If None, auto-detect by finding samples where only one
                      channel is non-zero (excluding black points).
        verbose: print progress.

    Returns:
        GOG model dict with keys: 'gain','offset','gamma','matrix'.
    """
    # Ensure inputs are arrays
    rgb = np.asarray(rgb, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    gains = np.zeros(3)
    offsets = np.zeros(3)
    gammas = np.ones(3) * 2.2
    cols = []

    for ch in range(3):
        if ramp_indices is not None:
            # Use explicit index range
            start, end = ramp_indices[ch]
            indices = np.arange(start, end)
            inputs = rgb[indices, ch]
            xyz_sel = xyz[indices, :]
        else:
            # Auto-detect: find samples where ONLY this channel is non-zero
            # Exclude black (all zeros) by requiring this channel > 0
            other1 = (ch + 1) % 3
            other2 = (ch + 2) % 3
            mask = (
                (rgb[:, ch] > 1e-6)  # This channel must be positive (exclude black)
                & np.isclose(rgb[:, other1], 0.0, atol=1e-6)
                & np.isclose(rgb[:, other2], 0.0, atol=1e-6)
            )
            # Also include the black point (input=0) for fitting
            # Find one black point where all channels are zero
            black_mask = (
                np.isclose(rgb[:, 0], 0.0, atol=1e-6)
                & np.isclose(rgb[:, 1], 0.0, atol=1e-6)
                & np.isclose(rgb[:, 2], 0.0, atol=1e-6)
            )
            # Take only the first black point to avoid duplicates
            black_indices = np.where(black_mask)[0]
            if len(black_indices) > 0:
                # Combine: primary samples + one black point
                primary_indices = np.where(mask)[0]
                indices = np.concatenate([[black_indices[0]], primary_indices])
            else:
                indices = np.where(mask)[0]

            inputs = rgb[indices, ch]
            xyz_sel = xyz[indices, :]

        if inputs.size < 4:
            raise ValueError(f"Not enough ramp samples for channel {ch}: found {inputs.size}")

        # Use Y (luminance) as the tone response measurement
        Y = xyz_sel[:, 1]

        # Sort by input
        order = np.argsort(inputs)
        inputs = inputs[order]
        Y = Y[order]
        xyz_sel_sorted = xyz_sel[order]

        # Define model: L = (gain * x + offset)^gamma, fit Y = a * L
        def _model(x, a, g, o, p):
            lin = np.maximum(g * x + o, 0.0)
            return a * np.power(lin, p)

        # Initial guesses and bounds
        p0 = [Y.max() if Y.max() > 0 else 1.0, 1.0, 0.001, 2.2]
        lower = [1e-8, 0.1, -0.1, 1.0]
        upper = [np.inf, 10.0, 0.5, 4.0]

        try:
            popt, _ = curve_fit(
                _model,
                inputs,
                Y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=20000,
            )
        except Exception:
            # Fallback to simple heuristic if fit fails
            popt = np.array(p0)
            if verbose:
                print(f"Warning: curve_fit failed for channel {ch}, using fallback params")

        a_ch, g_ch, o_ch, p_ch = popt
        gains[ch] = float(g_ch)
        offsets[ch] = float(o_ch)
        gammas[ch] = float(p_ch)

        # Find sample with maximum input (closest to 1.0)
        idx_max = np.argmax(inputs)
        input_max = float(inputs[idx_max])

        # Get XYZ at max input
        xyz_at_max = xyz_sel_sorted[idx_max]

        L_at_max = np.power(max(g_ch * input_max + o_ch, 0.0), p_ch)
        if L_at_max <= 0:
            L_at_max = 1e-12

        col = xyz_at_max / L_at_max
        cols.append(col)

        if verbose:
            print(f"Channel {ch}: {len(inputs)} samples, "
                  f"gain={g_ch:.4g}, offset={o_ch:.4g}, gamma={p_ch:.4g}")

    # Assemble matrix with columns as primaries
    matrix = np.column_stack(cols)

    gog_model = {
        "gain": gains,
        "offset": offsets,
        "gamma": gammas,
        "matrix": matrix,
    }

    return gog_model
