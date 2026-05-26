"""GOG (Gain-Offset-Gamma) model core implementation.

Standard GOG model formula:
    L = (gain * RGB + offset)^gamma   (per-channel tone response)
    XYZ = M @ L                        (3x3 matrix transform)

This implementation uses a unit-white constraint during optimization:
    For each channel, enforce L(1) = 1.
    (gain + offset)^gamma = 1  ->  gain + offset = 1  (assuming gain+offset > 0)
    Therefore, offset is derived and not independently optimized:
        offset = 1 - gain

Model is still returned as a dict with 18 values (gain/offset/gamma/matrix),
but only 15 degrees of freedom are optimized (gain, gamma, matrix).
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from lossgog.ucs import calculate_de2000, calculate_de_sucs


# ==============================================================================
# Parameter packing/unpacking utilities (unit-white constrained)
# ==============================================================================


def _pack_params_unit_white(
    gain: np.ndarray, gamma: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Pack GOG parameters for the unit-white constrained variant.

    This variant enforces per-channel L(1) = 1, i.e.
        (gain + offset)^gamma = 1  ->  gain + offset = 1 (assuming gain+offset>0)
    so offset is not optimized and is derived as:
        offset = 1 - gain

    Total 15 parameters:
        - 3 gain
        - 3 gamma
        - 9 matrix
    """
    return np.concatenate([gain, gamma, matrix.flatten()])


def _unpack_params_unit_white(params: np.ndarray) -> tuple:
    """Unpack 15-parameter unit-white constrained GOG parameters."""
    gain = params[0:3]
    gamma = params[3:6]
    matrix = params[6:15].reshape(3, 3)
    offset = 1.0 - gain
    return gain, offset, gamma, matrix


def _pack_params_white_constrained(
    gain: np.ndarray, gamma: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Pack GOG parameters for unit-white + white-point constrained variant.

    This variant enforces:
    1. L(1) = 1  =>  offset = 1 - gain
    2. XYZ(1,1,1) = WhitePoint  =>  M @ [1,1,1]^T = WhitePoint
       so M[:, 2] = WhitePoint - M[:, 0] - M[:, 1]

    Total 12 parameters:
        - 3 gain
        - 3 gamma
        - 6 matrix (first two columns, flattened)
    """
    return np.concatenate([gain, gamma, matrix[:, 0:2].flatten()])


def _unpack_params_white_constrained(
    params: np.ndarray, white_xyz: np.ndarray
) -> tuple:
    """Unpack 12-parameter white constrained GOG parameters."""
    gain = params[0:3]
    gamma = params[3:6]
    m_cols = params[6:12].reshape(3, 2)

    # Reconstruct 3x3 matrix using white point constraint
    matrix = np.zeros((3, 3))
    matrix[:, 0:2] = m_cols
    matrix[:, 2] = white_xyz - m_cols[:, 0] - m_cols[:, 1]

    offset = 1.0 - gain
    return gain, offset, gamma, matrix


# ==============================================================================
# Standard GOG Model: L = (gain * RGB + offset)^gamma
# ==============================================================================


def rgb_to_xyz_gog(rgb: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert RGB to XYZ using the GOG model.

    Standard GOG model formula:
        L = (gain * RGB + offset)^gamma  (per-channel)
        XYZ = M @ L + XYZ_black          (additive black correction)

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        gog_model: Dict containing GOG model parameters:
            - "gain": (3,) array - input linear scaling
            - "offset": (3,) array - black level offset
            - "gamma": (3,) array - non-linear power
            - "matrix": (3, 3) array - color transform
            - "xyz_black": (3,) array - optional additive black level

    Returns:
        XYZ values, array of shape (n_samples, 3).
    """
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]
    xyz_black = gog_model.get("xyz_black", None)

    # Standard GOG: L = (gain * RGB + offset)^gamma
    linear = np.maximum(gain * rgb + offset, 0.0)
    L = np.power(linear, gamma)

    # Apply 3x3 matrix transform: XYZ = L @ M.T (for row vectors)
    xyz = L @ matrix.T

    # Additive black correction if present
    if xyz_black is not None:
        xyz = xyz + xyz_black

    return xyz


def xyz_to_rgb_gog(xyz: np.ndarray, gog_model: dict) -> np.ndarray:
    """Convert XYZ to RGB using the inverse GOG model.

    Inverse GOG model formula:
        XYZ' = XYZ - XYZ_black
        L = M^(-1) @ XYZ'
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
    xyz_black = gog_model.get("xyz_black", None)

    # Subtract black level if present
    if xyz_black is not None:
        # Avoid modifying the original array if it was passed by reference
        xyz = xyz - xyz_black

    # Inverse matrix transform
    matrix_inv = np.linalg.inv(matrix)
    L = xyz @ matrix_inv.T

    # Inverse tone response: RGB = (L^(1/gamma) - offset) / gain
    L_positive = np.maximum(L, 0.0)
    rgb = (np.power(L_positive, 1.0 / gamma) - offset) / gain

    # Clamp to [0, inf)
    rgb = np.clip(rgb, 0.0, np.inf)
    return rgb


def _loss_function_unit_white(
    params: np.ndarray, rgb: np.ndarray, xyz_target: np.ndarray, mode: str = "xyz"
) -> float:
    """Loss for the unit-white constrained GOG variant (15 parameters)."""
    gain, offset, gamma, matrix = _unpack_params_unit_white(params)

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


def _loss_function_white_constrained(
    params: np.ndarray,
    rgb: np.ndarray,
    xyz_target: np.ndarray,
    white_xyz: np.ndarray,
    mode: str = "xyz",
) -> float:
    """Loss for the unit-white + white-point constrained GOG variant (12 parameters)."""
    gain, offset, gamma, matrix = _unpack_params_white_constrained(params, white_xyz)

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
    constrain_white: bool = True,
    correct_black: bool = False,
    verbose: bool = True,
) -> dict:
    """Create a GOG model from RGB and XYZ measurements using optimization.

    Standard GOG formula: L = (gain * RGB + offset)^gamma

    Optimization uses a unit-white constraint per channel:
        L(1) = 1  =>  offset = 1 - gain
    so only 15 degrees of freedom are optimized (gain, gamma, matrix).

    Optionally, a strict white-point constraint is applied:
        XYZ(1,1,1) = WhitePoint  =>  M @ [1,1,1]^T = WhitePoint
    reducing optimized degrees of freedom to 12.

    Optionally, a black level correction is applied:
        XYZ' = XYZ - XYZ_black
    subtracting the black level before optimization.

    Parameters:
        rgb: Array of shape (n_samples, 3), RGB values in [0, 1].
        xyz: Array of shape (n_samples, 3), corresponding XYZ values.
        mode: Loss mode for optimization - "xyz", "de2000", or "sucs".
        constrain_white: Whether to enforce strict white point constraint.
        correct_black: Whether to subtract black level before optimization.
        verbose: Whether to print optimization progress.

    Returns:
        Dict with keys:
            - "gain": array of shape (3,), input linear scaling.
            - "offset": array of shape (3,), black level offset.
            - "gamma": array of shape (3,), non-linear power.
            - "matrix": array of shape (3, 3), color transformation matrix.
            - "xyz_black": array of shape (3,), optional black level offset.
    """
    # Detect and subtract black point if correction is requested
    xyz_black = None
    if correct_black:
        black_idx = np.argmin(np.sum(rgb**2, axis=1))
        if not np.allclose(rgb[black_idx], 0.0, atol=1e-3):
            raise ValueError(
                "Black point RGB=[0,0,0] not found in training data. "
                "Cannot apply black level correction."
            )
        xyz_black = xyz[black_idx]
        # Subtract black level for optimization (returns a copy)
        xyz = xyz - xyz_black

    # Detect white point if constraint is requested
    white_xyz = None
    if constrain_white:
        white_idx = np.argmin(np.sum((rgb - 1.0) ** 2, axis=1))
        if not np.allclose(rgb[white_idx], 1.0, atol=1e-3):
            raise ValueError(
                "White point RGB=[1,1,1] not found in training data. "
                "Cannot apply white point constraint."
            )
        white_xyz = xyz[white_idx]

    # Initial parameter estimates
    init_gain = np.array([1.0, 1.0, 1.0])
    init_gamma = np.array([2.2, 2.2, 2.2])

    # Matrix: start with scaled identity to match XYZ range
    max_xyz = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])
    init_matrix = np.diag(max_xyz)

    # Pack initial parameters and setup bounds
    max_val = xyz.max()
    if constrain_white:
        init_params = _pack_params_white_constrained(init_gain, init_gamma, init_matrix)
        bounds = [(0.8, 1.2)] * 3 + [(1.0, 4.0)] * 3 + [(-2 * max_val, 2 * max_val)] * 6
        loss_func = _loss_function_white_constrained
        loss_args = (rgb, xyz, white_xyz, mode)
        init_loss = _loss_function_white_constrained(
            init_params, rgb, xyz, white_xyz, mode=mode
        )
    else:
        init_params = _pack_params_unit_white(init_gain, init_gamma, init_matrix)
        bounds = [(0.8, 1.2)] * 3 + [(1.0, 4.0)] * 3 + [(-2 * max_val, 2 * max_val)] * 9
        loss_func = _loss_function_unit_white
        loss_args = (rgb, xyz, mode)
        init_loss = _loss_function_unit_white(init_params, rgb, xyz, mode=mode)

    if verbose:
        msg = []
        if constrain_white:
            msg.append("white-point constrained")
        if correct_black:
            msg.append("black-corrected")
        msg_str = " + ".join(msg) if msg else "standard"
        print(f"Optimizing GOG model ({msg_str}): L(1)=1, offset=1-gain")
        print(f"  Initial MSE: {init_loss:.4f}")

    # Run optimization
    result = minimize(
        loss_func,
        init_params,
        args=loss_args,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )

    if verbose:
        print(f"  Final MSE: {result.fun:.4f}")
        print(f"  Optimization success: {result.success}")

    # Unpack optimized parameters
    if constrain_white:
        gain, offset, gamma, matrix = _unpack_params_white_constrained(
            result.x, white_xyz
        )
    else:
        gain, offset, gamma, matrix = _unpack_params_unit_white(result.x)

    model_dict = {
        "gain": gain,
        "offset": offset,
        "gamma": gamma,
        "matrix": matrix,
    }
    if xyz_black is not None:
        model_dict["xyz_black"] = xyz_black

    return model_dict


def classic_gog(
    rgb: np.ndarray,
    xyz: np.ndarray,
    ramp_indices: list[tuple[int, int]] | None = None,
    verbose: bool = True,
) -> dict:
    """Build a classic per-channel GOG from primary ramps.

    Classic GOG model:
        L_i = (gain_i * RGB_i + offset_i)^gamma_i   for i in {R, G, B}
        XYZ = M @ L

    where L is the linearized signal (normalized to [0, 1] at max input),
    and M is a 3x3 matrix with columns being the XYZ of each primary at max.

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
    primary_xyz = []  # XYZ of each primary at max input

    for ch in range(3):
        if ramp_indices is not None:
            # Use explicit index range
            start, end = ramp_indices[ch]
            indices = np.arange(start, end)
            inputs = rgb[indices, ch]
            xyz_sel = xyz[indices, :]
        else:
            # Auto-detect: find samples where ONLY this channel is non-zero
            other1 = (ch + 1) % 3
            other2 = (ch + 2) % 3
            mask = (
                (rgb[:, ch] > 1e-6)
                & np.isclose(rgb[:, other1], 0.0, atol=1e-6)
                & np.isclose(rgb[:, other2], 0.0, atol=1e-6)
            )
            black_mask = (
                np.isclose(rgb[:, 0], 0.0, atol=1e-6)
                & np.isclose(rgb[:, 1], 0.0, atol=1e-6)
                & np.isclose(rgb[:, 2], 0.0, atol=1e-6)
            )
            black_indices = np.where(black_mask)[0]
            if len(black_indices) > 0:
                primary_indices = np.where(mask)[0]
                indices = np.concatenate([[black_indices[0]], primary_indices])
            else:
                indices = np.where(mask)[0]

            inputs = rgb[indices, ch]
            xyz_sel = xyz[indices, :]

        if inputs.size < 4:
            raise ValueError(
                f"Not enough ramp samples for channel {ch}: found {inputs.size}"
            )

        # Sort by input
        order = np.argsort(inputs)
        inputs = inputs[order]
        xyz_sel_sorted = xyz_sel[order]

        # Use Y (luminance) for tone response fitting
        Y = xyz_sel_sorted[:, 1]

        # Normalize Y: L should go from ~0 to 1
        # Subtract black level and normalize by max
        Y_black = Y[0]  # Y at input=0 (black)
        Y_max = Y[-1]  # Y at input=1 (max)

        if Y_max - Y_black > 1e-10:
            L_measured = (Y - Y_black) / (Y_max - Y_black)
        else:
            L_measured = Y / max(Y_max, 1e-10)

        # Fit: L = (gain * input + offset)^gamma
        # Since L should be ~0 at input=0 and ~1 at input=1:
        #   At input=0: L = offset^gamma ≈ 0  -> offset ≈ 0
        #   At input=1: L = (gain + offset)^gamma ≈ 1 -> gain ≈ 1
        def _model(x, g, o, p):
            lin = np.maximum(g * x + o, 0.0)
            return np.power(lin, p)

        # Initial guesses
        p0 = [1.0, 0.001, 2.2]
        lower = [0.5, -0.05, 1.0]
        upper = [2.0, 0.2, 4.0]

        try:
            popt, _ = curve_fit(
                _model,
                inputs,
                L_measured,
                p0=p0,
                bounds=(lower, upper),
                maxfev=20000,
            )
            g_ch, o_ch, p_ch = popt
        except Exception as e:
            if verbose:
                print(
                    f"Warning: curve_fit failed for channel {ch}: {e}, using fallback"
                )
            g_ch, o_ch, p_ch = 1.0, 0.001, 2.2

        gains[ch] = float(g_ch)
        offsets[ch] = float(o_ch)
        gammas[ch] = float(p_ch)

        # Get XYZ at max input (primary color)
        # Subtract black XYZ to get pure primary contribution
        xyz_black = xyz_sel_sorted[0]
        xyz_max = xyz_sel_sorted[-1]
        xyz_primary = xyz_max - xyz_black
        primary_xyz.append(xyz_primary)

        if verbose:
            print(
                f"Channel {ch}: {len(inputs)} samples, "
                f"gain={g_ch:.4g}, offset={o_ch:.4g}, gamma={p_ch:.4g}"
            )

    # Matrix: columns are the XYZ of each primary (R, G, B)
    # XYZ = M @ L, where L = [L_R, L_G, L_B]^T
    matrix = np.column_stack(primary_xyz)

    gog_model = {
        "gain": gains,
        "offset": offsets,
        "gamma": gammas,
        "matrix": matrix,
        "xyz_black": xyz_black,
    }

    return gog_model
