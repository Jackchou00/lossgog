"""
3D LUT model construction.

This module provides functions to build 3D Look-Up Tables (LUTs) for display
characterization. Two types of LUTs are supported:

1. Forward LUT (RGB → XYZ):
   - Input: Device RGB values on a regular grid
   - Output: Measured XYZ tristimulus values
   - Method: Regular grid interpolation (cubic)

2. Inverse LUT (Mid Space → RGB):
   - Input: XYZ values converted to an intermediate "Mid Space"
   - Output: Device RGB values that produce those XYZ values
   - Method: Scattered data interpolation (linear or RBF)

The "Mid Space" is an intermediate color space designed to provide better
interpolation properties than raw XYZ. It uses:
   1. Normalization by white point luminance
   2. Conversion to BT.2020 RGB (wide gamut)
   3. Inverse EOTF (BT.1886) for perceptual uniformity

Authors: Jack Chou
Date: Dec 1, 2025
"""

import numpy as np
import colour
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator, griddata


# =============================================================================
# Mid Space Conversion
# =============================================================================


def xyz_to_mid_space(xyz: np.ndarray, white_point_luminance: float) -> np.ndarray:
    """Convert XYZ to a normalized intermediate space (Mid Space).

    The Mid Space transformation creates a more perceptually uniform and
    well-distributed color space for interpolation. The process:

    1. Normalize XYZ by white point luminance → scale-invariant XYZ
    2. Convert to BT.2020 RGB → wide gamut linear RGB
    3. Apply inverse BT.1886 EOTF → perceptually uniform encoding

    This space is suitable for building inverse LUTs because:
    - The values are bounded roughly in [0, 1]
    - The distribution is more uniform than raw XYZ
    - Interpolation errors are less perceptually visible

    Parameters:
        xyz: Array of shape (N, 3), XYZ tristimulus values.
        white_point_luminance: The Y value of the display white point (cd/m²).

    Returns:
        Array of shape (N, 3), Mid Space values approximately in [0, 1].

    Example:
        >>> white_Y = 503.27  # measured white point luminance
        >>> xyz = np.array([[250, 260, 300]])
        >>> mid = xyz_to_mid_space(xyz, white_Y)
    """
    # Step 1: Normalize by white point luminance for scale invariance
    # This maps white point XYZ to approximately (0.95, 1.0, 1.09) for D65
    norm_xyz = xyz / white_point_luminance

    # Step 2: Convert to BT.2020 linear RGB
    # BT.2020 is chosen for its wide gamut to minimize clipping
    mid_rgb = colour.XYZ_to_RGB(norm_xyz, colourspace="ITU-R BT.2020")

    # Step 3: Apply inverse EOTF for perceptual uniformity
    # BT.1886 EOTF approximates display gamma (~2.4)
    # Inverse EOTF compresses highlights and expands shadows
    mid_space = colour.eotf_inverse(mid_rgb, function="ITU-R BT.1886")

    return mid_space


def mid_space_to_xyz(mid: np.ndarray, white_point_luminance: float) -> np.ndarray:
    """Convert Mid Space back to XYZ tristimulus values.

    This is the inverse of xyz_to_mid_space(). The process:

    1. Apply BT.1886 EOTF → linear RGB
    2. Convert BT.2020 RGB to XYZ → normalized XYZ
    3. Scale by white point luminance → absolute XYZ

    Parameters:
        mid: Array of shape (N, 3), Mid Space values.
        white_point_luminance: The Y value of the display white point (cd/m²).

    Returns:
        Array of shape (N, 3), XYZ tristimulus values.
    """
    # Step 1: Apply EOTF to get linear RGB
    rgb_linear = colour.eotf(mid, function="ITU-R BT.1886")

    # Step 2: Convert BT.2020 RGB to XYZ
    norm_xyz = colour.RGB_to_XYZ(rgb_linear, colourspace="ITU-R BT.2020")

    # Step 3: Scale by white point luminance
    xyz = norm_xyz * white_point_luminance

    return xyz


# =============================================================================
# Forward LUT (RGB → XYZ)
# =============================================================================


def build_forward_lut(
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    source_grid_size: int,
    output_size: int = 17,
) -> np.ndarray:
    """Build a forward LUT (RGB → XYZ) from regular grid measurements.

    This function takes RGB-XYZ measurement pairs where the RGB values form
    a regular grid (e.g., 15×15×15), and creates a LUT that maps any RGB
    input to the corresponding XYZ output via cubic interpolation.

    The process:
    1. Reshape the flat measurement data into a 3D grid
    2. Create per-channel cubic interpolators
    3. Evaluate on a uniform output grid (e.g., 17×17×17)

    Parameters:
        source_rgb: Array of shape (N, 3), RGB values in [0, 1].
                    Must form a regular grid of size source_grid_size³.
        target_xyz: Array of shape (N, 3), corresponding XYZ values.
        source_grid_size: Number of samples per channel in source data
                          (e.g., 15 for a 15×15×15 grid, N = 15³ = 3375).
        output_size: Size of the output LUT per dimension
                     (e.g., 17 for a 17×17×17 LUT).

    Returns:
        Array of shape (output_size, output_size, output_size, 3),
        the 3D LUT mapping RGB → XYZ.

    Example:
        >>> # 15×15×15 measurements → 17×17×17 LUT
        >>> lut_data = build_forward_lut(rgb_measured, xyz_measured, 15, 17)
        >>> lut = colour.LUT3D(table=lut_data, size=17)
    """
    # Extract unique RGB values for each channel (should be source_grid_size values)
    r_vals = np.unique(source_rgb[:, 0])
    g_vals = np.unique(source_rgb[:, 1])
    b_vals = np.unique(source_rgb[:, 2])

    # Verify grid structure
    expected_points = source_grid_size**3
    if len(source_rgb) != expected_points:
        raise ValueError(
            f"Expected {expected_points} points for {source_grid_size}³ grid, "
            f"got {len(source_rgb)}"
        )

    # Reshape XYZ data to 3D grid: (R, G, B, channels)
    # Assumes data is ordered with B varying fastest, then G, then R
    target_grid = target_xyz.reshape(
        (source_grid_size, source_grid_size, source_grid_size, 3)
    )

    # Create cubic interpolator for each XYZ channel
    interpolators = []
    for channel in range(3):
        interp = RegularGridInterpolator(
            (r_vals, g_vals, b_vals),
            target_grid[:, :, :, channel],
            method="cubic",
            bounds_error=True,
            fill_value=None,
        )
        interpolators.append(interp)

    # Create uniform output grid in [0, 1]
    output_vals = np.linspace(0, 1, output_size)
    r_out, g_out, b_out = np.meshgrid(
        output_vals, output_vals, output_vals, indexing="ij"
    )

    # Flatten to query points: (output_size³, 3)
    query_points = np.stack([r_out.ravel(), g_out.ravel(), b_out.ravel()], axis=1)

    # Interpolate each channel
    output_data = np.zeros((output_size**3, 3), dtype=np.float32)
    for channel in range(3):
        output_data[:, channel] = interpolators[channel](query_points)

    # Reshape to 3D LUT format
    lut_data = output_data.reshape((output_size, output_size, output_size, 3))

    return lut_data


# =============================================================================
# Inverse LUT (Mid Space → RGB)
# =============================================================================


def build_inverse_lut(
    measured_xyz: np.ndarray,
    measured_rgb: np.ndarray,
    white_point_luminance: float,
    output_size: int = 17,
    method: str = "linear",
    rbf_kernel: str = "thin_plate_spline",
    rbf_smoothing: float = 0.0,
) -> np.ndarray:
    """Build an inverse LUT (Mid Space → RGB) from scattered measurements.

    This function creates a LUT for the inverse direction: given a target
    color (in XYZ or Mid Space), what RGB values should be sent to the
    display to achieve that color?

    The challenge is that measured data points are scattered in XYZ space
    (not on a regular grid), so we must use scattered data interpolation.
    To improve interpolation quality, we first convert XYZ to "Mid Space".

    The process:
    1. Convert measured XYZ points to Mid Space (more uniform distribution)
    2. Create a regular grid in Mid Space [0, 1]³
    3. Interpolate from scattered Mid Space points to the regular grid
    4. The result maps Mid Space → measured RGB

    To use this LUT for XYZ → RGB conversion:
    1. Convert input XYZ to Mid Space using xyz_to_mid_space()
    2. Apply the LUT to get RGB values

    Parameters:
        measured_xyz: Array of shape (N, 3), measured XYZ values.
        measured_rgb: Array of shape (N, 3), corresponding RGB values in [0, 1].
        white_point_luminance: The Y value of the display white point (cd/m²).
        output_size: Size of the output LUT per dimension.
        method: Interpolation method - "linear", "nearest", or "rbf".
        rbf_kernel: Kernel for RBF interpolation (if method="rbf").
                    Options: "thin_plate_spline", "multiquadric", "gaussian", etc.
        rbf_smoothing: Smoothing parameter for RBF (0 = exact interpolation).

    Returns:
        Array of shape (output_size, output_size, output_size, 3),
        the 3D LUT mapping Mid Space → RGB.

    Example:
        >>> lut_data = build_inverse_lut(xyz, rgb, white_Y, 17, method="rbf")
        >>> lut = colour.LUT3D(table=lut_data, size=17)
        >>> # To use: mid = xyz_to_mid_space(target_xyz, white_Y)
        >>> #         rgb = lut.apply(mid)
    """
    # Step 1: Convert scattered XYZ points to Mid Space
    mid_space_points = xyz_to_mid_space(measured_xyz, white_point_luminance)

    # Step 2: Create regular grid in Mid Space [0, 1]³
    grid_vals = np.linspace(0, 1, output_size)
    m1, m2, m3 = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    query_points = np.stack([m1.ravel(), m2.ravel(), m3.ravel()], axis=1)

    # Step 3: Interpolate from scattered points to regular grid
    if method in ["linear", "nearest"]:
        # scipy.griddata for linear/nearest neighbor interpolation
        interpolated_rgb = griddata(
            points=mid_space_points,
            values=measured_rgb,
            xi=query_points,
            method=method,
            fill_value=0.0,  # Fill extrapolated regions with 0
        )
    elif method == "rbf":
        # RBF interpolation (per channel)
        interpolated_rgb = np.zeros((len(query_points), 3), dtype=np.float32)
        for channel in range(3):
            rbf = RBFInterpolator(
                mid_space_points,
                measured_rgb[:, channel],
                kernel=rbf_kernel,
                smoothing=rbf_smoothing,
            )
            interpolated_rgb[:, channel] = rbf(query_points)
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            f"Supported: 'linear', 'nearest', 'rbf'"
        )

    # Step 4: Clip and reshape to LUT format
    interpolated_rgb = np.clip(interpolated_rgb, 0.0, 1.0)
    lut_data = interpolated_rgb.reshape((output_size, output_size, output_size, 3))

    return lut_data
