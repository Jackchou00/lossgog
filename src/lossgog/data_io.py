"""
Data I/O utilities for color measurement data.

Authors: Jack Chou
Date: Nov 30, 2025
"""

import os
import numpy as np
import pandas as pd
import colour


def read_cs2000_csv(file_path: str, spectral_length: int = 401, calculate_from_spec: bool = False) -> dict:
    """Read a CS2000 measurement CSV and return arrays for RGB, XYZ and spectral data.

    The CSV is expected to have at least the following columns in order:
    R, G, B, X, Y, Z, followed by spectral samples (one column per wavelength).

    Parameters:
        file_path: Path to the CSV file.
        spectral_length: Number of spectral columns expected per row (default 401).
        calculate_from_spec: If True, calculate XYZ from spectral data using CIE 1931 2° observer
                            with k=683 lm/W. If False, use the XYZ values from the CSV (default False).

    Returns:
        Dict with keys:
            - "RGB": array of shape (n_samples, 3), uint8 values.
            - "XYZ": array of shape (n_samples, 3), float values.
            - "spectral": array of shape (n_samples, spectral_length), float values.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: For column count mismatch or non-numeric data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path, header=0)

    n_rows, n_cols = df.shape
    required_cols = 6 + spectral_length
    if n_cols < required_cols:
        raise ValueError(
            f"CSV has {n_cols} columns but at least {required_cols} are required "
            f"(6 colorimetric + {spectral_length} spectral)."
        )

    df = df.iloc[:, :required_cols]

    # Convert numeric columns robustly
    try:
        rgb = (
            df.iloc[:, 0:3]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.uint8)
        )
        xyz = (
            df.iloc[:, 3:6]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        spectral = (
            df.iloc[:, 6 : 6 + spectral_length]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
    except Exception as exc:
        raise ValueError(f"Failed to convert CSV columns to numeric: {exc}")

    # Validate column counts
    if rgb.shape[1] != 3:
        raise ValueError(f"RGB array should have 3 columns, got {rgb.shape[1]}")
    if xyz.shape[1] != 3:
        raise ValueError(f"XYZ array should have 3 columns, got {xyz.shape[1]}")
    if spectral.shape[1] != spectral_length:
        raise ValueError(
            f"Spectral array should have {spectral_length} columns, got {spectral.shape[1]}"
        )

    # Check for NaNs introduced by non-numeric entries
    if np.isnan(xyz).any() or np.isnan(spectral).any():
        raise ValueError(
            "Non-numeric or missing values found in CSV. Please check the file."
        )

    # Calculate XYZ from spectral data if requested
    if calculate_from_spec:
        # Determine wavelength range from column names if available
        df_full = pd.read_csv(file_path, header=0)
        wavelengths = []
        
        # Try to extract wavelengths from column names
        for col in df_full.columns[6:6 + spectral_length]:
            if col.startswith('nm_'):
                try:
                    wl = int(col.split('_')[1])
                    wavelengths.append(wl)
                except (ValueError, IndexError):
                    pass
        
        # If we couldn't extract wavelengths from column names, use default range
        if len(wavelengths) != spectral_length:
            # Default: assume 380-780nm range with 1nm step for 401 wavelengths
            wavelengths = np.linspace(380, 780, spectral_length)
        else:
            wavelengths = np.array(wavelengths)
        
        # Get CIE 1931 2° Standard Observer
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        
        # Calculate XYZ for each spectrum using k=683 lm/W (CIE standard)
        xyz_calculated = np.zeros((spectral.shape[0], 3), dtype=np.float32)
        
        for i in range(spectral.shape[0]):
            # Create a SpectralDistribution object
            sd = colour.SpectralDistribution(
                dict(zip(wavelengths, spectral[i, :])),
                name=f'sample_{i}'
            )
            
            # Calculate XYZ with k=683 lm/W (CIE standard constant)
            # This converts radiometric quantities to photometric quantities
            xyz_calc = colour.sd_to_XYZ(sd, cmfs, k=683)
            xyz_calculated[i, :] = xyz_calc.astype(np.float32)
        
        xyz = xyz_calculated

    return {"RGB": rgb, "XYZ": xyz, "spectral": spectral}
