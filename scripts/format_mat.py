"""
To combine the RGB data and XYZ, SPD data into a single .csv file.

Author: Jack
Date: Dec, 6, 2025
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd

rgb_mat_path = "xgimi_data/Rgb96.mat"

# print all keys in the .mat file
mat_contents = loadmat(rgb_mat_path)
print(mat_contents.keys())

# load the RGB data
rgb_data = mat_contents["RGB"]
rgb_data = np.array(rgb_data)
print(f"RGB data shape: {rgb_data.shape}")  # should be (729, 3)


def parse_xgimi_csv(csv_path: str, expected_samples: int = 96) -> dict:
    """Parse the custom-format XGIMI CSV and return XYZ values.

    The CSV contains x, y chromaticity and Lv luminance values.
    XYZ is computed from these values.

    Returns:
        dict with 'XYZ' key containing (n_samples, 3) array.
    """
    df = pd.read_csv(csv_path, header=None, dtype=str)
    first_col = df.iloc[:, 0].astype(str)

    x_row = df[first_col == "x"]
    y_row = df[first_col == "y"]
    if x_row.empty or y_row.empty:
        raise ValueError("Failed to locate x/y chromaticity rows in CSV")

    x_idx = x_row.index[0]
    y_idx = y_row.index[0]

    lv_idx = df[first_col.str.startswith("Lv", na=False)].index
    if lv_idx.empty:
        raise ValueError("Failed to locate Lv row in CSV")
    lv_idx = lv_idx[0]

    def _row_values_as_float(row_index):
        vals = df.loc[row_index].iloc[1 : 1 + expected_samples]
        vals = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float)
        return vals

    x_vals = _row_values_as_float(x_idx)
    y_vals = _row_values_as_float(y_idx)
    Y_vals = _row_values_as_float(lv_idx)

    # Convert xyY to XYZ
    y_safe = np.where(y_vals == 0, 1e-12, y_vals)
    X = x_vals * (Y_vals / y_safe)
    Z = (1.0 - x_vals - y_vals) * (Y_vals / y_safe)
    XYZ = np.stack([X, Y_vals, Z], axis=1)

    # Read spectral power distribution (SPD) data
    # The SPD data starts from row where "Wavelength [nm]" appears
    spd_start_idx = df[first_col.str.contains("Wavelength", na=False)].index
    if spd_start_idx.empty:
        raise ValueError("Failed to locate SPD data starting with 'Wavelength' row")
    spd_start_idx = spd_start_idx[0] + 1  # SPD data starts from next row
    
    # Read all wavelength rows (from 230nm to 1000nm, 771 rows)
    spd_data = []
    for i in range(771):
        row_idx = spd_start_idx + i
        if row_idx >= len(df):
            break
        vals = df.loc[row_idx].iloc[1 : 1 + expected_samples]
        vals = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float)
        spd_data.append(vals)
    
    spd_data = np.array(spd_data).T  # Transpose to (n_samples, 771)
    
    return {"XYZ": XYZ, "SPD": spd_data}


csv_path = "xgimi_data/GOG_XGIMI.csv"
parsed_data = parse_xgimi_csv(csv_path, expected_samples=96)
xyz_data = parsed_data["XYZ"]
spd_data = parsed_data["SPD"]

print(f"XYZ data shape: {xyz_data.shape}")  # should be (96, 3)
print(f"SPD data shape: {spd_data.shape}")  # should be (96, 771)

# Convert RGB data to integers to ensure clean integer values
rgb_data_int = np.round(rgb_data).astype(np.int32)

# New header for the csv file
wavelengths = np.arange(230, 1001)  # 230 to 1000 inclusive, 771 values
header = "R,G,B,X,Y,Z," + ",".join([f"nm_{i}" for i in wavelengths])

# Combine the data
combined_data = np.hstack((rgb_data_int, xyz_data, spd_data))
print(f"Combined data shape: {combined_data.shape}")  # should be (96, 777)

# Save to csv file
output_csv_path = "xgimi_data/xgimi_96_gog.csv"

# Use pandas to write with better control over formatting
# Split RGB columns and float columns for different formatting
df_rgb = pd.DataFrame(rgb_data_int, columns=['R', 'G', 'B'])
df_xyz_spd = pd.DataFrame(
    np.hstack((xyz_data, spd_data)),
    columns=header.split(",")[3:]  # XYZ and SPD columns
)

# Combine and save
df_output = pd.concat([df_rgb, df_xyz_spd], axis=1)
df_output.to_csv(output_csv_path, index=False, float_format="%.6e")
print(f"Combined data saved to {output_csv_path}")
