"""
Train GOG model for XGIMI projector characterization.

Usage:
    uv run src/xgimi_train.py

This script:
 1. Loads XGIMI measurement data (GOG_XGIMI.csv + Rgb96.mat)
 2. Trains a GOG model using 72 samples, validates on 24 samples
 3. Saves results to xgimi_results/

Author: Jack Chou
Date: Dec 1, 2025
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from gog import make_gog, evaluate_gog
from plotting import plot_delta_e_histogram


def parse_xgimi_csv(csv_path: str, expected_samples: int = 96) -> dict:
    """Parse the custom-format XGIMI CSV and return XYZ values.

    The CSV contains x, y chromaticity and Lv luminance values.
    XYZ is computed from these values.

    Returns:
        dict with 'XYZ' key containing (n_samples, 3) array.
    """
    df = pd.read_csv(csv_path, header=None, dtype=str)
    first_col = df.iloc[:, 0].astype(str)

    x_row = df[first_col == 'x']
    y_row = df[first_col == 'y']
    if x_row.empty or y_row.empty:
        raise ValueError('Failed to locate x/y chromaticity rows in CSV')

    x_idx = x_row.index[0]
    y_idx = y_row.index[0]

    lv_idx = df[first_col.str.startswith('Lv', na=False)].index
    if lv_idx.empty:
        raise ValueError('Failed to locate Lv row in CSV')
    lv_idx = lv_idx[0]

    def _row_values_as_float(row_index):
        vals = df.loc[row_index].iloc[1:1 + expected_samples]
        vals = pd.to_numeric(vals, errors='coerce').to_numpy(dtype=float)
        return vals

    x_vals = _row_values_as_float(x_idx)
    y_vals = _row_values_as_float(y_idx)
    Y_vals = _row_values_as_float(lv_idx)

    # Convert xyY to XYZ
    y_safe = np.where(y_vals == 0, 1e-12, y_vals)
    X = x_vals * (Y_vals / y_safe)
    Z = (1.0 - x_vals - y_vals) * (Y_vals / y_safe)
    XYZ = np.stack([X, Y_vals, Z], axis=1)

    return {'XYZ': XYZ}


def load_rgb96(mat_path: str) -> np.ndarray:
    """Load RGB values from .mat file.

    Returns:
        (96, 3) array with RGB values in [0, 1].
    """
    mat = loadmat(mat_path)
    rgb = None
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray):
            if v.size == 0:
                continue
            if v.shape == (96, 3):
                rgb = v.astype(float)
                break
            if v.shape == (3, 96):
                rgb = v.T.astype(float)
                break
            if v.shape[0] == 96 and v.ndim == 2 and v.shape[1] >= 3:
                rgb = v[:, :3].astype(float)
                break

    if rgb is None:
        raise ValueError('Failed to find a 96x3 RGB array in mat file')

    # Normalize if values are in 0-255 range
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    return rgb


def plot_response_curves(gog_model: dict, out_path: str) -> None:
    """Plot GOG response curves.

    GOG formula: L = (gain * RGB + offset)^gamma
    """
    inputs = np.linspace(0.0, 1.0, 512)
    gain = gog_model['gain']
    offset = gog_model['offset']
    gamma = gog_model['gamma']

    L = np.zeros((inputs.size, 3))
    for i in range(3):
        L[:, i] = np.power(np.maximum(gain[i] * inputs + offset[i], 0.0), gamma[i])

    plt.figure(figsize=(6, 4))
    plt.plot(inputs, L[:, 0], label='R', color='red')
    plt.plot(inputs, L[:, 1], label='G', color='green')
    plt.plot(inputs, L[:, 2], label='B', color='blue')
    plt.xlabel('Input (normalized RGB)')
    plt.ylabel('L (post-tone)')
    plt.title('GOG Response: L = (gain·RGB + offset)^γ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Response curves saved to: {out_path}')


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, 'xgimi_data')
    csv_path = os.path.join(data_dir, 'GOG_XGIMI.csv')
    mat_path = os.path.join(data_dir, 'Rgb96.mat')

    out_dir = os.path.join(repo_root, 'xgimi_results')

    # Clean and create output directory
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('='*60)
    print('XGIMI Projector GOG Model Training')
    print('='*60)

    # Load data
    print('\nLoading data...')
    parsed = parse_xgimi_csv(csv_path, expected_samples=96)
    XYZ_all = parsed['XYZ']
    rgb96 = load_rgb96(mat_path)

    # Prepare data
    n = 96
    rgb_all = np.clip(rgb96[:n, :], 0.0, 1.0)

    # Normalize XYZ by white point Y
    white_y = XYZ_all[:, 1].max()
    xyz_all = XYZ_all / white_y

    print(f'  Total samples: {n}')
    print(f'  White point Y: {white_y:.2f} cd/m²')

    # Split: 72 train, 24 test
    train_n = 72
    rgb_train = rgb_all[:train_n, :]
    xyz_train = xyz_all[:train_n, :]
    rgb_test = rgb_all[train_n:n, :]
    xyz_test = xyz_all[train_n:n, :]

    print(f'  Training samples: {train_n}')
    print(f'  Test samples: {n - train_n}')

    # Train GOG model with de2000 loss
    mode = 'de2000'
    print(f'\nTraining GOG model (loss mode: {mode})...')
    gog_model = make_gog(rgb_train, xyz_train, mode=mode, verbose=True)

    # Evaluate
    train_result = evaluate_gog(gog_model, rgb_train, xyz_train)
    test_result = evaluate_gog(gog_model, rgb_test, xyz_test)

    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)
    print(f'\nTraining set:')
    print(f"  Mean ΔE: {train_result['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {train_result['max_delta_e']:.3f}")
    print(f'\nTest set:')
    print(f"  Mean ΔE: {test_result['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {test_result['max_delta_e']:.3f}")

    print('\nModel Parameters:')
    print(f"  Gain:   {gog_model['gain']}")
    print(f"  Offset: {gog_model['offset']}")
    print(f"  Gamma:  {gog_model['gamma']}")
    print(f"  Matrix:\n{gog_model['matrix']}")

    # Save model
    np.savez(os.path.join(out_dir, 'gog_model.npz'),
             gain=gog_model['gain'], offset=gog_model['offset'],
             gamma=gog_model['gamma'], matrix=gog_model['matrix'])

    # Save metrics
    metrics = {
        'mode': mode,
        'train_n': int(train_n),
        'test_n': int(n - train_n),
        'white_y': float(white_y),
        'train_mean_delta_e': float(train_result['mean_delta_e']),
        'train_max_delta_e': float(train_result['max_delta_e']),
        'test_mean_delta_e': float(test_result['mean_delta_e']),
        'test_max_delta_e': float(test_result['max_delta_e']),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save parameters as text
    with open(os.path.join(out_dir, 'gog_params.txt'), 'w') as f:
        f.write("XGIMI Projector GOG Model Parameters\n")
        f.write("="*50 + "\n")
        f.write("Formula: L = (gain * RGB + offset)^gamma\n")
        f.write("         XYZ = M @ L\n\n")
        f.write(f"Gain:   {gog_model['gain'].tolist()}\n")
        f.write(f"Offset: {gog_model['offset'].tolist()}\n")
        f.write(f"Gamma:  {gog_model['gamma'].tolist()}\n")
        f.write(f"Matrix:\n{gog_model['matrix'].tolist()}\n")

    # Plot histograms
    plot_delta_e_histogram(train_result['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_train.png'),
                           title='Training Set ΔE Distribution')
    plot_delta_e_histogram(test_result['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_test.png'),
                           title='Test Set ΔE Distribution')

    # Plot response curves
    plot_response_curves(gog_model, os.path.join(out_dir, 'response_curves.png'))

    print(f'\n{"="*60}')
    print(f'All outputs saved to: {out_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
