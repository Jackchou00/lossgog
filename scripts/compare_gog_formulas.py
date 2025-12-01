"""
Comprehensive comparison of two GOG formula variants.

This script compares:
 - v1 (legacy):   L = gain * (RGB + offset)^gamma  [gain outside power]
 - v2 (standard): L = (gain * RGB + offset)^gamma  [gain inside power]

Conclusion: Both formulas achieve equivalent fitting accuracy because
when offset ≈ 0, they become mathematically equivalent. However, v2
(standard form) has better parameter interpretability and matches
physical display models more closely.

Usage:
    uv run scripts/compare_gog_formulas.py

Author: Jack Chou
Date: Dec 1, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_io import read_cs2000_csv
from ucs import calculate_de2000, calculate_de_sucs
import colour


# ==============================================================================
# GOG v1 (Legacy): L = gain * (RGB + offset)^gamma
# NOTE: gain is OUTSIDE power, redundant with matrix
# ==============================================================================

def rgb_to_xyz_gog_v1(rgb, gog_model):
    """GOG v1: L = gain * (RGB + offset)^gamma"""
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]
    
    rgb_offset = np.maximum(rgb + offset, 0.0)
    L = gain * np.power(rgb_offset, gamma)
    xyz = L @ matrix.T
    return xyz


def _pack_params(gain, offset, gamma, matrix):
    return np.concatenate([gain, offset, gamma, matrix.flatten()])


def _unpack_params(params):
    gain = params[0:3]
    offset = params[3:6]
    gamma = params[6:9]
    matrix = params[9:18].reshape(3, 3)
    return gain, offset, gamma, matrix


def _loss_v1(params, rgb, xyz_target, mode):
    gain, offset, gamma, matrix = _unpack_params(params)
    gog = {"gain": gain, "offset": offset, "gamma": gamma, "matrix": matrix}
    xyz_pred = rgb_to_xyz_gog_v1(rgb, gog)
    
    if mode == "xyz":
        return float(np.mean((xyz_pred - xyz_target) ** 2))
    elif mode == "de2000":
        de = calculate_de2000(xyz_pred, xyz_target)
        return float(np.mean(de**2))
    elif mode == "sucs":
        de = calculate_de_sucs(xyz_pred, xyz_target)
        return float(np.mean(de**2))


def make_gog_v1(rgb, xyz, mode="xyz", verbose=False):
    """Train GOG v1: L = gain * (RGB + offset)^gamma"""
    init_gain = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])
    init_offset = np.array([0.001, 0.001, 0.001])
    init_gamma = np.array([2.2, 2.2, 2.2])
    init_matrix = np.eye(3)
    init_params = _pack_params(init_gain, init_offset, init_gamma, init_matrix)
    
    max_xyz = xyz.max()
    bounds = ([(0.0, 2*max_xyz)]*3 + [(-0.1, 0.1)]*3 + 
              [(1.0, 4.0)]*3 + [(-2*max_xyz, 2*max_xyz)]*9)
    
    result = minimize(_loss_v1, init_params, args=(rgb, xyz, mode),
                      method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    
    gain, offset, gamma, matrix = _unpack_params(result.x)
    return {"gain": gain, "offset": offset, "gamma": gamma, "matrix": matrix}


# ==============================================================================
# GOG v2 (Standard): L = (gain * RGB + offset)^gamma
# ==============================================================================

def rgb_to_xyz_gog_v2(rgb, gog_model):
    """GOG v2: L = (gain * RGB + offset)^gamma"""
    gain = gog_model["gain"]
    offset = gog_model["offset"]
    gamma = gog_model["gamma"]
    matrix = gog_model["matrix"]
    
    linear = np.maximum(gain * rgb + offset, 0.0)
    L = np.power(linear, gamma)
    xyz = L @ matrix.T
    return xyz


def _loss_v2(params, rgb, xyz_target, mode):
    gain, offset, gamma, matrix = _unpack_params(params)
    gog = {"gain": gain, "offset": offset, "gamma": gamma, "matrix": matrix}
    xyz_pred = rgb_to_xyz_gog_v2(rgb, gog)
    
    if mode == "xyz":
        return float(np.mean((xyz_pred - xyz_target) ** 2))
    elif mode == "de2000":
        de = calculate_de2000(xyz_pred, xyz_target)
        return float(np.mean(de**2))
    elif mode == "sucs":
        de = calculate_de_sucs(xyz_pred, xyz_target)
        return float(np.mean(de**2))


def make_gog_v2(rgb, xyz, mode="xyz", verbose=False):
    """Train GOG v2: L = (gain * RGB + offset)^gamma"""
    init_gain = np.array([1.0, 1.0, 1.0])
    init_offset = np.array([0.001, 0.001, 0.001])
    init_gamma = np.array([2.2, 2.2, 2.2])
    max_xyz = np.array([xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()])
    init_matrix = np.diag(max_xyz)
    init_params = _pack_params(init_gain, init_offset, init_gamma, init_matrix)
    
    max_val = xyz.max()
    bounds = ([(0.1, 10.0)]*3 + [(-0.1, 0.1)]*3 + 
              [(1.0, 4.0)]*3 + [(-2*max_val, 2*max_val)]*9)
    
    result = minimize(_loss_v2, init_params, args=(rgb, xyz, mode),
                      method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    
    gain, offset, gamma, matrix = _unpack_params(result.x)
    return {"gain": gain, "offset": offset, "gamma": gamma, "matrix": matrix}


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate(gog, rgb, xyz, version):
    """Evaluate GOG model using Delta E 2000."""
    if version == "v1":
        xyz_pred = rgb_to_xyz_gog_v1(rgb, gog)
    else:
        xyz_pred = rgb_to_xyz_gog_v2(rgb, gog)
    
    lab_measured = colour.XYZ_to_Lab(xyz)
    lab_pred = colour.XYZ_to_Lab(xyz_pred)
    delta_e = colour.delta_E(lab_measured, lab_pred, method="CIE 2000")
    
    return {
        "mean_delta_e": float(np.mean(delta_e)),
        "max_delta_e": float(np.max(delta_e)),
    }


# ==============================================================================
# Data Loading
# ==============================================================================

def parse_xgimi_csv(csv_path, expected_samples=96):
    """Parse XGIMI measurement CSV."""
    df = pd.read_csv(csv_path, header=None, dtype=str)
    first_col = df.iloc[:, 0].astype(str)
    
    x_idx = df[first_col == 'x'].index[0]
    y_idx = df[first_col == 'y'].index[0]
    lv_idx = df[first_col.str.startswith('Lv', na=False)].index[0]
    
    def _row_values(row_index):
        vals = df.loc[row_index].iloc[1:1 + expected_samples]
        return pd.to_numeric(vals, errors='coerce').to_numpy(dtype=float)
    
    x_vals = _row_values(x_idx)
    y_vals = _row_values(y_idx)
    Y_vals = _row_values(lv_idx)
    
    y_safe = np.where(y_vals == 0, 1e-12, y_vals)
    X = x_vals * (Y_vals / y_safe)
    Z = (1.0 - x_vals - y_vals) * (Y_vals / y_safe)
    
    return np.stack([X, Y_vals, Z], axis=1)


def load_rgb96(mat_path):
    """Load RGB from .mat file."""
    mat = loadmat(mat_path)
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray) and v.shape == (96, 3):
            rgb = v.astype(float)
            if rgb.max() > 1.5:
                rgb /= 255.0
            return rgb
    raise ValueError('Failed to find RGB array')


# ==============================================================================
# Main
# ==============================================================================

def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_root, 'xgimi_results', 'formula_comparison')
    
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load datasets
    datasets = {}
    
    # XGIMI
    print("Loading XGIMI dataset...")
    xgimi_csv = os.path.join(repo_root, 'xgimi_data', 'GOG_XGIMI.csv')
    xgimi_mat = os.path.join(repo_root, 'xgimi_data', 'Rgb96.mat')
    xyz_xgimi = parse_xgimi_csv(xgimi_csv)
    rgb_xgimi = load_rgb96(xgimi_mat)
    white_y = xyz_xgimi[:, 1].max()
    xyz_xgimi /= white_y
    
    datasets['XGIMI'] = {
        'rgb_train': rgb_xgimi[:72], 'xyz_train': xyz_xgimi[:72],
        'rgb_test': rgb_xgimi[72:], 'xyz_test': xyz_xgimi[72:],
    }
    
    # Measured Large
    print("Loading measured_data large...")
    large_file = os.path.join(repo_root, 'measured_data', 'cs2000_lut_measurements_20251124_163713.csv')
    large_data = read_cs2000_csv(large_file)
    rgb_large = large_data['RGB'] / 255.0
    xyz_large = large_data['XYZ'] / large_data['XYZ'][:, 1].max()
    
    np.random.seed(42)
    n = len(rgb_large)
    idx = np.random.permutation(n)
    train_n = int(0.8 * n)
    
    datasets['Measured_Large'] = {
        'rgb_train': rgb_large[idx[:train_n]], 'xyz_train': xyz_large[idx[:train_n]],
        'rgb_test': rgb_large[idx[train_n:]], 'xyz_test': xyz_large[idx[train_n:]],
    }
    
    # Measured Small
    print("Loading measured_data small...")
    small_file = os.path.join(repo_root, 'measured_data', 'cs2000_lut_measurements_20251124_194618.csv')
    small_data = read_cs2000_csv(small_file)
    rgb_small = small_data['RGB'] / 255.0
    xyz_small = small_data['XYZ'] / small_data['XYZ'][:, 1].max()
    
    n_s = len(rgb_small)
    idx_s = np.random.permutation(n_s)
    train_n_s = int(0.75 * n_s)
    
    datasets['Measured_Small'] = {
        'rgb_train': rgb_small[idx_s[:train_n_s]], 'xyz_train': xyz_small[idx_s[:train_n_s]],
        'rgb_test': rgb_small[idx_s[train_n_s:]], 'xyz_test': xyz_small[idx_s[train_n_s:]],
    }
    
    # Run comparisons
    modes = ['xyz', 'de2000', 'sucs']
    all_results = []
    
    print("\n" + "="*80)
    print("GOG FORMULA COMPARISON: v1 vs v2")
    print("  v1: L = gain * (RGB + offset)^gamma   [gain outside]")
    print("  v2: L = (gain * RGB + offset)^gamma   [gain inside - standard]")
    print("="*80)
    
    for ds_name, ds in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        for mode in modes:
            print(f"\n  Mode: {mode.upper()}")
            
            # Train both
            gog_v1 = make_gog_v1(ds['rgb_train'], ds['xyz_train'], mode)
            gog_v2 = make_gog_v2(ds['rgb_train'], ds['xyz_train'], mode)
            
            # Evaluate
            train_v1 = evaluate(gog_v1, ds['rgb_train'], ds['xyz_train'], 'v1')
            test_v1 = evaluate(gog_v1, ds['rgb_test'], ds['xyz_test'], 'v1')
            train_v2 = evaluate(gog_v2, ds['rgb_train'], ds['xyz_train'], 'v2')
            test_v2 = evaluate(gog_v2, ds['rgb_test'], ds['xyz_test'], 'v2')
            
            diff = test_v1['mean_delta_e'] - test_v2['mean_delta_e']
            winner = 'v2' if diff > 0.001 else ('v1' if diff < -0.001 else 'tie')
            
            print(f"    Test ΔE: v1={test_v1['mean_delta_e']:.4f}, v2={test_v2['mean_delta_e']:.4f}, diff={diff:+.4f} → {winner}")
            
            all_results.append({
                'dataset': ds_name, 'mode': mode,
                'v1_test': test_v1['mean_delta_e'],
                'v2_test': test_v2['mean_delta_e'],
                'diff': diff,
                'v1_offset': gog_v1['offset'].tolist(),
                'v2_offset': gog_v2['offset'].tolist(),
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    v1_wins = sum(1 for r in all_results if r['diff'] < -0.001)
    v2_wins = sum(1 for r in all_results if r['diff'] > 0.001)
    ties = len(all_results) - v1_wins - v2_wins
    
    print(f"  v1 wins: {v1_wins}")
    print(f"  v2 wins: {v2_wins}")
    print(f"  Ties:    {ties}")
    print(f"\n  Average diff (v1-v2): {np.mean([r['diff'] for r in all_results]):+.6f}")
    
    # Check offset values
    print("\n  Offset values (key to equivalence):")
    for r in all_results[:3]:
        print(f"    {r['dataset']}/{r['mode']}: v1={np.mean(np.abs(r['v1_offset'])):.6f}, v2={np.mean(np.abs(r['v2_offset'])):.6f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
  Both formulas achieve nearly IDENTICAL accuracy because:
  
  1. Offset values are very small (≈0.001), so:
     v1: gain * (RGB + 0)^gamma ≈ gain * RGB^gamma
     v2: (gain * RGB + 0)^gamma ≈ (gain)^gamma * RGB^gamma
     
  2. The scaling difference is absorbed by the matrix M.
  
  3. Therefore, v2 is PREFERRED because:
     - Parameters have clearer physical meaning
     - gain controls input amplification (before nonlinearity)
     - Matches standard display characterization literature
""")
    
    # Save results
    with open(os.path.join(out_dir, 'comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, 'comparison_summary.csv'), index=False)
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
