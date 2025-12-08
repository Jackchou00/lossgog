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
import matplotlib.pyplot as plt

from loss_gog import classic_gog, make_gog, evaluate_gog, plot_delta_e_histogram, read_cs2000_csv

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
    csv_path = os.path.join(data_dir, 'xgimi_96_gog.csv')

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
    parsed = read_cs2000_csv(csv_path, spectral_length=771)
    XYZ_all = parsed['XYZ']
    rgb_all = parsed['RGB'] / 255.0  # Normalize RGB to [0, 1]

    # Prepare data
    n = 96
    rgb_all = np.clip(rgb_all[:n, :], 0.0, 1.0)

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

    # =========================================================================
    # 1. Classic per-channel GOG (from primary ramps)
    # =========================================================================
    print('\n' + '='*60)
    print('1. CLASSIC GOG (per-channel fitting from primary ramps)')
    print('='*60)
    # XGIMI data layout: 18 R, 18 G, 18 B, 18 W (indices 0-17, 18-35, 36-53, 54-71)
    ramp_indices = [(0, 18), (18, 36), (36, 54)]
    gog_classic = classic_gog(rgb_train, xyz_train, ramp_indices=ramp_indices, verbose=True)

    # Evaluate classic GOG
    classic_train = evaluate_gog(gog_classic, rgb_train, xyz_train)
    classic_test = evaluate_gog(gog_classic, rgb_test, xyz_test)

    print(f'\nClassic GOG - Training set:')
    print(f"  Mean ΔE: {classic_train['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {classic_train['max_delta_e']:.3f}")
    print(f'\nClassic GOG - Test set:')
    print(f"  Mean ΔE: {classic_test['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {classic_test['max_delta_e']:.3f}")

    # =========================================================================
    # 2. Optimized GOG (full optimization with DE2000 loss)
    # =========================================================================
    print('\n' + '='*60)
    print('2. OPTIMIZED GOG (full optimization with ΔE2000 loss)')
    print('='*60)
    gog_optim = make_gog(rgb_train, xyz_train, mode='de2000', verbose=True)

    # Evaluate optimized GOG
    optim_train = evaluate_gog(gog_optim, rgb_train, xyz_train)
    optim_test = evaluate_gog(gog_optim, rgb_test, xyz_test)

    print(f'\nOptimized GOG - Training set:')
    print(f"  Mean ΔE: {optim_train['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {optim_train['max_delta_e']:.3f}")
    print(f'\nOptimized GOG - Test set:')
    print(f"  Mean ΔE: {optim_test['mean_delta_e']:.3f}")
    print(f"  Max  ΔE: {optim_test['max_delta_e']:.3f}")

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print('\n' + '='*60)
    print('COMPARISON SUMMARY')
    print('='*60)

    print('\n{:<25} {:>12} {:>12}'.format('Metric', 'Classic', 'Optimized'))
    print('-' * 50)
    print('{:<25} {:>12.3f} {:>12.3f}'.format('Train Mean ΔE',
          classic_train['mean_delta_e'], optim_train['mean_delta_e']))
    print('{:<25} {:>12.3f} {:>12.3f}'.format('Train Max ΔE',
          classic_train['max_delta_e'], optim_train['max_delta_e']))
    print('{:<25} {:>12.3f} {:>12.3f}'.format('Test Mean ΔE',
          classic_test['mean_delta_e'], optim_test['mean_delta_e']))
    print('{:<25} {:>12.3f} {:>12.3f}'.format('Test Max ΔE',
          classic_test['max_delta_e'], optim_test['max_delta_e']))

    print('\n' + '='*60)
    print('PARAMETER COMPARISON')
    print('='*60)

    print('\n--- Gain ---')
    print(f"  Classic:   {gog_classic['gain']}")
    print(f"  Optimized: {gog_optim['gain']}")

    print('\n--- Offset ---')
    print(f"  Classic:   {gog_classic['offset']}")
    print(f"  Optimized: {gog_optim['offset']}")

    print('\n--- Gamma ---')
    print(f"  Classic:   {gog_classic['gamma']}")
    print(f"  Optimized: {gog_optim['gamma']}")

    print('\n--- Matrix ---')
    print(f"  Classic:\n{gog_classic['matrix']}")
    print(f"\n  Optimized:\n{gog_optim['matrix']}")

    # =========================================================================
    # Save results
    # =========================================================================

    # Save classic model
    np.savez(os.path.join(out_dir, 'gog_classic.npz'),
             gain=gog_classic['gain'], offset=gog_classic['offset'],
             gamma=gog_classic['gamma'], matrix=gog_classic['matrix'])

    # Save optimized model
    np.savez(os.path.join(out_dir, 'gog_optimized.npz'),
             gain=gog_optim['gain'], offset=gog_optim['offset'],
             gamma=gog_optim['gamma'], matrix=gog_optim['matrix'])

    # Save comparison metrics
    metrics = {
        'train_n': int(train_n),
        'test_n': int(n - train_n),
        'white_y': float(white_y),
        'classic': {
            'train_mean_delta_e': float(classic_train['mean_delta_e']),
            'train_max_delta_e': float(classic_train['max_delta_e']),
            'test_mean_delta_e': float(classic_test['mean_delta_e']),
            'test_max_delta_e': float(classic_test['max_delta_e']),
        },
        'optimized': {
            'mode': 'de2000',
            'train_mean_delta_e': float(optim_train['mean_delta_e']),
            'train_max_delta_e': float(optim_train['max_delta_e']),
            'test_mean_delta_e': float(optim_test['mean_delta_e']),
            'test_max_delta_e': float(optim_test['max_delta_e']),
        }
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save parameters comparison as text
    with open(os.path.join(out_dir, 'gog_params_comparison.txt'), 'w') as f:
        f.write("XGIMI Projector GOG Model Comparison\n")
        f.write("="*60 + "\n")
        f.write("Formula: L = (gain * RGB + offset)^gamma\n")
        f.write("         XYZ = M @ L\n\n")

        f.write("-"*60 + "\n")
        f.write("CLASSIC GOG (per-channel fitting)\n")
        f.write("-"*60 + "\n")
        f.write(f"Gain:   {gog_classic['gain'].tolist()}\n")
        f.write(f"Offset: {gog_classic['offset'].tolist()}\n")
        f.write(f"Gamma:  {gog_classic['gamma'].tolist()}\n")
        f.write(f"Matrix:\n{gog_classic['matrix'].tolist()}\n\n")

        f.write("-"*60 + "\n")
        f.write("OPTIMIZED GOG (ΔE2000 loss)\n")
        f.write("-"*60 + "\n")
        f.write(f"Gain:   {gog_optim['gain'].tolist()}\n")
        f.write(f"Offset: {gog_optim['offset'].tolist()}\n")
        f.write(f"Gamma:  {gog_optim['gamma'].tolist()}\n")
        f.write(f"Matrix:\n{gog_optim['matrix'].tolist()}\n")

    # Plot histograms for both models
    plot_delta_e_histogram(classic_train['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_classic_train.png'),
                           title='Classic GOG - Training Set ΔE Distribution')
    plot_delta_e_histogram(classic_test['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_classic_test.png'),
                           title='Classic GOG - Test Set ΔE Distribution')
    plot_delta_e_histogram(optim_train['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_optim_train.png'),
                           title='Optimized GOG - Training Set ΔE Distribution')
    plot_delta_e_histogram(optim_test['delta_e'],
                           os.path.join(out_dir, 'delta_e_hist_optim_test.png'),
                           title='Optimized GOG - Test Set ΔE Distribution')

    # Plot response curves for both models
    plot_response_curves(gog_classic, os.path.join(out_dir, 'response_curves_classic.png'))
    plot_response_curves(gog_optim, os.path.join(out_dir, 'response_curves_optim.png'))

    # Create comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Training comparison
    x = np.arange(2)
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width/2,
                   [classic_train['mean_delta_e'], classic_train['max_delta_e']],
                   width, label='Classic', color='steelblue')
    bars2 = ax.bar(x + width/2,
                   [optim_train['mean_delta_e'], optim_train['max_delta_e']],
                   width, label='Optimized', color='darkorange')
    ax.set_ylabel('ΔE2000')
    ax.set_title('Training Set Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean ΔE', 'Max ΔE'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Test comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2,
                   [classic_test['mean_delta_e'], classic_test['max_delta_e']],
                   width, label='Classic', color='steelblue')
    bars2 = ax.bar(x + width/2,
                   [optim_test['mean_delta_e'], optim_test['max_delta_e']],
                   width, label='Optimized', color='darkorange')
    ax.set_ylabel('ΔE2000')
    ax.set_title('Test Set Performance (24-color chart)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean ΔE', 'Max ΔE'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_chart.png'), dpi=150)
    plt.close()
    print(f"Comparison chart saved to: {os.path.join(out_dir, 'comparison_chart.png')}")

    print(f'\n{"="*60}')
    print(f'All outputs saved to: {out_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
