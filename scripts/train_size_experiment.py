"""
Experiment: Evaluate GOG model performance vs training set size.

This script:
1. Loads two datasets (large for training, small for testing)
2. Randomly samples training points from the large dataset
3. Trains GOG models with different loss modes
4. Evaluates on the test set and plots results

Usage:
    python scripts/train_size_experiment.py

Authors: Jack Chou
Date: Nov 30, 2025
"""

import time
import numpy as np

from lossgog import (
    read_cs2000_csv,
    make_gog,
    evaluate_gog,
    plot_training_size_comparison,
)


# White point from measurements (D65 illuminant on display)
WHITE_POINT = np.array([474.94, 503.27, 579.53])


def run_experiment(
    train_rgb: np.ndarray,
    train_xyz: np.ndarray,
    test_rgb: np.ndarray,
    test_xyz: np.ndarray,
    sample_sizes: np.ndarray,
    n_repeats: int = 3,
    mode: str = "xyz",
    verbose: bool = False,
) -> dict:
    """Run training size experiment for a given loss mode.

    Parameters:
        train_rgb: Full training RGB data, shape (N, 3).
        train_xyz: Full training XYZ data, shape (N, 3).
        test_rgb: Test RGB data, shape (M, 3).
        test_xyz: Test XYZ data, shape (M, 3).
        sample_sizes: Array of training sample sizes to test.
        n_repeats: Number of random repeats per sample size.
        mode: Loss mode ('xyz', 'de2000', or 'sucs').
        verbose: Whether to print progress.

    Returns:
        Dict with keys:
            - 'sample_sizes': array of sample sizes
            - 'mean_de_all': (n_sizes, n_repeats) array of mean Delta E
            - 'max_de_all': (n_sizes, n_repeats) array of max Delta E
            - 'total_time': total execution time in seconds
    """
    start_time = time.time()
    n_train = train_rgb.shape[0]
    n_sizes = len(sample_sizes)

    mean_de_all = np.zeros((n_sizes, n_repeats))
    max_de_all = np.zeros((n_sizes, n_repeats))

    for i, size in enumerate(sample_sizes):
        size = int(size)
        if size > n_train:
            size = n_train

        if verbose:
            print(f"[{mode}] Sample size {size} ({i + 1}/{n_sizes})")

        for r in range(n_repeats):
            # Random sampling without replacement
            indices = np.random.choice(n_train, size=size, replace=False)
            rgb_sample = train_rgb[indices]
            xyz_sample = train_xyz[indices]

            # Train GOG model
            gog_model = make_gog(rgb_sample, xyz_sample, mode=mode, verbose=False)

            # Evaluate on test set
            metrics = evaluate_gog(gog_model, test_rgb, test_xyz)
            mean_de_all[i, r] = metrics["mean_delta_e"]
            max_de_all[i, r] = metrics["max_delta_e"]

            if verbose:
                print(
                    f"  Repeat {r + 1}: Mean ΔE = {metrics['mean_delta_e']:.3f}, "
                    f"Max ΔE = {metrics['max_delta_e']:.3f}"
                )

    total_time = time.time() - start_time
    return {
        "sample_sizes": sample_sizes,
        "mean_de_all": mean_de_all,
        "max_de_all": max_de_all,
        "total_time": total_time,
    }


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # --- Load datasets ---
    large_file = "measured_data/cs2000_lut_measurements_20251124_163713.csv"
    small_file = "measured_data/cs2000_lut_measurements_20251124_194618.csv"

    large_data = read_cs2000_csv(large_file)
    small_data = read_cs2000_csv(small_file)

    large_rgb = large_data["RGB"] / 255.0
    large_xyz = large_data["XYZ"] / WHITE_POINT[1]
    small_rgb = small_data["RGB"] / 255.0
    small_xyz = small_data["XYZ"] / WHITE_POINT[1]

    print(f"Large dataset (training pool): {large_rgb.shape[0]} points")
    print(f"Small dataset (test set): {small_rgb.shape[0]} points")

    # --- Define sample sizes ---
    n_intervals = 10
    sample_sizes = np.logspace(np.log10(10), np.log10(1000), n_intervals)
    sample_sizes = np.round(sample_sizes).astype(int)
    sample_sizes = np.unique(np.minimum(sample_sizes, large_rgb.shape[0]))

    print(f"\nSample sizes to test: {sample_sizes}")
    print("Number of repeats per size: 3")

    # --- Run experiments ---
    modes = ["xyz", "de2000", "sucs"]
    all_results = {}

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"Running experiment with {mode.upper()} loss mode...")
        print("=" * 60)
        results = run_experiment(
            large_rgb,
            large_xyz,
            small_rgb,
            small_xyz,
            sample_sizes,
            n_repeats=3,
            mode=mode,
            verbose=True,
        )
        all_results[mode] = results
        print(f"Total time: {results['total_time']:.2f} seconds")

    # --- Plot and save results ---
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    results_dict = {
        "XYZ MSE": all_results["xyz"]["mean_de_all"],
        "CIEDE2000": all_results["de2000"]["mean_de_all"],
        "DE sUCS": all_results["sucs"]["mean_de_all"],
    }

    plot_training_size_comparison(
        sample_sizes,
        results_dict,
        output_path="results/train_size_comparison.svg",
    )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    for mode in modes:
        results = all_results[mode]
        print(f"\n{mode.upper()} Mode:")
        print(f"  Min Mean ΔE: {np.min(np.mean(results['mean_de_all'], axis=1)):.3f}")
        print(f"  Min Max ΔE: {np.min(np.mean(results['max_de_all'], axis=1)):.3f}")


if __name__ == "__main__":
    main()
