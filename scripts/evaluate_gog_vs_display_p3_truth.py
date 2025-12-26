"""Evaluate how well GOG fits Display P3 ground truth.

Ground truth:
  XYZ = RGB_to_XYZ(cctf_decoding(RGB_encoded), colourspace=Display P3)

GOG model:
  L = (gain * RGB + offset)^gamma
  XYZ = L @ M.T

This script trains on a coarse RGB grid and evaluates on random samples.
"""

from __future__ import annotations

import numpy as np
import colour

from lossgog import make_gog, rgb_to_xyz_gog
from lossgog.ucs import calculate_de2000


def _rgb_grid(levels: int) -> np.ndarray:
    v = np.linspace(0.0, 1.0, int(levels), dtype=np.float64)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3)


def p3_truth_rgb_to_xyz(rgb_encoded: np.ndarray) -> np.ndarray:
    cs = colour.RGB_COLOURSPACES["Display P3"]
    rgb_linear = cs.cctf_decoding(rgb_encoded)
    xyz = colour.RGB_to_XYZ(rgb_linear, colourspace=cs)
    return np.asarray(xyz, dtype=np.float64)


def _report_errors(name: str, xyz_pred: np.ndarray, xyz_true: np.ndarray) -> None:
    diff = xyz_pred - xyz_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))

    de = calculate_de2000(xyz_pred, xyz_true)
    mean_de = float(np.mean(de))
    max_de = float(np.max(de))

    print(f"{name}:")
    print(f"  XYZ RMSE: {rmse:.6g}")
    print(f"  XYZ max |err|: {max_abs:.6g}")
    print(f"  ΔE00 mean: {mean_de:.6g}")
    print(f"  ΔE00 max:  {max_de:.6g}")


def main():
    train_levels = 9
    test_n = 20000
    seed = 0

    train_rgb = _rgb_grid(train_levels)
    train_xyz = p3_truth_rgb_to_xyz(train_rgb)

    white_xyz = p3_truth_rgb_to_xyz(np.array([[1.0, 1.0, 1.0]], dtype=np.float64))[0]
    train_xyz = train_xyz / white_xyz[1]

    gog = make_gog(train_rgb, train_xyz, mode="sucs", verbose=True)

    pred_train = rgb_to_xyz_gog(train_rgb, gog)
    _report_errors("Train grid", pred_train, train_xyz)

    rng = np.random.default_rng(seed)
    test_rgb = rng.random((test_n, 3), dtype=np.float64)
    test_xyz = p3_truth_rgb_to_xyz(test_rgb) / white_xyz[1]

    pred_test = rgb_to_xyz_gog(test_rgb, gog)
    _report_errors("Random test", pred_test, test_xyz)

    mid = (test_rgb > 0.02).all(axis=1)
    dark = (test_rgb < 0.02).any(axis=1)

    if np.any(mid):
        _report_errors(
            "Random test (RGB>0.02 all channels)", pred_test[mid], test_xyz[mid]
        )
    if np.any(dark):
        _report_errors(
            "Random test (any channel <0.02)", pred_test[dark], test_xyz[dark]
        )

    cs = colour.RGB_COLOURSPACES["Display P3"]
    true_m = np.asarray(cs.matrix_RGB_to_XYZ, dtype=np.float64)
    fit_m = np.asarray(gog["matrix"], dtype=np.float64)

    print("\nMatrix comparison (note: GOG matrix absorbs TRC mismatch):")
    print("  True Display P3 matrix_RGB_to_XYZ:\n", true_m)
    print("  Fitted GOG matrix:\n", fit_m)

    print("\nFitted per-channel parameters:")
    print("  gain :", np.asarray(gog["gain"]))
    print("  offset:", np.asarray(gog["offset"]))
    print("  gamma:", np.asarray(gog["gamma"]))


if __name__ == "__main__":
    main()
