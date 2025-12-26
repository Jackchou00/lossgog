"""Build a GOG model from synthetic Display P3 data and export an ICC profile.

This script does not use any measured dataset. It uses the Display P3 RGB
colourspace from the `colour-science` library to generate RGB->XYZ pairs.
"""

from __future__ import annotations

import numpy as np
import colour

from lossgog import make_gog
from lossgog.icc import generate_gog_icc_from_model


def _rgb_grid(levels: int) -> np.ndarray:
    v = np.linspace(0.0, 1.0, int(levels), dtype=np.float64)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    return rgb


def _display_p3_rgb_to_xyz(rgb_encoded: np.ndarray) -> np.ndarray:
    cs = colour.RGB_COLOURSPACES["Display P3"]

    rgb_linear = cs.cctf_decoding(rgb_encoded)
    xyz = colour.RGB_to_XYZ(rgb_linear, colourspace=cs)
    return np.asarray(xyz, dtype=np.float64)


def main():
    levels = 9
    out_icc = "display_p3_gog.icc"

    rgb = _rgb_grid(levels)
    xyz = _display_p3_rgb_to_xyz(rgb)

    white_xyz = _display_p3_rgb_to_xyz(np.array([[1.0, 1.0, 1.0]], dtype=np.float64))[0]
    xyz = xyz / white_xyz[1]

    gog_model = make_gog(rgb, xyz, mode="de2000", verbose=True)

    icc_bytes = generate_gog_icc_from_model(
        gog_model,
        description="Display P3 (synthetic) via GOG (Parametric)",
        assume_matrix_columns_are_primaries=True,
    )

    with open(out_icc, "wb") as f:
        f.write(icc_bytes)

    print(f"ICC saved to: {out_icc}")


if __name__ == "__main__":
    main()
