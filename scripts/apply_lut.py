"""
Apply LUT to images for round-trip color conversion testing.

This script demonstrates the full color management pipeline:
1. Input sRGB image → XYZ (standard sRGB to XYZ)
2. XYZ → Mid Space (normalized intermediate space)
3. Mid Space → Display RGB (via inverse LUT)
4. Display RGB → XYZ (via forward LUT)
5. XYZ → sRGB (for display/saving)

This round-trip test verifies the consistency of the forward and inverse LUTs.

Usage:
    python scripts/apply_lut.py

Authors: Jack Chou
Date: Dec 1, 2025
"""

import os
import numpy as np
import colour
from PIL import Image

from lossgog import xyz_to_mid_space


# Display white point from measurements
WHITE_POINT = np.array([474.94, 503.27, 579.53])


def apply_round_trip(
    image_path: str,
    forward_lut: colour.LUT3D,
    inverse_lut: colour.LUT3D,
    output_path: str,
    target_luminance: float = 100.0,
) -> None:
    """Apply round-trip LUT conversion to an image.

    Pipeline:
        sRGB → XYZ → Mid Space → RGB (inverse LUT) → XYZ (forward LUT) → sRGB

    Parameters:
        image_path: Path to input sRGB image.
        forward_lut: LUT mapping RGB → XYZ.
        inverse_lut: LUT mapping Mid Space → RGB.
        output_path: Path to save the result.
        target_luminance: Target peak luminance for XYZ scaling (cd/m²).
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Flatten and normalize to [0, 1]
    pixels = image_np.reshape(-1, 3) / 255.0

    # Step 1: sRGB → XYZ (scaled to target luminance)
    xyz_normalized = colour.sRGB_to_XYZ(pixels)
    xyz_absolute = xyz_normalized * target_luminance

    # Step 2: XYZ → Mid Space
    mid_space = xyz_to_mid_space(xyz_absolute, WHITE_POINT[1])

    # Step 3: Mid Space → Display RGB (via inverse LUT)
    display_rgb = inverse_lut.apply(np.clip(mid_space, 0, 1))

    # Step 4: Display RGB → XYZ (via forward LUT)
    xyz_roundtrip = forward_lut.apply(np.clip(display_rgb, 0, 1))

    # Step 5: XYZ → sRGB for display
    xyz_normalized_out = xyz_roundtrip / target_luminance
    srgb_out = colour.XYZ_to_sRGB(xyz_normalized_out)

    # Clip, convert to uint8, and save
    srgb_out = np.clip(srgb_out, 0, 1)
    output_pixels = (srgb_out * 255).astype(np.uint8)
    output_image = output_pixels.reshape(h, w, 3)
    Image.fromarray(output_image).save(output_path)
    print(f"Round-trip image saved to: {output_path}")


def main():
    # Load LUTs
    forward_lut_path = "results/cs2000_rgb_to_xyz_65.cube"
    inverse_lut_path = "results/cs2000_mid_to_rgb_65_rbf-tps.cube"

    if not os.path.exists(forward_lut_path):
        print(f"Forward LUT not found: {forward_lut_path}")
        print("Please run scripts/build_lut.py first.")
        return

    if not os.path.exists(inverse_lut_path):
        print(f"Inverse LUT not found: {inverse_lut_path}")
        print("Please run scripts/build_lut.py first.")
        return

    forward_lut = colour.io.read_LUT(forward_lut_path)
    inverse_lut = colour.io.read_LUT(inverse_lut_path)
    print(f"Loaded forward LUT: {forward_lut_path}")
    print(f"Loaded inverse LUT: {inverse_lut_path}")

    # Process test images
    test_images = ["test_image/1.jpg"]

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue

        print(f"\nProcessing: {image_path}")
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_lut_roundtrip{ext}"

        apply_round_trip(
            image_path,
            forward_lut,
            inverse_lut,
            output_path,
            target_luminance=100.0,
        )


if __name__ == "__main__":
    main()
