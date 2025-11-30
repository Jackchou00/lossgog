"""
Apply GOG model to images for round-trip color conversion testing.

This script demonstrates the GOG model pipeline:
1. Input sRGB image → XYZ (standard sRGB to XYZ)
2. XYZ → Display RGB (via inverse GOG model)
3. Display RGB → XYZ (via forward GOG model)
4. XYZ → sRGB (for display/saving)

This round-trip test verifies the consistency of the GOG forward/inverse transforms.

Usage:
    python scripts/apply_gog.py

Authors: Jack Chou
Date: Dec 1, 2025
"""

import os
import numpy as np
import colour
from PIL import Image

from gog import rgb_to_xyz_gog, xyz_to_rgb_gog


def load_gog_model(model_path: str) -> dict:
    """Load GOG model parameters from .npz file.

    Parameters:
        model_path: Path to the .npz file containing GOG model parameters.

    Returns:
        Dict with keys: 'gain', 'offset', 'gamma', 'matrix'.
    """
    data = np.load(model_path)
    return {
        "gain": data["gain"],
        "offset": data["offset"],
        "gamma": data["gamma"],
        "matrix": data["matrix"],
    }


def apply_round_trip(
    image_path: str,
    gog_model: dict,
    output_path: str,
    display_luminance: float = 500.0,
) -> None:
    """Apply round-trip GOG conversion to an image.

    Pipeline:
        sRGB → XYZ → Display RGB (inverse GOG) → XYZ (forward GOG) → sRGB

    Parameters:
        image_path: Path to input sRGB image.
        gog_model: GOG model dict with gain, offset, gamma, matrix.
        output_path: Path to save the result.
        display_luminance: Peak luminance of the target display (cd/m²).
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Flatten and normalize to [0, 1]
    pixels = image_np.reshape(-1, 3) / 255.0
    pixels = np.clip(pixels, 0, 1)

    # Step 1: sRGB → XYZ (scaled by display luminance)
    xyz_normalized = colour.sRGB_to_XYZ(pixels)
    xyz_absolute = xyz_normalized * display_luminance

    # Step 2: XYZ → Display RGB using inverse GOG model
    display_rgb = xyz_to_rgb_gog(xyz_absolute, gog_model)

    # Step 3: Display RGB → XYZ using forward GOG model
    xyz_roundtrip = rgb_to_xyz_gog(display_rgb, gog_model)

    # Step 4: XYZ → sRGB for display
    xyz_normalized_out = xyz_roundtrip / display_luminance
    srgb_out = colour.XYZ_to_sRGB(xyz_normalized_out)

    # Clip, convert to uint8, and save
    srgb_out = np.clip(srgb_out, 0, 1)
    output_pixels = (srgb_out * 255).astype(np.uint8)
    output_image = output_pixels.reshape(h, w, 3)
    Image.fromarray(output_image).save(output_path)
    print(f"Round-trip image saved to: {output_path}")


def main():
    # Load GOG model
    model_path = "results/gog_model.npz"

    if not os.path.exists(model_path):
        print(f"GOG model not found: {model_path}")
        print("Please run scripts/train_gog.py first to generate the model.")
        return

    gog_model = load_gog_model(model_path)
    print("GOG Model loaded:")
    print(f"  Gain: {gog_model['gain']}")
    print(f"  Offset: {gog_model['offset']}")
    print(f"  Gamma: {gog_model['gamma']}")

    # Process test images
    test_images = ["test_image/1.jpg"]

    print("\n" + "=" * 60)
    print("Processing images with GOG round-trip")
    print("=" * 60)

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue

        print(f"\nProcessing: {image_path}")
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_gog_roundtrip{ext}"

        apply_round_trip(
            image_path,
            gog_model,
            output_path,
            display_luminance=300.0,
        )


if __name__ == "__main__":
    main()
