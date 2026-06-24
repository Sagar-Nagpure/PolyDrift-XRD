"""
AM-Defect-2K: Synthetic Additive Manufacturing Defect Micrograph Generator
===========================================================================
Generates 2,000 synthetic grayscale micrographs simulating polished cross-sections
from laser powder-bed fusion (LPBF) additive manufacturing builds.

8-class classification with confounder biases, class imbalance, and a
compositional multi_defect trap designed to break autonomous AI agents.

Usage:
    python generate.py

Output:
    ./am_defect_2k/           <- dataset directory
    ./am_defect_2k.zip        <- ready for Eris upload

License: CC0 (Public Domain Dedication)
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = 512
N_IMAGES = 2000
SEED = 42

CLASS_NAMES = {
    0: 'porosity',
    1: 'lack_of_fusion',
    2: 'keyholing',
    3: 'balling',
    4: 'spatter',
    5: 'delamination',
    6: 'no_defect',
    7: 'multi_defect'
}

# Adjusted class prior (spatter/delamination bumped to 7%, no_defect to 20%)
CLASS_PRIOR = {
    0: 0.18,
    1: 0.18,
    2: 0.10,
    3: 0.10,
    4: 0.07,
    5: 0.07,
    6: 0.20,
    7: 0.10
}

MATERIALS = ['Ti64', 'IN718', 'AlSi']
MAGNIFICATIONS = [50, 200, 500]
ILLUMINATIONS = ['bright_field', 'dark_field', 'mixed']

MATERIAL_PARAMS = {
    'Ti64': {
        'grain_size': (15, 35),
        'base_intensity': 120,
        'contrast': 0.35,
        'melt_pool_width': (20, 40),
        'noise_std': 8
    },
    'IN718': {
        'grain_size': (20, 45),
        'base_intensity': 100,
        'contrast': 0.30,
        'melt_pool_width': (25, 50),
        'noise_std': 10
    },
    'AlSi': {
        'grain_size': (10, 25),
        'base_intensity': 140,
        'contrast': 0.40,
        'melt_pool_width': (15, 35),
        'noise_std': 6
    }
}

POLISHING_ARTIFACT_RATE = 0.03


# ============================================================
# Base Microstructure Renderer
# ============================================================

def render_base_microstructure(material, magnification, rng):
    params = MATERIAL_PARAMS[material]
    img = np.full((IMG_SIZE, IMG_SIZE), params['base_intensity'], dtype=np.float32)

    # Grain boundaries via Voronoi-like tessellation
    grain_min, grain_max = params['grain_size']
    scale_factor = 500.0 / magnification
    effective_grain_size = int(np.mean([grain_min, grain_max]) * scale_factor)
    effective_grain_size = max(5, effective_grain_size)

    n_grains = max(4, (IMG_SIZE // effective_grain_size) ** 2)
    seed_x = rng.integers(0, IMG_SIZE, size=n_grains)
    seed_y = rng.integers(0, IMG_SIZE, size=n_grains)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    grain_texture = np.zeros_like(img)
    for i in range(n_grains):
        dist = np.sqrt((xx - seed_x[i]) ** 2 + (yy - seed_y[i]) ** 2)
        grain_texture += np.exp(-(dist / (effective_grain_size * 0.5)) ** 2) * 5

    img -= grain_texture * params['contrast']

    # Melt-pool pattern
    pool_min, pool_max = params['melt_pool_width']
    pool_width = int(rng.uniform(pool_min, pool_max) * scale_factor)
    pool_width = max(10, pool_width)

    hatch_angle = rng.uniform(-0.3, 0.3)
    cos_a, sin_a = np.cos(hatch_angle), np.sin(hatch_angle)

    cx, cy = IMG_SIZE / 2.0, IMG_SIZE / 2.0
    xr = cos_a * (xx - cx) + sin_a * (yy - cy)
    pool_pattern = np.sin(2 * np.pi * xr / pool_width) * 8 * params['contrast']
    img += pool_pattern

    img += rng.normal(0, 2, img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


# ============================================================
# Defect Renderers
# ============================================================

def render_porosity(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_pores = int(rng.integers(3, 12))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_pores):
        cx = int(rng.integers(20, IMG_SIZE - 20))
        cy = int(rng.integers(20, IMG_SIZE - 20))
        radius = rng.uniform(5, 15) * scale_factor
        radius = max(3, min(radius, 40))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        void = np.exp(-(dist / radius) ** 2) * rng.uniform(40, 70)
        img -= void
    return img


def render_lack_of_fusion(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_regions = int(rng.integers(2, 6))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_regions):
        cx = int(rng.integers(30, IMG_SIZE - 30))
        cy = int(rng.integers(30, IMG_SIZE - 30))
        length = rng.uniform(30, 80) * scale_factor
        length = max(15, min(length, 150))
        width = rng.uniform(5, 12) * scale_factor
        width = max(3, min(width, 25))
        angle = rng.uniform(0, 2 * np.pi)

        dx = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        dy = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        modulation = 1 + 0.3 * np.sin(dy * 0.1)
        ellipse_dist = (dx / (length * modulation)) ** 2 + (dy / width) ** 2
        region = np.exp(-ellipse_dist * 2) * rng.uniform(50, 80)
        img -= region
    return img


def render_keyholing(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_keyholes = int(rng.integers(1, 4))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_keyholes):
        cx = int(rng.integers(40, IMG_SIZE - 40))
        cy = int(rng.integers(40, IMG_SIZE - 40))
        depth = rng.uniform(40, 80) * scale_factor
        depth = max(20, min(depth, 120))
        top_width = rng.uniform(8, 15) * scale_factor
        top_width = max(4, min(top_width, 25))

        dy = yy - cy
        dx = xx - cx
        mask = dy >= 0
        local_width = top_width * (1 - 0.7 * np.clip(dy / depth, 0, 1))
        local_width = np.maximum(local_width, 2)
        keyhole_dist = (dx / local_width) ** 2 + (dy / depth) ** 2
        keyhole = np.exp(-keyhole_dist * 3) * rng.uniform(60, 90)
        keyhole[~mask] *= 0.3
        img -= keyhole
    return img


def render_balling(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_beads = int(rng.integers(5, 15))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_beads):
        cx = int(rng.integers(15, IMG_SIZE - 15))
        cy = int(rng.integers(15, IMG_SIZE - 15))
        radius = rng.uniform(8, 18) * scale_factor
        radius = max(4, min(radius, 35))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bead = np.where(
            dist < radius,
            30 * np.cos(np.pi * dist / (2 * radius)) + 20,
            0
        )
        ring = np.exp(-((dist - radius) / 3) ** 2) * 20
        img += bead - ring
    return img


def render_spatter(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_particles = int(rng.integers(20, 60))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_particles):
        cx = int(rng.integers(0, IMG_SIZE))
        cy = int(rng.integers(0, IMG_SIZE))
        radius = rng.uniform(1, 4) * scale_factor
        radius = max(1, min(radius, 8))
        brightness = rng.uniform(30, 60)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        particle = np.exp(-(dist / radius) ** 2) * brightness
        img += particle
    return img


def render_delamination(img, magnification, rng):
    scale_factor = 500.0 / magnification
    n_cracks = int(rng.integers(1, 3))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    for _ in range(n_cracks):
        y_pos = int(rng.integers(50, IMG_SIZE - 50))
        crack_length = rng.uniform(200, 400) * scale_factor
        crack_length = min(crack_length, IMG_SIZE - 20)
        crack_width = rng.uniform(2, 5) * scale_factor
        crack_width = max(1, min(crack_width, 8))

        wave = rng.normal(0, 3, IMG_SIZE).cumsum()
        wave = wave - wave.mean()
        wave = np.clip(wave, -10, 10)

        crack_y = y_pos + wave[np.newaxis, :]
        crack_dist = np.abs(yy - crack_y)
        crack = np.exp(-(crack_dist / crack_width) ** 2) * rng.uniform(60, 90)

        x_center = IMG_SIZE / 2.0
        length_mask = np.exp(-((xx - x_center) / (crack_length / 2)) ** 2)
        crack *= length_mask
        img -= crack
    return img


DEFECT_RENDERERS = {
    0: render_porosity,
    1: render_lack_of_fusion,
    2: render_keyholing,
    3: render_balling,
    4: render_spatter,
    5: render_delamination,
}


# ============================================================
# Illumination Model
# ============================================================

def apply_illumination(img, illumination, rng):
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]

    if illumination == 'bright_field':
        gradient_angle = rng.uniform(0, 2 * np.pi)
        gradient = (xx * np.cos(gradient_angle) + yy * np.sin(gradient_angle)) / IMG_SIZE
        img += gradient * 15

    elif illumination == 'dark_field':
        cy, cx = IMG_SIZE / 2.0, IMG_SIZE / 2.0
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = (dist / (IMG_SIZE * 0.5)) ** 2 * 25
        img = img * 0.7 + ring

    elif illumination == 'mixed':
        split_pos = int(rng.integers(IMG_SIZE // 4, 3 * IMG_SIZE // 4))
        split_angle = rng.uniform(0, 2 * np.pi)
        threshold = xx * np.cos(split_angle) + yy * np.sin(split_angle)
        mask = threshold > split_pos
        img = img.copy()
        img[~mask] *= 0.6

    return np.clip(img, 0, 255)


# ============================================================
# Post-Processing
# ============================================================

def apply_magnification_effects(img, magnification, rng):
    if magnification == 500:
        img += rng.normal(0, 3, img.shape).astype(np.float32)
    elif magnification == 200:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
    elif magnification == 50:
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


def add_polishing_artifacts(img, rng):
    if rng.random() < POLISHING_ARTIFACT_RATE:
        start_x = int(rng.integers(0, IMG_SIZE))
        start_y = int(rng.integers(0, IMG_SIZE))
        length = int(rng.integers(100, 300))
        angle = rng.uniform(0, 2 * np.pi)
        thickness = int(rng.integers(1, 3))

        end_x = int(np.clip(start_x + length * np.cos(angle), 0, IMG_SIZE - 1))
        end_y = int(np.clip(start_y + length * np.sin(angle), 0, IMG_SIZE - 1))

        scratch_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        cv2.line(scratch_img, (start_x, start_y), (end_x, end_y), 1.0, thickness)
        scratch_img = cv2.GaussianBlur(scratch_img, (5, 5), 1.0)

        if rng.random() < 0.5:
            img += scratch_img * 30
        else:
            img -= scratch_img * 30
    return img


def add_sensor_noise(img, material, rng):
    params = MATERIAL_PARAMS[material]
    img += rng.normal(0, params['noise_std'], img.shape).astype(np.float32)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy, cx = IMG_SIZE / 2.0, IMG_SIZE / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vignette = 1 - 0.15 * (dist / (IMG_SIZE * 0.5)) ** 2
    img *= vignette
    return np.clip(img, 0, 255)


# ============================================================
# Confounder Sampling
# ============================================================

def sample_confounders(label, rng):
    material_probs = np.array([0.34, 0.33, 0.33])
    mag_probs = np.array([0.34, 0.33, 0.33])
    illum_probs = np.array([0.34, 0.33, 0.33])

    if label == 0:
        material_probs = np.array([0.60, 0.20, 0.20])
    elif label == 1:
        material_probs = np.array([0.20, 0.60, 0.20])

    if label == 4:
        mag_probs = np.array([0.20, 0.20, 0.60])
    elif label == 5:
        mag_probs = np.array([0.60, 0.20, 0.20])

    if label == 6:
        illum_probs = np.array([0.60, 0.20, 0.20])

    material = MATERIALS[int(rng.choice(3, p=material_probs))]
    magnification = MAGNIFICATIONS[int(rng.choice(3, p=mag_probs))]
    illumination = ILLUMINATIONS[int(rng.choice(3, p=illum_probs))]
    return material, magnification, illumination


# ============================================================
# Single Image Generation
# ============================================================

def generate_single_image(label, rng):
    material, magnification, illumination = sample_confounders(label, rng)
    img = render_base_microstructure(material, magnification, rng)

    if label == 7:
        available = list(DEFECT_RENDERERS.keys())
        n_defects = int(rng.integers(2, min(4, len(available) + 1)))
        chosen = rng.choice(available, size=n_defects, replace=False)
        for d in chosen:
            img = DEFECT_RENDERERS[int(d)](img, magnification, rng)
    elif label in DEFECT_RENDERERS:
        img = DEFECT_RENDERERS[label](img, magnification, rng)

    img = apply_illumination(img, illumination, rng)
    img = apply_magnification_effects(img, magnification, rng)
    img = add_polishing_artifacts(img, rng)
    img = add_sensor_noise(img, material, rng)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, material, magnification, illumination


# ============================================================
# Requirements File Content
# ============================================================

REQUIREMENTS_TXT = """numpy>=1.24
Pillow>=10.0
opencv-python>=4.8
pandas>=2.0
"""


# ============================================================
# Dataset Description Markdown
# ============================================================

DATASET_DESCRIPTION = """# Dataset Description

## Overview

AM-Defect-2K is a fully synthetic dataset of 2,000 grayscale micrographs (512x512 px, PNG) simulating optical microscopy of polished cross-sections from laser powder-bed fusion (LPBF) additive manufacturing builds. Each image depicts a microstructure region that may contain one of six defect types, no defect, or a co-occurrence of multiple defect types. The dataset is stratified across three confounder axes -- material alloy, optical magnification, and simulated illumination -- whose distributions vary across defect classes, reflecting natural biases in real-world AM data collection.

This dataset is 100% synthetically generated using a procedural pipeline that renders physically-inspired microstructures (Voronoi grain boundaries, melt-pool patterns) and parametric defect morphologies. The generation script (`generate.py`) is included in the archive and is fully reproducible with a hardcoded random seed (42). No external or third-party data was used.

## File Structure

```
am_defect_2k/
├── generate.py          # Fully reproducible synthesis script
├── requirements.txt     # Python dependencies
└── raw/
    ├── metadata.csv     # Image metadata and labels (2000 rows)
    └── images/
        ├── 00000.png
        ├── 00001.png
        ├── ...
        └── 01999.png
```

## Features

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | `str` | Unique identifier (5-digit zero-padded), maps to `{image_id}.png` |
| `material` | `str` | Alloy code: `Ti64` (Ti-6Al-4V), `IN718` (Inconel 718), `AlSi` (AlSi10Mg) |
| `magnification` | `int` | Objective magnification: `50`, `200`, or `500` |
| `illumination` | `str` | Illumination mode: `bright_field`, `dark_field`, or `mixed` |
| `label` | `int` | Defect class (0-7), see class table below |

### Defect Classes

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | `porosity` | Small, near-circular dark voids (10-40 um apparent diameter) |
| 1 | `lack_of_fusion` | Irregular elongated dark regions at melt-pool boundaries |
| 2 | `keyholing` | Deep, narrow keyhole-shaped voids with tapered profile |
| 3 | `balling` | Spherical bead clusters on the surface |
| 4 | `spatter` | Small bright particulate ejecta scattered across field |
| 5 | `delamination` | Horizontal crack-like separations between layers |
| 6 | `no_defect` | Clean, well-formed melt-pool microstructure |
| 7 | `multi_defect` | Co-occurrence of 2 or more of the above defect types |

### Contextual Notes

- **Image format**: Grayscale PNG, 512x512 pixels, 8-bit depth
- **Material variation**: Each alloy (Ti64, IN718, AlSi) exhibits distinct grain size distributions, base intensity, contrast, and sensor noise characteristics
- **Magnification effects**: Higher magnification (500x) increases apparent defect size and fine detail; lower magnification (50x) introduces blur and reduces visible field of view
- **Illumination modes**: Bright-field (uniform with slight directional gradient), dark-field (inverted ring pattern with darker center), and mixed (partial illumination with one half darkened) significantly alter defect visibility and contrast
- **Controlled complexities**: The dataset includes simulated polishing artifacts (streaks/scratches) in approximately 3% of images, and natural parameter variation in defect renderers creates borderline cases where defect severity is near the classification threshold
- **Reproducibility**: The entire dataset is regenerated by running `python generate.py --output_dir ./am_defect_2k --n_images 2000 --seed 42`
"""


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate AM-Defect-2K synthetic dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./am_defect_2k',
        help='Output directory for dataset (default: ./am_defect_2k)'
    )
    parser.add_argument(
        '--n_images', type=int, default=N_IMAGES,
        help=f'Number of images to generate (default: {N_IMAGES})'
    )
    parser.add_argument(
        '--seed', type=int, default=SEED,
        help=f'Random seed for reproducibility (default: {SEED})'
    )
    parser.add_argument(
        '--no_zip', action='store_true',
        help='Skip ZIP creation'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'raw' / 'images'
    raw_dir = output_dir / 'raw'

    # Clean existing directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Seed both legacy and modern RNG for full determinism
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("AM-Defect-2K: Synthetic AM Defect Micrograph Generator")
    print("=" * 60)
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Total images     : {args.n_images}")
    print(f"  Image size       : {IMG_SIZE}x{IMG_SIZE} grayscale")
    print(f"  Random seed      : {args.seed}")
    print(f"  Classes          : {len(CLASS_NAMES)}")
    print("=" * 60)
    print()

    # Sample labels from class prior
    class_ids = list(CLASS_PRIOR.keys())
    class_probs = list(CLASS_PRIOR.values())
    raw_labels = rng.choice(class_ids, size=args.n_images, p=class_probs)

    metadata = []

    for i in range(args.n_images):
        label = int(raw_labels[i])
        img, material, magnification, illumination = generate_single_image(label, rng)

        img_id = f"{i:05d}"
        img_path = images_dir / f"{img_id}.png"
        Image.fromarray(img, mode='L').save(img_path)

        metadata.append({
            'image_id': img_id,
            'material': material,
            'magnification': magnification,
            'illumination': illumination,
            'label': label
        })

        if (i + 1) % 200 == 0:
            pct = (i + 1) / args.n_images * 100
            print(f"  [{pct:5.1f}%] Generated {i + 1}/{args.n_images} images")

    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = raw_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)

    print()
    print("-" * 60)
    print("GENERATION COMPLETE")
    print("-" * 60)

    # Print class distribution
    print()
    print("Class Distribution:")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = int((df['label'] == cls_id).sum())
        pct = count / args.n_images * 100
        bar = '#' * int(pct / 2)
        print(f"  {cls_id} {cls_name:<16s} {count:5d}  ({pct:4.1f}%)  {bar}")

    # Print confounder distributions
    print()
    print("Material Distribution:")
    for mat in MATERIALS:
        count = int((df['material'] == mat).sum())
        pct = count / args.n_images * 100
        print(f"  {mat:<8s} {count:5d}  ({pct:4.1f}%)")

    print()
    print("Magnification Distribution:")
    for mag in MAGNIFICATIONS:
        count = int((df['magnification'] == mag).sum())
        pct = count / args.n_images * 100
        print(f"  {mag:>4d}x   {count:5d}  ({pct:4.1f}%)")

    print()
    print("Illumination Distribution:")
    for illum in ILLUMINATIONS:
        count = int((df['illumination'] == illum).sum())
        pct = count / args.n_images * 100
        print(f"  {illum:<14s} {count:5d}  ({pct:4.1f}%)")

    # Copy generate.py and requirements.txt into output directory
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, output_dir / 'generate.py')

    req_path = output_dir / 'requirements.txt'
    req_path.write_text(REQUIREMENTS_TXT)

    # Write dataset description
    desc_path = output_dir / 'DATASET_DESCRIPTION.md'
    desc_path.write_text(DATASET_DESCRIPTION)

    print()
    print("-" * 60)
    print("FILES WRITTEN:")
    print(f"  {output_dir / 'generate.py'}")
    print(f"  {output_dir / 'requirements.txt'}")
    print(f"  {output_dir / 'DATASET_DESCRIPTION.md'}")
    print(f"  {metadata_path}")
    print(f"  {images_dir}/  ({args.n_images} PNG files)")

    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in output_dir.rglob('*') if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    print(f"  Total size: {size_mb:.1f} MB")

    # Create ZIP archive
    if not args.no_zip:
        print()
        print("-" * 60)
        print("Creating ZIP archive...")

        zip_base = str(output_dir.resolve())
        zip_path = shutil.make_archive(
            base_name=zip_base,
            format='zip',
            root_dir=str(output_dir.parent.resolve()),
            base_dir=output_dir.name
        )

        zip_size = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  ZIP created: {zip_path}")
        print(f"  ZIP size: {zip_size:.1f} MB")

    print()
    print("=" * 60)
    print("ALL DONE!")
    if not args.no_zip:
        print(f"  Upload this file to Eris: {zip_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
