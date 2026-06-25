# OrbitMosaic: Multi-System Phase-Portrait Mosaics of Discrete 2D Dynamics

## Overview

OrbitMosaic is a fully procedural synthetic computer-vision dataset for
inverse identification of two-dimensional discrete dynamical systems
from sparse phase-portrait imagery. Each sample is a 384×384 RGB image
arranged as a 3×3 mosaic of nine 128×128 phase-portrait tiles. Each
mosaic is rendered from a mixture of **two to four different**
underlying maps drawn from six classical 2D dynamical systems; every
non-dropout tile is assigned to one of the maps in the mixture and is
rendered with independent zoom, offset, rotation, flip, and color
stylization. Four of nine tiles per mosaic are pure-noise distractors.

Six discrete maps:

| Index | Map             | Update                                                            | Reference                |
|-------|-----------------|-------------------------------------------------------------------|--------------------------|
| 0     | Hénon           | `x' = 1 − a x² + y;  y' = b x`                                     | Hénon 1976               |
| 1     | Lozi            | `x' = 1 − a |x| + y;  y' = b x`                                    | Lozi 1978                |
| 2     | Standard        | `p' = p + K sin(θ) (mod 2π);  θ' = θ + p' (mod 2π)`                | Chirikov 1979            |
| 3     | Tinkerbell      | `x' = x² − y² + c1 x + c2 y;  y' = 2 x y + c3 x + c4 y`            | Davis 1993               |
| 4     | Ikeda           | `x' = 1 + u (x cos t − y sin t);  y' = u (x sin t + y cos t)`      | Ikeda 1979               |
| 5     | Gingerbreadman  | `x' = 1 − y + |x|;  y' = x`                                         | Devaney 1984             |

Five classification targets are attached to each image:

1. **`m_0..m_5`** — six binary presence flags, one per map.
2. **`dominant_map`** — integer in `{ 0, 1, 2, 3, 4, 5 }`, the map covering the most non-dropout tiles (lower index wins ties).
3. **`parameter_regime`** — integer in `{ 0, 1, 2, 3 }`, regime band of the dominant map: `0 = stable / periodic`, `1 = period-doubling cascade`, `2 = intermittency / mixed`, `3 = fully developed chaos`.
4. **`attractor_type`** — integer in `{ 0, 1, 2, 3 }`: `0 = fixed point / short orbit`, `1 = periodic cycle`, `2 = quasi-periodic / invariant curve`, `3 = strange attractor`.
5. **`is_chaotic`** — `1` if the largest Lyapunov exponent of the dominant map is positive.

Hidden image-generation nuisances:

- Multi-system mixture (2–4 maps per mosaic).
- Tile dropout: 4 of 9 tiles are pure noise.
- Per-tile framing: independent zoom in `[ 0.25, 1.15 ]`, offset, 90° rotation, horizontal flip.
- Trajectory-budget covariate shift (long-dense train, short-sparse test).
- Ten cubic-polynomial LUTs per tile.
- Per-image gamma `[ 0.5, 2.0 ]`, hue rotation `±25°`, Gaussian blur `σ ∈ [ 0, 1.2 ]`.
- Three 16×16 black occlusion squares per image (bboxes published).
- JPEG re-encoding at quality `[ 18, 55 ]`.

## Source

Raw payload is one `data.csv` manifest plus `images/` and
`reference_images/` directories, all procedurally generated from seed
`0xC0FFEE17` via `generate.py`. No third-party data.

## File Structure

- `data.csv` — labels, regime tag, occlusion bboxes for all samples.
- `images/{sample_id}.png` — 384×384 RGB mosaics.
- `reference_images/map_{0..5}_clean.png` — six clean canonical single-system mosaics, fully-developed regime, LUT `L0`, no nuisances.

## Features

### Columns in `data.csv`

| Column            | Type   | Description                                                                                |
|-------------------|--------|--------------------------------------------------------------------------------------------|
| sample_id         | string | 12-character HMAC identifier; matches `images/{sample_id}.png`.                             |
| m_0..m_5          | int    | Binary presence flag per map, `0` or `1`.                                                   |
| dominant_map      | int    | In `{ 0, 1, 2, 3, 4, 5 }`.                                                                  |
| parameter_regime  | int    | In `{ 0, 1, 2, 3 }`.                                                                        |
| attractor_type    | int    | In `{ 0, 1, 2, 3 }`.                                                                        |
| is_chaotic        | int    | `0` or `1`.                                                                                 |
| regime_kind       | string | `"long"` or `"short"`; used by `prepare.py` for covariate-shifted split.                    |
| occ1_x..occ3_y    | int    | Three 16×16 occlusion-square top-left coordinates.                                          |

### Image assets

| Asset                                | Format | Dimensions  | Notes                                              |
|--------------------------------------|--------|-------------|----------------------------------------------------|
| `images/{sample_id}.png`             | PNG    | 384×384 RGB | All corruptions applied.                            |
| `reference_images/map_{m}_clean.png` | PNG    | 384×384 RGB | Single-system canonical reference.                  |

## Notes

- Parameter regimes are drawn from textbook regions for each map.
- Lyapunov exponents are estimated from finite-difference Jacobians.
- All randomness derives from the master seed `0xC0FFEE17` via per-sample HMAC-SHA256 keying; byte-for-byte reproducible.
- License: CC BY 4.0. Fully creator-owned.
