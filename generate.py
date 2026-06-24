"""
PolyDrift-XRD: deterministic synthetic powder XRD dataset generator.

Reproducibility:
    python generate.py --seed 42 --out ./polydrift_xrd_raw

Provenance of reference structures (Crystallography Open Database, CC0):
    alpha   : COD 1010368  (orthorhombic reference)
    beta    : COD 1011000  (monoclinic  reference)
    gamma   : COD 1528823  (triclinic   reference)
    amorphous: synthetic broad-halo model (no crystalline reference)

NOTE: The peak lists below are *stylized* approximations chosen so the three
crystalline phases share most low-angle peaks and differ mainly in intensity
ratios and high-angle fingerprints. This is intentional: the benchmark tests
mixture quantification under overlap, not phase ID from disjoint peak sets.

License: CC BY 4.0
"""

from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Global grid
# ---------------------------------------------------------------------------
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 90.0
N_POINTS = 4000
GRID = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, N_POINTS, dtype=np.float64)
STEP = GRID[1] - GRID[0]  # ~0.02125 deg

# ---------------------------------------------------------------------------
# Stylized peak lists: (2theta_deg, relative_intensity)
# Deliberately overlapping in 5-40 deg; fingerprint peaks live above 40 deg.
# ---------------------------------------------------------------------------
PEAKS = {
    "alpha": [
        (8.42, 1.00), (12.65, 0.55), (16.88, 0.40), (21.10, 0.72),
        (25.32, 0.30), (29.55, 0.48), (33.78, 0.22), (44.10, 0.62),
        (52.40, 0.35), (61.18, 0.28), (74.55, 0.18),
    ],
    "beta": [
        (8.55, 0.95), (12.70, 0.80), (17.05, 0.25), (21.22, 0.60),
        (25.40, 0.50), (29.61, 0.30), (33.90, 0.45), (47.85, 0.70),
        (55.20, 0.40), (66.30, 0.25), (78.10, 0.20),
    ],
    "gamma": [
        (8.48, 0.70), (12.60, 0.65), (16.95, 0.55), (21.15, 0.85),
        (25.36, 0.42), (29.58, 0.60), (33.84, 0.38), (50.95, 0.55),
        (58.40, 0.48), (69.75, 0.30), (82.40, 0.22),
    ],
}
PHASES_CRYST = ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Physics-flavored building blocks
# ---------------------------------------------------------------------------
def pseudo_voigt(x: np.ndarray, center: float, fwhm: float, eta: float) -> np.ndarray:
    """Normalized pseudo-Voigt (linear combo of Gaussian + Lorentzian)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    gamma = fwhm / 2.0
    lorentz = (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def scherrer_fwhm(two_theta_deg: float, crystallite_nm: float,
                  wavelength_a: float = 1.5406, K: float = 0.9) -> float:
    """Scherrer broadening in degrees 2-theta."""
    theta = np.deg2rad(two_theta_deg / 2.0)
    beta_rad = (K * (wavelength_a * 0.1)) / (crystallite_nm * np.cos(theta))  # nm-> A handled
    # Convert to deg 2theta. (Approximation; constants absorbed for benchmark stylization.)
    return float(np.rad2deg(beta_rad))


def march_dollase(intensities: np.ndarray, two_thetas: np.ndarray, r: float) -> np.ndarray:
    """Stylized preferred-orientation correction."""
    # Use 2theta as a proxy for hkl angle; produces smooth angle-dependent rescale.
    theta = np.deg2rad(two_thetas / 2.0)
    factor = (r ** 2 * np.cos(theta) ** 2 + np.sin(theta) ** 2 / r) ** (-1.5)
    return intensities * factor


def displacement_shift(two_theta: np.ndarray, s: float, R: float = 240.0) -> np.ndarray:
    """Sample displacement error: nonlinear angle-dependent 2theta shift in deg."""
    return -2.0 * s / R * np.cos(np.deg2rad(two_theta / 2.0)) * (180.0 / np.pi)


# ---------------------------------------------------------------------------
# Per-phase pattern synthesis
# ---------------------------------------------------------------------------
def synthesize_phase(phase: str,
                     crystallite_nm: float,
                     pref_orient_r: float,
                     zero_shift: float,
                     displacement_s: float,
                     rng: np.random.Generator) -> np.ndarray:
    pattern = np.zeros_like(GRID)
    for center, rel_i in PEAKS[phase]:
        # Drift the center: zero shift + sample displacement
        center_drifted = center + zero_shift
        center_drifted = center_drifted + displacement_shift(
            np.array([center_drifted]), displacement_s)[0]
        # Broadening: Scherrer (size) + small instrumental floor
        fwhm = max(0.08, scherrer_fwhm(center, crystallite_nm) + 0.05)
        eta = float(rng.uniform(0.2, 0.7))
        # Per-peak intensity jitter to avoid memorizable exact ratios
        jitter = float(rng.uniform(0.85, 1.15))
        pattern += rel_i * jitter * pseudo_voigt(GRID, center_drifted, fwhm, eta)
    # Preferred orientation rescale
    pattern = march_dollase(pattern, GRID, pref_orient_r)
    # Normalize per-phase to unit max so mass fractions translate to amplitude
    if pattern.max() > 0:
        pattern /= pattern.max()
    return pattern


def synthesize_amorphous(rng: np.random.Generator) -> np.ndarray:
    halo_center = float(rng.uniform(15.0, 25.0))
    halo_width = float(rng.uniform(8.0, 14.0))
    halo = np.exp(-0.5 * ((GRID - halo_center) / halo_width) ** 2)
    # Optional second broad bump
    if rng.random() < 0.5:
        c2 = float(rng.uniform(35.0, 45.0))
        w2 = float(rng.uniform(10.0, 18.0))
        halo += 0.4 * np.exp(-0.5 * ((GRID - c2) / w2) ** 2)
    halo /= halo.max()
    return halo


# ---------------------------------------------------------------------------
# Mixture + acquisition artifacts
# ---------------------------------------------------------------------------
def sample_phase_fractions(rng: np.random.Generator) -> np.ndarray:
    """Dirichlet draw, with a small chance of a near-pure phase."""
    if rng.random() < 0.15:
        # Near-pure: spike one phase
        idx = rng.integers(0, 4)
        alpha = np.full(4, 0.5)
        alpha[idx] = 12.0
    else:
        alpha = rng.uniform(0.4, 2.5, size=4)
    f = rng.dirichlet(alpha)
    return f.astype(np.float64)


def acquisition_params(regime: str, rng: np.random.Generator) -> dict:
    if regime == "sim_A":
        return dict(
            zero_shift=float(rng.uniform(-0.10, 0.10)),
            displacement_s=float(rng.uniform(-0.10, 0.10)),
            pref_orient_r=float(rng.uniform(0.85, 1.15)),
            bg_slope=float(rng.uniform(-2.0, 4.0)),
            bg_intercept=float(rng.uniform(2.0, 8.0)),
            counts_scale=float(rng.uniform(800, 1500)),
        )
    else:  # sim_B: harsher
        return dict(
            zero_shift=float(rng.uniform(-0.30, 0.30)),
            displacement_s=float(rng.uniform(-0.25, 0.25)),
            pref_orient_r=float(rng.uniform(0.70, 1.35)),
            bg_slope=float(rng.uniform(-5.0, 10.0)),
            bg_intercept=float(rng.uniform(3.0, 15.0)),
            counts_scale=float(rng.uniform(300, 900)),
        )


def build_pattern(fractions: np.ndarray,
                  crystallite_nm: float,
                  acq: dict,
                  rng: np.random.Generator) -> np.ndarray:
    f_alpha, f_beta, f_gamma, f_amorph = fractions
    p_a = synthesize_phase("alpha", crystallite_nm, acq["pref_orient_r"],
                           acq["zero_shift"], acq["displacement_s"], rng)
    p_b = synthesize_phase("beta", crystallite_nm, 1.0 / acq["pref_orient_r"],
                           acq["zero_shift"], acq["displacement_s"], rng)
    p_g = synthesize_phase("gamma", crystallite_nm, acq["pref_orient_r"] ** 0.5,
                           acq["zero_shift"], acq["displacement_s"], rng)
    p_m = synthesize_amorphous(rng)

    pattern = f_alpha * p_a + f_beta * p_b + f_gamma * p_g + f_amorph * p_m

    # Linear background drift
    bg = acq["bg_intercept"] + acq["bg_slope"] * (GRID - GRID.mean()) / (GRID.max() - GRID.min())
    bg = np.clip(bg, 0.0, None)
    pattern = pattern * acq["counts_scale"] + bg

    # Poisson counting noise
    pattern = rng.poisson(np.clip(pattern, 0.0, None)).astype(np.float32)

    # Tiny residual smoothing to mimic detector response
    pattern = gaussian_filter1d(pattern, sigma=0.6).astype(np.float32)
    return pattern


# ---------------------------------------------------------------------------
# Reference patterns (clean, no drift, no noise)
# ---------------------------------------------------------------------------
def build_reference(phase: str) -> np.ndarray:
    rng = np.random.default_rng(0)  # deterministic, no jitter
    pattern = np.zeros_like(GRID)
    for center, rel_i in PEAKS[phase]:
        fwhm = 0.10
        pattern += rel_i * pseudo_voigt(GRID, center, fwhm, eta=0.5)
    if pattern.max() > 0:
        pattern /= pattern.max()
    return pattern.astype(np.float32)


def build_reference_amorphous() -> np.ndarray:
    halo = np.exp(-0.5 * ((GRID - 20.0) / 10.0) ** 2)
    halo /= halo.max()
    return halo.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--out", type=str, default="./polydrift_xrd_raw")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "patterns").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    (out / "generation").mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(args.seed)

    # Reference patterns
    np.save(out / "reference" / "alpha.npy", build_reference("alpha"))
    np.save(out / "reference" / "beta.npy", build_reference("beta"))
    np.save(out / "reference" / "gamma.npy", build_reference("gamma"))
    np.save(out / "reference" / "amorphous.npy", build_reference_amorphous())

    rows = []
    for i in range(args.n):
        # Per-sample RNG seeded from master => fully reproducible regardless of order
        sub_seed = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)

        # Deterministic UUID from sub_seed for stable filenames across reruns
        sample_id = uuid.UUID(int=(sub_seed << 96) | (i & 0xFFFFFFFF) & ((1 << 128) - 1)).hex[:16]

        regime = "sim_A" if rng.random() < 0.55 else "sim_B"
        acq = acquisition_params(regime, rng)
        crystallite_nm = float(rng.uniform(15.0, 200.0))
        fractions = sample_phase_fractions(rng)

        pattern = build_pattern(fractions, crystallite_nm, acq, rng)
        np.save(out / "patterns" / f"{sample_id}.npy", pattern)

        rows.append({
            "sample_id": sample_id,
            "f_alpha": float(fractions[0]),
            "f_beta": float(fractions[1]),
            "f_gamma": float(fractions[2]),
            "f_amorphous": float(fractions[3]),
            "zero_shift": acq["zero_shift"],
            "crystallite_nm": crystallite_nm,
            "acquisition": regime,
        })

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out / "data.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f", lineterminator="\n")

    # Copy this script into the generation/ folder for provenance
    this_file = Path(__file__).resolve()
    (out / "generation" / "generate.py").write_bytes(this_file.read_bytes())

    # LICENSE
    (out / "LICENSE.md").write_text(
        "PolyDrift-XRD is released under CC BY 4.0.\n"
        "Reference structure provenance: COD entries 1010368, 1011000, 1528823 (CC0).\n",
        encoding="utf-8",
    )

    # Reproducibility hash
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "seed": args.seed,
        "n_samples": args.n,
        "n_points_per_pattern": N_POINTS,
        "two_theta_range_deg": [TWO_THETA_MIN, TWO_THETA_MAX],
        "data_csv_sha256": sha,
    }
    (out / "generation" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
