"""generate.py — OrbitMosaic v4 local generator.

Usage:
    pip install numpy pandas pillow scikit-learn
    python generate.py --n 8000 --out ./orbitmosaic_raw
"""
import argparse, hashlib, hmac, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

SEED = 0xC0FFEE17
ID_SECRET = b"orbitmosaic_v4::token_salt::9f2a"
MOSAIC_W = MOSAIC_H = 384
TILE_S = 128
TILE_GRID = (3, 3)
N_TILES = 9
N_DROPOUT = 4
OCC = 16
N_OCC = 3

LUT_FAMILY = [
    dict(r=(0.267, 0.005, 1.247, -0.520), g=(0.005, 1.404, -0.490, 0.084),
         b=(0.329, 1.385, -2.560, 1.146)),
    dict(r=(0.040, 2.100, -1.200, 0.080), g=(0.020, 0.300, 1.400, -0.700),
         b=(0.020, 0.100, 0.400, 0.500)),
    dict(r=(0.000, 0.200, 0.500, 0.300), g=(0.000, 1.300, -0.500, 0.200),
         b=(0.300, 1.400, -1.200, 0.500)),
    dict(r=(0.000, 1.000, 0.000, 0.000), g=(0.000, 1.000, 0.000, 0.000),
         b=(0.000, 1.000, 0.000, 0.000)),
    dict(r=(0.001, 0.700, 1.500, -1.200), g=(0.000, 0.100, 0.300, 0.600),
         b=(0.015, 1.700, -2.300, 1.600)),
    dict(r=(0.050, 2.300, -1.800, 0.450), g=(0.030, 0.100, 0.700, 0.170),
         b=(0.530, 0.500, -1.000, 0.000)),
    dict(r=(0.200, 0.400, 0.300, 0.100), g=(0.100, 0.900, -0.200, 0.200),
         b=(0.050, 0.600, 0.300, 0.050)),
    dict(r=(0.500, -0.300, 1.000, -0.200), g=(0.300, 0.500, 0.300, -0.100),
         b=(0.100, 0.300, 0.700, -0.100)),
    dict(r=(0.100, 0.800, 0.200, -0.100), g=(0.400, -0.200, 1.100, -0.300),
         b=(0.700, -0.500, 0.500, 0.300)),
    dict(r=(0.300, 1.100, -0.700, 0.300), g=(0.150, 0.300, 0.900, -0.300),
         b=(0.250, 0.700, -0.400, 0.400)),
]

# Tightened regime bands — adjacent regimes share visual cues.
REGIMES = {
    0: [{"a": (1.02, 1.10), "b": (0.28, 0.32)},
        {"a": (1.10, 1.20), "b": (0.28, 0.32)},
        {"a": (1.20, 1.28), "b": (0.28, 0.32)},
        {"a": (1.28, 1.40), "b": (0.28, 0.32)}],
    1: [{"a": (1.42, 1.55), "b": (0.30, 0.40)},
        {"a": (1.55, 1.65), "b": (0.30, 0.40)},
        {"a": (1.65, 1.74), "b": (0.30, 0.40)},
        {"a": (1.74, 1.85), "b": (0.30, 0.40)}],
    2: [{"K": (0.20, 0.60)}, {"K": (0.60, 0.97)},
        {"K": (0.97, 1.50)}, {"K": (1.50, 2.50)}],
    3: [{"c1": (0.85, 0.88), "c2": (-0.65, -0.60),
         "c3": (1.95, 2.00), "c4": (0.45, 0.50)},
        {"c1": (0.88, 0.90), "c2": (-0.60, -0.58),
         "c3": (2.00, 2.03), "c4": (0.50, 0.52)},
        {"c1": (0.90, 0.92), "c2": (-0.58, -0.56),
         "c3": (2.03, 2.05), "c4": (0.52, 0.54)},
        {"c1": (0.92, 0.95), "c2": (-0.56, -0.55),
         "c3": (2.05, 2.05), "c4": (0.54, 0.55)}],
    4: [{"u": (0.82, 0.86)}, {"u": (0.86, 0.89)},
        {"u": (0.89, 0.92)}, {"u": (0.92, 0.95)}],
    5: [{"_ic_band": (0, 1)}, {"_ic_band": (1, 2)},
        {"_ic_band": (2, 3)}, {"_ic_band": (3, 4)}],
}


def _opaque_id(idx):
    return hmac.new(ID_SECRET, f"om4::{idx}".encode(),
                    hashlib.sha256).hexdigest()[:12]


def _rng(idx, tag):
    h = hashlib.sha256(f"{tag}::{idx}".encode()).digest()
    s = int.from_bytes(h[:8], "big") ^ SEED
    return np.random.default_rng(s & 0x7FFFFFFFFFFFFFFF)


def _sample_params(rng, m, regime):
    band = REGIMES[m][regime]
    params = {}
    for k, v in band.items():
        if k.startswith("_"): continue
        params[k] = float(rng.uniform(v[0], v[1]))
    return params


def henon(x, y, p): return (1.0 - p["a"] * x * x + y, p["b"] * x)
def lozi(x, y, p):  return (1.0 - p["a"] * abs(x) + y, p["b"] * x)
def standard(x, y, p):
    p_new = (y + p["K"] * np.sin(x)) % (2 * np.pi)
    th_new = (x + p_new) % (2 * np.pi)
    return (th_new, p_new)
def tinkerbell(x, y, p):
    return (x*x - y*y + p["c1"]*x + p["c2"]*y,
            2*x*y + p["c3"]*x + p["c4"]*y)
def ikeda(x, y, p):
    t = 0.4 - 6.0 / (1.0 + x*x + y*y)
    return (1.0 + p["u"] * (x * np.cos(t) - y * np.sin(t)),
            p["u"] * (x * np.sin(t) + y * np.cos(t)))
def gingerbread(x, y, p): return (1.0 - y + abs(x), x)

MAP_FNS = [henon, lozi, standard, tinkerbell, ikeda, gingerbread]
NATIVE = [(-1.6, 1.6, -0.5, 0.5),
          (-1.6, 1.6, -0.6, 0.6),
          (0.0, 2 * np.pi, 0.0, 2 * np.pi),
          (-1.5, 1.5, -1.5, 1.5),
          (-2.5, 2.5, -2.5, 2.5),
          (-6.0, 6.0, -6.0, 6.0)]


def _trajectory(m, params, n_iters, n_transient, rng, ic_band=0):
    fn = MAP_FNS[m]
    n_ic = int(rng.integers(3, 7))
    xs, ys = [], []
    for _ in range(n_ic):
        if m == 2:
            x0 = float(rng.uniform(0, 2 * np.pi))
            y0 = float(rng.uniform(0, 2 * np.pi))
        elif m == 5:
            x0 = float(rng.uniform(-0.5 + ic_band, 0.5 + ic_band))
            y0 = float(rng.uniform(-0.5 + ic_band, 0.5 + ic_band))
        else:
            x0 = float(rng.uniform(-0.5, 0.5))
            y0 = float(rng.uniform(-0.5, 0.5))
        x, y = x0, y0
        for _ in range(n_transient):
            x, y = fn(x, y, params)
            if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e6 or abs(y) > 1e6:
                x, y = 0.0, 0.0
        per_ic = max(1, n_iters // n_ic)
        for _ in range(per_ic):
            x, y = fn(x, y, params)
            if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e6 or abs(y) > 1e6:
                break
            xs.append(x); ys.append(y)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _lyapunov(m, params, rng):
    fn = MAP_FNS[m]
    if m == 2:
        x, y = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))
    else:
        x, y = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))
    for _ in range(300):
        x, y = fn(x, y, params)
        if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e6:
            return 0.0
    eps = 1e-7; log_sum = 0.0; count = 0
    for _ in range(400):
        x1, y1 = fn(x + eps, y, params)
        x2, y2 = fn(x, y + eps, params)
        x_new, y_new = fn(x, y, params)
        if not (np.isfinite(x_new) and np.isfinite(y_new)) or abs(x_new) > 1e6:
            break
        j = np.array([[(x1 - x_new) / eps, (x2 - x_new) / eps],
                      [(y1 - y_new) / eps, (y2 - y_new) / eps]])
        sv = np.linalg.svd(j, compute_uv=False)
        if sv[0] > 0:
            log_sum += np.log(sv[0]); count += 1
        x, y = x_new, y_new
    if count == 0: return 0.0
    return log_sum / count


def _classify_attractor(xs, ys, lyap):
    if len(xs) < 50: return 0
    pts = np.stack([xs[-min(2000, len(xs)):],
                    ys[-min(2000, len(ys)):]], axis=1)
    spread = np.std(pts, axis=0).mean()
    if spread < 0.02: return 0
    if lyap > 0.02: return 3
    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    if np.median(diffs) > 0.5: return 1
    return 2


def _render_tile(xs, ys, framing, point_budget, rng):
    x_min, x_max, y_min, y_max = framing
    img = Image.new("L", (TILE_S, TILE_S), 0)
    d = ImageDraw.Draw(img)
    mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    xs_v = xs[mask]; ys_v = ys[mask]
    if len(xs_v) == 0:
        return np.zeros((TILE_S, TILE_S), dtype=np.float32)
    if len(xs_v) > point_budget:
        idx = rng.choice(len(xs_v), size=point_budget, replace=False)
        xs_v = xs_v[idx]; ys_v = ys_v[idx]
    px = ((xs_v - x_min) / max(x_max - x_min, 1e-9) * (TILE_S - 1)).astype(int)
    py = ((ys_v - y_min) / max(y_max - y_min, 1e-9) * (TILE_S - 1)).astype(int)
    for x_, y_ in zip(px, py):
        d.point((x_, TILE_S - 1 - y_), fill=255)
    arr = np.asarray(img, np.float32) / 255.0
    soft = Image.fromarray((arr * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(0.6))
    return np.asarray(soft, np.float32) / 255.0


def _lut_apply(gray01, lut_id):
    c = LUT_FAMILY[lut_id]
    g = np.clip(gray01, 0, 1).astype(np.float32)
    def poly(k): return np.clip(k[0] + k[1]*g + k[2]*g**2 + k[3]*g**3, 0, 1)
    rgb = np.stack([poly(c["r"]), poly(c["g"]), poly(c["b"])], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _yuv_hue_shift(rgb, deg):
    arr = rgb.astype(np.float32)
    yuv = np.stack([
        0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2],
        -0.169 * arr[..., 0] - 0.331 * arr[..., 1] + 0.500 * arr[..., 2],
        0.500 * arr[..., 0] - 0.419 * arr[..., 1] - 0.081 * arr[..., 2],
    ], axis=-1)
    theta = np.deg2rad(deg)
    c_, s_ = np.cos(theta), np.sin(theta)
    u_new = yuv[..., 1] * c_ - yuv[..., 2] * s_
    v_new = yuv[..., 1] * s_ + yuv[..., 2] * c_
    out = np.stack([
        yuv[..., 0] + 1.402 * v_new,
        yuv[..., 0] - 0.344 * u_new - 0.714 * v_new,
        yuv[..., 0] + 1.772 * u_new,
    ], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_mosaic(idx, regime_kind, forced_regime=None):
    rng = _rng(idx, "img4")
    # 2-4 maps per mosaic (heavier mixture)
    n_maps = int(rng.choice([2, 3, 4], p=[0.40, 0.40, 0.20]))
    maps_in_mix = rng.choice(6, size=n_maps, replace=False).tolist()

    map_traj = {}
    for i, m in enumerate(maps_in_mix):
        # Force regime on the (eventual) dominant map for balancing.
        regime_for_m = int(rng.integers(0, 4))
        params = _sample_params(rng, m, regime_for_m)
        ic_band = float(regime_for_m) if m == 5 else 0.0
        if regime_kind == "long":
            n_iters = 4000; n_trans = int(rng.integers(800, 1200))
        else:
            n_iters = int(rng.integers(800, 1800))
            n_trans = int(rng.integers(20, 120))
        xs, ys = _trajectory(m, params, n_iters, n_trans, rng, ic_band)
        lyap = _lyapunov(m, params, rng)
        chaos = int(lyap > 0.02)
        atype = _classify_attractor(xs, ys, lyap)
        map_traj[m] = dict(xs=xs, ys=ys, regime=regime_for_m,
                           attractor=atype, chaotic=chaos)

    tile_slots = list(range(N_TILES)); rng.shuffle(tile_slots)
    drop_slots = set(tile_slots[:N_DROPOUT])
    live_slots = tile_slots[N_DROPOUT:]
    assignments = {}
    for i, slot in enumerate(live_slots):
        assignments[slot] = maps_in_mix[i % len(maps_in_mix)]

    counts = {m: 0 for m in maps_in_mix}
    for slot, mm in assignments.items():
        counts[mm] += 1
    dominant = min(m for m, c in counts.items() if c == max(counts.values()))

    # Override dominant's regime if we need to balance class distribution.
    if forced_regime is not None:
        new_regime = int(forced_regime)
        params = _sample_params(rng, dominant, new_regime)
        ic_band = float(new_regime) if dominant == 5 else 0.0
        if regime_kind == "long":
            n_iters = 4000; n_trans = int(rng.integers(800, 1200))
        else:
            n_iters = int(rng.integers(800, 1800))
            n_trans = int(rng.integers(20, 120))
        xs, ys = _trajectory(dominant, params, n_iters, n_trans, rng, ic_band)
        lyap = _lyapunov(dominant, params, rng)
        map_traj[dominant] = dict(xs=xs, ys=ys, regime=new_regime,
                                  attractor=_classify_attractor(xs, ys, lyap),
                                  chaotic=int(lyap > 0.02))

    mosaic = np.zeros((MOSAIC_H, MOSAIC_W, 3), dtype=np.uint8)
    for tile_idx in range(N_TILES):
        if tile_idx in drop_slots:
            tile_gray = rng.random((TILE_S, TILE_S)).astype(np.float32) * 0.7
        else:
            m = assignments[tile_idx]
            native = NATIVE[m]
            nx0, nx1, ny0, ny1 = native
            zoom = float(rng.uniform(0.25, 1.15))  # widened
            cx = float(rng.uniform(nx0 + 0.1, nx1 - 0.1))
            cy = float(rng.uniform(ny0 + 0.1, ny1 - 0.1))
            half_x = (nx1 - nx0) * zoom * 0.5
            half_y = (ny1 - ny0) * zoom * 0.5
            framing = (cx - half_x, cx + half_x, cy - half_y, cy + half_y)
            per_tile_budget = int(rng.integers(120, 600))
            tile_gray = _render_tile(map_traj[m]["xs"], map_traj[m]["ys"],
                                     framing, per_tile_budget, rng)

        lut_id = int(rng.integers(0, len(LUT_FAMILY)))
        tile_rgb = _lut_apply(tile_gray, lut_id)
        k_rot = int(rng.integers(0, 4))
        if k_rot:
            tile_rgb = np.rot90(tile_rgb, k=k_rot).copy()
        if rng.random() < 0.5:
            tile_rgb = tile_rgb[:, ::-1, :].copy()
        r = tile_idx // TILE_GRID[1]; col = tile_idx % TILE_GRID[1]
        y0 = r * TILE_S; x0 = col * TILE_S
        mosaic[y0:y0 + TILE_S, x0:x0 + TILE_S] = tile_rgb

    occs = []
    for _ in range(N_OCC):
        ox = int(rng.integers(0, MOSAIC_W - OCC + 1))
        oy = int(rng.integers(0, MOSAIC_H - OCC + 1))
        mosaic[oy:oy + OCC, ox:ox + OCC, :] = 0
        occs.append((ox, oy))

    gamma = float(rng.uniform(0.5, 2.0))
    arr = (mosaic.astype(np.float32) / 255.0) ** gamma
    mosaic = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    mosaic = _yuv_hue_shift(mosaic, float(rng.uniform(-25, 25)))
    sigma = float(rng.uniform(0.0, 1.2))
    if sigma > 0.05:
        mosaic = np.asarray(
            Image.fromarray(mosaic, "RGB").filter(ImageFilter.GaussianBlur(sigma)))
    q = int(rng.integers(18, 56))
    buf = io.BytesIO()
    Image.fromarray(mosaic, "RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    mosaic = np.asarray(Image.open(buf).convert("RGB"))

    presence = [int(m in maps_in_mix) for m in range(6)]
    return (mosaic, presence, int(dominant),
            int(map_traj[dominant]["regime"]),
            int(map_traj[dominant]["attractor"]),
            int(map_traj[dominant]["chaotic"]), occs)


def _make_reference(m):
    rng = _rng(99_000 + m, "ref")
    params = _sample_params(rng, m, regime=3)
    xs, ys = _trajectory(m, params, n_iters=4000, n_transient=1500, rng=rng)
    mosaic = np.zeros((MOSAIC_H, MOSAIC_W, 3), dtype=np.uint8)
    for tile_idx in range(N_TILES):
        tile_gray = _render_tile(xs, ys, NATIVE[m], 1500, rng)
        tile_rgb = _lut_apply(tile_gray, 0)
        r = tile_idx // TILE_GRID[1]; col = tile_idx % TILE_GRID[1]
        y0 = r * TILE_S; x0 = col * TILE_S
        mosaic[y0:y0 + TILE_S, x0:x0 + TILE_S] = tile_rgb
    return mosaic


def _save_png(arr, path):
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG", optimize=True)
    path.write_bytes(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--out", type=str, default="./orbitmosaic_raw")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "reference_images").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    for m in range(6):
        _save_png(_make_reference(m),
                  out / "reference_images" / f"map_{m}_clean.png")

    # Cycle forced regime to keep parameter_regime balanced ~25% each.
    regime_cycle = np.tile([0, 1, 2, 3], args.n // 4 + 1)[: args.n]
    rng.shuffle(regime_cycle)

    rows = []
    for i in range(args.n):
        regime_kind = "long" if rng.random() < 0.55 else "short"
        forced = int(regime_cycle[i])
        (mosaic, presence, dom, dom_reg, dom_attr,
         dom_chao, occs) = _make_mosaic(i, regime_kind, forced_regime=forced)
        sid = _opaque_id(i)
        _save_png(mosaic, out / "images" / f"{sid}.png")
        row = {"sample_id": sid}
        for k in range(6):
            row[f"m_{k}"] = presence[k]
        row.update({"dominant_map": dom, "parameter_regime": dom_reg,
                    "attractor_type": dom_attr, "is_chaotic": dom_chao,
                    "regime_kind": regime_kind})
        for j, (ox, oy) in enumerate(occs, start=1):
            row[f"occ{j}_x"] = ox; row[f"occ{j}_y"] = oy
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  generated {i+1}/{args.n}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "data.csv", index=False)
    (out / "LICENSE.md").write_text(
        "OrbitMosaic synthetic benchmark, released under CC BY 4.0.\n"
        "Procedurally generated from seed 0xC0FFEE17. No third-party data.\n",
        encoding="utf-8")
    sha = hashlib.sha256((out / "data.csv").read_bytes()).hexdigest()
    print(json.dumps({"n_samples": args.n, "data_csv_sha256": sha}, indent=2))


if __name__ == "__main__":
    main()
