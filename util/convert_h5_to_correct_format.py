from __future__ import annotations
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import h5py
from typing import Optional


# ----------------------------
# Read / write
# ----------------------------
def load_csi_amps(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        if "CSI_amps" in f:
            key = "CSI_amps"
        elif "/CSI_amps" in f:
            key = "/CSI_amps"
        else:
            # h5py usually exposes keys without leading '/'
            raise KeyError("Cannot find dataset 'CSI_amps' or '/CSI_amps' in file.")
        x = np.array(f[key])
    if np.iscomplexobj(x):
        x = np.abs(x)
    return x, key


def save_h5_same_key(out_path: Path, key: str, x_fixed: np.ndarray, keep_raw: bool, x_raw: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Normalize key for h5py create_dataset (no leading '/')
    key_norm = key.lstrip("/")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(key_norm, data=x_fixed.astype(np.float32),
                         compression="gzip", compression_opts=4)
        if keep_raw:
            f.create_dataset(key_norm + "_raw", data=np.asarray(x_raw, dtype=np.float32),
                             compression="gzip", compression_opts=4)

        f.attrs["note"] = "shape preserved; CSI_amps corrected in-place if signature matched."


# ----------------------------
# Corruption signature + fix
# ----------------------------
def reinterpret_fortran_reshape_ft(ft: np.ndarray) -> np.ndarray:
    """ft: (F,T) -> same shape, corrected for reshape memory-order mismatch."""
    F, T = ft.shape
    return ft.ravel(order="C").reshape(F, T, order="F")


def wrong_h5_signature(ft: np.ndarray, max_lag: int = 40, max_pairs: int = 40):
    """
    Strong signature to avoid false positives on real CSI:
      - adjacent frequency rows are near-identical up to constant lag
      - lags are consistent (low std)
      - correlation at that lag is very high
    Returns: (is_wrong, median_lag, lag_std, median_corr)
    """
    if ft.ndim != 2:
        return False, 0, 0.0, 0.0
    F, T = ft.shape
    if F < 8 or T < 80:
        return False, 0, 0.0, 0.0

    pairs = min(F - 1, max_pairs)
    lags, corrs = [], []

    for f in range(pairs):
        a = ft[f].astype(np.float64)
        b = ft[f + 1].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        na = float(np.linalg.norm(a)) + 1e-12
        nb = float(np.linalg.norm(b)) + 1e-12

        best_lag, best_val, best_corr = 0, -1e30, 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                aa, bb = a[-lag:], b[:T + lag]
            elif lag > 0:
                aa, bb = a[:T - lag], b[lag:]
            else:
                aa, bb = a, b
            val = float(np.dot(aa, bb))
            if val > best_val:
                best_val = val
                best_lag = lag
                best_corr = val / (na * nb)

        lags.append(best_lag)
        corrs.append(best_corr)

    med_lag = int(np.median(lags))
    lag_std = float(np.std(lags))
    med_corr = float(np.median(corrs))

    # Conservative thresholds
    is_wrong = (abs(med_lag) >= 2) and (lag_std <= 0.75) and (med_corr >= 0.85)
    return is_wrong, med_lag, lag_std, med_corr


# ----------------------------
# Shape-preserving 3D fixer
# ----------------------------
def choose_axes_for_ft(x: np.ndarray, expected_time: Optional[int], max_freq: int):
    """
    Decide which 2 axes are (freq,time) and which axis is slice axis.
    Returns: (freq_axis, time_axis, slice_axis, flipped)
    where flipped=True means original plane is (time,freq) instead of (freq,time).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.ndim}D shape={x.shape}")

    shape = x.shape
    axes = [0, 1, 2]

    # Score each choice of slice_axis; remaining two axes are the 2D plane.
    best = None
    for slice_axis in axes:
        plane = [a for a in axes if a != slice_axis]
        a0, a1 = plane
        d0, d1 = shape[a0], shape[a1]

        # Decide which is time: prefer expected_time if provided; else larger dim
        if expected_time is not None and (d0 == expected_time) != (d1 == expected_time):
            time_axis = a0 if d0 == expected_time else a1
        else:
            time_axis = a0 if d0 >= d1 else a1

        freq_axis = a1 if time_axis == a0 else a0

        T = shape[time_axis]
        F = shape[freq_axis]

        # Scoring: time should be reasonably large; freq should not be enormous
        score = 0
        if expected_time is not None and T == expected_time:
            score += 10
        if T >= 64:
            score += 3
        if 2 <= F <= max_freq:
            score += 3
        if shape[slice_axis] == 1:
            score += 2  # common case: trailing singleton slice dim

        # prefer slice_axis being the singleton if exists
        if best is None or score > best[0]:
            best = (score, freq_axis, time_axis, slice_axis)

    _, freq_axis, time_axis, slice_axis = best
    flipped = False
    # flipped means the plane is actually stored as (T,F) when we want (F,T)
    # We'll handle by reading plane and converting.
    # If freq_axis dimension is larger than time_axis, it might be swapped; treat as flipped.
    if x.shape[freq_axis] > x.shape[time_axis] and (expected_time is None or x.shape[time_axis] != expected_time):
        flipped = True

    return freq_axis, time_axis, slice_axis, flipped


def fix_csi_3d_preserve_shape(x: np.ndarray,
                             expected_time: int | None = None,
                             max_freq: int = 8192,
                             enable_fix: bool = True):
    """
    Preserve original shape exactly; only fix the 2D (freq,time) plane per slice if signature matches.
    Returns: (x_fixed, meta)
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.abs(x)

    freq_axis, time_axis, slice_axis, flipped = choose_axes_for_ft(x, expected_time, max_freq)

    # Move to canonical layout: (F, T, S)
    perm = [freq_axis, time_axis, slice_axis]
    inv_perm = np.argsort(perm)
    y = np.transpose(x, perm)  # (F, T, S)
    F, T, S = y.shape

    meta = {
        "orig_shape": tuple(x.shape),
        "used_axes": {"freq_axis": int(freq_axis), "time_axis": int(time_axis), "slice_axis": int(slice_axis)},
        "F": int(F),
        "T": int(T),
        "S": int(S),
        "fixed_any": False,
        "signature": []
    }

    y_fixed = np.empty_like(y, dtype=np.float32)

    for s in range(S):
        ft = y[:, :, s].astype(np.float32)  # (F,T)

        # If the plane likely stored as (T,F) under this permutation heuristic, transpose.
        # (This is rare once axes are chosen correctly, but keep as safety.)
        if flipped:
            ft = ft.T
            # Now ft is (F,T) again if truly swapped; else it will be (T,F) and signature will fail.

        if enable_fix:
            is_wrong, med_lag, lag_std, med_corr = wrong_h5_signature(ft)
        else:
            is_wrong, med_lag, lag_std, med_corr = (False, 0, 0.0, 0.0)

        if enable_fix and is_wrong:
            ft2 = reinterpret_fortran_reshape_ft(ft)
            meta["fixed_any"] = True
        else:
            ft2 = ft

        meta["signature"].append({
            "slice": int(s),
            "is_wrong": bool(is_wrong),
            "median_lag": int(med_lag),
            "lag_std": float(lag_std),
            "median_corr": float(med_corr),
        })

        if flipped:
            ft2 = ft2.T  # revert to match y layout

        y_fixed[:, :, s] = ft2.astype(np.float32)

    # Back to original axis order and shape
    x_fixed = np.transpose(y_fixed, inv_perm)
    x_fixed = x_fixed.reshape(x.shape).astype(np.float32)
    return x_fixed, meta


# ----------------------------
# CLI driver
# ----------------------------
def convert_one(in_path: Path, out_path: Path, expected_time: int | None, max_freq: int, enable_fix: bool, keep_raw: bool):
    x, key = load_csi_amps(in_path)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D CSI_amps in {in_path.name}, got shape={x.shape}")

    x_fixed, meta = fix_csi_3d_preserve_shape(
        x,
        expected_time=expected_time,
        max_freq=max_freq,
        enable_fix=enable_fix
    )

    # Ensure shape preserved
    assert x_fixed.shape == x.shape, (x.shape, x_fixed.shape)

    save_h5_same_key(out_path, key, x_fixed, keep_raw=keep_raw, x_raw=x)
    print(f"[OK] {in_path} -> {out_path} | shape {x.shape} | fixed_any={meta['fixed_any']} | axes={meta['used_axes']}")
    # If you want details per slice, uncomment:
    # print(meta["signature"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .h5 file or directory")
    ap.add_argument("--output", required=True, help="Output .h5 file or directory")
    ap.add_argument("--expected-time", type=int, default=-1,
                    help="Expected time length (e.g., 500). Use -1 to auto-detect.")
    ap.add_argument("--max-freq", type=int, default=8192,
                    help="Max allowed freq dimension for axis detection (default 8192).")
    ap.add_argument("--no-fix", action="store_true",
                    help="Disable auto-fix; only repack/save (shape preserved).")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Also save CSI_amps_raw dataset for debugging (increases size).")
    ap.add_argument("--glob", default="*.h5", help="When input is dir: glob pattern (default *.h5)")
    args = ap.parse_args()

    expected_time = None if args.expected_time == -1 else args.expected_time
    enable_fix = not args.no_fix

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        if out_path.is_dir():
            out_file = out_path / (in_path.stem + "_fixed.h5")
        else:
            out_file = out_path
        convert_one(in_path, out_file, expected_time, args.max_freq, enable_fix, args.keep_raw)
        return

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        for fp in sorted(in_path.rglob(args.glob)):
            rel = fp.relative_to(in_path)
            out_file = out_path / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            convert_one(fp, out_file, expected_time, args.max_freq, enable_fix, args.keep_raw)
        return

    raise FileNotFoundError(f"Input not found: {in_path}")


if __name__ == "__main__":
    main()
