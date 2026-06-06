"""Audit every (task, split) combination for files referenced in metadata
that are missing on disk.  Run this once after syncing CSI-Bench from S3 to
catch stale CSV references *before* launching a training sweep.

Usage:
    python util/audit_missing_files.py \
        --dataset-root ~/Data/CSI-Bench \
        [--tasks FallDetection BreathingDetection ...] \
        [--report missing_report.csv]

If ``--report`` is provided a CSV listing every missing reference is written
to disk.  In all cases a per-task / per-split summary is printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# Make ``load.supervised.*`` importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

DEFAULT_TASKS = [
    "FallDetection",
    "BreathingDetection",
    "Localization",
    "MotionSourceRecognition",
    "HumanActivityRecognition",
    "HumanIdentification",
    "ProximityRecognition",
]
_MULTITASK_TASKS = {
    "HumanActivityRecognition",
    "HumanIdentification",
    "ProximityRecognition",
}
_IGNORED_SPLIT_RE = re.compile(r"_p\d+(?:_|\.|$)")


def _resolve_task_dir(dataset_root: Path, task: str) -> Path | None:
    candidates = [
        dataset_root / task,
        dataset_root / "Multitask" / task,
        dataset_root / "tasks" / task,
    ]
    for c in candidates:
        if (c / "metadata").is_dir() and (c / "splits").is_dir():
            return c
    return None


def _resolve_filepath(task_dir: Path, dataset_root: Path,
                      original: str) -> Path | None:
    if os.path.isabs(original) and os.path.exists(original):
        return Path(original)
    candidates = [
        os.path.normpath(os.path.join(str(task_dir), original)),
        os.path.normpath(os.path.join(str(task_dir), "metadata", original)),
        os.path.normpath(os.path.join(str(dataset_root), original)),
        os.path.normpath(os.path.join(str(dataset_root), "tasks",
                                      task_dir.name, original)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return Path(c)
    return None


def _load_split_ids(splits_dir: Path) -> dict[str, set]:
    out: dict[str, set] = {}
    for p in sorted(splits_dir.glob("*.json")):
        if _IGNORED_SPLIT_RE.search(p.name):
            continue
        try:
            with p.open() as f:
                ids = set(json.load(f))
        except Exception as e:
            print(f"  [skip] failed to read {p.name}: {e}")
            continue
        out[p.stem] = ids
    return out


def audit_task(dataset_root: Path, task: str,
               writer: csv.writer | None) -> dict:
    task_dir = _resolve_task_dir(dataset_root, task)
    if task_dir is None:
        print(f"[{task}] task directory NOT FOUND under {dataset_root}")
        return {"task": task, "found": False, "splits": {}}

    metadata_path = task_dir / "metadata" / "sample_metadata.csv"
    if not metadata_path.exists():
        print(f"[{task}] metadata CSV missing at {metadata_path}")
        return {"task": task, "found": False, "splits": {}}

    df = pd.read_csv(metadata_path)
    id_col = "id" if "id" in df.columns else "sample_id"
    by_id = dict(zip(df[id_col].astype(str), df["file_path"].astype(str)))

    splits = _load_split_ids(task_dir / "splits")
    print(f"\n[{task}]  task_dir={task_dir}")
    print(f"  metadata rows: {len(df)}  splits: {len(splits)}")

    summary: dict = {"task": task, "found": True, "splits": {}}
    for split_name, ids in splits.items():
        n_total = len(ids)
        if n_total == 0:
            summary["splits"][split_name] = (0, 0)
            print(f"  - {split_name}: empty split")
            continue
        missing: list[str] = []
        for sid in ids:
            fp = by_id.get(str(sid))
            if fp is None:
                missing.append(f"<id {sid} absent from CSV>")
                continue
            if _resolve_filepath(task_dir, dataset_root, fp) is None:
                missing.append(fp)
        n_missing = len(missing)
        summary["splits"][split_name] = (n_missing, n_total)
        marker = "OK" if n_missing == 0 else "MISSING"
        pct = 100.0 * n_missing / n_total if n_total else 0
        print(f"  - {split_name:<22s} [{marker:>7s}] "
              f"{n_missing}/{n_total} missing ({pct:.1f}%)")
        if writer is not None and missing:
            for m in missing:
                writer.writerow([task, split_name, m])
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path,
                   help="Local mirror of s3://.../CSI-Bench/")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                   help="Tasks to audit (default: all 7)")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional CSV with one row per missing reference")
    args = p.parse_args(list(argv) if argv is not None else None)

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}")
        return 2

    writer = None
    fh = None
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        fh = args.report.open("w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["task", "split", "missing_file_path"])

    summaries = [audit_task(dataset_root, t, writer) for t in args.tasks]
    if fh is not None:
        fh.close()
        print(f"\nDetailed missing-file CSV written to: {args.report}")

    print("\n" + "=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    any_missing = False
    for s in summaries:
        if not s["found"]:
            print(f"  {s['task']:<26s}  TASK DIR NOT FOUND")
            any_missing = True
            continue
        n_miss = sum(m for m, _ in s["splits"].values())
        n_tot = sum(t for _, t in s["splits"].values())
        marker = "OK" if n_miss == 0 else "MISSING"
        if n_miss:
            any_missing = True
        print(f"  {s['task']:<26s}  [{marker:>7s}]  "
              f"{n_miss}/{n_tot} missing across {len(s['splits'])} splits")
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
