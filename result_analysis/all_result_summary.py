"""Aggregate CSI-Bench training results into paper-style CSV tables.

Walks a results directory, parses the per-experiment ``*_results.json`` and
``*_config.json`` files emitted by ``scripts/train_supervised.py`` and
``scripts/train_multitask_adapter.py``, and produces the per-table CSVs
referenced in the CSI-Bench paper (Tab. 3, 4, 8-14).

By default the script reports a single ``acc`` / ``f1`` value per
(task, model) pair (suitable for the Phase E single-seed sweep).  When
``--seeds`` lists multiple seeds, the script reports ``mean +/- std``
columns matching the legacy ``*_with_error_bar.csv`` files in this folder.

Examples
--------
Single-seed (Phase E default)::

    python result_analysis/all_result_summary.py \
        --results-dir ./results/csi_bench \
        --multitask-results-dir ./results/csi_bench_multitask \
        --output-dir result_analysis/csi_bench_new

Three seeds (Phase F mean +/- std)::

    python result_analysis/all_result_summary.py \
        --results-dir ./results/csi_bench \
        --multitask-results-dir ./results/csi_bench_multitask \
        --output-dir result_analysis/csi_bench_new_3seed \
        --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MODELS_ORDER = [
    "mlp",
    "lstm",
    "resnet18",
    "transformer",
    "vit",
    "patchtst",
    "timesformer1d",
]

SINGLE_TASKS = [
    "FallDetection",
    "BreathingDetection",
    "Localization",
    "MotionSourceRecognition",
]

MULTITASK_TASKS = [
    "HumanActivityRecognition",
    "HumanIdentification",
    "ProximityRecognition",
]

# Difficulty splits per task (varies in suffix: ``_id`` vs none).
DIFFICULTY_SPLITS = {
    "FallDetection": ("test_easy", "test_medium", "test_hard"),
    "BreathingDetection": ("test_easy_id", "test_medium_id", "test_hard_id"),
    "Localization": ("test_easy_id", "test_medium_id", "test_hard_id"),
    "MotionSourceRecognition": ("test_easy", "test_medium", "test_hard"),
}

OOD_SPLITS = ("test_cross_device", "test_cross_env", "test_cross_user")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_FEWSHOT_SUFFIX = "_few_shot_results.json"
_TEST_RESULTS_SUFFIX = "_test_results.json"
_RESULTS_SUFFIX = "_results.json"


def _is_relevant_results_file(path: Path) -> bool:
    """Match the main results JSON written by either train script.

    Includes:
      - ``<model>_<task>_results.json``       (supervised)
      - ``<model>_<task>_test_results.json``  (multi-task)
    Excludes:
      - ``<model>_<task>_few_shot_results.json``
    """
    name = path.name
    if name.endswith(_FEWSHOT_SUFFIX):
        return False
    return name.endswith(_RESULTS_SUFFIX)


def _glob_experiments(results_dir: Path) -> list[Path]:
    """Find every experiment folder that contains a main results JSON.

    Returns unique exp_dirs (an experiment dir is counted only once even if
    multiple matching ``_results.json`` files happen to live there).
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for path in results_dir.rglob("*_results.json"):
        if not _is_relevant_results_file(path):
            continue
        if path.parent in seen:
            continue
        seen.add(path.parent)
        out.append(path.parent)
    return out


def _find_main_results(exp_dir: Path) -> Optional[Path]:
    """Return the main results JSON for an experiment directory.

    Prefers ``*_test_results.json`` (multi-task) over ``*_results.json``
    (supervised) when both happen to be present; ignores few-shot results.
    """
    candidates = sorted(
        p for p in exp_dir.glob("*_results.json")
        if _is_relevant_results_file(p)
    )
    if not candidates:
        return None
    # Prefer the explicit _test_results.json variant when present.
    test_variants = [p for p in candidates if p.name.endswith(_TEST_RESULTS_SUFFIX)]
    return (test_variants or candidates)[0]


def _parse_exp(exp_dir: Path) -> Optional[dict]:
    """Return ``{task, model, seed, splits}`` from one experiment folder."""
    cfg_path = next(iter(exp_dir.glob("*_config.json")), None)
    res_path = _find_main_results(exp_dir)
    if cfg_path is None or res_path is None:
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        res = json.loads(res_path.read_text())
    except Exception as e:
        print(f"[warn] failed to parse {exp_dir}: {e}")
        return None
    task = cfg.get("task_name") or cfg.get("task") or _infer_field(res_path, 1)
    model = cfg.get("model") or _infer_field(res_path, 0)
    seed = cfg.get("seed")
    if seed is None:
        # Some legacy runs forgot to write seed; default to 42 so we still
        # report a single row instead of dropping the run silently.
        seed = 42
    splits = {k: v for k, v in res.items() if isinstance(v, dict)}
    return {"task": task, "model": model, "seed": int(seed),
            "exp_dir": str(exp_dir), "splits": splits}


def _infer_field(results_json_path: Path, idx: int) -> Optional[str]:
    """Filenames look like ``{model}_{task}_results.json`` or
    ``{model}_{task}_test_results.json``.  Returns ``model`` (idx=0) or
    ``task`` (idx=1)."""
    stem = results_json_path.stem
    # Strip the trailing ``_test_results`` / ``_results`` suffix.
    if stem.endswith("_test_results"):
        stem = stem[: -len("_test_results")]
    elif stem.endswith("_results"):
        stem = stem[: -len("_results")]
    parts = stem.split("_", 1)
    if idx == 0:
        return parts[0] if parts else None
    return parts[1] if len(parts) > 1 else None


# The benchmark loader renames the ``test_id`` test split to ``test`` in the
# loaders dict (and therefore in the per-experiment results JSON).  Normalize
# back to ``test_id`` for downstream paper-table lookups.
_SPLIT_ALIASES = {
    "test": "test_id",
}


def _canonical_split(name: str) -> str:
    return _SPLIT_ALIASES.get(name, name)


def collect_results(results_dir: Path) -> pd.DataFrame:
    """Return a long-form dataframe with one row per (exp, split, metric)."""
    rows = []
    for exp_dir in _glob_experiments(results_dir):
        rec = _parse_exp(exp_dir)
        if rec is None:
            continue
        for split, metrics in rec["splits"].items():
            canonical = _canonical_split(split)
            row = {
                "task": rec["task"],
                "model": rec["model"],
                "seed": rec["seed"],
                "split": canonical,
                "split_raw": split,
                "exp_dir": rec["exp_dir"],
            }
            for m, v in metrics.items():
                row[m] = v
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["task", "model", "seed", "split", "accuracy", "f1_score"]
        )
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg_cell(values: pd.Series, fmt: str = "{:.4f}") -> Optional[str]:
    """Aggregate a list of seed values into a single cell string."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    if len(values) == 1:
        return fmt.format(float(values.iloc[0]))
    return f"{fmt.format(float(values.mean()))}\u00b1{fmt.format(float(values.std(ddof=0)))}"


def _build_split_table(
    df: pd.DataFrame,
    tasks: list[str],
    splits: list[str],
    models: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    """Build a paper-style (task, model) x (split_acc/split_f1) table."""
    sub = df[df["seed"].isin(seeds) & df["task"].isin(tasks) & df["split"].isin(splits)]
    rows = []
    for task in tasks:
        for model in models:
            row = {"task": task, "model": model}
            empty = True
            for split in splits:
                cell = sub[(sub.task == task) & (sub.model == model) & (sub.split == split)]
                acc = _agg_cell(cell["accuracy"]) if "accuracy" in cell.columns else None
                f1 = _agg_cell(cell["f1_score"]) if "f1_score" in cell.columns else None
                row[f"{split}_accuracy"] = acc
                row[f"{split}_f1_score"] = f1
                if acc is not None or f1 is not None:
                    empty = False
            if not empty:
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-table writers
# ---------------------------------------------------------------------------

def _write(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        print(f"  [skip] {path.name}: no rows")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def emit_tab3(df: pd.DataFrame, out_dir: Path, models: list[str], seeds: list[int]) -> None:
    """Tab. 3: supervised on test_id, all 7 tasks (4 single-task + 3 multi-task).

    When run on the optional ``train_supervised.py`` baselines for the
    multitask tasks, those rows appear here too.
    """
    all_tasks = SINGLE_TASKS + MULTITASK_TASKS
    out = _build_split_table(df, all_tasks, ["test_id"], models, seeds)
    if not out.empty:
        out = out.rename(columns={"test_id_accuracy": "test_accuracy",
                                  "test_id_f1_score": "test_f1_score"})
    _write(out, out_dir / "tab3_supervised.csv")


def emit_tab4(df_supervised: pd.DataFrame, df_multitask: pd.DataFrame,
              out_dir: Path, seeds: list[int]) -> None:
    """Tab. 4: multitask Transformer vs single-task Transformer on the three
    multitask tasks."""
    rows = []
    for task in MULTITASK_TASKS:
        # multitask transformer row
        mt = df_multitask[(df_multitask.task == task) & (df_multitask.model == "transformer")
                          & (df_multitask.seed.isin(seeds)) & (df_multitask.split == "test_id")]
        row_mt = {"task": task, "pipeline": "multitask",
                  "test_accuracy": _agg_cell(mt["accuracy"]) if "accuracy" in mt else None,
                  "test_f1_score": _agg_cell(mt["f1_score"]) if "f1_score" in mt else None}
        # single-task transformer row (if user trained transformer-only on each)
        st = df_supervised[(df_supervised.task == task) & (df_supervised.model == "transformer")
                           & (df_supervised.seed.isin(seeds)) & (df_supervised.split == "test_id")]
        row_st = {"task": task, "pipeline": "single-task",
                  "test_accuracy": _agg_cell(st["accuracy"]) if "accuracy" in st else None,
                  "test_f1_score": _agg_cell(st["f1_score"]) if "f1_score" in st else None}
        rows.extend([row_st, row_mt])
    _write(pd.DataFrame(rows), out_dir / "tab4_multitask_vs_singletask.csv")


def emit_difficulty_tables(df: pd.DataFrame, out_dir: Path,
                           models: list[str], seeds: list[int]) -> None:
    """Tab. 8/9/10/11: difficulty breakdowns per single-task."""
    table_id_map = {
        "FallDetection": "tab8_fall_difficulty.csv",
        "BreathingDetection": "tab9_breath_difficulty.csv",
        "Localization": "tab10_loc_difficulty.csv",
        "MotionSourceRecognition": "tab11_msr_difficulty.csv",
    }
    for task, filename in table_id_map.items():
        splits = DIFFICULTY_SPLITS[task]
        out = _build_split_table(df, [task], list(splits), models, seeds)
        _write(out, out_dir / filename)


def emit_multitask_ood_tables(df_multitask: pd.DataFrame, out_dir: Path,
                              models: list[str], seeds: list[int]) -> None:
    """Tab. 12/13/14: per-task OOD breakdown for multitask training.

    Each row contains ``test_cross_device``, ``test_cross_env``,
    ``test_cross_user`` columns for accuracy + f1.
    """
    table_id_map = {
        "HumanActivityRecognition": "tab12_har_ood.csv",
        "HumanIdentification": "tab13_uid_ood.csv",
        "ProximityRecognition": "tab14_prox_ood.csv",
    }
    for task, filename in table_id_map.items():
        out = _build_split_table(df_multitask, [task], list(OOD_SPLITS), models, seeds)
        _write(out, out_dir / filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True,
                    help="Root directory containing supervised results "
                         "(e.g. ./results/csi_bench).")
    ap.add_argument("--multitask-results-dir", default=None,
                    help="Optional root for multi-task results "
                         "(e.g. ./results/csi_bench_multitask).  If omitted, "
                         "multitask tables are skipped.")
    ap.add_argument("--output-dir", default="result_analysis/csi_bench_new",
                    help="Where to write the aggregated CSV files.")
    ap.add_argument("--seeds", default="42",
                    help="Comma-separated list of seeds to include "
                         "(default: 42 -> single-seed reporting).")
    ap.add_argument("--models", default=",".join(MODELS_ORDER),
                    help="Comma-separated list of models to report.")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Seeds: {seeds}  ({'single-seed mode' if len(seeds) == 1 else 'mean +/- std mode'})")
    print(f"Models: {models}")

    sup_dir = Path(args.results_dir).expanduser().resolve()
    print(f"\nCollecting supervised results from {sup_dir}")
    df_sup = collect_results(sup_dir)
    print(f"  -> {len(df_sup)} (exp, split) rows")

    df_mt = pd.DataFrame()
    if args.multitask_results_dir:
        mt_dir = Path(args.multitask_results_dir).expanduser().resolve()
        print(f"\nCollecting multitask results from {mt_dir}")
        df_mt = collect_results(mt_dir)
        print(f"  -> {len(df_mt)} (exp, split) rows")

    # Persist the raw long-form dataframe for traceability.
    if not df_sup.empty:
        df_sup.to_csv(out_dir / "_supervised_raw.csv", index=False)
    if not df_mt.empty:
        df_mt.to_csv(out_dir / "_multitask_raw.csv", index=False)

    print("\nWriting paper tables:")
    if not df_sup.empty:
        emit_tab3(df_sup, out_dir, models, seeds)
        emit_difficulty_tables(df_sup, out_dir, models, seeds)
    if not df_mt.empty:
        emit_tab4(df_sup, df_mt, out_dir, seeds)
        emit_multitask_ood_tables(df_mt, out_dir, models, seeds)
        # Also dump a generic test_id snapshot for the multitask sweep.
        _write(
            _build_split_table(df_mt, MULTITASK_TASKS, ["test_id"], models, seeds)
                .rename(columns={"test_id_accuracy": "test_accuracy",
                                 "test_id_f1_score": "test_f1_score"}),
            out_dir / "multitask_test_id.csv",
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
