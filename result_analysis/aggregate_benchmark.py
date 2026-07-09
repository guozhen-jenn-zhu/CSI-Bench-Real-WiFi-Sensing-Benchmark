#!/usr/bin/env python3
"""Aggregate benchmark matrix per-run JSONs into a single CSV.

Reads <runs_dir>/*/run.json (or <runs_dir>/runs/*/run.json) and emits one row per
(model, dataset, seed, split, phase) with accuracy and F1.

For PAG runs, three phases are recorded: dense / pruned_zero_shot / pruned_finetuned.
For supervised baselines, one phase 'final' is recorded.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def _gather(run_record, rows):
    rid = run_record["run_id"]
    model = run_record["model"]
    dataset = f"{run_record['dataset_kind']}:{run_record['dataset']}"
    seed = run_record["seed"]
    if run_record["status"] != "ok":
        rows.append({
            "run_id": rid, "model": model, "dataset": dataset, "seed": seed,
            "phase": "-", "split": "-",
            "status": run_record["status"],
            "accuracy": "", "f1": "",
            "reason": run_record.get("reason", run_record.get("error", "")),
        })
        return

    outcome = run_record.get("outcome", {})
    if outcome.get("trainer") == "pag":
        for phase, splits in outcome["results"]["test"].items():
            for split, m in splits.items():
                rows.append({
                    "run_id": rid, "model": model, "dataset": dataset, "seed": seed,
                    "phase": phase, "split": split, "status": "ok",
                    "accuracy": m.get("accuracy", ""), "f1": m.get("f1_macro", ""),
                    "reason": "",
                })
    elif outcome.get("trainer") == "supervised":
        for split, m in outcome.get("test", {}).items():
            rows.append({
                "run_id": rid, "model": model, "dataset": dataset, "seed": seed,
                "phase": "final", "split": split, "status": "ok",
                "accuracy": m.get("accuracy", ""), "f1": m.get("f1_score", ""),
                "reason": "",
            })


def _summarize(rows):
    """Return aggregated mean/std per (model, dataset, phase, split) over seeds."""
    bucket = defaultdict(list)
    for r in rows:
        if r["status"] != "ok" or r["accuracy"] == "":
            continue
        key = (r["model"], r["dataset"], r["phase"], r["split"])
        bucket[key].append((float(r["accuracy"]), float(r["f1"]) if r["f1"] != "" else float("nan")))
    summary = []
    for (model, dataset, phase, split), vals in bucket.items():
        accs = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        n = len(vals)
        mean_acc = sum(accs) / n
        mean_f1 = sum(f1s) / n
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / max(n - 1, 1)
        var_f1 = sum((f - mean_f1) ** 2 for f in f1s) / max(n - 1, 1)
        summary.append({
            "model": model, "dataset": dataset, "phase": phase, "split": split,
            "n_seeds": n,
            "accuracy_mean": mean_acc,
            "accuracy_std": var_acc ** 0.5,
            "f1_mean": mean_f1,
            "f1_std": var_f1 ** 0.5,
        })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True,
                   help="Either output_dir or output_dir/runs from run_benchmark_matrix.py")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--summary_csv", default=None)
    args = p.parse_args()

    candidates = sorted(glob.glob(os.path.join(args.runs_dir, "**", "run.json"), recursive=True))
    if not candidates:
        raise SystemExit(f"No run.json files found under {args.runs_dir}")

    rows = []
    for path in candidates:
        with open(path) as f:
            try:
                rec = json.load(f)
            except Exception as e:
                print(f"WARN: could not parse {path}: {e}")
                continue
        _gather(rec, rows)

    out_csv = args.out_csv or os.path.join(args.runs_dir, "all_runs.csv")
    fieldnames = ["run_id", "model", "dataset", "seed", "phase", "split", "status",
                  "accuracy", "f1", "reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

    summary = _summarize(rows)
    summary_csv = args.summary_csv or os.path.join(args.runs_dir, "summary_by_model_dataset.csv")
    sfields = ["model", "dataset", "phase", "split", "n_seeds",
               "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {len(summary)} summary rows to {summary_csv}")


if __name__ == "__main__":
    main()
