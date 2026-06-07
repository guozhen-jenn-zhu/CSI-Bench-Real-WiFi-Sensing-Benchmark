"""Drive the full CSI-Bench paper reproduction sweep.

Use this script on a g5.12xlarge box where ``~/Data/CSI-Bench/`` is already
populated.  It loops over (task, model, seed) for the supervised tasks and
then runs the multi-task adapter pipeline.  Each individual run is forwarded
to ``scripts/local_runner.py`` so the behaviour matches a normal local run.

Default behaviour (no ``--seeds``):
    seed=42 only -- this is the deliverable for Phase E.

Phase F (opt-in mean +/- std):
    ``--seeds 42,43,44``

Examples::

    # Phase E (single seed) -- the default deliverable
    python scripts/run_seed_sweep.py \
        --config configs/csi_bench_local_config.json

    # Phase F (three seeds) -- mean +/- std re-run
    python scripts/run_seed_sweep.py \
        --config configs/csi_bench_local_config.json \
        --seeds 42,43,44

    # Multi-task adapter (Tab. 4 Transformer-only)
    python scripts/run_seed_sweep.py \
        --config configs/csi_bench_multitask_config.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNNER = REPO_ROOT / "scripts" / "local_runner.py"


def _parse_seeds(spec: str) -> list[int]:
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def _make_per_seed_config(base: dict, seeds: list[int]) -> dict:
    """Return a config copy with ``seed`` / ``seeds`` set explicitly."""
    cfg = copy.deepcopy(base)
    cfg["seeds"] = seeds
    cfg["seed"] = seeds[0]
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Path to one of configs/csi_bench_{local,multitask}_config.json")
    ap.add_argument("--seeds", default="42",
                    help="Comma-separated list of seeds (default: 42)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the per-seed commands without launching them")
    ap.add_argument("--skip-existing", "--skip_existing",
                    dest="skip_existing", action="store_true",
                    help="Resume an interrupted sweep: skip (task, model, seed) "
                         "combinations whose results already exist on disk.")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="Pass-through args appended to the local_runner call. "
                         "May be repeated, e.g. --extra-arg=--debug")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    base_cfg = json.loads(cfg_path.read_text())
    seeds = _parse_seeds(args.seeds)

    # Materialize a temp config that lists the seeds so ``local_runner.py`` can
    # iterate over (task, model, seed) on its own.  Written to a system temp
    # directory so we don't pollute the repo's ``configs/`` folder.
    materialised = _make_per_seed_config(base_cfg, seeds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="csi_bench_sweep_"))
    tmp_cfg = tmp_dir / f"{cfg_path.stem}_seeds{'-'.join(str(s) for s in seeds)}.json"
    tmp_cfg.write_text(json.dumps(materialised, indent=2))
    print(f"[run_seed_sweep] wrote materialized sweep config -> {tmp_cfg}")
    print(f"[run_seed_sweep] seeds: {seeds}")
    if base_cfg.get("pipeline") == "multitask":
        print(f"[run_seed_sweep] multitask tasks: {base_cfg.get('tasks')}")
    else:
        print(f"[run_seed_sweep] supervised tasks: "
              f"{base_cfg.get('available_tasks') or [base_cfg.get('task')]}")
    print(f"[run_seed_sweep] models: {base_cfg.get('available_models')}")

    cmd = [
        sys.executable,
        str(LOCAL_RUNNER),
        "--config", str(tmp_cfg),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    cmd += list(args.extra_arg)

    print(f"\n[run_seed_sweep] launching: {' '.join(cmd)}\n")
    if args.dry_run:
        return 0

    try:
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = (time.time() - t0) / 60
        print(f"\n[run_seed_sweep] finished in {dt:.1f} min, rc={rc}")
        return rc
    finally:
        # Always remove the temp config so repeated invocations stay tidy.
        try:
            tmp_cfg.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
