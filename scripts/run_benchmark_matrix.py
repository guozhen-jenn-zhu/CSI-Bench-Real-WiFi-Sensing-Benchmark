#!/usr/bin/env python3
"""Unified benchmark matrix runner: (model x dataset x seed).

Datasets:
  - CSI-Bench tasks (loaded via load_benchmark_supervised; require
    --training_dir pointing at the wifi_benchmark_dataset root).
  - External datasets from prunedAttentionGRU (ARIL/HAR-1/HAR-3/SignFi/StanFi),
    loaded via load.external.load_external_dataset; require PAG_REPO_ROOT.
    Datasets whose raw files are missing are reported as 'skipped'.

Models:
  - Built-in baselines: mlp, lstm, resnet18, transformer, vit
  - pruned_attention_gru: trained with the 3-phase PAGTrainer.
  - newmodel: trained with the standard supervised trainer.

Outputs: one JSON per run at <output_dir>/runs/<model>__<dataset>__seed<S>.json
"""
import argparse
import json
import math
import os
import random
import sys
import time
import traceback

# Make repo importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn

from engine.supervised.pag_trainer import PAGTrainer
from engine.supervised.task_trainer import TaskTrainer
from load.external import SkipDataset, list_external_datasets, load_external_dataset
from load.supervised.benchmark_loader import load_benchmark_supervised
from model.supervised.models import (
    LSTMClassifier,
    MLPClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier,
)
from model.supervised.newmodel import NewModel
from model.supervised.pruned_attention_gru import PrunedAttentionGRUClassifier

BUILTIN_MODELS = {
    "mlp": MLPClassifier,
    "lstm": LSTMClassifier,
    "resnet18": ResNet18Classifier,
    "transformer": TransformerClassifier,
    "vit": ViTClassifier,
    "newmodel": NewModel,
}
PAG_MODEL_NAME = "pruned_attention_gru"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


def _build_model(model_name, num_classes, feature_size, win_len, overrides):
    if model_name == PAG_MODEL_NAME:
        return PrunedAttentionGRUClassifier(
            num_classes=num_classes,
            feature_size=feature_size,
            win_len=win_len,
            hidden_dim=int(overrides.get("hidden_dim", 128)),
            attention_dim=int(overrides.get("attention_dim", 32)),
        )
    if model_name == "newmodel":
        return NewModel(
            num_classes=num_classes,
            feature_size=feature_size,
            win_len=win_len,
            hidden_dim=int(overrides.get("hidden_dim", 128)),
            dropout=float(overrides.get("dropout", 0.1)),
        )
    cls = BUILTIN_MODELS[model_name]
    if model_name in ("mlp", "vit"):
        return cls(win_len=win_len, feature_size=feature_size, num_classes=num_classes)
    if model_name == "lstm":
        return cls(feature_size=feature_size, num_classes=num_classes)
    if model_name == "resnet18":
        return cls(win_len=win_len, feature_size=feature_size, num_classes=num_classes)
    if model_name == "transformer":
        return cls(feature_size=feature_size, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def _load_csibench(training_dir, task_name, batch_size):
    return load_benchmark_supervised(
        dataset_root=training_dir,
        task_name=task_name,
        batch_size=batch_size,
        file_format="h5",
        data_key="CSI_amps",
        num_workers=0,
        test_splits="all",
        use_root_as_task_dir=False,
        collate_fn=_custom_collate_fn,
        pin_memory=False,
    )


def _run_pag(model, loaders, num_classes, save_dir, cfg):
    test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    trainer = PAGTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders.get("val", loaders["train"]),
        test_loaders=test_loaders or {"test": loaders.get("test", loaders["train"])},
        num_classes=num_classes,
        save_path=save_dir,
        config=cfg,
    )
    _, results = trainer.train()
    return {"trainer": "pag", "results": results}


def _run_supervised(model, loaders, num_classes, save_dir, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(cfg.get("epochs", 100))
    lr = float(cfg.get("learning_rate", 5e-4))
    wd = float(cfg.get("weight_decay", 1e-5))
    warmup = int(cfg.get("warmup_epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    num_steps = max(len(loaders["train"]) * epochs, 1)
    warmup_steps = max(len(loaders["train"]) * warmup, 1)

    def schedule(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    if not test_loaders:
        test_loaders = {"test": loaders.get("test", loaders["val"])}

    trainer = TaskTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders.get("val", loaders["train"]),
        test_loader=test_loaders,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=save_dir,
        num_classes=num_classes,
        config=cfg,
    )
    _, training_results = trainer.train()
    test_results = {}
    for split_name, loader in test_loaders.items():
        loss, acc = trainer.evaluate(loader)
        try:
            f1, _ = trainer.calculate_metrics(loader)
        except Exception:
            f1 = float("nan")
        test_results[split_name] = {"loss": float(loss), "accuracy": float(acc), "f1_score": float(f1)}
    return {
        "trainer": "supervised",
        "best_epoch": training_results.get("best_epoch"),
        "best_val_accuracy": training_results.get("best_val_accuracy"),
        "test": test_results,
    }


def _run_one(model_name, dataset_kind, dataset_id, seed, matrix_cfg, output_dir):
    """Returns a dict ready to dump to JSON. Never raises."""
    run_id = f"{model_name}__{dataset_kind}_{dataset_id}__seed{seed}"
    save_dir = os.path.join(output_dir, "runs", run_id)
    os.makedirs(save_dir, exist_ok=True)
    started = time.time()
    try:
        set_seed(seed)
        if dataset_kind == "csibench":
            cfg = dict(matrix_cfg.get("csibench", {}))
            data = _load_csibench(
                training_dir=cfg["training_dir"],
                task_name=dataset_id,
                batch_size=int(cfg.get("batch_size", 8)),
            )
            loaders = data["loaders"]
            num_classes = data["num_classes"]
            feature_size = int(cfg.get("feature_size", 232))
            win_len = int(cfg.get("win_len", 500))
        elif dataset_kind == "external":
            cfg = dict(matrix_cfg.get("csibench", {}))  # reuse batch_size
            data = load_external_dataset(
                dataset_id, batch_size=int(cfg.get("batch_size", 32)), seed=seed,
            )
            loaders = data["loaders"]
            num_classes = data["num_classes"]
            feature_size = int(data["feature_size"])
            win_len = -1  # not used for external
        else:
            raise ValueError(f"Unknown dataset kind: {dataset_kind}")

        is_pag = model_name == PAG_MODEL_NAME
        overrides = matrix_cfg.get(
            "pag_overrides" if is_pag else "newmodel_overrides", {},
        )
        model = _build_model(model_name, num_classes, feature_size, win_len, overrides)

        run_cfg = dict(matrix_cfg.get("csibench", {}))
        run_cfg.update(overrides)
        run_cfg["seed"] = seed

        if is_pag:
            outcome = _run_pag(model, loaders, num_classes, save_dir, run_cfg)
        else:
            outcome = _run_supervised(model, loaders, num_classes, save_dir, run_cfg)

        record = {
            "run_id": run_id,
            "status": "ok",
            "model": model_name,
            "dataset_kind": dataset_kind,
            "dataset": dataset_id,
            "seed": seed,
            "num_classes": num_classes,
            "feature_size": feature_size,
            "win_len": win_len,
            "elapsed_s": time.time() - started,
            "outcome": outcome,
        }
    except SkipDataset as e:
        record = {
            "run_id": run_id,
            "status": "skipped",
            "model": model_name,
            "dataset_kind": dataset_kind,
            "dataset": dataset_id,
            "seed": seed,
            "reason": str(e),
            "elapsed_s": time.time() - started,
        }
    except Exception as e:
        record = {
            "run_id": run_id,
            "status": "error",
            "model": model_name,
            "dataset_kind": dataset_kind,
            "dataset": dataset_id,
            "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_s": time.time() - started,
        }

    with open(os.path.join(save_dir, "run.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[{record['status']}] {run_id} ({record['elapsed_s']:.1f}s)")
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT_DIR, "configs", "benchmark_matrix.json"))
    p.add_argument("--models", default=None, help="Comma-separated subset of models to run.")
    p.add_argument("--datasets", default=None, help="Comma-separated dataset ids (any kind).")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    p.add_argument("--training_dir", default=None, help="Override CSI-Bench dataset_root.")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        matrix_cfg = json.load(f)

    if args.training_dir:
        matrix_cfg.setdefault("csibench", {})["training_dir"] = args.training_dir
    output_dir = args.output_dir or matrix_cfg.get("output_dir", "./results/benchmark_matrix")
    os.makedirs(os.path.join(output_dir, "runs"), exist_ok=True)

    models = matrix_cfg.get("models", [])
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    csibench_tasks = matrix_cfg.get("csibench_tasks", [])
    external_datasets = matrix_cfg.get("external_datasets", list_external_datasets())
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        csibench_tasks = [t for t in csibench_tasks if t in wanted]
        external_datasets = [d for d in external_datasets if d in wanted]

    seeds = matrix_cfg.get("seeds", [0, 1, 2])
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    summary = []
    for model_name in models:
        for task in csibench_tasks:
            for s in seeds:
                summary.append(_run_one(model_name, "csibench", task, s, matrix_cfg, output_dir))
        for ds in external_datasets:
            for s in seeds:
                summary.append(_run_one(model_name, "external", ds, s, matrix_cfg, output_dir))

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote summary to {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
