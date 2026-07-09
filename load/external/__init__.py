"""Adapter for the prunedAttentionGRU repo's per-dataset loaders.

The external datasets (ARIL, HAR-1, HAR-3, SignFi, StanFi) live in the upstream
parkis2002/prunedAttentionGRU repo and require raw .mat files that are NOT
shipped here. This module wraps the upstream loaders so they can plug into the
unified benchmark runner, and raises SkipDataset when raw data is unavailable.

Usage: set env var PAG_REPO_ROOT to the absolute path of a clone of
prunedAttentionGRU (with raw .mat files in ARIL/, HAR/, SignFi/, StanFi/) and
the loaders will find them. If unset or files missing, the runner skips.
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class SkipDataset(Exception):
    """Raised when raw external data is not available; runner skips this dataset."""


@contextmanager
def _add_to_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _resolve_pag_root():
    root = os.environ.get("PAG_REPO_ROOT")
    if not root:
        raise SkipDataset(
            "PAG_REPO_ROOT not set. Point it at a clone of prunedAttentionGRU "
            "containing raw dataset files."
        )
    if not os.path.isdir(root):
        raise SkipDataset(f"PAG_REPO_ROOT={root} is not a directory.")
    return root


def _check_data_files(pag_root, files):
    missing = [f for f in files if not os.path.exists(os.path.join(pag_root, f))]
    if missing:
        raise SkipDataset(
            f"Missing raw data files: {missing}. See dataset .txt guides in the "
            "prunedAttentionGRU repo for download instructions."
        )


def _to_class_indices(y):
    """Convert one-hot or already-indexed labels to a 1D long tensor of class indices."""
    if isinstance(y, torch.Tensor):
        arr = y
    else:
        arr = torch.as_tensor(np.asarray(y))
    if arr.dim() == 2 and arr.shape[1] > 1:
        return arr.argmax(dim=1).long()
    return arr.long().reshape(-1)


def _wrap_as_bench_inputs(X, feature_size):
    """Convert (N, T, F) to (N, 1, T, F) so it matches BenchmarkCSIDataset output.
    If X is (N, F, T) it is transposed to (N, T, F) first.
    """
    X = torch.as_tensor(np.asarray(X)).float()
    if X.dim() == 3:
        if X.shape[-1] != feature_size and X.shape[-2] == feature_size:
            X = X.transpose(1, 2)
        X = X.unsqueeze(1)  # (N, 1, T, F)
    return X


def _make_loaders(X_train, y_train, X_test, y_test, feature_size, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    Xtr = _wrap_as_bench_inputs(X_train, feature_size)
    Xte = _wrap_as_bench_inputs(X_test, feature_size)
    ytr = _to_class_indices(y_train)
    yte = _to_class_indices(y_test)
    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# (name, feature_size, num_classes, required_files, loader_callable_name)
_DATASETS = {
    "aril":   (52, 6,
               ["ARIL/train_data_split_amp.mat", "ARIL/test_data_split_amp.mat"],
               "aril"),
    "har_1":  (104, 4,
               ["HAR/Data1.mat"],
               "har1"),
    "har_3":  (256, 5,
               ["HAR/Data3.mat"],
               "har3"),
    "signfi": (90, 276,
               ["SignFi/dataset_home_276.mat"],
               "signfi"),
    "stanfi": (90, 6,
               ["StanFi/csid_lab.mat", "StanFi/label_lab.mat"],
               "stanfi"),
}


def load_external_dataset(name, batch_size=32, seed=42):
    """Return dict matching CSI-Bench's load_benchmark_supervised result shape.

    Keys: loaders={'train','val','test'}, num_classes, feature_size.
    Raises SkipDataset if raw data is unavailable.
    """
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(f"Unknown external dataset: {name}")
    feature_size, num_classes, required_files, fn_name = _DATASETS[name]

    pag_root = _resolve_pag_root()
    _check_data_files(pag_root, required_files)

    cwd0 = os.getcwd()
    try:
        os.chdir(pag_root)
        with _add_to_path(pag_root):
            if name == "aril":
                from ARIL.aril import aril as _fn
            elif name == "har_1":
                from HAR.har import har1 as _fn
            elif name == "har_3":
                from HAR.har import har3 as _fn
            elif name == "signfi":
                from SignFi.signfi import signfi as _fn
            elif name == "stanfi":
                from StanFi.stanfi import stanfi as _fn
            else:
                raise SkipDataset(name)
            X_train, y_train, X_test, y_test = _fn()
    finally:
        os.chdir(cwd0)

    train_loader, val_loader, test_loader = _make_loaders(
        X_train, y_train, X_test, y_test, feature_size, batch_size, seed,
    )
    return {
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "num_classes": num_classes,
        "feature_size": feature_size,
        "label_mapper": None,
        "name": name,
    }


def list_external_datasets():
    return list(_DATASETS.keys())
