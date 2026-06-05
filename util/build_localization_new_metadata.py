"""Build sample_metadata.csv, label_mapping.json, and split JSONs for the
updated Localization dataset (Localization_New) on S3.

Mirrors the existing Localization layout at
``s3://rnd-sagemaker/Data/fm_downstream/Localization/`` while adding a new
``device`` column to capture the device-triplet directory present in the new
dataset, and including the new ``000`` (empty) class.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import boto3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METADATA_COLUMNS = [
    "id",
    "file_path",
    "subject",
    "user",
    "activity",
    "environment",
    "num_sub",
    "c_freq",
    "difficulty",
    "deplyoment",
    "session",
    "is_multi_subject",
    "label",
    "device",
]

SESSION_RE = re.compile(r"^session_(\d+)__freq(\d+)\.h5$")


def list_h5_keys(bucket: str, prefix: str) -> list[str]:
    """List all .h5 keys under prefix using paginated S3 listing."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if k.endswith(".h5"):
                keys.append(k)
    return keys


def parse_key(key: str, prefix: str) -> dict | None:
    """Parse one S3 key into a metadata-row dict.

    Returns ``None`` if the key does not match the expected layout.
    """
    rel = key[len(prefix):] if key.startswith(prefix) else key
    parts = rel.split("/")
    if len(parts) < 2:
        return None

    env = diff = dpl = act = user = mot = dev = ""
    fname = parts[-1]
    for seg in parts[:-1]:
        if seg.startswith("env_"):
            env = seg[len("env_"):]
        elif seg.startswith("Diff_"):
            diff = seg[len("Diff_"):]
        elif seg.startswith("Dpl_"):
            dpl = seg[len("Dpl_"):]
        elif seg.startswith("act_"):
            act = seg[len("act_"):]
        elif seg.startswith("user_"):
            user = seg[len("user_"):]
        elif seg.startswith("motionsrc_"):
            mot = seg[len("motionsrc_"):]
        elif seg.startswith("device_"):
            dev = seg[len("device_"):]

    m = SESSION_RE.match(fname)
    if not m:
        return None
    session, freq = m.group(1), m.group(2)

    if act == "empty" and not user:
        user = "Empty"

    if "-5G" in dev:
        c_freq = "5G"
    elif "-2G" in dev:
        c_freq = "2G"
    else:
        c_freq = ""

    is_multi = "TRUE" if act == "twouser_walking" else "FALSE"

    # ``mot`` is a zero-padded 3-digit string from the directory name
    # (e.g. ``001``).  The old ``label`` column stores the int form
    # (e.g. ``1``) so we follow that convention here.
    try:
        label_int = int(mot)
    except ValueError:
        label_int = 0

    sample_id = (
        f"Human_{user}_{env}_{act}_{dpl}_{diff}_{mot}_{freq}_{session}"
    )

    rel_path = "./" + rel

    return {
        "id": sample_id,
        "file_path": rel_path,
        "subject": "Human",
        "user": user,
        "activity": act,
        "environment": env,
        "num_sub": freq,
        "c_freq": c_freq,
        "difficulty": diff,
        "deplyoment": dpl,
        "session": session,
        "is_multi_subject": is_multi,
        "label": label_int,
        "device": dev,
    }


def build_dataframe(keys: Iterable[str], prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    skipped = 0
    for k in keys:
        row = parse_key(k, prefix)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        print(f"[warn] skipped {skipped} keys that did not match expected layout")
    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    return df


def make_splits(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})"
        )

    ids = df["id"].tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # remainder goes to test so all rows are accounted for
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    test_set = set(test_ids)
    test_meta = df[df["id"].isin(test_set)]
    test_easy = test_meta[test_meta["difficulty"] == "easy"]["id"].tolist()
    test_medium = test_meta[test_meta["difficulty"] == "medium"]["id"].tolist()
    test_hard = test_meta[test_meta["difficulty"] == "hard"]["id"].tolist()

    return {
        "train_id": train_ids,
        "val_id": val_ids,
        "test_id": test_ids,
        "test_easy_id": test_easy,
        "test_medium_id": test_medium,
        "test_hard_id": test_hard,
    }


def write_artifacts(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    out_dir: Path,
    label_column: str = "label",
) -> dict[str, Path]:
    metadata_dir = out_dir / "metadata"
    splits_dir = out_dir / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / "sample_metadata.csv"
    df.to_csv(metadata_path, index=False)

    # Build label_mapping.json in the same format as the old Localization
    # dataset (compatible with ``LabelMapper.load``). Native Python ints are
    # required because ``LabelMapper.fit`` uses ``np.unique`` internally,
    # which produces ``np.int64`` keys that ``json.dump`` cannot serialize.
    unique_labels = sorted({int(v) for v in df[label_column].tolist()})
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    mapping = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "num_classes": len(unique_labels),
    }
    mapping_path = metadata_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote label mapping with {len(unique_labels)} classes to {mapping_path}")
    for lbl, idx in label_to_idx.items():
        print(f"  {lbl} -> {idx}")

    split_paths: dict[str, Path] = {}
    for split_name, ids in splits.items():
        sp = splits_dir / f"{split_name}.json"
        with open(sp, "w") as f:
            json.dump(ids, f)
        split_paths[split_name] = sp

    return {
        "metadata": metadata_path,
        "label_mapping": mapping_path,
        **split_paths,
    }


def verify(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    expected_count: int | None = None,
) -> None:
    n = len(df)
    print("\n==== Verification ====")
    print(f"total rows: {n}")
    print(f"unique ids: {df['id'].nunique()}")
    if expected_count is not None and n != expected_count:
        print(f"[warn] expected {expected_count} rows, got {n}")
    assert df["id"].nunique() == n, "id collisions detected"

    train = set(splits["train_id"])
    val = set(splits["val_id"])
    test = set(splits["test_id"])
    assert train.isdisjoint(val), "train/val overlap"
    assert train.isdisjoint(test), "train/test overlap"
    assert val.isdisjoint(test), "val/test overlap"
    assert (
        len(train) + len(val) + len(test) == n
    ), f"split sum {len(train) + len(val) + len(test)} != n {n}"

    easy = set(splits["test_easy_id"])
    medium = set(splits["test_medium_id"])
    hard = set(splits["test_hard_id"])
    assert easy.issubset(test), "test_easy not subset of test_id"
    assert medium.issubset(test), "test_medium not subset of test_id"
    assert hard.issubset(test), "test_hard not subset of test_id"
    assert easy | medium | hard == test, "easy ∪ medium ∪ hard != test_id"
    assert easy.isdisjoint(medium) and easy.isdisjoint(hard) and medium.isdisjoint(hard), (
        "test difficulty buckets overlap"
    )

    print("split sizes:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")

    print("\nlabel distribution (full metadata):")
    print(df["label"].value_counts().sort_index().to_string())

    train_meta = df[df["id"].isin(train)]
    val_meta = df[df["id"].isin(val)]
    test_meta = df[df["id"].isin(test)]

    print("\ndifficulty distribution per split:")
    for name, m in [("train_id", train_meta), ("val_id", val_meta), ("test_id", test_meta)]:
        print(f"  {name}: {m['difficulty'].value_counts().to_dict()}")

    print("\nenv distribution per split:")
    for name, m in [("train_id", train_meta), ("val_id", val_meta), ("test_id", test_meta)]:
        print(f"  {name}: {m['environment'].value_counts().to_dict()}")
    print("=====================\n")


def s3_head_check(bucket: str, prefix: str, df: pd.DataFrame, n: int = 5) -> None:
    """Spot-check that the file_path keys exist on S3."""
    s3 = boto3.client("s3")
    sample = df.sample(n=min(n, len(df)), random_state=0)
    print(f"Spot-checking {len(sample)} S3 keys exist:")
    for fp in sample["file_path"]:
        rel = fp[2:] if fp.startswith("./") else fp
        key = prefix + rel
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"  OK  s3://{bucket}/{key}")
        except Exception as e:
            print(f"  ERR s3://{bucket}/{key} -> {e}")


def upload_artifacts(
    out_dir: Path,
    bucket: str,
    target_prefix: str,
) -> None:
    s3 = boto3.client("s3")
    if not target_prefix.endswith("/"):
        target_prefix += "/"

    pairs: list[tuple[Path, str]] = []
    metadata_dir = out_dir / "metadata"
    splits_dir = out_dir / "splits"
    for fp in sorted(metadata_dir.glob("*")):
        if fp.is_file():
            pairs.append((fp, f"{target_prefix}metadata/{fp.name}"))
    for fp in sorted(splits_dir.glob("*.json")):
        pairs.append((fp, f"{target_prefix}splits/{fp.name}"))

    print(f"Uploading {len(pairs)} files to s3://{bucket}/{target_prefix}")
    for local, key in pairs:
        s3.upload_file(str(local), bucket, key)
        print(f"  -> s3://{bucket}/{key}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="rnd-sagemaker")
    p.add_argument(
        "--prefix",
        default="Data/fm_downstream/Localization_New/Localization_h5/",
        help="S3 prefix that contains sub_Human/. Must end with '/'.",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "_loc_new_artifacts"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--expected-count", type=int, default=10732)

    upload_grp = p.add_mutually_exclusive_group()
    upload_grp.add_argument("--upload", dest="upload", action="store_true")
    upload_grp.add_argument("--no-upload", dest="upload", action="store_false")
    p.set_defaults(upload=True)

    p.add_argument(
        "--target-prefix",
        default=None,
        help="Override S3 prefix to upload metadata/ and splits/ to. "
        "Defaults to --prefix (i.e. Localization_h5 root).",
    )

    args = p.parse_args()

    if not args.prefix.endswith("/"):
        args.prefix = args.prefix + "/"

    list_prefix = args.prefix + "sub_Human/"
    print(f"Listing keys under s3://{args.bucket}/{list_prefix}")
    keys = list_h5_keys(args.bucket, list_prefix)
    print(f"Found {len(keys)} .h5 keys")

    df = build_dataframe(keys, args.prefix)

    splits = make_splits(
        df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    out_dir = Path(args.out_dir)
    paths = write_artifacts(df, splits, out_dir)
    print("Wrote artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    verify(df, splits, expected_count=args.expected_count)

    s3_head_check(args.bucket, args.prefix, df, n=5)

    if args.upload:
        target_prefix = args.target_prefix or args.prefix
        upload_artifacts(out_dir, args.bucket, target_prefix)
    else:
        print("--no-upload: skipping S3 upload")


if __name__ == "__main__":
    main()
