"""Rebuild the 4-class MotionSourceRecognition label mapping.

The ``label_mapping.json`` currently uploaded at
``s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/MotionSourceRecognition/metadata/label_mapping.json``
collapses the four source classes (``Fan``, ``Human``, ``IRobot``, ``Pet``) into
a binary ``Human`` vs ``Nonhuman`` mapping.  The published CSI-Bench paper
(Table 3 / Table 11) reports motion-source recognition as a *four-class*
classification task.

This script rewrites ``label_mapping.json`` so that each motion source receives
its own index::

    {"Fan": 0, "Human": 1, "IRobot": 2, "Pet": 3}

By default the script uploads the corrected file back to S3 in-place.  Use
``--no-upload`` to stage it locally only.

Usage::

    python util/build_msr_label_mapping.py                    # stage + upload to S3
    python util/build_msr_label_mapping.py --no-upload        # local staging only
    python util/build_msr_label_mapping.py --download-csv     # also pull the CSV
                                                              # to sanity-check
                                                              # the class set
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

BUCKET = "rnd-sagemaker"
TASK_PREFIX = "Data/fm_downstream/CSI-Bench/MotionSourceRecognition"
LABEL_KEY = f"{TASK_PREFIX}/metadata/label_mapping.json"
CSV_KEY = f"{TASK_PREFIX}/metadata/sample_metadata.csv"

# Paper-accurate 4-class mapping (sorted alphabetically).
LABEL_TO_IDX = {"Fan": 0, "Human": 1, "IRobot": 2, "Pet": 3}
IDX_TO_LABEL = {str(v): k for k, v in LABEL_TO_IDX.items()}

NEW_MAPPING = {
    "label_to_idx": LABEL_TO_IDX,
    "idx_to_label": IDX_TO_LABEL,
    "num_classes": len(LABEL_TO_IDX),
}


def _stage_local(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / "label_mapping.json"
    local_path.write_text(json.dumps(NEW_MAPPING, indent=2) + "\n")
    print(f"Wrote 4-class label mapping to {local_path}")
    return local_path


def _read_s3_csv_labels(bucket: str, key: str) -> set[str]:
    """Spot-check: read the existing CSV and return the unique label set.

    Used only as a sanity-check that the four expected classes are present.
    """
    import boto3
    import pandas as pd

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    if "label" not in df.columns:
        raise ValueError(f"CSV at s3://{bucket}/{key} has no 'label' column")
    return set(df["label"].dropna().unique().tolist())


def _upload(local_path: Path, bucket: str, key: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"Uploaded -> s3://{bucket}/{key}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--key", default=LABEL_KEY,
                    help="S3 key of the label_mapping.json to overwrite")
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "_msr_label_mapping_artifacts"),
                    help="Local staging directory for the rebuilt JSON")
    ap.add_argument("--verify",
                    action="store_true",
                    help="Read sample_metadata.csv from S3 and sanity-check "
                         "that the four expected classes (Fan/Human/IRobot/"
                         "Pet) are present before writing.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--upload", dest="upload", action="store_true")
    grp.add_argument("--no-upload", dest="upload", action="store_false")
    ap.set_defaults(upload=True)
    args = ap.parse_args()

    if args.verify:
        print(f"Reading s3://{args.bucket}/{CSV_KEY} for verification")
        labels = _read_s3_csv_labels(args.bucket, CSV_KEY)
        print(f"  unique labels in CSV: {sorted(labels)}")
        missing = set(LABEL_TO_IDX) - labels
        extra = labels - set(LABEL_TO_IDX)
        if missing:
            print(f"  [warn] labels expected but missing in CSV: {sorted(missing)}")
        if extra:
            print(f"  [warn] unexpected labels in CSV: {sorted(extra)}")
        if not labels.issubset(set(LABEL_TO_IDX)) and extra:
            raise SystemExit("CSV contains labels not in the 4-class mapping; "
                             "aborting to be safe.")

    local_path = _stage_local(Path(args.out_dir))

    if args.upload:
        _upload(local_path, args.bucket, args.key)
    else:
        print("--no-upload: local staging only; S3 not modified")


if __name__ == "__main__":
    main()
