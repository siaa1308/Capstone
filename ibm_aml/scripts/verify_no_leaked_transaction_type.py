#!/usr/bin/env python3
"""Regression check: transaction_type_model_safe must never re-enter the model pipeline.

transaction_type_model_safe was created in preprocessing as a "redacted" replacement for
transaction_type_raw, but the redaction flag itself still correlates strongly with the
laundering label and was later confirmed to be leakage. It was independently hardcoded as
an approved feature in the final-dataset builder, so removing it from one stage was not
enough -- this script is a lightweight, dependency-free (stdlib only) guard that checks
every stage at once and fails loudly if any of them silently reintroduces the column.

Run after regenerating prepared_data/ and/or final_temporal_dataset/, or wire into CI.
Does not load full CSV files into memory -- only the first (header) line of each gzip file
is read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

FORBIDDEN_FEATURE = "transaction_type_model_safe"
TRAINING_SCRIPTS = [
    "run_xgboost_baseline.py",
    "run_xgboost_enhanced.py",
    "run_xgboost_tuned.py",
]

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_gzip_header(path: Path) -> str:
    with gzip.open(path, "rt", newline="") as fh:
        return fh.readline()


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def check_prepared_data(prepared_dir: Path, failures: list[str]) -> None:
    manifest_path = prepared_dir / "model_feature_columns.json"
    if not manifest_path.exists():
        print(f"[SKIP] prepared_data manifest not found at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text())
    safe_cols = set(manifest.get("safe_feature_columns", []))
    check(
        FORBIDDEN_FEATURE not in safe_cols,
        f"prepared_data/model_feature_columns.json safe_feature_columns excludes {FORBIDDEN_FEATURE}",
        failures,
    )


def check_final_dataset_config(dataset_dir: Path, failures: list[str]) -> None:
    config_path = dataset_dir / "configuration" / "model_feature_columns.json"
    if not config_path.exists():
        print(f"[SKIP] final dataset config not found at {config_path}")
        return
    config = json.loads(config_path.read_text())
    for key in ("tabular_safe_features", "graph_edge_safe_features"):
        cols = set(config.get(key, []))
        check(
            FORBIDDEN_FEATURE not in cols,
            f"final_temporal_dataset/configuration/model_feature_columns.json[{key}] excludes {FORBIDDEN_FEATURE}",
            failures,
        )


def check_csv_headers(dataset_dir: Path, filename: str, failures: list[str]) -> None:
    matches = sorted(dataset_dir.glob(f"*/*/{filename}"))
    if not matches:
        print(f"[SKIP] no {filename} files found under {dataset_dir}")
        return
    for path in matches:
        columns = [c.strip() for c in read_gzip_header(path).strip().split(",")]
        rel = path.relative_to(dataset_dir)
        check(FORBIDDEN_FEATURE not in columns, f"{rel} header excludes {FORBIDDEN_FEATURE}", failures)


def check_training_guards(scripts_dir: Path, failures: list[str]) -> None:
    for name in TRAINING_SCRIPTS:
        path = scripts_dir / name
        if not path.exists():
            print(f"[SKIP] training script not found at {path}")
            continue
        text = path.read_text()
        check(
            f'"{FORBIDDEN_FEATURE}"' in text,
            f"{name} still references {FORBIDDEN_FEATURE} in its forbidden-feature guard",
            failures,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "prepared_data",
        help="Path to prepared_data/ (output of preprocess_ibm_amlsim_style.py). Skipped if absent.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=resolve_existing(
            SCRIPT_DIR.parent / "final_temporal_dataset",
            SCRIPT_DIR.parent / "data" / "final_temporal_dataset",
        ),
        help="Path to final_temporal_dataset/ (output of build_final_temporal_dataset.py).",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing the run_xgboost_*.py training scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    print(f"Checking for forbidden feature: {FORBIDDEN_FEATURE}\n")
    check_prepared_data(args.prepared_dir, failures)
    check_final_dataset_config(args.dataset_dir, failures)
    check_csv_headers(args.dataset_dir, "transactions.csv.gz", failures)
    check_csv_headers(args.dataset_dir, "edge_list.csv.gz", failures)
    check_training_guards(args.scripts_dir, failures)

    print()
    if failures:
        print(f"REGRESSION CHECK FAILED ({len(failures)} failure(s))")
        for message in failures:
            print(" -", message)
        sys.exit(1)
    print("REGRESSION CHECK PASSED")


if __name__ == "__main__":
    main()
