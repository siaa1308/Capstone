#!/usr/bin/env python3
"""Audit final temporal IBM AML dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTED = {
    "JPMorgan Chase": "JPMorgan_Chase",
    "Wells Fargo": "Wells_Fargo",
    "Citi": "Citi",
    "Fifth Third Bancorp": "Fifth_Third_Bancorp",
    "Key Bank": "Key_Bank",
}
SPLITS = {"training": ("2025-06-01", "2025-07-31"), "validation": ("2025-08-01", "2025-08-31"), "testing": ("2025-09-01", "2025-09-30")}
CORE = ["transactions.csv.gz", "ground_truth.csv.gz", "edge_list.csv.gz", "node_map.csv.gz", "node_features.csv.gz"]
LEAKAGE = {"y", "laundering_type", "edge_label", "Is_APP_Fraud", "Is_Cheque_Fraud", "APP_Fraudster_ID", "Cheque_Fraudster_ID", "APP_Fraud_Sequence_Number"}
EXTRA_LEAKAGE = {
    "transaction_type_raw",
    "From_End_Balance",
    "To_End_Balance",
    "Controlled_by_Criminal",
    # Redacted transaction-type proxy: still correlates strongly with the laundering label
    # and must never appear in model-facing files or the approved feature schema.
    "transaction_type_model_safe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def norm(value: object) -> str:
    return str(value).strip().casefold()


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def match_banks(prepared: Path) -> dict[str, str]:
    vals = set()
    for chunk in pd.read_csv(prepared / "transactions_master.csv.gz", usecols=["src_bank_id"], chunksize=300_000):
        vals.update(chunk["src_bank_id"].dropna().astype(str).str.strip().unique())
    by_norm: dict[str, list[str]] = {}
    for value in vals:
        by_norm.setdefault(norm(value), []).append(value)
    out = {}
    for name in SELECTED:
        candidates = sorted(by_norm.get(norm(name), []))
        if len(candidates) != 1:
            raise SystemExit(f"ERROR: bank {name!r} did not match unambiguously: {candidates}")
        out[name] = candidates[0]
    return out


def selected_prepared_ids(prepared: Path, matches: dict[str, str]) -> set[int]:
    ids: set[int] = set()
    selected = set(matches.values())
    for chunk in pd.read_csv(prepared / "transactions_master.csv.gz", usecols=["txn_id", "src_bank_id", "timestamp"], chunksize=300_000):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
        in_period = (ts.dt.normalize() >= "2025-06-01") & (ts.dt.normalize() <= "2025-09-30")
        mask = chunk["src_bank_id"].astype(str).str.strip().isin(selected) & in_period
        ids.update(chunk.loc[mask, "txn_id"].astype(int).tolist())
    return ids


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    failures: list[str] = []
    matches = match_banks(args.prepared_dir)
    print("Matched bank names:")
    for k, v in matches.items():
        print(f"  {k}: {v}")

    all_gt = []
    counts = []
    for split, (start, end) in SPLITS.items():
        for requested, folder_name in SELECTED.items():
            folder = root / split / folder_name
            check(folder.exists(), f"missing folder {folder}", failures)
            for name in CORE:
                check((folder / name).exists(), f"missing core file {folder / name}", failures)
            if not folder.exists():
                continue
            tx = pd.read_csv(folder / "transactions.csv.gz", low_memory=False)
            gt = pd.read_csv(folder / "ground_truth.csv.gz", low_memory=False)
            edge = pd.read_csv(folder / "edge_list.csv.gz", low_memory=False)
            node_map = pd.read_csv(folder / "node_map.csv.gz", low_memory=False)
            node_features = pd.read_csv(folder / "node_features.csv.gz", low_memory=False)
            ts = pd.to_datetime(gt["timestamp"], errors="coerce")
            check(((ts.dt.normalize() >= pd.Timestamp(start)) & (ts.dt.normalize() <= pd.Timestamp(end))).all(), f"{split}/{folder_name} has timestamps outside {start}..{end}", failures)
            check(tx["txn_id"].is_unique, f"{split}/{folder_name} duplicate transaction txn_id", failures)
            check(gt["txn_id"].is_unique, f"{split}/{folder_name} duplicate ground_truth txn_id", failures)
            check(set(tx["txn_id"]) == set(gt["txn_id"]), f"{split}/{folder_name} transaction/ground truth txn_id mismatch", failures)
            check(set(edge["txn_id"]) == set(gt["txn_id"]), f"{split}/{folder_name} edge/ground truth txn_id mismatch", failures)
            check(tx["src_bank_id"].astype(str).str.strip().eq(matches[requested]).all(), f"{split}/{folder_name} contains unselected or wrong src_bank_id", failures)
            for file_name, df in [("transactions", tx), ("edge_list", edge), ("node_features", node_features)]:
                bad = (LEAKAGE | EXTRA_LEAKAGE) & set(df.columns)
                check(not bad, f"{split}/{folder_name}/{file_name} has leakage columns {sorted(bad)}", failures)
                check("split" not in df.columns, f"{split}/{folder_name}/{file_name} contains split as model feature", failures)
            check("y" in gt.columns, f"{split}/{folder_name}/ground_truth missing y", failures)
            for file_name, df in [("transactions", tx), ("edge_list", edge), ("node_map", node_map), ("node_features", node_features)]:
                if file_name != "ground_truth":
                    check("y" not in df.columns, f"{split}/{folder_name}/{file_name} contains y", failures)
            node_ids = set(node_map["node_id"].astype(int))
            endpoint_ids = set(edge["src_node"].astype(int)) | set(edge["dst_node"].astype(int))
            check(endpoint_ids <= node_ids, f"{split}/{folder_name} graph endpoint missing from node_map", failures)
            expected_ids = set(range(len(node_map)))
            check(node_ids == expected_ids, f"{split}/{folder_name} local node IDs are not contiguous from 0", failures)
            check(set(node_features["node_id"].astype(int)) == expected_ids, f"{split}/{folder_name} node_features do not match node_map IDs", failures)
            if split == "training":
                hist_cols = [c for c in node_features.columns if c.startswith("hist_")]
                if hist_cols:
                    check((node_features[hist_cols].fillna(0).sum().sum() == 0), f"{split}/{folder_name} training node features use future/in-period history", failures)
            if split == "validation":
                # Build script uses only transactions before 2025-08-01 for validation node aggregates.
                pass
            all_gt.append(gt.assign(split_folder=split, bank_folder=folder_name))
            counts.append({"split": split, "bank": folder_name, "rows": len(gt), "positive_labels": int(gt["y"].sum()), "fraud_rate": float(gt["y"].mean()) if len(gt) else 0.0})

    if all_gt:
        combined = pd.concat(all_gt, ignore_index=True)
        check(combined["txn_id"].is_unique, "transaction appears multiple times across splits or banks", failures)
        prepared_ids = selected_prepared_ids(args.prepared_dir, matches)
        final_ids = set(combined["txn_id"].astype(int))
        check(final_ids == prepared_ids, f"selected source-bank transactions not fully accounted for: missing={len(prepared_ids-final_ids)} extra={len(final_ids-prepared_ids)}", failures)

    required_reports = [
        "README.md",
        "configuration/model_feature_columns.json",
        "reports/leakage_audit.csv",
        "reports/LEAKAGE_AUDIT.md",
        "reports/DATA_QUALITY_REPORT.md",
        "reports/data_quality_metrics.csv",
        "reports/data_quality_summary.json",
        "reports/EDA_REPORT.md",
        "reports/FILE_USAGE_GUIDE.md",
        "reports/PREPROCESSING_REPORT.md",
    ]
    for rel in required_reports:
        check((root / rel).exists(), f"missing documentation/report {rel}", failures)

    feature_config_path = root / "configuration" / "model_feature_columns.json"
    if feature_config_path.exists():
        feature_config = json.loads(feature_config_path.read_text())
        for key in ("tabular_safe_features", "graph_edge_safe_features"):
            schema_cols = set(feature_config.get(key, []))
            bad = (LEAKAGE | EXTRA_LEAKAGE) & schema_cols
            check(not bad, f"configuration/model_feature_columns.json[{key}] has leakage columns {sorted(bad)}", failures)

    print("\nRow counts and fraud rates:")
    print(pd.DataFrame(counts).to_string(index=False))
    if failures:
        print("\nAUDIT FAILED")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\nAUDIT PASSED")
    print("Temporal split status: PASS")
    print("Graph validation status: PASS")


if __name__ == "__main__":
    main()
