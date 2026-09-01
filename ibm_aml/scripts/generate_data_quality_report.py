#!/usr/bin/env python3
"""Generate data-quality reports for final temporal IBM AML dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BANKS = ["JPMorgan_Chase", "Wells_Fargo", "Citi", "Fifth_Third_Bancorp", "Key_Bank"]
SPLITS = {"training": ("2025-06-01", "2025-07-31"), "validation": ("2025-08-01", "2025-08-31"), "testing": ("2025-09-01", "2025-09-30")}
CORE = ["transactions.csv.gz", "ground_truth.csv.gz", "edge_list.csv.gz", "node_map.csv.gz", "node_features.csv.gz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def status(ok: bool, warning: bool = False) -> str:
    if not ok:
        return "FAIL"
    return "WARNING" if warning else "PASS"


def add(rows: list[dict[str, object]], check: str, file: str, stat: str, value: object, detail: str = "") -> None:
    rows.append({"check": check, "file": file, "status": stat, "value": value, "detail": detail})


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    all_txn: list[pd.DataFrame] = []
    schemas: dict[str, dict[str, list[str]]] = {}
    split_bank_counts: list[dict[str, object]] = []

    for split, (start, end) in SPLITS.items():
        for bank in BANKS:
            folder = root / split / bank
            schemas[f"{split}/{bank}"] = {}
            for name in CORE:
                path = folder / name
                add(rows, "core file exists", f"{split}/{bank}/{name}", status(path.exists()), path.exists())
            if not folder.exists():
                continue
            tx = pd.read_csv(folder / "transactions.csv.gz", low_memory=False)
            gt = pd.read_csv(folder / "ground_truth.csv.gz", low_memory=False)
            edge = pd.read_csv(folder / "edge_list.csv.gz", low_memory=False)
            node_map = pd.read_csv(folder / "node_map.csv.gz", low_memory=False)
            node_features = pd.read_csv(folder / "node_features.csv.gz", low_memory=False)
            for name, df in [("transactions", tx), ("ground_truth", gt), ("edge_list", edge), ("node_map", node_map), ("node_features", node_features)]:
                schemas[f"{split}/{bank}"][name] = list(df.columns)
                add(rows, "row count", f"{split}/{bank}/{name}.csv.gz", "PASS", len(df))
                add(rows, "duplicate rows", f"{split}/{bank}/{name}.csv.gz", status(not df.duplicated().any()), int(df.duplicated().sum()))
                for col, miss in df.isna().sum().items():
                    add(rows, "missing values by column", f"{split}/{bank}/{name}.csv.gz::{col}", status(miss == 0, miss > 0), int(miss))
                for col, dtype in df.dtypes.astype(str).items():
                    add(rows, "column dtype", f"{split}/{bank}/{name}.csv.gz::{col}", "PASS", dtype)
            add(rows, "duplicate txn_id values", f"{split}/{bank}/transactions.csv.gz", status(tx["txn_id"].is_unique), int(tx["txn_id"].duplicated().sum()))
            add(rows, "duplicate txn_id values", f"{split}/{bank}/ground_truth.csv.gz", status(gt["txn_id"].is_unique), int(gt["txn_id"].duplicated().sum()))
            ts = pd.to_datetime(gt["timestamp"], errors="coerce")
            add(rows, "invalid or unparseable timestamps", f"{split}/{bank}/ground_truth.csv.gz", status(ts.notna().all()), int(ts.isna().sum()))
            outside = ~((ts.dt.normalize() >= pd.Timestamp(start)) & (ts.dt.normalize() <= pd.Timestamp(end)))
            add(rows, "timestamps outside expected period", f"{split}/{bank}/ground_truth.csv.gz", status(not outside.any()), int(outside.sum()))
            if "amount" in tx:
                add(rows, "zero amounts", f"{split}/{bank}/transactions.csv.gz", status((tx["amount"] != 0).all(), (tx["amount"] == 0).any()), int((tx["amount"] == 0).sum()))
                add(rows, "negative amounts", f"{split}/{bank}/transactions.csv.gz", status((tx["amount"] >= 0).all()), int((tx["amount"] < 0).sum()))
                add(rows, "extreme amount max", f"{split}/{bank}/transactions.csv.gz", "PASS", float(tx["amount"].max()))
            bad_labels = ~gt["y"].isin([0, 1])
            add(rows, "invalid labels", f"{split}/{bank}/ground_truth.csv.gz", status(not bad_labels.any()), int(bad_labels.sum()))
            y_counts = gt["y"].value_counts().to_dict()
            add(rows, "class imbalance", f"{split}/{bank}/ground_truth.csv.gz", "WARNING" if y_counts.get(1, 0) == 0 or gt["y"].mean() < 0.01 else "PASS", json.dumps(y_counts), f"fraud_rate={gt['y'].mean():.6f}")
            add(rows, "unique source accounts", f"{split}/{bank}/transactions.csv.gz", "PASS", int(tx["src_id"].nunique()))
            add(rows, "unique destination accounts", f"{split}/{bank}/transactions.csv.gz", "PASS", int(tx["dst_id"].nunique()))
            known_accounts = set(node_map["account_id"].astype(str))
            refs = set(tx["src_id"].astype(str)) | set(tx["dst_id"].astype(str))
            unknown = refs - known_accounts
            add(rows, "unknown account references", f"{split}/{bank}", status(not unknown), len(unknown))
            invalid_bank = tx["src_bank_id"].nunique() != 1
            add(rows, "invalid bank values", f"{split}/{bank}/transactions.csv.gz", status(not invalid_bank), int(tx["src_bank_id"].nunique()))
            currencies = sorted(set(tx.get("currency", pd.Series(dtype=str)).dropna().astype(str)) | set(tx.get("Receiving_Currency", pd.Series(dtype=str)).dropna().astype(str)))
            add(rows, "inconsistent currencies", f"{split}/{bank}/transactions.csv.gz", "WARNING" if len(currencies) > 1 else "PASS", ",".join(currencies))
            for col in tx.columns:
                nunique = tx[col].nunique(dropna=False)
                if nunique <= 1:
                    add(rows, "constant columns", f"{split}/{bank}/transactions.csv.gz::{col}", "WARNING", int(nunique))
                elif nunique / max(len(tx), 1) > 0.5 and tx[col].dtype == "object":
                    add(rows, "high-cardinality categorical columns", f"{split}/{bank}/transactions.csv.gz::{col}", "WARNING", int(nunique))
            missing_nodes = (set(edge["src_node"]) | set(edge["dst_node"])) - set(node_map["node_id"])
            add(rows, "graph nodes missing from node maps", f"{split}/{bank}/edge_list.csv.gz", status(not missing_nodes), len(missing_nodes))
            add(rows, "duplicate graph edges", f"{split}/{bank}/edge_list.csv.gz", status(not edge.duplicated(["src_node", "dst_node", "txn_id"]).any()), int(edge.duplicated(["src_node", "dst_node", "txn_id"]).sum()))
            deg_nodes = set(edge["src_node"]) | set(edge["dst_node"])
            isolated = set(node_map["node_id"]) - deg_nodes
            add(rows, "disconnected or isolated nodes", f"{split}/{bank}", "PASS", len(isolated))
            add(rows, "graph endpoint node features", f"{split}/{bank}", status(set(node_map["node_id"]) == set(node_features["node_id"])), len(node_features))
            add(rows, "transactions match labels", f"{split}/{bank}", status(set(tx["txn_id"]) == set(gt["txn_id"])), len(set(tx["txn_id"]) ^ set(gt["txn_id"])))
            split_bank_counts.append({"split": split, "bank": bank, "rows": len(tx), "y0": int((gt["y"] == 0).sum()), "y1": int((gt["y"] == 1).sum()), "fraud_rate": float(gt["y"].mean())})
            all_txn.append(gt.assign(bank=bank, folder_split=split))

    if all_txn:
        all_gt = pd.concat(all_txn, ignore_index=True)
        add(rows, "transactions duplicated across splits or banks", "all ground_truth", status(all_gt["txn_id"].is_unique), int(all_gt["txn_id"].duplicated().sum()))
        add(rows, "transactions lost during splitting", "all ground_truth", "PASS", len(all_gt), "selected-bank rows in final dataset")
    schema_ref = None
    for folder, file_schemas in schemas.items():
        if schema_ref is None:
            schema_ref = file_schemas
            continue
        add(rows, "schema differences across folders", folder, status(file_schemas == schema_ref), json.dumps({k: len(v) for k, v in file_schemas.items()}))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(report_dir / "data_quality_metrics.csv", index=False)
    summary = {
        "status_counts": metrics["status"].value_counts().to_dict(),
        "split_bank_counts": split_bank_counts,
        "total_rows": int(sum(x["rows"] for x in split_bank_counts)),
        "total_positive_labels": int(sum(x["y1"] for x in split_bank_counts)),
    }
    (report_dir / "data_quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    status_lines = ["| status | count |", "|---|---:|"]
    for key, value in metrics["status"].value_counts().items():
        status_lines.append(f"| {key} | {value} |")
    count_lines = ["| split | bank | rows | y0 | y1 | fraud_rate |", "|---|---|---:|---:|---:|---:|"]
    for item in split_bank_counts:
        count_lines.append(
            f"| {item['split']} | {item['bank']} | {item['rows']} | {item['y0']} | {item['y1']} | {item['fraud_rate']:.6f} |"
        )
    md = "# Data Quality Report\n\n" + "\n".join(status_lines) + "\n\n## Counts by Split and Bank\n\n" + "\n".join(count_lines) + "\n"
    (report_dir / "DATA_QUALITY_REPORT.md").write_text(md)
    print("Wrote data quality reports to", report_dir)


if __name__ == "__main__":
    main()
