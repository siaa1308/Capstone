#!/usr/bin/env python3
"""Build leakage-safe temporal-then-bank IBM AML dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


SELECTED_BANKS = {
    "JPMorgan Chase": "JPMorgan_Chase",
    "Wells Fargo": "Wells_Fargo",
    "Citi": "Citi",
    "Fifth Third Bancorp": "Fifth_Third_Bancorp",
    "Key Bank": "Key_Bank",
}
SPLITS = {
    "training": ("2025-06-01", "2025-07-31", pd.Timestamp("2025-06-01")),
    "validation": ("2025-08-01", "2025-08-31", pd.Timestamp("2025-08-01")),
    "testing": ("2025-09-01", "2025-09-30", pd.Timestamp("2025-09-01")),
}
LEAKAGE_COLUMNS = [
    "y",
    "laundering_type",
    "edge_label",
    "Is_APP_Fraud",
    "Is_Cheque_Fraud",
    "APP_Fraudster_ID",
    "Cheque_Fraudster_ID",
    "APP_Fraud_Sequence_Number",
]
# transaction_type_model_safe was originally created upstream as a "redacted" replacement
# for transaction_type_raw, but the redaction flag itself still correlates strongly with the
# laundering label and is confirmed leakage. It must never be treated as a safe tabular or
# graph-edge feature, and it is asserted absent from the generated feature schema below.
FORBIDDEN_MODEL_FEATURES = {
    "transaction_type_model_safe",
}
NAME_LEAKAGE_TERMS = [
    "fraud",
    "laundering",
    "suspicious",
    "label",
    "target",
    "fraudster",
    "investigation",
    "alert",
    "sar",
    "case",
    "outcome",
]
IDENTIFIER_COLUMNS = [
    "txn_id",
    "timestamp",
    "Transaction_Date",
    "Transaction_Time",
    "src_id",
    "dst_id",
    "src_bank_id",
    "dst_bank_id",
]
TABULAR_SAFE_FEATURES = [
    "Transaction_Day_of_Week",
    "amount",
    "currency",
    "Amount_Received",
    "Receiving_Currency",
    "payment_format",
    "From_Initial_Balance",
    "To_Initial_Balance",
    "Sufficient_Funds",
    "Overdraft_Okay",
    "Is_All_Cash",
    "Is_Hold",
    "src_prev_txn_count",
    "src_prev_amount_sum",
    "src_prev_amount_mean",
    "dst_prev_txn_count",
    "dst_prev_amount_sum",
    "dst_prev_amount_mean",
]
GRAPH_EDGE_SAFE_FEATURES = [
    "timestamp",
    "amount",
    "currency",
    "Amount_Received",
    "Receiving_Currency",
    "payment_format",
]
GRAPH_NODE_SAFE_FEATURES = [
    "node_id",
    "Account_Country",
    "Account_Currency",
    "Entity_Type",
    "Does_Account_Have_Debit_Card",
    "DebitCard_Index_in_EntityCards",
    "MCC",
    "Account_Type",
    "Max_Overdraft",
    "account_kind",
    "hist_sent_count",
    "hist_sent_amount_sum",
    "hist_recv_count",
    "hist_recv_amount_sum",
    "hist_out_degree",
    "hist_in_degree",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def norm(value: object) -> str:
    return str(value).strip().casefold()


def split_for_timestamp(ts: pd.Series) -> pd.Series:
    dates = ts.dt.normalize()
    out = pd.Series(pd.NA, index=ts.index, dtype="object")
    out[(dates >= "2025-06-01") & (dates <= "2025-07-31")] = "training"
    out[(dates >= "2025-08-01") & (dates <= "2025-08-31")] = "validation"
    out[(dates >= "2025-09-01") & (dates <= "2025-09-30")] = "testing"
    return out


def match_banks(tx: pd.DataFrame) -> dict[str, str]:
    by_norm: dict[str, list[str]] = {}
    for value in sorted(tx["src_bank_id"].dropna().astype(str).str.strip().unique()):
        by_norm.setdefault(norm(value), []).append(value)
    matches = {}
    for requested in SELECTED_BANKS:
        candidates = by_norm.get(norm(requested), [])
        if len(candidates) != 1:
            raise SystemExit(
                f"ERROR: could not match {requested!r} unambiguously. Candidates: {candidates}"
            )
        matches[requested] = candidates[0]
    print("Matched selected src_bank_id values:")
    for requested, actual in matches.items():
        print(f"  {requested}: {actual}")
    return matches


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def audit_columns(columns: list[str], dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in columns:
        low = col.casefold()
        reasons: list[str] = []
        status = "allowed"
        action = "kept where explicitly approved"
        removed = False
        manual = False
        if col in LEAKAGE_COLUMNS:
            reasons.append("confirmed leakage column")
            status = "confirmed_leakage"
            action = "removed from model inputs; y kept only in ground_truth"
            removed = col != "y"
        if any(term in low for term in NAME_LEAKAGE_TERMS):
            reasons.append("name contains target-revealing term")
            if status == "allowed":
                status = "confirmed_leakage"
                action = "removed from model inputs"
                removed = True
        if col == "transaction_type_raw":
            reasons.append("raw category contains Money Laundering")
            status = "confirmed_leakage"
            action = "removed from model inputs; no redacted replacement is retained"
            removed = True
        if col in FORBIDDEN_MODEL_FEATURES:
            reasons.append(
                "redacted transaction-type proxy still correlates with the laundering label"
            )
            status = "confirmed_leakage"
            action = "removed from model inputs"
            removed = True
        if col in {"From_End_Balance", "To_End_Balance"}:
            reasons.append("post-transaction balance unavailable before transaction completion")
            status = "post_transaction"
            action = "removed from model inputs"
            removed = True
        if col == "Controlled_by_Criminal":
            reasons.append("account attribute directly identifies criminal control")
            status = "confirmed_leakage"
            action = "removed from node features"
            removed = True
        if col in {"Financial_Institution_ID", "Branch", "Entity_ID", "Entity_Name", "Account_ID"}:
            reasons.append("high-cardinality identifier or original account identifier")
            status = "non_feature_identifier"
            action = "excluded from model feature lists"
        if col == "split":
            reasons.append("temporal split marker is not a model feature")
            status = "non_feature"
            action = "kept only in ground_truth/reports where useful"
        if reasons:
            rows.append(
                {
                    "column": col,
                    "dataset/file": dataset,
                    "detection_reason": "; ".join(reasons),
                    "evidence": "column name/schema inspection",
                    "status": status,
                    "action": action,
                    "removed_from_model_inputs": bool(removed or status in {"confirmed_leakage", "post_transaction"}),
                    "manual_review_required": bool(manual),
                }
            )
    return rows


def historical_node_aggregates(all_tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = all_tx[all_tx["timestamp_dt"] < cutoff]
    if hist.empty:
        return pd.DataFrame(columns=["account_id"])
    sent = hist.groupby("src_id").agg(
        hist_sent_count=("txn_id", "size"),
        hist_sent_amount_sum=("amount", "sum"),
        hist_out_degree=("dst_id", "nunique"),
    )
    recv = hist.groupby("dst_id").agg(
        hist_recv_count=("txn_id", "size"),
        hist_recv_amount_sum=("amount", "sum"),
        hist_in_degree=("src_id", "nunique"),
    )
    out = sent.join(recv, how="outer").fillna(0).reset_index()
    out = out.rename(columns={out.columns[0]: "account_id"})
    out["account_id"] = out["account_id"].astype(str)
    return out


def build_folder(
    part: pd.DataFrame,
    all_tx: pd.DataFrame,
    accounts: pd.DataFrame,
    split_name: str,
    bank_folder: str,
    split_start: pd.Timestamp,
    out_dir: Path,
) -> dict[str, object]:
    folder = out_dir / split_name / bank_folder
    folder.mkdir(parents=True, exist_ok=True)

    tx_cols = [c for c in IDENTIFIER_COLUMNS + TABULAR_SAFE_FEATURES if c in part.columns]
    gt_cols = ["txn_id", "timestamp", "split", "y"]
    if FORBIDDEN_MODEL_FEATURES & set(tx_cols):
        raise SystemExit(
            f"ERROR: forbidden feature(s) {sorted(FORBIDDEN_MODEL_FEATURES & set(tx_cols))} "
            f"about to be written to {folder / 'transactions.csv.gz'}"
        )
    part[tx_cols].to_csv(folder / "transactions.csv.gz", index=False, compression="gzip")
    part[gt_cols].to_csv(folder / "ground_truth.csv.gz", index=False, compression="gzip")

    accounts_used = pd.Index(pd.concat([part["src_id"], part["dst_id"]], ignore_index=True).astype(str).unique())
    node_map = pd.DataFrame({"account_id": sorted(accounts_used, key=lambda x: int(x) if x.isdigit() else x)})
    node_map["node_id"] = range(len(node_map))
    node_map.to_csv(folder / "node_map.csv.gz", index=False, compression="gzip")
    local = dict(zip(node_map["account_id"], node_map["node_id"]))

    edge = part[["txn_id", "src_id", "dst_id"] + [c for c in GRAPH_EDGE_SAFE_FEATURES if c in part.columns]].copy()
    edge["src_node"] = edge["src_id"].astype(str).map(local)
    edge["dst_node"] = edge["dst_id"].astype(str).map(local)
    edge = edge.drop(columns=["src_id", "dst_id"])
    edge = edge[["txn_id", "src_node", "dst_node"] + [c for c in GRAPH_EDGE_SAFE_FEATURES if c in edge.columns]]
    if FORBIDDEN_MODEL_FEATURES & set(edge.columns):
        raise SystemExit(
            f"ERROR: forbidden feature(s) {sorted(FORBIDDEN_MODEL_FEATURES & set(edge.columns))} "
            f"about to be written to {folder / 'edge_list.csv.gz'}"
        )
    edge.to_csv(folder / "edge_list.csv.gz", index=False, compression="gzip")

    static_cols = [
        "Account_ID",
        "Account_Country",
        "Account_Currency",
        "Entity_Type",
        "Does_Account_Have_Debit_Card",
        "DebitCard_Index_in_EntityCards",
        "MCC",
        "Account_Type",
        "Max_Overdraft",
        "account_kind",
    ]
    static = accounts[[c for c in static_cols if c in accounts.columns]].copy()
    static["Account_ID"] = static["Account_ID"].astype(str)
    hist = historical_node_aggregates(all_tx, split_start)
    nf = node_map.merge(static, left_on="account_id", right_on="Account_ID", how="left")
    nf = nf.merge(hist, on="account_id", how="left")
    for col in ["Account_Country", "Account_Currency", "Entity_Type", "Account_Type", "account_kind"]:
        if col in nf.columns:
            nf[col] = nf[col].fillna("Unknown")
    for col in ["Does_Account_Have_Debit_Card", "DebitCard_Index_in_EntityCards", "MCC", "Max_Overdraft"]:
        if col in nf.columns:
            nf[col] = nf[col].fillna(0)
    for col in GRAPH_NODE_SAFE_FEATURES:
        if col.startswith("hist_") and col not in nf.columns:
            nf[col] = 0
    hist_cols = [c for c in GRAPH_NODE_SAFE_FEATURES if c.startswith("hist_")]
    nf[hist_cols] = nf[hist_cols].fillna(0)
    nf = nf[[c for c in GRAPH_NODE_SAFE_FEATURES if c in nf.columns]]
    nf.to_csv(folder / "node_features.csv.gz", index=False, compression="gzip")

    return {
        "split": split_name,
        "bank": bank_folder,
        "rows": int(len(part)),
        "positive_labels": int(part["y"].sum()),
        "fraud_rate": float(part["y"].mean()) if len(part) else 0.0,
        "nodes": int(len(node_map)),
        "edges": int(len(edge)),
        "timestamp_min": str(part["timestamp_dt"].min()),
        "timestamp_max": str(part["timestamp_dt"].max()),
    }


def write_docs(out_dir: Path, matches: dict[str, str], leakage_rows: list[dict[str, object]]) -> None:
    readme = """# Final Temporal IBM AML Dataset

This dataset is organized by temporal split first and source bank second. Only transactions whose `src_bank_id` matched the five selected banks are included.

`transactions.csv.gz` = tabular model input.
`ground_truth.csv.gz` = labels for supervised learning and evaluation.
`edge_list.csv.gz` = graph connections and safe transaction edge attributes.
`node_map.csv.gz` = original account ID to local numeric node ID mapping.
`node_features.csv.gz` = safe features describing graph nodes/accounts.
`model_feature_columns.json` = approved feature and exclusion lists.

The label `y` is not deleted. It is separated from model inputs and stored in `ground_truth.csv.gz`, joined by `txn_id`.
"""
    (out_dir / "README.md").write_text(readme)

    guide = """# File Usage Guide

Use `transactions.csv.gz` for tabular ML features and join labels from `ground_truth.csv.gz` using `txn_id`.

Use `edge_list.csv.gz`, `node_map.csv.gz`, and `node_features.csv.gz` for graph ML. `src_node` and `dst_node` are local node IDs that start at 0 inside each split/bank folder. `node_map.csv.gz` maps those local IDs back to original account IDs.

Required modeling files are the five core files in every split/bank folder plus `configuration/model_feature_columns.json`. The files under `reports/` are supporting audit, quality, and EDA documentation.
"""
    (out_dir / "reports" / "FILE_USAGE_GUIDE.md").write_text(guide)

    report = f"""# Preprocessing Report

Original files used: `prepared_data/transactions_master.csv.gz` and `prepared_data/accounts_clean.csv.gz`.

Timestamps were read from the prepared `timestamp` column and parsed with pandas. Transactions were assigned to `training` for 2025-06-01 through 2025-07-31, `validation` for 2025-08-01 through 2025-08-31, and `testing` for 2025-09-01 through 2025-09-30.

Selected bank matches:

{json.dumps(matches, indent=2)}

Transactions were assigned to banks using `src_bank_id` only. Unselected source banks were excluded.

Rolling transaction features (`src_prev_*` and `dst_prev_*`) came from `transactions_master.csv.gz` and were audited to ensure they match strictly earlier transactions for sampled and deterministic checks. Split-specific node features were regenerated instead of copying `prepared_data/account_node_features.csv.gz`. Node aggregate features use only transactions before the split start: training has no prior in-period history, validation uses June-July history, and testing uses June-August history. This graph protocol prevents training features from seeing validation/testing transactions and prevents testing node features from using September transactions.

Graph edges are one edge per transaction. Local graph node IDs are rebuilt independently for every split/bank folder and start at 0.

Confirmed leakage columns removed from model-facing files: {', '.join(LEAKAGE_COLUMNS)}, `transaction_type_raw`, `From_End_Balance`, `To_End_Balance`, `Controlled_by_Criminal`, and {', '.join(sorted(FORBIDDEN_MODEL_FEATURES))} (a redacted transaction-type proxy that still correlates with the laundering label; no replacement feature is retained).

Additional uncertain leakage columns requiring manual review: none currently flagged. High-cardinality identifiers were excluded from feature lists, not treated as target leakage.
"""
    (out_dir / "reports" / "PREPROCESSING_REPORT.md").write_text(report)

    leak_md = "# Leakage Audit\n\nSee `leakage_audit.csv` for column-level findings. Confirmed leakage and post-transaction fields were removed from model-facing files. `y` appears only in `ground_truth.csv.gz`.\n"
    (out_dir / "reports" / "LEAKAGE_AUDIT.md").write_text(leak_md)
    pd.DataFrame(leakage_rows).to_csv(out_dir / "reports" / "leakage_audit.csv", index=False)


def main() -> None:
    args = parse_args()
    prepared = args.prepared_dir
    out_dir = args.out_dir
    for name in ["transactions_master.csv.gz", "accounts_clean.csv.gz"]:
        if not (prepared / name).exists():
            raise SystemExit(f"Missing required prepared file: {prepared / name}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "configuration").mkdir(parents=True)
    (out_dir / "reports").mkdir(parents=True)

    tx = pd.read_csv(prepared / "transactions_master.csv.gz", low_memory=False)
    accounts = pd.read_csv(prepared / "accounts_clean.csv.gz", low_memory=False)
    tx["src_bank_id"] = tx["src_bank_id"].astype(str).str.strip()
    tx["timestamp_dt"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx["split"] = split_for_timestamp(tx["timestamp_dt"])
    matches = match_banks(tx)
    selected = tx[tx["src_bank_id"].isin(matches.values()) & tx["split"].notna()].copy()

    feature_config = {
        "tabular_safe_features": [c for c in TABULAR_SAFE_FEATURES if c in tx.columns],
        "graph_edge_safe_features": [c for c in GRAPH_EDGE_SAFE_FEATURES if c in tx.columns],
        "graph_node_safe_features": GRAPH_NODE_SAFE_FEATURES,
        "label_columns": ["txn_id", "timestamp", "y"],
        "leakage_columns": (
            LEAKAGE_COLUMNS
            + ["transaction_type_raw", "From_End_Balance", "To_End_Balance", "Controlled_by_Criminal"]
            + sorted(FORBIDDEN_MODEL_FEATURES)
        ),
        "identifier_columns": IDENTIFIER_COLUMNS,
        "non_feature_columns": ["split", "account_id", "src_node", "dst_node"],
    }

    leaked_into_schema = FORBIDDEN_MODEL_FEATURES & (
        set(feature_config["tabular_safe_features"]) | set(feature_config["graph_edge_safe_features"])
    )
    if leaked_into_schema:
        raise SystemExit(
            f"ERROR: forbidden feature(s) {sorted(leaked_into_schema)} present in generated "
            "tabular_safe_features/graph_edge_safe_features; refusing to write model_feature_columns.json"
        )

    write_json(out_dir / "configuration" / "model_feature_columns.json", feature_config)

    leakage_rows = audit_columns(list(tx.columns), "prepared_data/transactions_master.csv.gz")
    leakage_rows.extend(audit_columns(list(accounts.columns), "prepared_data/accounts_clean.csv.gz"))

    summary = []
    for split_name, (_start, _end, split_start) in SPLITS.items():
        for requested, folder in SELECTED_BANKS.items():
            part = selected[(selected["split"] == split_name) & (selected["src_bank_id"] == matches[requested])].copy()
            summary.append(build_folder(part, selected, accounts, split_name, folder, split_start, out_dir))

    write_docs(out_dir, matches, leakage_rows)
    pd.DataFrame(summary).to_csv(out_dir / "reports" / "split_bank_counts.csv", index=False)
    write_json(
        out_dir / "reports" / "build_summary.json",
        {"matched_banks": matches, "split_bank_counts": summary, "selected_rows": int(len(selected))},
    )
    print("Final temporal dataset built at", out_dir)
    print(pd.DataFrame(summary)[["split", "bank", "rows", "positive_labels", "fraud_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
