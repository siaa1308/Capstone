#!/usr/bin/env python3
"""Generate EDA tables and plots for final temporal IBM AML dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BANKS = ["JPMorgan_Chase", "Wells_Fargo", "Citi", "Fifth_Third_Bancorp", "Key_Bank"]
SPLITS = ["training", "validation", "testing"]
SEED = 42
SAMPLE_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def write_svg(path: Path, body: str, width: int = 900, height: int = 420) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white"/>{body}</svg>\n'
    )


def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    width, height, left, bottom = 900, 420, 80, 70
    plot_w, plot_h = width - left - 30, height - bottom - 50
    vals = df[y].astype(float).tolist()
    labels = df[x].astype(str).tolist()
    max_v = max(vals) if vals else 1
    bar_w = plot_w / max(len(vals), 1)
    parts = [f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>']
    for i, (label, val) in enumerate(zip(labels, vals)):
        h = 0 if max_v == 0 else (val / max_v) * plot_h
        bx = left + i * bar_w + 8
        by = height - bottom - h
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bar_w-16, 2):.1f}" height="{h:.1f}" fill="#386cb0"/>')
        parts.append(f'<text x="{bx + bar_w/2 - 8:.1f}" y="{height-42}" text-anchor="end" transform="rotate(-30 {bx + bar_w/2 - 8:.1f},{height-42})" font-family="Arial" font-size="11">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="50" x2="{left}" y2="{height-bottom}" stroke="#222"/>')
    write_svg(path, "".join(parts), width, height)


def save_hist(series: pd.Series, path: Path, title: str, bins: int = 50) -> None:
    values = series.dropna().astype(float)
    if values.empty:
        write_svg(path, f'<text x="20" y="40">{title}: no data</text>')
        return
    counts = pd.cut(values, bins=bins).value_counts().sort_index().reset_index(drop=True)
    df = pd.DataFrame({"bin": list(range(len(counts))), "count": counts})
    save_bar(df, "bin", "count", path, title)


def save_line(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    width, height, left, bottom = 1000, 420, 70, 45
    plot_w, plot_h = width - left - 30, height - bottom - 50
    vals = df[y].astype(float).tolist()
    max_v = max(vals) if vals else 1
    points = []
    for i, val in enumerate(vals):
        px = left + (i / max(len(vals) - 1, 1)) * plot_w
        py = height - bottom - (0 if max_v == 0 else (val / max_v) * plot_h)
        points.append(f"{px:.1f},{py:.1f}")
    body = f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>'
    body += f'<polyline points="{" ".join(points)}" fill="none" stroke="#386cb0" stroke-width="2"/>'
    body += f'<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="#222"/>'
    body += f'<line x1="{left}" y1="50" x2="{left}" y2="{height-bottom}" stroke="#222"/>'
    write_svg(path, body, width, height)


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    report_dir = root / "reports"
    table_dir = report_dir / "eda_tables"
    plot_dir = report_dir / "eda_plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tx_parts = []
    gt_parts = []
    graph_rows = []
    node_activity = []
    for split in SPLITS:
        for bank in BANKS:
            folder = root / split / bank
            tx = pd.read_csv(folder / "transactions.csv.gz", low_memory=False)
            gt = pd.read_csv(folder / "ground_truth.csv.gz", low_memory=False)
            edge = pd.read_csv(folder / "edge_list.csv.gz", low_memory=False)
            node_map = pd.read_csv(folder / "node_map.csv.gz", low_memory=False)
            tx_parts.append(tx.assign(split=split, bank=bank))
            gt_parts.append(gt.assign(split=split, bank=bank))
            graph_rows.append({"split": split, "bank": bank, "nodes": len(node_map), "edges": len(edge)})
            sent = edge.groupby("src_node").size().rename("sent_count")
            recv = edge.groupby("dst_node").size().rename("recv_count")
            deg = pd.concat([sent, recv], axis=1).fillna(0).reset_index().rename(columns={"index": "node_id"})
            deg["degree"] = deg["sent_count"] + deg["recv_count"]
            deg["split"] = split
            deg["bank"] = bank
            node_activity.append(deg)

    tx_all = pd.concat(tx_parts, ignore_index=True)
    gt_all = pd.concat(gt_parts, ignore_index=True)
    data = tx_all.merge(gt_all[["txn_id", "y"]], on="txn_id", how="left")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["date"] = data["timestamp"].dt.date
    data["week"] = data["timestamp"].dt.to_period("W").astype(str)
    data["month"] = data["timestamp"].dt.to_period("M").astype(str)

    tables = {
        "transactions_by_bank.csv": data.groupby("bank").size().reset_index(name="transactions"),
        "positive_labels_by_bank.csv": data.groupby("bank")["y"].sum().reset_index(name="positive_labels"),
        "fraud_rate_by_bank.csv": data.groupby("bank")["y"].mean().reset_index(name="fraud_rate"),
        "label_distribution_by_split.csv": data.groupby(["split", "y"]).size().reset_index(name="transactions"),
        "amount_distribution.csv": data["amount"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).reset_index(),
        "amount_distribution_by_label.csv": data.groupby("y")["amount"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).reset_index(),
        "payment_format_distribution.csv": data["payment_format"].value_counts(dropna=False).reset_index(name="transactions").rename(columns={"payment_format": "value"}),
        "currency_distribution.csv": data["currency"].value_counts(dropna=False).reset_index(name="transactions").rename(columns={"currency": "value"}),
        # transaction_type_distribution.csv was removed: it was sourced from
        # transaction_type_model_safe, a redacted transaction-type proxy that was found to
        # still correlate with the laundering label and is no longer generated as a
        # model-facing column (see build_final_temporal_dataset.py FORBIDDEN_MODEL_FEATURES).
        "transactions_by_day.csv": data.groupby("date").size().reset_index(name="transactions"),
        "transactions_by_week.csv": data.groupby("week").size().reset_index(name="transactions"),
        "transactions_by_month.csv": data.groupby("month").size().reset_index(name="transactions"),
        "fraud_transactions_over_time.csv": data.groupby("date")["y"].sum().reset_index(name="positive_labels"),
        "source_account_activity.csv": data.groupby("src_id").size().describe(percentiles=[0.5, 0.9, 0.99]).reset_index(),
        "destination_account_activity.csv": data.groupby("dst_id").size().describe(percentiles=[0.5, 0.9, 0.99]).reset_index(),
        "graph_node_edge_counts.csv": pd.DataFrame(graph_rows),
        "split_bank_distribution.csv": data.groupby(["split", "bank"]).agg(transactions=("txn_id", "size"), positives=("y", "sum"), fraud_rate=("y", "mean"), amount_mean=("amount", "mean"), amount_median=("amount", "median")).reset_index(),
    }
    activity = pd.concat(node_activity, ignore_index=True)
    tables["top_account_degree_distribution.csv"] = activity["degree"].describe(percentiles=[0.5, 0.9, 0.99]).reset_index()
    tables["sent_received_count_distribution.csv"] = activity[["sent_count", "recv_count"]].describe(percentiles=[0.5, 0.9, 0.99]).reset_index()
    for name, df in tables.items():
        df.to_csv(table_dir / name, index=False)

    sample_n = min(SAMPLE_SIZE, len(data))
    sample = data.groupby("y", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, int(sample_n * len(g) / len(data)))), random_state=SEED)
    )
    sample.to_csv(table_dir / "plot_sample_manifest.csv", index=False)

    save_bar(tables["transactions_by_bank.csv"], "bank", "transactions", plot_dir / "transactions_by_bank.svg", "Transactions by Bank")
    save_bar(tables["fraud_rate_by_bank.csv"], "bank", "fraud_rate", plot_dir / "fraud_rate_by_bank.svg", "Fraud Rate by Bank")
    save_hist(sample["amount"].clip(upper=sample["amount"].quantile(0.99)), plot_dir / "amount_distribution.svg", f"Amount Distribution, stratified sample n={len(sample)}, seed={SEED}")
    save_line(tables["transactions_by_day.csv"], "date", "transactions", plot_dir / "transactions_by_day.svg", "Transactions by Day")

    md = f"""# EDA Report

Tables were computed from the full final dataset. Expensive plots use a reproducible stratified sample with random seed `{SEED}` and sample size `{len(sample)}`.

## Class Imbalance

The positive class is rare across all selected banks and splits. See `eda_tables/label_distribution_by_split.csv` and `eda_tables/fraud_rate_by_bank.csv`.

## Temporal And Bank Drift

Use `eda_tables/split_bank_distribution.csv`, `transactions_by_day.csv`, `transactions_by_week.csv`, and `fraud_transactions_over_time.csv` to compare training, validation, and testing distributions. Bank-level differences are visible in transaction volume, fraud rate, graph size, and amount summaries, so graph and tabular evaluation should report split and bank-level metrics instead of only pooled metrics.

## Included Tables

- total transactions by bank
- total positive labels by bank
- fraud rate by bank
- label distribution by split
- amount and amount-by-label distributions
- payment-format, currency, and safe transaction-type distributions
- transactions by day, week, and month
- fraud transactions over time
- source and destination account activity
- account-degree, sent-count, and received-count distributions
- graph node and edge counts
"""
    (report_dir / "EDA_REPORT.md").write_text(md)
    print("Wrote EDA report to", report_dir / "EDA_REPORT.md")


if __name__ == "__main__":
    main()
