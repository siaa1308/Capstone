#!/usr/bin/env python3
"""Quantify non-IID evidence for the active AML federated cohort."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "final_temporal_dataset")
    parser.add_argument(
        "--fedavg-metrics", type=Path,
        default=ROOT / "artifacts" / "optimized_fedavg" / "development" / "stability" / "seed_42" / "metrics.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "optimized_fedavg" / "diagnostics",
    )
    return parser.parse_args()


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    terms = []
    for values in (left, right):
        mask = values > 0
        terms.append(float(np.sum(values[mask] * np.log2(values[mask] / middle[mask]))))
    return 0.5 * sum(terms)


def categorical_js(frames: dict[str, pd.DataFrame], column: str) -> dict[str, float]:
    categories = sorted(set().union(*(set(frame[column].fillna("<NA>").astype(str)) for frame in frames.values())))
    distributions = {
        bank: frame[column].fillna("<NA>").astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(float)
        for bank, frame in frames.items()
    }
    return {
        f"{left}__{right}": js_divergence(distributions[left], distributions[right])
        for left, right in combinations(BANKS, 2)
    }


def main() -> None:
    args = parse_args()
    edge_frames: dict[str, pd.DataFrame] = {}
    bank_rows = []
    for bank in BANKS:
        directory = args.dataset_dir / "training" / bank
        edges = pd.read_csv(directory / "edge_list.csv.gz")
        truth = pd.read_csv(directory / "ground_truth.csv.gz")
        nodes = pd.read_csv(directory / "node_map.csv.gz")
        joined = edges.merge(truth[["txn_id", "y"]], on="txn_id", validate="one_to_one")
        edge_frames[bank] = joined
        bank_rows.append({
            "bank": bank,
            "transactions": int(len(joined)),
            "positives": int(joined["y"].sum()),
            "fraud_rate": float(joined["y"].mean()),
            "nodes": int(len(nodes)),
            "edges_per_node": float(len(joined) / len(nodes)),
            "amount_mean": float(joined["amount"].mean()),
            "amount_std": float(joined["amount"].std()),
        })

    bank_table = pd.DataFrame(bank_rows).set_index("bank")
    label_distributions = {
        bank: np.array([len(frame) - frame["y"].sum(), frame["y"].sum()], dtype=float)
        for bank, frame in edge_frames.items()
    }
    label_js = {
        f"{left}__{right}": js_divergence(label_distributions[left], label_distributions[right])
        for left, right in combinations(BANKS, 2)
    }
    amount_smd = {}
    for left, right in combinations(BANKS, 2):
        a, b = bank_table.loc[left], bank_table.loc[right]
        pooled = np.sqrt((a["amount_std"] ** 2 + b["amount_std"] ** 2) / 2)
        amount_smd[f"{left}__{right}"] = float(abs(a["amount_mean"] - b["amount_mean"]) / pooled) if pooled else 0.0

    update_norms: dict[str, dict[str, float]] = {}
    if args.fedavg_metrics.exists():
        metrics = json.loads(args.fedavg_metrics.read_text())
        for bank in BANKS:
            values = [float(row["client_update_l2"][bank]) for row in metrics["rounds"]]
            update_norms[bank] = {
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "max": float(np.max(values)),
            }

    report = {
        "active_banks": list(BANKS),
        "training_bank_statistics": bank_rows,
        "volume_max_to_min_ratio": float(bank_table["transactions"].max() / bank_table["transactions"].min()),
        "fraud_rate_max_to_min_ratio": float(bank_table["fraud_rate"].max() / bank_table["fraud_rate"].min()),
        "pairwise_label_js_divergence_bits": label_js,
        "pairwise_currency_js_divergence_bits": categorical_js(edge_frames, "currency"),
        "pairwise_payment_format_js_divergence_bits": categorical_js(edge_frames, "payment_format"),
        "pairwise_amount_standardized_mean_difference": amount_smd,
        "fedavg_client_update_l2": update_norms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "non_iid_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    bank_table.to_csv(args.output_dir / "training_bank_statistics.csv")

    lines = [
        "# Active-cohort non-IID audit", "",
        "All statistics below use training data only. They diagnose heterogeneity; they do not use validation or test labels.", "",
        bank_table.to_markdown(), "",
        f"Transaction-volume max/min ratio: {report['volume_max_to_min_ratio']:.3f}.",
        f"Fraud-rate max/min ratio: {report['fraud_rate_max_to_min_ratio']:.3f}.", "",
        "Pairwise Jensen-Shannon divergences and client-update norms are recorded in `non_iid_audit.json`.",
    ]
    (args.output_dir / "NON_IID_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
