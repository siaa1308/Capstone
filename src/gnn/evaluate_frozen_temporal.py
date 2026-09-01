#!/usr/bin/env python3
"""Evaluate frozen validation-selected temporal GNN checkpoints exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_temporal_graphsage import (
    BANKS, DEFAULT_DATASET, CausalTemporalGraphSAGE, alert_budget_metrics,
    build_bank_data, choose_threshold, fit_shared_feature_encoders, metric_block,
    score_stream, set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("local", "continual", "fedavg", "fedprox"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alert-k", default="10,25,50")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def checkpoint_path(method: str, directory: Path, bank: str) -> Path:
    if method == "local":
        return directory / f"{bank}_causal_temporal_graphsage.pt"
    if method == "continual":
        return directory / f"{bank}_continual.pt"
    return directory / "global_model.pt"


def evaluate_model(model, static, streams, batch_size: int, budgets: list[int]) -> dict[str, object]:
    state = model.initial_state(static)
    _, _, state = score_stream(model, state, streams["training"], batch_size)
    validation_probs, validation_labels, state = score_stream(model, state, streams["validation"], batch_size)
    threshold = choose_threshold(validation_labels, validation_probs)
    testing_probs, testing_labels, _ = score_stream(model, state, streams["testing"], batch_size)
    return {
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "validation": metric_block(validation_labels, validation_probs, threshold),
        "testing": metric_block(testing_labels, testing_probs, threshold),
        "validation_alert_metrics": alert_budget_metrics(validation_labels, validation_probs, budgets),
        "testing_alert_metrics": alert_budget_metrics(testing_labels, testing_probs, budgets),
    }


def main() -> None:
    cfg = parse_args()
    cfg.dataset_dir = cfg.dataset_dir.resolve()
    budgets = [int(value) for value in cfg.alert_k.split(",") if value.strip()]
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("--alert-k must contain positive integers")
    device = torch.device(cfg.device)
    node_encoder, edge_encoder = fit_shared_feature_encoders(cfg.dataset_dir, BANKS)
    shared_checkpoint = None
    if cfg.method in {"fedavg", "fedprox"}:
        shared_checkpoint = torch.load(checkpoint_path(cfg.method, cfg.checkpoint_dir, BANKS[0]), map_location="cpu", weights_only=False)

    results = []
    for bank in BANKS:
        checkpoint = shared_checkpoint or torch.load(
            checkpoint_path(cfg.method, cfg.checkpoint_dir, bank), map_location="cpu", weights_only=False,
        )
        saved_args = checkpoint["args"]
        seed = int(saved_args.get("seed", 42))
        set_seed(seed)
        static, streams, node_columns, edge_columns = build_bank_data(
            cfg.dataset_dir, bank, node_encoder, edge_encoder,
        )
        static = static.to(device)
        streams = {name: events.to(device) for name, events in streams.items()}
        model = CausalTemporalGraphSAGE(
            static.shape[1], streams["training"].edge_attr.shape[1],
            int(saved_args["hidden_channels"]), float(saved_args.get("dropout", 0.25)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        batch_size = int(saved_args.get("event_batch_size", saved_args.get("batch_size", 512)))
        row = {"bank": bank, **evaluate_model(model, static, streams, batch_size, budgets)}
        row["node_feature_columns"] = node_columns
        row["edge_feature_columns"] = edge_columns
        results.append(row)
        print(f"{bank}: test PR-AUC={row['testing']['pr_auc']:.5f}")

    testing_macro = [float(row["testing"]["pr_auc"]) for row in results]
    validation_macro = [float(row["validation"]["pr_auc"]) for row in results]
    report = {
        "method": cfg.method,
        "checkpoint_dir": str(cfg.checkpoint_dir.resolve()),
        "evaluation_status": "frozen confirmatory diagnostic; September was inspected in legacy experiments",
        "macro_validation_pr_auc": float(np.mean(validation_macro)),
        "macro_testing_pr_auc": float(np.mean(testing_macro)),
        "per_bank": results,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
