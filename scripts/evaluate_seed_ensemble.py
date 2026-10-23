#!/usr/bin/env python3
"""Freeze a fixed equal-probability ensemble on validation, then score once.

Ensembling is established practice, not a claimed research novelty. September
results remain exploratory because that period was already inspected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gnn"))
from adaptive_continual_federated import development_fingerprint, fast_choose_threshold, json_write
from causal_temporal_graphsage import (BANKS, DEFAULT_DATASET, CausalTemporalGraphSAGE,
    build_bank_data, fit_shared_feature_encoders, score_stream, metric_block, alert_budget_metrics)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_probabilities(predictions):
    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Expected aligned finite member probabilities")
    return array.mean(axis=0)


def score(entries, include_test=False, thresholds=None):
    encoders = fit_shared_feature_encoders(DEFAULT_DATASET, BANKS)
    splits = ("training", "validation", "testing") if include_test else ("training", "validation")
    rows = []
    checkpoints = [torch.load(entry["path"], weights_only=True, map_location="cpu") for entry in entries]
    for bank in BANKS:
        static, streams, *_ = build_bank_data(DEFAULT_DATASET, bank, *encoders, splits=splits)
        predictions = {split: [] for split in splits if split != "training"}
        for checkpoint in checkpoints:
            cfg = checkpoint["args"]
            model = CausalTemporalGraphSAGE(static.shape[1], streams["training"].edge_attr.shape[1],
                                             cfg["hidden_channels"], 0.25)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            state = model.initial_state(static)
            for split in splits:
                probabilities, _, state = score_stream(model, state, streams[split], cfg["batch_size"])
                if split != "training":
                    predictions[split].append(probabilities)
        validation_scores = mean_probabilities(predictions["validation"])
        threshold = (thresholds[bank] if thresholds is not None else
                     fast_choose_threshold(streams["validation"].labels.numpy(), validation_scores))
        row = {"bank": bank, "threshold": threshold}
        for split, members in predictions.items():
            probabilities = mean_probabilities(members)
            labels = streams[split].labels.numpy()
            row[split] = metric_block(labels, probabilities, threshold)
            row[split + "_alert_metrics"] = alert_budget_metrics(labels, probabilities, [10, 25, 50])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research_seed_ensemble")
    parser.add_argument("--acknowledge-exploratory-september", action="store_true")
    cfg = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = cfg.output_dir / "FREEZE.json"
    if cfg.action == "freeze":
        if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
            raise ValueError("Freeze requires a new or empty output directory")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for seed in (42, 52, 62):
            path = ROOT / f"artifacts/research_adaptive_fcl/warm_start/federated/context_harmonized/seed_{seed}/global_model.pt"
            entries.append({"seed": seed, "path": str(path), "sha256": digest(path)})
        # Fixed before any validation/test evaluation; no weight fitting or member selection.
        design = {"method": "equal arithmetic mean of probabilities", "entries": entries,
                  "test_loaded": False, "source_sha256": digest(__file__),
                  "data_sha256": development_fingerprint(DEFAULT_DATASET),
                  "novelty": "None claimed; variance-reduction baseline",
                  "test_status": "Exploratory historically inspected September; no tuning after evaluation"}
        json_write(cfg.output_dir / "DESIGN.json", design)
        rows = score(entries)
        macro = float(np.mean([r["validation"]["pr_auc"] for r in rows]))
        record = {**design, "per_bank": rows, "macro_validation_pr_auc": macro,
                  "thresholds": {r["bank"]: r["threshold"] for r in rows}}
        json_write(frozen, record)
        print(json.dumps({"macro_validation_pr_auc": macro, "per_bank": rows}, indent=2))
    else:
        if not cfg.acknowledge_exploratory_september:
            parser.error("Explicit exploratory acknowledgement required")
        record = json.loads(frozen.read_text())
        if record["source_sha256"] != digest(__file__):
            raise ValueError("Evaluator changed after freeze")
        for entry in record["entries"]:
            if digest(entry["path"]) != entry["sha256"]:
                raise ValueError("Checkpoint changed after freeze")
        if development_fingerprint(DEFAULT_DATASET) != record["data_sha256"]:
            raise ValueError("Development data changed after freeze")
        output = cfg.output_dir / "exploratory_september"
        output.mkdir()  # Existing/partial evaluation is never repeated silently.
        json_write(output / "STARTED.json", {"freeze_sha256": digest(frozen)})
        rows = score(record["entries"], True, record["thresholds"])
        result = {"status": record["test_status"], "freeze_sha256": digest(frozen), "per_bank": rows,
                  "macro_validation_pr_auc": float(np.mean([r["validation"]["pr_auc"] for r in rows])),
                  "macro_testing_pr_auc": float(np.mean([r["testing"]["pr_auc"] for r in rows])),
                  "member_count": 3, "inference_cost": "three model passes; not a single-seed result"}
        json_write(output / "metrics.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
