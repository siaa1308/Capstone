#!/usr/bin/env python3
"""Freeze validation-selected checkpoints, then explicitly score September once.

September has already been inspected in this project's history. Results from this
tool are exploratory diagnostics, not an untouched confirmatory evaluation.
The freeze action reads only saved validation metrics and checkpoint bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(study_root: Path, candidate: str, output: Path, mode: str = "federated"):
    if output.exists():
        raise ValueError("Freeze record already exists; it must not be overwritten")
    selection = json.loads((study_root / "selection.json").read_text())
    if candidate != selection["winner"]:
        raise ValueError("Candidate does not match the frozen seed-42 selection")
    entries = []
    for variant in ("baseline", candidate):
        for seed in (42, 52, 62):
            directory = study_root / mode / variant / f"seed_{seed}"
            metrics_path = directory / "metrics.json"
            checkpoint = directory / ("global_model.pt" if mode == "federated" else "JPMorgan_Chase_continual.pt")
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("test_loaded") is not False or any("testing" in row for row in metrics["per_bank"]):
                raise ValueError("Selection evidence must be validation-only")
            entries.append({"variant": variant, "seed": seed, "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": digest(checkpoint), "metrics": str(metrics_path.resolve()),
                            "metrics_sha256": digest(metrics_path),
                            "macro_validation_pr_auc": metrics["best_macro_validation_pr_auc"],
                            "best_round": metrics["best_round"]})
            if mode == "local":
                entries[-1]["bank_checkpoints"] = {
                    bank: {"path": str((directory / f"{bank}_continual.pt").resolve()),
                           "sha256": digest(directory / f"{bank}_continual.pt")}
                    for bank in ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")}
    record = {"status": "frozen_before_exploratory_september", "candidate": candidate, "mode": mode,
              "study_root": str(study_root.resolve()), "selection_sha256": digest(study_root / "selection.json"),
              "seeds": [42, 52, 62], "entries": entries,
              "evaluation_status": "Exploratory: September was already inspected during earlier project development. No post-September tuning permitted in this study."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("FROZEN", output)


def evaluate(freeze_path: Path, output: Path):
    if output.exists():
        raise ValueError("Evaluation directory already exists; never silently repeat or overwrite evaluation")
    record = json.loads(freeze_path.read_text())
    # Verify everything before reading the test split or creating an output.
    for entry in record["entries"]:
        if digest(Path(entry["checkpoint"])) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        if digest(Path(entry["metrics"])) != entry["metrics_sha256"]:
            raise ValueError("Validation metrics changed after freeze")
        for item in entry.get("bank_checkpoints", {}).values():
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Bank checkpoint changed after freeze")
    selection = Path(record["study_root"]) / "selection.json"
    if digest(selection) != record["selection_sha256"]:
        raise ValueError("Selection record changed after freeze")
    sys.path.insert(0, str(ROOT / "src" / "gnn"))
    import torch
    from causal_temporal_graphsage import BANKS, DEFAULT_DATASET, CausalTemporalGraphSAGE, build_bank_data, fit_shared_feature_encoders
    from evaluate_frozen_temporal import evaluate_model
    from adaptive_continual_federated import development_fingerprint
    current_data = development_fingerprint(DEFAULT_DATASET)
    for entry in record["entries"]:
        if json.loads(Path(entry["metrics"]).read_text())["data_sha256"] != current_data:
            raise ValueError("Development data changed after training")
    output.mkdir(parents=True)
    (output / "STARTED.json").write_text(json.dumps({"freeze_sha256": digest(freeze_path), "status": record["evaluation_status"],
                                                   "development_data_fingerprint_verified": True}, indent=2) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoders = fit_shared_feature_encoders(DEFAULT_DATASET, BANKS)
    banks = {bank: build_bank_data(DEFAULT_DATASET, bank, *encoders)[:2] for bank in BANKS}
    reports = []
    for entry in record["entries"]:
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        saved = checkpoint["args"]
        per_bank = []
        for bank, (static, streams) in banks.items():
            if "bank_checkpoints" in entry:
                checkpoint = torch.load(entry["bank_checkpoints"][bank]["path"], map_location="cpu", weights_only=True)
                saved = checkpoint["args"]
            model = CausalTemporalGraphSAGE(static.shape[1], streams["training"].edge_attr.shape[1],
                                          int(saved["hidden_channels"]), float(saved.get("dropout", .25)))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            per_bank.append({"bank": bank, **evaluate_model(model, static, streams, int(saved["batch_size"]), [10, 25, 50])})
        report = {**entry, "per_bank": per_bank,
                  "macro_testing_pr_auc": statistics.mean(row["testing"]["pr_auc"] for row in per_bank),
                  "recomputed_macro_validation_pr_auc": statistics.mean(row["validation"]["pr_auc"] for row in per_bank),
                  "evaluation_status": record["evaluation_status"]}
        path = output / f"{entry['variant']}_seed_{entry['seed']}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(entry["variant"], entry["seed"], report["macro_testing_pr_auc"], flush=True)
    summary = {"evaluation_status": record["evaluation_status"], "mode": record["mode"], "freeze_sha256": digest(freeze_path), "methods": {}}
    for variant in ("baseline", record["candidate"]):
        rows = [row for row in reports if row["variant"] == variant]
        values = [row["macro_testing_pr_auc"] for row in rows]
        summary["methods"][variant] = {"mean_pr_auc": statistics.mean(values), "sample_sd": statistics.stdev(values),
                                       "seeds": {str(row["seed"]): row["macro_testing_pr_auc"] for row in rows},
                                       "per_bank_mean_pr_auc": {bank: statistics.mean(
                                           next(r["testing"]["pr_auc"] for r in row["per_bank"] if r["bank"] == bank)
                                           for row in rows) for bank in BANKS}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    first = sub.add_parser("freeze")
    first.add_argument("--study-root", type=Path, required=True)
    first.add_argument("--candidate", required=True)
    first.add_argument("--output", type=Path, required=True)
    first.add_argument("--mode", choices=("federated", "local"), default="federated")
    second = sub.add_parser("evaluate")
    second.add_argument("--freeze", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    second.add_argument("--acknowledge-exploratory-september", action="store_true", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.study_root, args.candidate, args.output, args.mode)
    else:
        evaluate(args.freeze, args.output)


if __name__ == "__main__":
    main()
