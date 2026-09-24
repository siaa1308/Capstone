#!/usr/bin/env python3
"""Summarize saved development metrics; never load data or evaluate a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def summarize(root: Path):
    results = {}
    for mode in ("federated", "local"):
        for directory in sorted((root / mode).glob("*")):
            rows = [json.loads(p.read_text()) for p in sorted(directory.glob("seed_*/metrics.json"))]
            if not rows:
                continue
            values = [row["best_macro_validation_pr_auc"] for row in rows]
            result = {"macro_validation_pr_auc": describe(values),
                      "seeds": {str(row["configuration"]["seed"]): row["best_macro_validation_pr_auc"] for row in rows},
                      "best_rounds": {str(row["configuration"]["seed"]): row["best_round"] for row in rows},
                      "per_bank": {}}
            for bank in ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank"):
                bank_rows = [next(item for item in row["per_bank"] if item["bank"] == bank) for row in rows]
                result["per_bank"][bank] = {
                    "validation_pr_auc": describe([r["validation"]["pr_auc"] for r in bank_rows]),
                    "recall_at_25": describe([r["validation_alert_metrics"]["25"]["recall_at_k"] for r in bank_rows]),
                    "precision_at_25": describe([r["validation_alert_metrics"]["25"]["precision_at_k"] for r in bank_rows])}
            if mode == "local":
                result["macro_june_forgetting"] = describe([
                    statistics.mean(item["retention"]["forgetting"] for item in row["per_bank"]) for row in rows])
            results[f"{mode}/{directory.name}"] = result
    for name, result in results.items():
        baseline = results.get(name.split("/")[0] + "/baseline")
        if baseline and not name.endswith("/baseline"):
            shared = sorted(set(result["seeds"]) & set(baseline["seeds"]))
            result["paired_delta_vs_baseline"] = {s: result["seeds"][s] - baseline["seeds"][s] for s in shared}
            result["paired_delta_summary"] = describe(list(result["paired_delta_vs_baseline"].values()))
    return results


def freeze_selection(root: Path, study: dict, study_path: Path):
    path = root / "selection.json"
    if path.exists():
        raise ValueError("Selection is already frozen; do not overwrite it")
    records = {}
    for variant in study["variants"]:
        metrics_path = root / "federated" / variant / f"seed_{study['screen_seed']}" / "metrics.json"
        record = json.loads(metrics_path.read_text())
        records[variant] = {"validation": record["best_macro_validation_pr_auc"],
                            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()}
    winner = max((name for name in records if name != "baseline"), key=lambda name: records[name]["validation"])
    value = {"criterion": f"Highest seed-42 macro August validation PR-AUC among all {len(records)-1} screened non-baseline variants; see study protocol for adaptive follow-ups",
             "winner": winner, "screen": records, "test_loaded": False,
             "confirmation_seeds": study["confirmation_seeds"],
             "configuration": {**study["common"], **study["variants"][winner]},
             "study_config_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest()}
    path.write_text(json.dumps(value, indent=2) + "\n")
    print("FROZEN WINNER", winner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/research_adaptive_fcl")
    parser.add_argument("--freeze-selection", action="store_true")
    parser.add_argument("--study-config", type=Path, default=ROOT / "configs/adaptive_fcl_study.json")
    args = parser.parse_args()
    study = json.loads(args.study_config.read_text())
    if args.freeze_selection:
        freeze_selection(args.root, study, args.study_config)
    results = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Adaptive FCL validation results", "", "Development evidence only. No September scores are included.", "",
             "| Method | Seeds | Macro August PR-AUC | Paired gain over matched baseline |", "|---|---|---:|---:|"]
    for name, result in results.items():
        stat = result["macro_validation_pr_auc"]
        score = f"{stat['mean']:.4f}" + (f" ± {stat['sample_sd']:.4f}" if stat["sample_sd"] is not None else " (one seed)")
        delta = result.get("paired_delta_summary", {}).get("mean")
        delta_text = "—" if delta is None else f"{delta:+.4f}"
        lines.append(f"| {name} | {', '.join(result['seeds'])} | {score} | {delta_text} |")
    lines += ["", "Each seed first averages PR-AUC equally over banks. ± is sample SD across seeds.",
              "Single-seed screen rows are not comparable in certainty to three-seed confirmation.",
              "The historical final FedAvg validation reference is 0.7083 ± 0.0937; its September reference is 0.7590 ± 0.0360.",
              "Do not compare a new August score to the historical September score as an improvement estimate.", "",
              "The matched new baseline uses the same corrected calendar boundary, runtime, RNG policy, and compute budget as the candidates."]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
