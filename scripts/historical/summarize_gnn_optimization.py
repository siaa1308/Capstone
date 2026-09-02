#!/usr/bin/env python3
"""Summarize validation selection and frozen-checkpoint temporal GNN results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")
SEEDS = (42, 52, 62)
METHODS = {
    "Local Temporal GNN": "optimized_local_temporal",
    "+ Continual Learning": "optimized_cl_replay",
    "+ FedAvg": "optimized_fedavg",
    "+ FedProx": "optimized_fedprox",
}


def mean_std(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def load_final(method_dir: str) -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        report = json.loads((ROOT / "artifacts" / method_dir / "final_test" / f"seed_{seed}" / "metrics.json").read_text())
        for row in report["per_bank"]:
            rows.append({"seed": seed, **row})
    return rows


def summarize_method(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = ("pr_auc", "roc_auc", "precision", "recall", "f1")
    macro_by_seed: dict[str, dict[str, float]] = {str(seed): {} for seed in SEEDS}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        for split in ("validation", "testing"):
            for metric in metrics:
                macro_by_seed[str(seed)][f"{split}_{metric}"] = float(np.mean([row[split][metric] for row in selected]))
            for budget in (10, 25, 50):
                for metric in ("precision_at_k", "recall_at_k"):
                    key = f"testing_{metric}_{budget}"
                    macro_by_seed[str(seed)][key] = float(np.mean([
                        row["testing_alert_metrics"][str(budget)][metric] for row in selected
                    ]))

    summary: dict[str, object] = {"macro_by_seed": macro_by_seed}
    for key in next(iter(macro_by_seed.values())):
        summary[key] = mean_std([macro_by_seed[str(seed)][key] for seed in SEEDS])
    summary["per_bank_testing_pr_auc"] = {
        bank: mean_std([row["testing"]["pr_auc"] for row in rows if row["bank"] == bank]) for bank in BANKS
    }
    return summary


def candidate_macro(path: Path) -> float:
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else data["per_bank"]
    return float(np.mean([row["validation"]["pr_auc"] for row in rows if row["bank"] in BANKS]))


def candidate_tables() -> dict[str, list[dict[str, object]]]:
    patterns = {
        "local": ROOT / "artifacts" / "optimized_local_temporal" / "development" / "reproducible_candidates",
        "continual": ROOT / "artifacts" / "optimized_cl_replay" / "development" / "candidates",
        "fedavg": ROOT / "artifacts" / "optimized_fedavg" / "development" / "candidates",
        "fedprox": ROOT / "artifacts" / "optimized_fedprox" / "development" / "candidates",
    }
    result = {}
    for method, directory in patterns.items():
        rows = []
        for path in sorted(directory.glob("*/metrics.json")):
            candidate = path.parent.name
            if method == "fedprox" and candidate == "mu_0":
                candidate += " (FedAvg equivalence control)"
            rows.append({"candidate": candidate, "macro_validation_pr_auc": candidate_macro(path)})
        result[method] = sorted(rows, key=lambda row: row["macro_validation_pr_auc"], reverse=True)
    return result


def fmt(stat: dict[str, object]) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def main() -> None:
    summaries = {name: summarize_method(load_final(directory)) for name, directory in METHODS.items()}
    candidates = candidate_tables()
    continual_rows = []
    for seed in SEEDS:
        seed_rows = json.loads((
            ROOT / "artifacts" / "optimized_cl_replay" / "development" / "stability" / f"seed_{seed}" / "metrics.json"
        ).read_text())
        continual_rows.extend({"run_seed": seed, **row} for row in seed_rows)
    retention_by_bank = {
        bank: mean_std([float(row["forgetting"]) for row in continual_rows if row["bank"] == bank]) for bank in BANKS
    }
    retention_macro = mean_std([
        float(np.mean([row["forgetting"] for row in continual_rows if row["run_seed"] == seed])) for seed in SEEDS
    ])
    report = {
        "seeds": list(SEEDS), "banks": list(BANKS), "methods": summaries,
        "validation_candidates": candidates,
        "continual_forgetting": {"macro": retention_macro, "per_bank": retention_by_bank},
    }
    output = ROOT / "artifacts" / "optimized_comparison_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for method, summary in summaries.items():
        rows.append({
            "method": method,
            "validation_pr_auc_mean": summary["validation_pr_auc"]["mean"],
            "validation_pr_auc_std": summary["validation_pr_auc"]["std"],
            "test_pr_auc_mean": summary["testing_pr_auc"]["mean"],
            "test_pr_auc_std": summary["testing_pr_auc"]["std"],
            **{f"{bank}_test_pr_auc": summary["per_bank_testing_pr_auc"][bank]["mean"] for bank in BANKS},
        })
    pd.DataFrame(rows).to_csv(ROOT / "artifacts" / "optimized_comparison_summary.csv", index=False)

    validation_lines = []
    for method, rows_for_method in candidates.items():
        for row in rows_for_method:
            validation_lines.append(f"| {method} | {row['candidate']} | {row['macro_validation_pr_auc']:.4f} |")

    final_lines = []
    for method, summary in summaries.items():
        bank_values = [summary["per_bank_testing_pr_auc"][bank]["mean"] for bank in BANKS]
        final_lines.append(
            f"| {method} | {fmt(summary['testing_pr_auc'])} | " + " | ".join(f"{value:.3f}" for value in bank_values) + " |"
        )

    metric_lines = []
    display = [
        ("PR-AUC", "testing_pr_auc"), ("Recall@10", "testing_recall_at_k_10"),
        ("Precision@10", "testing_precision_at_k_10"), ("Recall@25", "testing_recall_at_k_25"),
        ("Precision@25", "testing_precision_at_k_25"), ("Recall@50", "testing_recall_at_k_50"),
        ("Precision@50", "testing_precision_at_k_50"), ("Recall", "testing_recall"),
        ("Precision", "testing_precision"), ("F1", "testing_f1"), ("ROC-AUC", "testing_roc_auc"),
    ]
    for method, summary in summaries.items():
        metric_lines.append("| " + method + " | " + " | ".join(fmt(summary[key]) for _label, key in display) + " |")

    doc = f"""# Temporal GNN optimization report

## Scope and evaluation status

The active cohort is JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp data and historical artifacts were left intact and were not used in new modeling runs. All configuration decisions in this optimization were made from August validation PR-AUC. September was scored only after configurations and checkpoints were frozen.

September cannot honestly be called a genuinely untouched final holdout: legacy repository artifacts already contain repeated September results, and the three-bank cohort was chosen after historical performance inspection. The repository contains no post-September raw period. Consequently, the numbers below are **frozen confirmatory test diagnostics**, not unbiased untouched-holdout estimates. A later temporal period is required for a defensible final paper claim.

## A. Historical baseline

The historical active-cohort reference is approximately 0.383 macro test PR-AUC for the three-seed local causal temporal model, 0.295 for ten-round FedAvg, and 0.263 for federated continual learning with replay. These runs used earlier code/protocols and are retained for auditability, not treated as directly comparable optimized results.

## B. Problems discovered

- Temporal micro-batches with no positive label were skipped, coupling negative sampling to positive-event locations and discarding most ordinary transactions from the loss.
- Local checkpoint selection used an account-memory state generated while weights changed during the epoch; restored checkpoints did not reproduce the recorded validation PR-AUC.
- Exploratory local runs had no validation-only mode and therefore always inspected September.
- Continual and older federated scripts did not consistently select checkpoints/rounds using macro validation PR-AUC.
- Client volume differs by 5.64× and fraud rate by 8.68×. JPMorgan also has the largest mean FedAvg update norm; sample-count weighting gives it about 63% of aggregation mass.
- The single-process simulation centrally constructs a common training-only feature transform before federated training. No raw rows are exchanged during rounds, but a production privacy claim requires a pre-agreed schema or privacy-preserving aggregation of preprocessing statistics.
- September has already been inspected historically, so a truly untouched final estimate is unavailable.

## C. Optimizations performed

- Replaced batch-conditioned sampling with one seeded, stream-wide negative sample per epoch while preserving causal updates from every event.
- Rebuilt causal training memory under frozen end-of-epoch weights before validation checkpoint selection.
- Added validation-only execution, frozen-checkpoint evaluation, Recall/Precision@10/25/50, shared training-only encoders, client-weighting alternatives, FedProx, update-norm diagnostics, and separate non-overwriting experiment directories.
- Tuned only against validation PR-AUC. Final candidates used seeds 42, 52, and 62.

## D. Validation candidates (seed 42 development sweep)

| Stage | Candidate | Macro validation PR-AUC |
|---|---|---:|
{chr(10).join(validation_lines)}

## E. Frozen configurations

- Local: hidden 32, dropout 0.25, AdamW learning rate 1e-3, weight decay 1e-4, negative ratio 20, batch 1024, maximum 25 epochs, patience 5.
- CL: same backbone/optimizer, 20 epochs per June/July task, replay capacity 2,000.
- FedAvg: same backbone, 2 local epochs, at most 12 rounds, round patience 3, sample-count weighting.
- FedProx: same backbone, 1 local epoch, at most 12 rounds, round patience 3, sample-count weighting, μ=1e-4.

The μ=0 FedProx control exactly reproduced FedAvg, confirming the implementation boundary. Every positive μ candidate was worse on validation. The nonzero μ=1e-4/one-local-epoch setting was frozen because it was the strongest actual FedProx candidate at seed 42; multi-seed evaluation then exposed its instability.

## F. Multi-seed validation stability

| Method | Macro validation PR-AUC |
|---|---:|
{chr(10).join(f'| {method} | {fmt(summary["validation_pr_auc"])} |' for method, summary in summaries.items())}

## G. Frozen confirmatory September results

| Method | Macro PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---:|---:|---:|---:|
{chr(10).join(final_lines)}

| Method | {' | '.join(label for label, _key in display)} |
|---|{'|'.join('---:' for _ in display)}|
{chr(10).join(metric_lines)}

All entries are macro-over-bank values summarized as mean ± sample standard deviation over seeds, except the per-bank PR-AUC columns, which are seed means.

## H. Progression analysis

Validation does not show a reliable monotonic progression: Local {fmt(summaries['Local Temporal GNN']['validation_pr_auc'])}, CL {fmt(summaries['+ Continual Learning']['validation_pr_auc'])}, FedAvg {fmt(summaries['+ FedAvg']['validation_pr_auc'])}, FedProx {fmt(summaries['+ FedProx']['validation_pr_auc'])}. FedAvg's small validation gain over local overlaps seed variation and masks weak Wells Fargo performance.

Replay retention is heterogeneous. Mean June forgetting (positive means degradation) is {fmt(retention_macro)} macro across seeds: JPMorgan {fmt(retention_by_bank['JPMorgan_Chase'])}, Wells Fargo {fmt(retention_by_bank['Wells_Fargo'])}, and Key Bank {fmt(retention_by_bank['Key_Bank'])}. Replay reliably preserves/improves Key Bank's older task, but JPMorgan and Wells Fargo are not stable enough to claim general forgetting prevention.

On the frozen September diagnostic, CL is strongest, FedAvg improves over local but not CL, and FedProx is worst. Replay therefore appears useful for later-period adaptation despite weaker/noisier August selection, whereas the chosen proximal constraint over-regularizes local learning and does not solve the observed client heterogeneity.

## I. Scientific conclusion

The evidence does **not** support Local < CL < FedAvg < FedProx. It supports a narrower claim: replay-based continual learning improved the already-inspected later-period diagnostic, and FedAvg produced a modest gain over local, but FedProx did not improve stability or PR-AUC. These conclusions require confirmation on a genuinely untouched post-September temporal holdout before being presented as generalization results.
"""
    (ROOT / "docs" / "OPTIMIZATION_REPORT.md").write_text(doc)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
