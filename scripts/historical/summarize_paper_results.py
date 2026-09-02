#!/usr/bin/env python3
"""Create a comparable Markdown table from saved AML experiment metrics.

The script intentionally consumes only already-written metric files.  It never
trains models or reads labels, keeping result aggregation separate from model
selection.  Missing methods are listed rather than silently omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "paper_evaluation_summary.md"
METHODS = {
    "Leakage-safe tabular XGBoost": ROOT / "artifacts" / "xgboost_paper_baseline" / "metrics.json",
    "Static GraphSAGE": ROOT / "artifacts" / "local_graphsage" / "metrics.json",
    "Local temporal GNN": ROOT / "artifacts" / "causal_temporal_multiseed" / "summary.json",
    "Local continual temporal GNN": ROOT / "artifacts" / "continual_temporal_optimized" / "metrics.json",
    "FedAvg temporal GNN": ROOT / "artifacts" / "fedavg_optimized" / "metrics.json",
    "FedProx temporal GNN": ROOT / "artifacts" / "fedprox_optimized" / "metrics.json",
}
METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")
ACTIVE_BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rows_from_summary(data: list[dict]) -> list[dict]:
    rows = []
    for result in data:
        if result["bank"] not in ACTIVE_BANKS:
            continue
        row = {"bank": result["bank"]}
        for metric in METRICS:
            value = result.get(f"testing_{metric}", {})
            row[metric] = value.get("mean") if isinstance(value, dict) else None
            row[f"{metric}_std"] = value.get("std") if isinstance(value, dict) else None
        rows.append(row)
    return rows


def rows_from_metrics(data: dict) -> list[dict]:
    rows = []
    for result in data["per_bank"]:
        if result["bank"] not in ACTIVE_BANKS:
            continue
        row = {"bank": result["bank"]}
        row.update({metric: result["testing"][metric] for metric in METRICS})
        rows.append(row)
    return rows


def rows_from_list(data: list[dict]) -> list[dict]:
    rows = []
    for result in data:
        if result["bank"] not in ACTIVE_BANKS:
            continue
        row = {"bank": result["bank"]}
        row.update({metric: result["testing"][metric] for metric in METRICS})
        rows.append(row)
    return rows


def render_method(name: str, path: Path) -> str:
    if not path.exists():
        return f"## {name}\n\n_Not yet run: `{path.relative_to(ROOT)}`._\n"
    data = json.loads(path.read_text())
    rows = rows_from_summary(data) if path.name == "summary.json" else (rows_from_list(data) if isinstance(data, list) else rows_from_metrics(data))
    if not rows:
        return f"## {name}\n\n_No saved results for the active three-bank cohort._\n"
    lines = [f"## {name}", "", "| Bank | PR-AUC | ROC-AUC | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {bank} | {pr_auc:.3f} | {roc_auc:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |".format(**row))
    macro = {metric: mean(row[metric] for row in rows) for metric in METRICS}
    spread = {metric: pstdev(row[metric] for row in rows) for metric in METRICS}
    lines.append("| **Macro mean** | {pr_auc:.3f} | {roc_auc:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |".format(**macro))
    lines.append("")
    lines.append("Macro standard deviation across banks: " + ", ".join(f"{metric}={spread[metric]:.3f}" for metric in METRICS) + ".")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sections = [
        "# AML Paper Evaluation Summary",
        "",
        "Primary metric: PR-AUC. Active cohort: JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp are excluded from new experiments based on prior development PR-AUC and remain in historical artifacts for auditability. Results are development diagnostics until the final configuration is frozen and evaluated once on the held-out test period.",
        "",
    ]
    sections.extend(render_method(name, path) for name, path in METHODS.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
