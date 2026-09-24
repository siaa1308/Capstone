#!/usr/bin/env python3
"""Plot saved validation evidence without model evaluation or dataset access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"baseline": "FedAvg control", "context_replay": "Context replay",
          "context_retention": "Context + retention", "temporal_aggregation": "Temporal aggregation",
          "combined": "Context + temporal\n+ retention"}
LABELS["context_harmonized"] = "Context + temporal\naggregation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for directory in sorted((args.root / "federated").glob("*")):
        path = directory / "seed_42" / "metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        history = data["rounds"]
        axes[0].plot([r["round"] for r in history], [r["macro_validation_pr_auc"] for r in history],
                     marker=".", linewidth=1.5, label=LABELS.get(directory.name, directory.name))
    axes[0].set(title="Seed 42: August validation trajectory", xlabel="Communication round", ylabel="Macro-bank PR-AUC", ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="best")
    rows = [(name.split("/")[-1], value) for name, value in summary.items() if name.startswith("federated/")]
    for index, (name, result) in enumerate(rows):
        values = list(result["seeds"].values())
        jitter = np.linspace(-.12, .12, len(values)) if len(values) > 1 else [0]
        axes[1].scatter(np.asarray(jitter) + index, values, s=35, zorder=3)
        stat = result["macro_validation_pr_auc"]
        axes[1].errorbar(index, stat["mean"], yerr=stat["sample_sd"] or 0, fmt="_", color="black", capsize=4, markersize=15)
    axes[1].set(xticks=range(len(rows)), xticklabels=[LABELS.get(name, name) for name, _ in rows],
                title="Selected checkpoints: individual seeds, mean ± SD", ylabel="Macro-bank August PR-AUC", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Development evidence only — September is excluded", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(args.root / f"validation_comparison.{extension}", dpi=180)
    print(args.root / "validation_comparison.png")


if __name__ == "__main__":
    main()
