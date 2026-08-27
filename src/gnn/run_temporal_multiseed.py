#!/usr/bin/env python3
"""Run the causal temporal baseline with fixed seeds and summarize mean ± std.

All model selection remains validation-based inside each run. This runner is for
robustness reporting, not for selecting a setting by September test performance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "gnn" / "causal_temporal_graphsage.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--bank", default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--calibration", choices=("none", "platt"), default="platt")
    parser.add_argument("--alert-k", default="10,25,50")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "causal_temporal_multiseed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise SystemExit("Use at least two seeds for a robustness experiment")
    results: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = args.output_dir / f"seed_{seed}"
        command = [
            sys.executable, str(MODEL), "--bank", args.bank, "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--negative-ratio", str(args.negative_ratio), "--calibration", args.calibration, "--alert-k", args.alert_k,
            "--seed", str(seed), "--output-dir", str(run_dir),
        ]
        print("\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        results.extend(json.loads((run_dir / "metrics.json").read_text()))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["bank"])].append(result)
    summary = []
    for bank, rows in grouped.items():
        def stat(path: str) -> dict[str, float]:
            values = np.asarray([float(row["testing"][path]) for row in rows])  # type: ignore[index]
            return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        val = np.asarray([float(row["best_validation_pr_auc"]) for row in rows])
        summary.append({
            "bank": bank, "runs": len(rows),
            "validation_pr_auc": {"mean": float(val.mean()), "std": float(val.std(ddof=1))},
            "testing_pr_auc": stat("pr_auc"), "testing_roc_auc": stat("roc_auc"),
            "testing_precision": stat("precision"), "testing_recall": stat("recall"), "testing_f1": stat("f1"),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_runs.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nMean ± standard deviation summary")
    for row in summary:
        print(f"{row['bank']}: test PR-AUC={row['testing_pr_auc']['mean']:.5f} ± {row['testing_pr_auc']['std']:.5f}")


if __name__ == "__main__":
    main()
