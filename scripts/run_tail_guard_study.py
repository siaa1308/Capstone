#!/usr/bin/env python3
"""Execute the fixed validation-only comparison; preserve completed runs."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/research_tail_guard"


def run(seed, guarded):
    variant = "tail_guard" if guarded else "continuation"
    output = OUTPUT / variant / f"seed_{seed}"
    if (output / "metrics.json").exists():
        return str(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial run requires inspection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / f"artifacts/research_adaptive_fcl/warm_start/federated/context_harmonized/seed_{seed}/global_model.pt"
    command = [sys.executable, "-B", str(ROOT / "src/gnn/adaptive_continual_federated.py"),
               "--seed", str(seed), "--rng-policy", "global", "--threads", "1",
               "--rounds", "4", "--local-epochs", "3", "--replay-size", "2000",
               "--replay", "risk_context", "--aggregation", "temporal_harmonized",
               "--distill", "0", "--initial-checkpoint", str(checkpoint), "--output-dir", str(output)]
    if guarded:
        command += ["--tail-guard", "--tail-guard-tolerance", "0.05"]
    with (output.parent / f"seed_{seed}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(output)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = ROOT / "configs/causal_tail_guard_study.json"
    snapshot = OUTPUT / "study_design.json"
    if snapshot.exists() and snapshot.read_bytes() != design.read_bytes():
        raise RuntimeError("Study design changed; use a separate output location")
    snapshot.write_bytes(design.read_bytes())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, seed, guard) for seed in (42, 52, 62) for guard in (False, True)]
        for future in concurrent.futures.as_completed(futures):
            print("COMPLETE", future.result(), flush=True)
    rows = []
    for seed in (42, 52, 62):
        row = {"seed": seed}
        for variant in ("continuation", "tail_guard"):
            metrics = json.loads((OUTPUT / variant / f"seed_{seed}/metrics.json").read_text())
            row[variant] = {"validation_ap": metrics["best_macro_validation_pr_auc"],
                            "best_round": metrics["best_round"], "seconds": metrics["elapsed_seconds"]}
        row["paired_gain"] = row["tail_guard"]["validation_ap"] - row["continuation"]["validation_ap"]
        rows.append(row)
    (OUTPUT / "summary.json").write_text(json.dumps({"test_loaded": False, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
