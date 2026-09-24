#!/usr/bin/env python3
"""Run a specified validation-only study with bounded CPU concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirm", "local"), required=True)
    parser.add_argument("--winner", help="Screen winner required for confirmation")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--study-config", type=Path, default=ROOT / "configs/adaptive_fcl_study.json")
    parser.add_argument("--variants", nargs="+", help="Run a subset of screen variants; freeze still requires the complete screen")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/research_adaptive_fcl")
    args = parser.parse_args()
    study = json.loads(args.study_config.read_text())
    if args.jobs not in (1, 2):
        parser.error("Use 1 or 2 CPU jobs")
    if args.phase == "screen":
        variants, seeds, mode = list(study["variants"]), [study["screen_seed"]], "federated"
        if args.variants:
            if set(args.variants) - set(variants):
                parser.error("Unknown screen variant")
            variants = args.variants
    elif args.phase == "confirm":
        if args.winner not in study["variants"] or args.winner == "baseline":
            parser.error("Choose a non-baseline screened winner")
        selection_path = args.output_root / "selection.json"
        if not selection_path.exists() or json.loads(selection_path.read_text())["winner"] != args.winner:
            parser.error("Freeze the complete screen with summarize_adaptive_fcl_study.py --freeze-selection first")
        if json.loads(selection_path.read_text())["study_config_sha256"] != hashlib.sha256(args.study_config.read_bytes()).hexdigest():
            parser.error("Study settings differ from the frozen selection")
        variants, seeds, mode = ["baseline", args.winner], study["confirmation_seeds"], "federated"
    else:
        variants = study["local_cl_comparison"]["variants"]
        seeds, mode = study["local_cl_comparison"]["seeds"], "local"
    if args.variants and args.phase != "screen":
        parser.error("--variants is only available for the screen phase")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, seed) for seed in seeds for variant in variants]

    def run(job):
        variant, seed = job
        directory = args.output_root / mode / variant / f"seed_{seed}"
        # Resume orchestration only, never training or a partial run.
        if (directory / "metrics.json").exists():
            print(f"EXISTS {mode}/{variant}/{seed}", flush=True)
            return 0
        options = {**study["common"], **study["variants"][variant], "seed": seed, "mode": mode,
                   "output_dir": str(directory)}
        if mode == "local":
            options["local_epochs"] = study["local_cl_comparison"]["local_epochs"]
        command = [sys.executable, "-B", str(ROOT / "src/gnn/adaptive_continual_federated.py")]
        for key, value in options.items():
            if key == "initial_checkpoint":
                value = str(value).format(seed=seed)
            command.extend(["--" + key.replace("_", "-"), str(value)])
        log_path = args.output_root / f"{mode}_{variant}_{seed}.log"
        with log_path.open("a") as log:
            log.write("COMMAND " + json.dumps(command) + "\n")
            log.flush()
            print(f"START {mode}/{variant}/{seed}", flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {mode}/{variant}/{seed}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = [future.result() for future in as_completed([pool.submit(run, job) for job in jobs])]
    raise SystemExit(1 if any(results) else 0)


if __name__ == "__main__":
    main()
