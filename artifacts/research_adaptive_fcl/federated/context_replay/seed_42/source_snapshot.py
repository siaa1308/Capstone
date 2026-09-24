#!/usr/bin/env python3
"""Validation-only research on context replay and temporal update harmonization.

This module leaves the frozen paper implementation and artifacts untouched.
No test split is accepted by the training entry point. See the research report
for the distinction between a project contribution and established prior art.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from causal_temporal_graphsage import (
    BANKS, DEFAULT_DATASET, CausalTemporalGraphSAGE, Events,
    alert_budget_metrics, batches, build_bank_data, choose_threshold,
    epoch_sample_mask, fit_shared_feature_encoders, metric_block,
    score_stream, set_seed, train_temporal_epoch,
)
from continual_temporal_graphsage import concat_chronological, replay_sample, subset

TEMPORAL_PREFIXES = ("time_projection", "message_projection", "memory_update")
ROOT = Path(__file__).resolve().parents[2]


def json_write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def calendar_boundary(dataset: Path, bank: str) -> float:
    frame = pd.read_csv(dataset / "training" / bank / "edge_list.csv.gz", usecols=["timestamp"])
    origin = pd.to_datetime(frame.timestamp).min()
    return float((pd.Timestamp("2025-07-01") - origin).total_seconds())


def development_fingerprint(dataset: Path) -> dict:
    """Fingerprint the actual CSV content, not gzip timestamps or test data."""
    import gzip
    result = {}
    for split in ("training", "validation"):
        for bank in BANKS:
            for name in ("edge_list", "ground_truth", "node_map", "node_features"):
                path = dataset / split / bank / f"{name}.csv.gz"
                with gzip.open(path, "rb") as stream:
                    result[str(path.relative_to(dataset))] = hashlib.sha256(stream.read()).hexdigest()
    path = dataset / "configuration" / "model_feature_columns.json"
    result[str(path.relative_to(dataset))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def risk_context_indices(history: Events, scores: np.ndarray, capacity: int, seed: int,
                         hard_fraction: float = 0.25, context_fraction: float = 0.25) -> torch.Tensor:
    """Bounded replay: positives, hard negatives, earlier endpoint context, coverage.

    Only completed-task labels/scores enter selection. Context is chosen round
    robin across anchors, so a high-activity account cannot consume the budget.
    Unused context slots become uniformly sampled coverage slots. If positives
    exceed capacity, uniformly sample them instead of silently exceeding memory.
    """
    if capacity < 0 or hard_fraction < 0 or context_fraction < 0 or hard_fraction + context_fraction > 1:
        raise ValueError("Invalid replay budget or fractions")
    if len(scores) != len(history) or not np.isfinite(scores).all():
        raise ValueError("Replay scores must be finite and aligned with history")
    if len(history) > 1 and bool(torch.any(history.timestamp[1:] < history.timestamp[:-1])):
        raise ValueError("Replay history must be chronological")
    capacity = min(capacity, len(history))
    rng = np.random.default_rng(seed)
    labels = history.labels.cpu().numpy()
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) >= capacity:
        chosen = rng.choice(positive, capacity, replace=False)
        return torch.tensor(np.sort(chosen), dtype=torch.long, device=history.labels.device)
    selected = set(positive.tolist())
    remaining = capacity - len(selected)
    hard_count = int(remaining * hard_fraction)
    hard = negative[np.argsort(-np.asarray(scores)[negative], kind="stable")[:hard_count]]
    selected.update(hard.tolist())
    context_budget = int(remaining * context_fraction)
    anchors = np.concatenate((positive, hard)).tolist()
    rng.shuffle(anchors)
    src, dst = history.src.cpu().tolist(), history.dst.cpu().tolist()
    times = history.timestamp.cpu().numpy()
    account_history: dict[int, list[int]] = {}
    for index, (left, right) in enumerate(zip(src, dst)):
        for node in {left, right}:
            account_history.setdefault(node, []).append(index)
    queues = []
    for anchor in anchors:
        candidates = set()
        for node in {src[anchor], dst[anchor]}:
            indices = account_history[node]
            cutoff = np.searchsorted(indices, anchor)
            # Strictly earlier timestamps; no same-time or future neighbors.
            candidates.update(i for i in indices[:cutoff] if times[i] < times[anchor])
        queues.append(iter(sorted(candidates, reverse=True)))
    added = 0
    active = list(queues)
    while active and added < context_budget:
        following = []
        for queue in active:
            candidate = next((i for i in queue if i not in selected), None)
            if candidate is not None:
                selected.add(candidate)
                added += 1
                following.append(queue)
            if added >= context_budget:
                break
        active = following
    pool = np.asarray([i for i in range(len(history)) if i not in selected], dtype=np.int64)
    missing = capacity - len(selected)
    if missing:
        selected.update(rng.choice(pool, missing, replace=False).tolist())
    return torch.tensor(sorted(selected), dtype=torch.long, device=history.labels.device)


def temporal_harmonized_average(states: list[dict], global_state: dict, strength: float = 0.5,
                                norm_cap: float = 2.0) -> tuple[dict, dict]:
    """Equal-weight FedAvg for decoder/static; harmonize temporal deltas only.

    Cap temporal update norms at norm_cap * median client norm. For negative
    pairwise dot products, remove `strength` of the projection onto each ORIGINAL
    peer direction. Average peer corrections to make client order irrelevant.
    At strength=0 and norm_cap=inf this is exactly ordinary equal-weight FedAvg.
    Only parameter updates are required; no data, labels, or validation scores.
    """
    if not states or not 0 <= strength <= 1 or norm_cap <= 0:
        raise ValueError("Invalid harmonization inputs")
    keys = [key for key in global_state if key.startswith(TEMPORAL_PREFIXES)]
    result = {key: torch.stack([state[key].float() for state in states]).mean(0)
              for key in global_state}
    if not keys:
        return result, {"conflicting_pairs": 0, "mean_cosine": 0.0}
    deltas = torch.stack([torch.cat([(state[k] - global_state[k]).float().reshape(-1)
                                  for k in keys]) for state in states])
    norms = deltas.norm(dim=1)
    cap = torch.quantile(norms, 0.5) * norm_cap
    if np.isfinite(norm_cap):
        deltas = deltas * (cap / norms.clamp_min(1e-12)).clamp(max=1).unsqueeze(1)
    original = deltas.clone()
    conflicts, cosines = 0, []
    for i in range(len(states)):
        correction = torch.zeros_like(original[i])
        for j in range(len(states)):
            if i == j:
                continue
            dot = torch.dot(original[i], original[j])
            if i < j:
                cosines.append(float(dot / (original[i].norm() * original[j].norm()).clamp_min(1e-12)))
                conflicts += int(dot < 0)
            if dot < 0:
                correction += dot / original[j].square().sum().clamp_min(1e-12) * original[j]
        deltas[i] = original[i] - strength * correction / max(len(states) - 1, 1)
    average = deltas.mean(0)
    offset = 0
    for key in keys:
        count = global_state[key].numel()
        result[key] = global_state[key] + average[offset:offset + count].reshape_as(global_state[key])
        offset += count
    return result, {"conflicting_pairs": conflicts, "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
                    "temporal_update_norms": norms.tolist(), "temporal_norm_cap": float(cap) if np.isfinite(norm_cap) else None}


def distillation_epoch(model, static, events, optimizer, sample_mask, batch_size, tbptt_steps,
                       teacher_logits, replay_mask, alpha):
    """BCE plus class-balanced logit retention on completed-task replay only."""
    model.train()
    state = model.initial_state(static)
    optimizer.zero_grad()
    pending_bce, pending_count = None, 0
    pending_distill = {0: [], 1: []}
    event_batches = list(batches(events, batch_size))
    offset = 0
    for batch_index, event_batch in enumerate(event_batches, 1):
        logits, state = model.score_and_update(state, event_batch)
        stop = offset + len(event_batch)
        mask = sample_mask[offset:stop]
        if bool(mask.any()):
            loss = F.binary_cross_entropy_with_logits(logits[mask], event_batch.labels[mask], reduction="sum")
            pending_bce = loss if pending_bce is None else pending_bce + loss
            pending_count += int(mask.sum())
        retain = replay_mask[offset:stop]
        for label in (0, 1):
            selected = retain & (event_batch.labels == label)
            if bool(selected.any()):
                pending_distill[label].append((logits[selected] - teacher_logits[offset:stop][selected]).square())
        offset = stop
        if batch_index % tbptt_steps == 0 or batch_index == len(event_batches):
            terms = [torch.cat(items).mean() for items in pending_distill.values() if items]
            objective = pending_bce / pending_count if pending_count else None
            if terms:
                retention = alpha * torch.stack(terms).mean()
                objective = retention if objective is None else objective + retention
            if objective is not None:
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError("Non-finite training objective")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
            state = state.detached()
            pending_bce, pending_count = None, 0
            pending_distill = {0: [], 1: []}


def train_client(model, static, training, boundary, cfg, seed):
    set_seed(seed)
    june = subset(training, training.timestamp < boundary)
    july = subset(training, training.timestamp >= boundary)
    if not len(june) or not len(july):
        raise ValueError("Both calendar tasks must contain events")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(cfg.local_epochs):
        train_temporal_epoch(model, static, june, optimizer,
                             epoch_sample_mask(june.labels, 20, generator), cfg.batch_size, 2)
    june_scores, june_labels, _ = score_stream(model, model.initial_state(static), june, cfg.batch_size)
    if cfg.replay == "risk_context":
        indices = risk_context_indices(june, june_scores, cfg.replay_size, seed)
        replay = subset(june, indices)
        targets = torch.from_numpy(np.log(np.clip(june_scores[indices.numpy()], 1e-5, 1-1e-5) /
                                         (1-np.clip(june_scores[indices.numpy()], 1e-5, 1-1e-5)))).float()
    else:
        replay = replay_sample(june, cfg.replay_size, seed)
        if replay is not None:
            order = torch.argsort(replay.timestamp, stable=True)
            replay = subset(replay, order)
        targets = None
    task = july if replay is None else concat_chronological(replay, july)
    replay_count = 0 if replay is None else len(replay)
    teacher = torch.zeros(len(task))
    replay_mask = torch.zeros(len(task), dtype=torch.bool)
    if targets is not None:
        teacher[:replay_count] = targets
        replay_mask[:replay_count] = True
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(cfg.local_epochs):
        mask = epoch_sample_mask(task.labels, 20, generator)
        if cfg.distill > 0:
            distillation_epoch(model, static, task, optimizer, mask, cfg.batch_size, 2,
                               teacher, replay_mask, cfg.distill)
        else:
            train_temporal_epoch(model, static, task, optimizer, mask, cfg.batch_size, 2)
    after, _, _ = score_stream(model, model.initial_state(static), june, cfg.batch_size)
    before_ap = float(average_precision_score(june_labels, june_scores))
    after_ap = float(average_precision_score(june_labels, after))
    return {"june_events": len(june), "july_events": len(july), "replay_events": replay_count,
            "replay_positives": int(replay.labels.sum()) if replay is not None else 0,
            "june_pr_auc_before": before_ap, "june_pr_auc_after": after_ap,
            "forgetting": before_ap - after_ap}


@torch.no_grad()
def validation(model, static, streams, batch_size):
    state = model.initial_state(static)
    _, _, state = score_stream(model, state, streams["training"], batch_size)
    scores, labels, _ = score_stream(model, state, streams["validation"], batch_size)
    threshold = choose_threshold(labels, scores)
    return {"validation": metric_block(labels, scores, threshold), "threshold": threshold,
            "validation_alert_metrics": alert_budget_metrics(labels, scores, [10, 25, 50])}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("federated", "local"), default="federated")
    parser.add_argument("--replay", choices=("uniform", "risk_context"), default="uniform")
    parser.add_argument("--aggregation", choices=("fedavg", "temporal_harmonized"), default="fedavg")
    parser.add_argument("--distill", type=float, default=0.0)
    parser.add_argument("--harmonization-strength", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--replay-size", type=int, default=2000)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    cfg = parser.parse_args()
    if min(cfg.rounds, cfg.local_epochs, cfg.hidden_channels, cfg.batch_size, cfg.threads) < 1:
        parser.error("Counts must be positive")
    if cfg.replay_size < 0 or cfg.distill < 0 or cfg.learning_rate <= 0 or not 0 <= cfg.harmonization_strength <= 1:
        parser.error("Invalid training setting")
    if cfg.distill and cfg.replay != "risk_context":
        parser.error("Distillation requires risk_context replay")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; completed evidence is never overwritten")
    return cfg


def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(cfg.seed)
    started = time.monotonic()
    encoders = fit_shared_feature_encoders(cfg.dataset_dir, BANKS)
    clients = {}
    for bank in BANKS:
        static, streams, *_ = build_bank_data(cfg.dataset_dir, bank, *encoders, splits=("training", "validation"))
        clients[bank] = (static, streams, calendar_boundary(cfg.dataset_dir, bank))
    static, streams, _ = clients[BANKS[0]]
    global_model = CausalTemporalGraphSAGE(static.shape[1], streams["training"].edge_attr.shape[1], cfg.hidden_channels, 0.25)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(cfg).items()}
    metadata = {"configuration": configuration, "test_loaded": False,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "data_sha256": development_fingerprint(cfg.dataset_dir)}
    json_write(cfg.output_dir / "manifest.json", metadata)
    best_score, best_round, history, best_rows = -1.0, 0, [], []
    if cfg.mode == "local":
        rows = []
        for index, (bank, (static, streams, boundary)) in enumerate(clients.items()):
            set_seed(cfg.seed + index * 10_000)
            model = CausalTemporalGraphSAGE(static.shape[1], streams["training"].edge_attr.shape[1], cfg.hidden_channels, 0.25)
            diagnostic = train_client(model, static, streams["training"], boundary, cfg, cfg.seed + index * 10_000)
            row = {"bank": bank, **validation(model, static, streams, cfg.batch_size), "retention": diagnostic}
            rows.append(row)
            torch.save({"state_dict": model.state_dict(), "args": configuration}, cfg.output_dir / f"{bank}_continual.pt")
            print(bank, row["validation"]["pr_auc"], flush=True)
        best_rows = rows
        best_score = float(np.mean([row["validation"]["pr_auc"] for row in rows]))
    else:
        for round_id in range(1, cfg.rounds + 1):
            old = {key: value.detach().clone() for key, value in global_model.state_dict().items()}
            states, diagnostics = [], {}
            for index, (bank, (static, streams, boundary)) in enumerate(clients.items()):
                model = copy.deepcopy(global_model)
                diagnostics[bank] = train_client(model, static, streams["training"], boundary, cfg,
                                                 cfg.seed + round_id * 100 + index)
                states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            if cfg.aggregation == "temporal_harmonized":
                state, aggregation_diagnostics = temporal_harmonized_average(states, old, cfg.harmonization_strength)
            else:
                state = {key: torch.stack([item[key] for item in states]).mean(0) for key in old}
                aggregation_diagnostics = {}
            global_model.load_state_dict(state)
            rows = [{"bank": bank, **validation(global_model, static, streams, cfg.batch_size)}
                    for bank, (static, streams, _) in clients.items()]
            macro = float(np.mean([row["validation"]["pr_auc"] for row in rows]))
            record = {"round": round_id, "macro_validation_pr_auc": macro, "per_bank": rows,
                      "local_retention": diagnostics, "aggregation": aggregation_diagnostics,
                      "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            if macro > best_score:
                best_score, best_round, best_rows = macro, round_id, rows
                torch.save({"state_dict": state, "args": configuration, "best_round": best_round}, cfg.output_dir / "global_model.pt")
            json_write(cfg.output_dir / "progress.json", {"best_macro_validation_pr_auc": best_score, "best_round": best_round, "rounds": history})
            print(f"round={round_id:02d} macro={macro:.6f} best={best_score:.6f} banks=" +
                  ",".join(f"{row['bank']}:{row['validation']['pr_auc']:.4f}" for row in rows) +
                  f" seconds={time.monotonic()-started:.1f}", flush=True)
    json_write(cfg.output_dir / "metrics.json", {**metadata, "best_macro_validation_pr_auc": best_score,
               "best_round": best_round, "per_bank": best_rows, "rounds": history,
               "elapsed_seconds": time.monotonic()-started})
    print(f"COMPLETE validation={best_score:.6f} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
