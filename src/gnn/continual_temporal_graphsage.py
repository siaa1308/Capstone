#!/usr/bin/env python3
"""Local continual Causal Temporal GraphSAGE with June-to-July replay.

Replay contains only completed June events. Validation selects the alert threshold,
and September test labels are used only for the final frozen evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from causal_temporal_graphsage import (
    BANKS, DEFAULT_DATASET, DEFAULT_OUTPUT, CausalTemporalGraphSAGE, Events,
    alert_budget_metrics, batches, build_bank_data, choose_threshold, metric_block,
    epoch_sample_mask, fit_shared_feature_encoders, masked_loss, score_stream, set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=("all", *BANKS), default="all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--replay-size", type=int, default=2000)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alert-k", default="10,25,50")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip the test stream during hyperparameter searches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT.parent / "continual_temporal_graphsage")
    args = parser.parse_args()
    if (args.epochs_per_task < 1 or args.replay_size < 0 or args.negative_ratio < 0 or
            args.batch_size < 1 or args.weight_decay < 0):
        parser.error("epochs-per-task and batch-size must be positive; replay-size and negative-ratio cannot be negative")
    return args


def subset(events: Events, mask: torch.Tensor) -> Events:
    return Events(events.src[mask], events.dst[mask], events.edge_attr[mask], events.timestamp[mask], events.labels[mask])


def concat_chronological(first: Events, second: Events) -> Events:
    order = torch.argsort(torch.cat((first.timestamp, second.timestamp)), stable=True)
    pairs = zip(
        (first.src, first.dst, first.edge_attr, first.timestamp, first.labels),
        (second.src, second.dst, second.edge_attr, second.timestamp, second.labels),
    )
    return Events(*(torch.cat((left, right))[order] for left, right in pairs))


def train_task(model, static, events, optimizer, cfg, seed: int) -> None:
    generator = torch.Generator(device=static.device).manual_seed(seed)
    for _ in range(cfg.epochs_per_task):
        model.train()
        state = model.initial_state(static)
        epoch_mask = epoch_sample_mask(events.labels, cfg.negative_ratio, generator)
        offset = 0
        for event_batch in batches(events, cfg.batch_size):
            optimizer.zero_grad()
            logits, state = model.score_and_update(state, event_batch)
            batch_mask = epoch_mask[offset:offset + len(event_batch)]
            offset += len(event_batch)
            loss = masked_loss(logits, event_batch.labels, batch_mask)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            state = state.detached()


def retention_metrics(model, static, events, batch_size: int) -> dict[str, float]:
    probabilities, labels, _ = score_stream(model, model.initial_state(static), events, batch_size)
    return metric_block(labels, probabilities, 0.5)


def replay_sample(history: Events, capacity: int, seed: int) -> Events | None:
    if capacity == 0:
        return None
    positives = torch.where(history.labels == 1)[0]
    negatives = torch.where(history.labels == 0)[0]
    generator = torch.Generator(device=history.labels.device).manual_seed(seed)
    if len(positives) >= capacity:
        picked = positives[torch.randperm(len(positives), generator=generator, device=history.labels.device)[:capacity]]
    else:
        negative_count = min(len(negatives), capacity - len(positives))
        sampled_negatives = negatives[
            torch.randperm(len(negatives), generator=generator, device=history.labels.device)[:negative_count]
        ]
        picked = torch.cat((positives, sampled_negatives))
    return subset(history, picked)


def run_bank(
    cfg, bank: str, device: torch.device, budgets: list[int], node_encoder, edge_encoder,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    seed = cfg.seed + BANKS.index(bank) * 10_000
    set_seed(seed)
    static, streams, node_columns, edge_columns = build_bank_data(
        cfg.dataset_dir, bank, node_encoder, edge_encoder,
    )
    static = static.to(device)
    streams = {name: events.to(device) for name, events in streams.items()}
    training = streams["training"]

    # June has 30 days. Everything at or after the July boundary is task two.
    july_boundary = 30 * 86_400
    june = subset(training, training.timestamp < july_boundary)
    july = subset(training, training.timestamp >= july_boundary)
    model = CausalTemporalGraphSAGE(
        static.shape[1], training.edge_attr.shape[1], cfg.hidden_channels, cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_task(model, static, june, optimizer, cfg, seed)
    june_before = retention_metrics(model, static, june, cfg.batch_size)
    replay = replay_sample(june, cfg.replay_size, seed)
    second_task = july if replay is None else concat_chronological(replay, july)
    train_task(model, static, second_task, optimizer, cfg, seed + 1)
    june_after = retention_metrics(model, static, june, cfg.batch_size)

    # Rebuild causal state from the full training stream before held-out evaluation.
    state = model.initial_state(static)
    _, _, state = score_stream(model, state, training, cfg.batch_size)
    validation_probs, validation_labels, state = score_stream(model, state, streams["validation"], cfg.batch_size)
    threshold = choose_threshold(validation_labels, validation_probs)
    result = {
        "bank": bank,
        "seed": seed,
        "june_events": len(june),
        "july_events": len(july),
        "replay_events": 0 if replay is None else len(replay),
        "june_before_july": june_before,
        "june_after_july": june_after,
        "forgetting": june_before["pr_auc"] - june_after["pr_auc"],
        "threshold": threshold,
        "validation": metric_block(validation_labels, validation_probs, threshold),
        "validation_alert_metrics": alert_budget_metrics(validation_labels, validation_probs, budgets),
        "node_feature_columns": node_columns,
        "edge_feature_columns": edge_columns,
    }
    if not cfg.validation_only:
        testing_probs, testing_labels, _ = score_stream(model, state, streams["testing"], cfg.batch_size)
        result["testing"] = metric_block(testing_labels, testing_probs, threshold)
        result["testing_alert_metrics"] = alert_budget_metrics(testing_labels, testing_probs, budgets)
    state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return result, state_dict


def main() -> None:
    cfg = parse_args()
    cfg.dataset_dir = cfg.dataset_dir.resolve()
    device = torch.device(cfg.device)
    selected = BANKS if cfg.bank == "all" else (cfg.bank,)
    budgets = [int(value) for value in cfg.alert_k.split(",") if value.strip()]
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("--alert-k must contain positive integers")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    node_encoder, edge_encoder = fit_shared_feature_encoders(cfg.dataset_dir, BANKS)
    results = []
    for bank in selected:
        result, state_dict = run_bank(cfg, bank, device, budgets, node_encoder, edge_encoder)
        results.append(result)
        torch.save({"state_dict": state_dict, "args": vars(cfg)}, cfg.output_dir / f"{bank}_continual.pt")
        test_text = "" if cfg.validation_only else f"; test PR-AUC={result['testing']['pr_auc']:.5f}"
        print(f"{bank}: validation PR-AUC={result['validation']['pr_auc']:.5f}{test_text}; "
              f"forgetting={result['forgetting']:.5f}")
    (cfg.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
