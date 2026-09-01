#!/usr/bin/env python3
"""Federated continual Causal Temporal GraphSAGE for simulated AML clients.

Each federated round is a chronological local task.  A bank receives the global
model, trains it on its next local time window plus a replay buffer sampled only
from *earlier* local windows, and returns model weights for data-size-weighted
FedAvg.  Raw events, labels, account memories, and replay examples never leave a
client.

The supplied final dataset has June--July training data, so the default 31-day
window creates two rounds (June and July).  Validation and test streams are used
only after the final aggregation for threshold selection and reporting.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from causal_temporal_graphsage import (
    BANKS,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    CausalTemporalGraphSAGE,
    Events,
    alert_budget_metrics,
    batches,
    build_bank_data,
    choose_threshold,
    epoch_sample_mask,
    fit_shared_feature_encoders,
    masked_loss,
    metric_block,
    score_stream,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=("all", *BANKS), default="all")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-days", type=float, default=31.0,
                        help="Chronological duration of one local continual-learning task.")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Optional cap on chronological tasks/FedAvg rounds.")
    parser.add_argument("--fed-rounds-per-task", type=int, default=3,
                        help="FedAvg aggregations performed for each chronological local task.")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=2000)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alert-k", default="10,25,50",
                        help="Comma-separated analyst alert budgets for precision@K and recall@K.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT.parent / "federated_continual_temporal_graphsage")
    args = parser.parse_args()
    if args.task_days <= 0 or args.local_epochs <= 0 or args.fed_rounds_per_task <= 0 or args.replay_size < 0:
        parser.error("--task-days, --local-epochs, and --fed-rounds-per-task must be positive; --replay-size cannot be negative")
    if args.max_rounds is not None and args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    return args


def subset(events: Events, mask: torch.Tensor) -> Events:
    return Events(events.src[mask], events.dst[mask], events.edge_attr[mask], events.timestamp[mask], events.labels[mask])


def concat_chronological(first: Events, second: Events) -> Events:
    order = torch.argsort(torch.cat((first.timestamp, second.timestamp)), stable=True)
    values = zip(
        (first.src, first.dst, first.edge_attr, first.timestamp, first.labels),
        (second.src, second.dst, second.edge_attr, second.timestamp, second.labels),
    )
    return Events(*(torch.cat((left, right))[order] for left, right in values))


def chronological_tasks(events: Events, task_days: float, max_rounds: int | None) -> list[Events]:
    """Split one client stream into non-overlapping timestamp windows."""
    width = task_days * 86_400
    task_index = torch.floor((events.timestamp - events.timestamp[0]) / width).to(torch.long)
    tasks = [subset(events, task_index == index) for index in range(int(task_index.max().item()) + 1)]
    tasks = [task for task in tasks if len(task)]
    return tasks if max_rounds is None else tasks[:max_rounds]


def select_replay(history: Events | None, capacity: int, seed: int) -> Events | None:
    """Keep all historical positives first, then a seeded negative sample.

    The history contains only events completed in previous local tasks.  If there
    are more positives than capacity, positives are sampled deterministically;
    otherwise no current-task or evaluation event can enter replay.
    """
    if history is None or capacity == 0 or len(history) == 0:
        return None
    positives = torch.where(history.labels == 1)[0]
    negatives = torch.where(history.labels == 0)[0]
    generator = torch.Generator(device=history.labels.device).manual_seed(seed)
    if len(positives) > capacity:
        picked = positives[torch.randperm(len(positives), generator=generator, device=history.labels.device)[:capacity]]
    else:
        need = min(len(negatives), capacity - len(positives))
        sampled_negatives = negatives[torch.randperm(len(negatives), generator=generator, device=history.labels.device)[:need]]
        picked = torch.cat((positives, sampled_negatives))
    return subset(history, picked)


def local_train(model: CausalTemporalGraphSAGE, static: torch.Tensor, events: Events, cfg: argparse.Namespace, seed: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=static.device).manual_seed(seed)
    for _ in range(cfg.local_epochs):
        model.train()
        state = model.initial_state(static)
        epoch_mask = epoch_sample_mask(events.labels, cfg.negative_ratio, generator)
        offset = 0
        for batch in batches(events, cfg.batch_size):
            optimizer.zero_grad()
            logits, state = model.score_and_update(state, batch)
            batch_mask = epoch_mask[offset:offset + len(batch)]
            offset += len(batch)
            loss = masked_loss(logits, batch.labels, batch_mask)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            state = state.detached()


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    return {name: sum(state[name].float() * (weight / total) for state, weight in zip(states, weights)) for name in states[0]}


def evaluate_global(model: CausalTemporalGraphSAGE, clients: dict[str, tuple[torch.Tensor, dict[str, Events]]], batch_size: int, alert_budgets: list[int]) -> list[dict[str, object]]:
    results = []
    for bank, (static, streams) in clients.items():
        state = model.initial_state(static)
        _, _, state = score_stream(model, state, streams["training"], batch_size)
        validation_probs, validation_labels, state = score_stream(model, state, streams["validation"], batch_size)
        testing_probs, testing_labels, _ = score_stream(model, state, streams["testing"], batch_size)
        threshold = choose_threshold(validation_labels, validation_probs)
        results.append({
            "bank": bank,
            "threshold": threshold,
            "validation": metric_block(validation_labels, validation_probs, threshold),
            "testing": metric_block(testing_labels, testing_probs, threshold),
            "validation_alert_metrics": alert_budget_metrics(validation_labels, validation_probs, alert_budgets),
            "testing_alert_metrics": alert_budget_metrics(testing_labels, testing_probs, alert_budgets),
        })
    return results


def main() -> None:
    cfg = parse_args()
    cfg.dataset_dir = cfg.dataset_dir.resolve()
    device = torch.device(cfg.device)
    selected = BANKS if cfg.bank == "all" else (cfg.bank,)
    set_seed(cfg.seed)

    node_encoder, edge_encoder = fit_shared_feature_encoders(cfg.dataset_dir, selected)
    clients: dict[str, tuple[torch.Tensor, dict[str, Events]]] = {}
    tasks: dict[str, list[Events]] = {}
    for bank in selected:
        static, streams, _, _ = build_bank_data(cfg.dataset_dir, bank, node_encoder, edge_encoder)
        streams = {name: values.to(device) for name, values in streams.items()}
        clients[bank] = (static.to(device), streams)
        tasks[bank] = chronological_tasks(streams["training"], cfg.task_days, cfg.max_rounds)

    round_count = min(len(bank_tasks) for bank_tasks in tasks.values())
    if round_count == 0:
        raise ValueError("No chronological training tasks were created")
    if any(len(bank_tasks) != round_count for bank_tasks in tasks.values()):
        raise ValueError("Clients have unequal task counts; use --max-rounds to align them")

    first_static, first_streams = next(iter(clients.values()))
    global_model = CausalTemporalGraphSAGE(
        first_static.shape[1], first_streams["training"].edge_attr.shape[1], cfg.hidden_channels, 0.25,
    ).to(device)
    histories: dict[str, Events | None] = {bank: None for bank in selected}
    rounds: list[dict[str, object]] = []

    global_round = 0
    for task_index in range(round_count):
        # A replay sample is fixed for the whole task and comes strictly from
        # history before the task.  Current events become replay-eligible only
        # after every FedAvg update for this task has completed.
        task_inputs: dict[str, tuple[Events, Events | None]] = {}
        for client_index, bank in enumerate(selected):
            current = tasks[bank][task_index]
            replay = select_replay(histories[bank], cfg.replay_size, cfg.seed + task_index * 10_000 + client_index)
            task_inputs[bank] = (current, replay)

        for task_round in range(cfg.fed_rounds_per_task):
            states: list[dict[str, torch.Tensor]] = []
            weights: list[int] = []
            client_rounds: list[dict[str, object]] = []
            for client_index, bank in enumerate(selected):
                static, _streams = clients[bank]
                current, replay = task_inputs[bank]
                local_events = current if replay is None else concat_chronological(replay, current)
                local_model = copy.deepcopy(global_model)
                local_train(
                    local_model, static, local_events, cfg,
                    cfg.seed + task_index * 10_000 + task_round * 100 + client_index,
                )
                states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
                weights.append(len(local_events))
                client_rounds.append({
                    "bank": bank,
                    "current_events": len(current),
                    "current_positives": int(current.labels.sum().item()),
                    "replay_events": 0 if replay is None else len(replay),
                    "aggregation_weight": len(local_events),
                })
            global_model.load_state_dict(fedavg(states, weights))
            global_round += 1
            rounds.append({
                "round": global_round,
                "task": task_index + 1,
                "fed_round_within_task": task_round + 1,
                "clients": client_rounds,
            })
            print(f"Completed continual FedAvg round {global_round}/{round_count * cfg.fed_rounds_per_task} "
                  f"(task {task_index + 1}, pass {task_round + 1}/{cfg.fed_rounds_per_task})")

        for bank in selected:
            current, _replay = task_inputs[bank]
            histories[bank] = current if histories[bank] is None else concat_chronological(histories[bank], current)

    alert_budgets = [int(value) for value in cfg.alert_k.split(",") if value.strip()]
    if not alert_budgets or any(value < 1 for value in alert_budgets):
        raise ValueError("--alert-k must contain positive integers")
    results = evaluate_global(global_model, clients, cfg.batch_size, alert_budgets)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": global_model.state_dict(), "args": vars(cfg)}, cfg.output_dir / "global_model.pt")
    (cfg.output_dir / "metrics.json").write_text(json.dumps({
        "method": "federated continual Causal Temporal GraphSAGE",
        "rounds": rounds,
        "per_bank": results,
    }, indent=2) + "\n")
    for result in results:
        print(f"{result['bank']}: test PR-AUC={result['testing']['pr_auc']:.5f}; F1={result['testing']['f1']:.5f}")


if __name__ == "__main__":
    main()
