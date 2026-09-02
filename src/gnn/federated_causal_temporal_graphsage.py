#!/usr/bin/env python3
"""Validation-selected FedAvg/FedProx for causal temporal GraphSAGE clients.

Only model parameters are aggregated. Raw transactions, labels, account memories,
and evaluation state remain client-local. Communication-round selection uses the
macro validation PR-AUC; test streams are evaluated once after restoring the best
validation-selected global model.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from causal_temporal_graphsage import (
    BANKS, DEFAULT_DATASET, DEFAULT_OUTPUT, CausalTemporalGraphSAGE,
    alert_budget_metrics, batches, build_bank_data, choose_threshold,
    epoch_sample_mask, fit_shared_feature_encoders, masked_loss, metric_block,
    score_stream, set_seed, train_temporal_epoch,
)
from continual_temporal_graphsage import concat_chronological, replay_sample, subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=0.01,
                        help="FedProx proximal coefficient; ignored by FedAvg.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many unimproved validation rounds; 0 disables.")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--local-training", choices=("full_stream", "continual_replay"), default="full_stream",
                        help="Client procedure: repeated full stream or June then July plus historical replay.")
    parser.add_argument("--replay-size", type=int, default=2000,
                        help="Historical replay capacity for continual_replay; ignored by full_stream.")
    parser.add_argument("--client-weighting", choices=("samples", "sqrt_samples", "uniform"), default="samples")
    parser.add_argument("--server-learning-rate", type=float, default=1.0,
                        help="Interpolate all aggregated parameters toward the client average.")
    parser.add_argument("--temporal-server-learning-rate", type=float, default=None,
                        help="Optional separate interpolation for time/message/GRU parameters.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tbptt-steps", type=int, default=2,
                        help="Temporal batches per optimizer step; must remain >1 to train memory-update modules.")
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum; ignored by AdamW.")
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="AdamW numerical-stability epsilon; ignored by SGD.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alert-k", default="10,25,50")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip the test stream during hyperparameter searches.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT.parent / "federated_causal_temporal_graphsage")
    args = parser.parse_args()
    if (args.rounds < 1 or args.local_epochs < 1 or args.batch_size < 1 or
            args.tbptt_steps < 2 or args.negative_ratio < 0 or args.replay_size < 0):
        parser.error("rounds, local-epochs, and batch-size must be positive; tbptt-steps must be at least 2")
    if (args.prox_mu < 0 or args.patience < 0 or args.weight_decay < 0 or
            not 0 <= args.momentum < 1 or args.adam_eps <= 0 or
            not 0 < args.server_learning_rate <= 1 or
            (args.temporal_server_learning_rate is not None and not 0 <= args.temporal_server_learning_rate <= 1)):
        parser.error("prox-mu, patience, and weight-decay cannot be negative; momentum must be in [0,1); adam-eps must be positive")
    return args


TEMPORAL_PREFIXES = ("time_projection", "message_projection", "memory_update")


def make_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, eps=cfg.adam_eps,
    )


def local_train(model, static, events, cfg, seed, global_parameters) -> dict[str, float | int]:
    optimizer = make_optimizer(model, cfg)
    generator = torch.Generator(device=static.device).manual_seed(seed)
    optimizer_steps = 0
    temporal_gradient_steps = 0
    data_loss_total = 0.0
    proximal_total = 0.0
    for _ in range(cfg.local_epochs):
        model.train()
        state = model.initial_state(static)
        epoch_mask = epoch_sample_mask(events.labels, cfg.negative_ratio, generator)
        offset = 0
        pending_weighted_loss = None
        pending_examples = 0
        optimizer.zero_grad()
        event_batches = list(batches(events, cfg.batch_size))
        for batch_index, event_batch in enumerate(event_batches, start=1):
            logits, state = model.score_and_update(state, event_batch)
            batch_mask = epoch_mask[offset:offset + len(event_batch)]
            offset += len(event_batch)
            loss = masked_loss(logits, event_batch.labels, batch_mask)
            if loss is not None:
                selected = int(batch_mask.sum().item())
                weighted = loss * selected
                pending_weighted_loss = weighted if pending_weighted_loss is None else pending_weighted_loss + weighted
                pending_examples += selected
                data_loss_total += float(loss.detach().item()) * selected

            window_end = batch_index % cfg.tbptt_steps == 0 or batch_index == len(event_batches)
            if window_end and pending_examples:
                objective = pending_weighted_loss / pending_examples
                proximal_value = 0.0
                if cfg.algorithm == "fedprox" and cfg.prox_mu > 0:
                    proximal = sum(
                        torch.sum((local - global_value) ** 2)
                        for local, global_value in zip(model.parameters(), global_parameters)
                    )
                    proximal_value = float(proximal.detach().item())
                    objective = objective + 0.5 * cfg.prox_mu * proximal
                objective.backward()
                if any(
                    parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                    for name, parameter in model.named_parameters() if name.startswith(TEMPORAL_PREFIXES)
                ):
                    temporal_gradient_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                proximal_total += proximal_value
                optimizer.zero_grad()
                pending_weighted_loss = None
                pending_examples = 0
            if window_end:
                state = state.detached()
    return {
        "optimizer_steps": optimizer_steps,
        "temporal_gradient_steps": temporal_gradient_steps,
        "mean_sampled_data_loss": data_loss_total / max(int(epoch_mask.sum().item()) * cfg.local_epochs, 1),
        "mean_proximal_squared_l2": proximal_total / max(optimizer_steps, 1),
    }


def continual_local_train(model, static, events, cfg, seed) -> dict[str, float | int | str]:
    """Match local CL: June training, then July plus a June-only replay sample."""
    july_boundary = 30 * 86_400
    june = subset(events, events.timestamp < july_boundary)
    july = subset(events, events.timestamp >= july_boundary)
    replay = replay_sample(june, cfg.replay_size, seed)
    second_task = july if replay is None else concat_chronological(replay, july)
    optimizer = make_optimizer(model, cfg)
    temporal_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
        if name.startswith(TEMPORAL_PREFIXES)
    }
    optimizer_steps = 0
    for task_index, task in enumerate((june, second_task)):
        generator = torch.Generator(device=static.device).manual_seed(seed + task_index)
        for _ in range(cfg.local_epochs):
            epoch_mask = epoch_sample_mask(task.labels, cfg.negative_ratio, generator)
            _loss, steps = train_temporal_epoch(
                model, static, task, optimizer, epoch_mask, cfg.batch_size, cfg.tbptt_steps,
            )
            optimizer_steps += steps
    temporal_changed = any(
        not torch.equal(temporal_before[name], parameter.detach())
        for name, parameter in model.named_parameters() if name in temporal_before
    )
    return {
        "optimizer_steps": optimizer_steps,
        "temporal_parameters_changed": temporal_changed,
        "june_events": len(june),
        "july_events": len(july),
        "replay_events": 0 if replay is None else len(replay),
        "optimizer_state_policy": "one optimizer retained across June and July+replay; reset at next FedAvg round",
    }


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    return {
        name: sum(state[name].float() * (weight / total) for state, weight in zip(states, weights))
        for name in states[0]
    }


def aggregation_weights(sample_counts: list[int], strategy: str) -> list[float]:
    if strategy == "uniform":
        return [1.0] * len(sample_counts)
    if strategy == "sqrt_samples":
        return [float(np.sqrt(count)) for count in sample_counts]
    return [float(count) for count in sample_counts]


def update_l2_norm(local_state: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> float:
    squared = sum(torch.sum((local_state[name].float() - global_state[name].cpu().float()) ** 2).item() for name in local_state)
    return float(np.sqrt(squared))


def grouped_distance(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, float]:
    groups = {"temporal_gru": 0.0, "static_edge_decoder": 0.0}
    for name in left:
        group = "temporal_gru" if name.startswith(TEMPORAL_PREFIXES) else "static_edge_decoder"
        groups[group] += torch.sum((left[name].float() - right[name].float()) ** 2).item()
    return {name: float(np.sqrt(value)) for name, value in groups.items()}


def pairwise_grouped_distances(states: list[dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    result = {}
    for left_index in range(len(states)):
        for right_index in range(left_index + 1, len(states)):
            result[f"{BANKS[left_index]}__{BANKS[right_index]}"] = grouped_distance(
                states[left_index], states[right_index],
            )
    return result


def interpolated_average(states, weights, global_state, server_lr: float, temporal_server_lr: float | None):
    average = fedavg(states, weights)
    result = {}
    for name, value in average.items():
        alpha = temporal_server_lr if name.startswith(TEMPORAL_PREFIXES) and temporal_server_lr is not None else server_lr
        result[name] = global_state[name].float() + alpha * (value.float() - global_state[name].float())
    return result


@torch.no_grad()
def bank_validation_score(model, static, streams, batch_size: int) -> float:
    state = model.initial_state(static)
    _, _, state = score_stream(model, state, streams["training"], batch_size)
    probabilities, labels, _ = score_stream(model, state, streams["validation"], batch_size)
    return float(average_precision_score(labels, probabilities))


@torch.no_grad()
def validation_scores(model, clients, batch_size: int) -> tuple[float, dict[str, float]]:
    per_bank = {}
    for bank, (static, streams) in clients.items():
        state = model.initial_state(static)
        _, _, state = score_stream(model, state, streams["training"], batch_size)
        probabilities, labels, _ = score_stream(model, state, streams["validation"], batch_size)
        per_bank[bank] = float(average_precision_score(labels, probabilities))
    return float(np.mean(list(per_bank.values()))), per_bank


@torch.no_grad()
def final_evaluation(model, clients, batch_size: int, budgets: list[int], validation_only: bool) -> list[dict[str, object]]:
    results = []
    for bank, (static, streams) in clients.items():
        state = model.initial_state(static)
        _, _, state = score_stream(model, state, streams["training"], batch_size)
        validation_probs, validation_labels, state = score_stream(model, state, streams["validation"], batch_size)
        threshold = choose_threshold(validation_labels, validation_probs)
        result = {
            "bank": bank,
            "threshold": threshold,
            "validation": metric_block(validation_labels, validation_probs, threshold),
            "validation_alert_metrics": alert_budget_metrics(validation_labels, validation_probs, budgets),
        }
        if not validation_only:
            testing_probs, testing_labels, _ = score_stream(model, state, streams["testing"], batch_size)
            result["testing"] = metric_block(testing_labels, testing_probs, threshold)
            result["testing_alert_metrics"] = alert_budget_metrics(testing_labels, testing_probs, budgets)
        results.append(result)
    return results


def main() -> None:
    cfg = parse_args()
    cfg.dataset_dir = cfg.dataset_dir.resolve()
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    budgets = [int(value) for value in cfg.alert_k.split(",") if value.strip()]
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("--alert-k must contain positive integers")

    node_encoder, edge_encoder = fit_shared_feature_encoders(cfg.dataset_dir, BANKS)
    clients = {}
    development_splits = ("training", "validation") if cfg.validation_only else ("training", "validation", "testing")
    for bank in BANKS:
        static, streams, _, _ = build_bank_data(
            cfg.dataset_dir, bank, node_encoder, edge_encoder, development_splits,
        )
        clients[bank] = (static.to(device), {name: stream.to(device) for name, stream in streams.items()})

    first_static, first_streams = next(iter(clients.values()))
    global_model = CausalTemporalGraphSAGE(
        first_static.shape[1], first_streams["training"].edge_attr.shape[1],
        cfg.hidden_channels, cfg.dropout,
    ).to(device)
    best_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
    best_validation, best_round, stale = -np.inf, 0, 0
    history = []
    initial_macro_validation, initial_per_bank_validation = validation_scores(global_model, clients, cfg.batch_size)

    for round_index in range(1, cfg.rounds + 1):
        states, sample_counts, training_diagnostics, local_validation = [], [], {}, {}
        global_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
        global_parameters = [parameter.detach().clone() for parameter in global_model.parameters()]
        before_macro_validation, before_per_bank_validation = validation_scores(global_model, clients, cfg.batch_size)
        for client_index, (bank, (static, streams)) in enumerate(clients.items()):
            local_model = copy.deepcopy(global_model)
            if cfg.local_training == "continual_replay":
                training_diagnostics[bank] = continual_local_train(
                    local_model, static, streams["training"], cfg,
                    cfg.seed + round_index * 100 + client_index,
                )
            else:
                training_diagnostics[bank] = local_train(
                    local_model, static, streams["training"], cfg,
                    cfg.seed + round_index * 100 + client_index, global_parameters,
                )
                training_diagnostics[bank]["replay_events"] = 0
                training_diagnostics[bank]["optimizer_state_policy"] = "reset for every client at every FedAvg round"
            local_validation[bank] = bank_validation_score(local_model, static, streams, cfg.batch_size)
            states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
            sample_counts.append(len(streams["training"]))
        weights = aggregation_weights(sample_counts, cfg.client_weighting)
        global_model.load_state_dict(interpolated_average(
            states, weights, global_state, cfg.server_learning_rate, cfg.temporal_server_learning_rate,
        ))
        macro_validation, per_bank_validation = validation_scores(global_model, clients, cfg.batch_size)
        history.append({
            "round": round_index,
            "global_validation_before_aggregation": before_macro_validation,
            "global_validation_by_bank_before_aggregation": before_per_bank_validation,
            "local_validation_pr_auc_before_aggregation": local_validation,
            "client_sample_counts": dict(zip(BANKS, sample_counts)),
            "client_weights": dict(zip(BANKS, weights)),
            "normalized_client_weights": {
                bank: float(weight / sum(weights)) for bank, weight in zip(BANKS, weights)
            },
            "client_update_l2": {
                bank: update_l2_norm(state, global_state) for bank, state in zip(BANKS, states)
            },
            "client_update_l2_by_parameter_group": {
                bank: grouped_distance(state, global_state) for bank, state in zip(BANKS, states)
            },
            "pairwise_client_l2_by_parameter_group": pairwise_grouped_distances(states),
            "client_training_diagnostics": training_diagnostics,
            "macro_validation_pr_auc": macro_validation,
            "validation_pr_auc_by_bank": per_bank_validation,
        })
        print(f"Round {round_index}/{cfg.rounds}: macro validation PR-AUC={macro_validation:.5f}")
        if macro_validation > best_validation:
            best_validation, best_round, stale = macro_validation, round_index, 0
            best_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
        else:
            stale += 1
        if cfg.patience and stale >= cfg.patience:
            print(f"Early stopping after round {round_index}; best round={best_round}")
            break

    global_model.load_state_dict(best_state)
    results = final_evaluation(global_model, clients, cfg.batch_size, budgets, cfg.validation_only)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(cfg), "best_round": best_round}, cfg.output_dir / "global_model.pt")
    (cfg.output_dir / "metrics.json").write_text(json.dumps({
        "method": cfg.algorithm,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(cfg).items()},
        "prox_mu": cfg.prox_mu if cfg.algorithm == "fedprox" else 0.0,
        "best_round": best_round,
        "best_macro_validation_pr_auc": best_validation,
        "initial_macro_validation_pr_auc": initial_macro_validation,
        "initial_validation_pr_auc_by_bank": initial_per_bank_validation,
        "rounds": history,
        "per_bank": results,
    }, indent=2) + "\n")
    for result in results:
        if cfg.validation_only:
            print(f"{result['bank']}: validation PR-AUC={result['validation']['pr_auc']:.5f}")
        else:
            print(f"{result['bank']}: test PR-AUC={result['testing']['pr_auc']:.5f}")


if __name__ == "__main__":
    main()
