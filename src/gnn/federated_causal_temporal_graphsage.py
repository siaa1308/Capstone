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
    score_stream, set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=0.01,
                        help="FedProx proximal coefficient; ignored by FedAvg.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many unimproved validation rounds; 0 disables.")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--client-weighting", choices=("samples", "sqrt_samples", "uniform"), default="samples")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alert-k", default="10,25,50")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip the test stream during hyperparameter searches.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT.parent / "federated_causal_temporal_graphsage")
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.batch_size < 1 or args.negative_ratio < 0:
        parser.error("rounds, local-epochs, and batch-size must be positive; negative-ratio cannot be negative")
    if args.prox_mu < 0 or args.patience < 0 or args.weight_decay < 0:
        parser.error("prox-mu and patience cannot be negative")
    return args


def local_train(model, static, events, cfg, seed, global_parameters) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device=static.device).manual_seed(seed)
    for _ in range(cfg.local_epochs):
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
                if cfg.algorithm == "fedprox" and cfg.prox_mu > 0:
                    proximal = sum(
                        torch.sum((local - global_value) ** 2)
                        for local, global_value in zip(model.parameters(), global_parameters)
                    )
                    loss = loss + 0.5 * cfg.prox_mu * proximal
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            state = state.detached()


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
    for bank in BANKS:
        static, streams, _, _ = build_bank_data(cfg.dataset_dir, bank, node_encoder, edge_encoder)
        clients[bank] = (static.to(device), {name: stream.to(device) for name, stream in streams.items()})

    first_static, first_streams = next(iter(clients.values()))
    global_model = CausalTemporalGraphSAGE(
        first_static.shape[1], first_streams["training"].edge_attr.shape[1],
        cfg.hidden_channels, cfg.dropout,
    ).to(device)
    best_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
    best_validation, best_round, stale = -np.inf, 0, 0
    history = []

    for round_index in range(1, cfg.rounds + 1):
        states, sample_counts = [], []
        global_state = {name: value.detach().cpu().clone() for name, value in global_model.state_dict().items()}
        global_parameters = [parameter.detach().clone() for parameter in global_model.parameters()]
        for client_index, (bank, (static, streams)) in enumerate(clients.items()):
            local_model = copy.deepcopy(global_model)
            local_train(
                local_model, static, streams["training"], cfg,
                cfg.seed + round_index * 100 + client_index, global_parameters,
            )
            states.append({name: value.detach().cpu() for name, value in local_model.state_dict().items()})
            sample_counts.append(len(streams["training"]))
        weights = aggregation_weights(sample_counts, cfg.client_weighting)
        global_model.load_state_dict(fedavg(states, weights))
        macro_validation, per_bank_validation = validation_scores(global_model, clients, cfg.batch_size)
        history.append({
            "round": round_index,
            "client_sample_counts": dict(zip(BANKS, sample_counts)),
            "client_weights": dict(zip(BANKS, weights)),
            "client_update_l2": {
                bank: update_l2_norm(state, global_state) for bank, state in zip(BANKS, states)
            },
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
        "prox_mu": cfg.prox_mu if cfg.algorithm == "fedprox" else 0.0,
        "best_round": best_round,
        "best_macro_validation_pr_auc": best_validation,
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
