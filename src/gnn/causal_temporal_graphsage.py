#!/usr/bin/env python3
"""Causal Temporal GraphSAGE for local AML transaction (edge) classification.

The model processes each bank's transactions in timestamp order. A transaction is
scored from the account states available *before* its micro-batch is incorporated,
then that batch updates account state. This conservative micro-batch protocol never
lets a later transaction influence an earlier score. Complete validation/testing
sets remain untouched and are used only for evaluation.

This is intentionally a GraphSAGE-style temporal extension, not a replacement with
a separate TGN architecture: account state is updated from historical neighbour
messages, time deltas, and safe transaction attributes, then an edge decoder scores
the source/destination pair.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "final_temporal_dataset"
DEFAULT_OUTPUT = ROOT / "artifacts" / "causal_temporal_graphsage"
# Active three-client cohort selected by the paper's primary metric (PR-AUC).
# Citi and Fifth Third remain in the dataset and historical artifacts for auditability.
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")
SPLITS = ("training", "validation", "testing")
FORBIDDEN = {
    "node_id", "txn_id", "timestamp", "Transaction_Date", "Transaction_Time", "src_id", "dst_id",
    "src_bank_id", "dst_bank_id", "split", "y", "laundering_type", "edge_label", "Is_APP_Fraud",
    "Is_Cheque_Fraud", "APP_Fraudster_ID", "Cheque_Fraudster_ID", "APP_Fraud_Sequence_Number",
    "transaction_type_raw", "transaction_type_model_safe", "From_End_Balance", "To_End_Balance",
    "Controlled_by_Criminal",
}


@dataclass
class Events:
    src: Tensor
    dst: Tensor
    edge_attr: Tensor
    timestamp: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> "Events":
        return Events(*(value.to(device) for value in (self.src, self.dst, self.edge_attr, self.timestamp, self.labels)))

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class TemporalState:
    memory: Tensor
    last_seen: Tensor

    def detached(self) -> "TemporalState":
        return TemporalState(self.memory.detach(), self.last_seen.detach())


class FeatureEncoder:
    """Dense feature transform fitted strictly on the training split."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.transformer: ColumnTransformer | None = None

    @staticmethod
    def _one_hot() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        numeric = [name for name in self.columns if pd.api.types.is_numeric_dtype(frame[name])]
        categorical = [name for name in self.columns if name not in numeric]
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")), ("onehot", self._one_hot()),
            ]), categorical))
        self.transformer = ColumnTransformer(transformers, sparse_threshold=0).fit(frame[self.columns])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Feature encoder has not been fitted")
        return np.asarray(self.transformer.transform(frame[self.columns]), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bank", choices=("all", *BANKS), default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-batch-size", type=int, default=512)
    parser.add_argument("--tbptt-steps", type=int, default=2,
                        help="Temporal batches per optimizer step; must remain >1 to train memory-update modules.")
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--loss-pos-weight", type=float, default=1.0)
    parser.add_argument("--full-training-weighted-bce", action="store_true")
    parser.add_argument("--calibration", choices=("none", "platt"), default="none", help="Fit score calibration using validation labels only.")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip the test stream during hyperparameter searches.")
    parser.add_argument("--alert-k", default="10,25,50", help="Comma-separated alert budgets for precision@K and recall@K.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-encoder", action="store_true",
                        help="Use one active-cohort training-only feature transform for fair local/federated comparison.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def columns_from_manifest(dataset: Path, node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    manifest = json.loads((dataset / "configuration" / "model_feature_columns.json").read_text())
    node_columns = [name for name in manifest["graph_node_safe_features"] if name in node_frame and name not in FORBIDDEN]
    edge_columns = [name for name in manifest["graph_edge_safe_features"] if name in edge_frame and name not in FORBIDDEN]
    if not node_columns or not edge_columns:
        raise ValueError("Approved causal node or edge feature list is empty")
    return node_columns, edge_columns


def load_bank_frames(
    dataset: Path, bank: str, splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    nodes, maps, edges = {}, {}, {}
    for split in splits:
        directory = dataset / split / bank
        nodes[split] = pd.read_csv(directory / "node_features.csv.gz")
        maps[split] = pd.read_csv(directory / "node_map.csv.gz")
        edges[split] = pd.read_csv(directory / "edge_list.csv.gz")
        truth = pd.read_csv(directory / "ground_truth.csv.gz")
        if nodes[split]["node_id"].duplicated().any() or maps[split]["node_id"].duplicated().any():
            raise ValueError(f"{bank}/{split}: duplicate local node ID")
        if edges[split]["txn_id"].duplicated().any() or truth["txn_id"].duplicated().any():
            raise ValueError(f"{bank}/{split}: duplicate transaction ID")
        edges[split] = edges[split].merge(truth[["txn_id", "y"]], on="txn_id", how="inner", validate="one_to_one")
        if len(edges[split]) != len(truth):
            raise ValueError(f"{bank}/{split}: edge/label join changed row count")
    return nodes, maps, edges


def make_events(
    frame: pd.DataFrame, local_to_global: dict[int, int], edge_encoder: FeatureEncoder, origin: pd.Timestamp,
) -> Events:
    source = frame["src_node"].map(local_to_global)
    destination = frame["dst_node"].map(local_to_global)
    if source.isna().any() or destination.isna().any():
        raise ValueError("An edge endpoint is missing from node_map")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    seconds = ((timestamps - origin).dt.total_seconds().to_numpy(dtype=np.float32))[order]
    return Events(
        src=torch.tensor(source.to_numpy(dtype=np.int64)[order]),
        dst=torch.tensor(destination.to_numpy(dtype=np.int64)[order]),
        edge_attr=torch.from_numpy(edge_encoder.transform(frame)[order]),
        timestamp=torch.from_numpy(seconds),
        labels=torch.tensor(frame["y"].to_numpy(dtype=np.float32)[order]),
    )


def build_bank_data(
    dataset: Path, bank: str, node_encoder: FeatureEncoder | None = None, edge_encoder: FeatureEncoder | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> tuple[Tensor, dict[str, Events], list[str], list[str]]:
    if "training" not in splits:
        raise ValueError("training must be included when building bank data")
    nodes, maps, edges = load_bank_frames(dataset, bank, splits)
    node_columns, edge_columns = columns_from_manifest(dataset, nodes["training"], edges["training"])
    node_encoder = node_encoder or FeatureEncoder(node_columns).fit(nodes["training"])
    edge_encoder = edge_encoder or FeatureEncoder(edge_columns).fit(edges["training"])

    # Local node IDs are deliberately split-specific.  Convert through account_id
    # to one stable bank-local ID space so that account memory persists over time.
    account_ids = pd.concat([maps[split]["account_id"] for split in splits], ignore_index=True).drop_duplicates().tolist()
    account_to_global = {account_id: index for index, account_id in enumerate(account_ids)}
    local_to_global: dict[str, dict[int, int]] = {}
    account_features: list[pd.DataFrame] = []
    for split in splits:
        joined = nodes[split].merge(maps[split], on="node_id", how="inner", validate="one_to_one")
        local_to_global[split] = dict(zip(joined["node_id"].astype(int), joined["account_id"].map(account_to_global).astype(int)))
        # First available pre-period account snapshot is used. Training snapshots
        # take precedence; accounts first observed later use that split's snapshot.
        joined["global_id"] = joined["account_id"].map(account_to_global)
        account_features.append(joined)
    selected_features = pd.concat(account_features, ignore_index=True).drop_duplicates("global_id", keep="first")
    static = np.zeros((len(account_to_global), len(node_encoder.transform(nodes["training"])[0])), dtype=np.float32)
    transformed = node_encoder.transform(selected_features)
    static[selected_features["global_id"].to_numpy(dtype=np.int64)] = transformed
    origin = min(pd.to_datetime(edges[split]["timestamp"]).min() for split in splits)
    events = {split: make_events(edges[split], local_to_global[split], edge_encoder, origin) for split in splits}
    return torch.from_numpy(static), events, node_columns, edge_columns


def fit_shared_feature_encoders(dataset: Path, banks: tuple[str, ...] = BANKS) -> tuple[FeatureEncoder, FeatureEncoder]:
    """Fit one training-only schema so FedAvg clients have compatible tensors."""
    node_frames, edge_frames = [], []
    for bank in banks:
        nodes, _maps, edges = load_bank_frames(dataset, bank, ("training",))
        node_frames.append(nodes["training"])
        edge_frames.append(edges["training"])
    node_columns, edge_columns = columns_from_manifest(dataset, node_frames[0], edge_frames[0])
    return FeatureEncoder(node_columns).fit(pd.concat(node_frames, ignore_index=True)), FeatureEncoder(edge_columns).fit(pd.concat(edge_frames, ignore_index=True))


class CausalTemporalGraphSAGE(nn.Module):
    """GraphSAGE-style account memory updated only from earlier event batches."""

    def __init__(self, node_features: int, edge_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.static_projection = nn.Linear(node_features, hidden)
        self.edge_projection = nn.Sequential(nn.Linear(edge_features, hidden), nn.ReLU())
        self.time_projection = nn.Sequential(nn.Linear(2, hidden), nn.ReLU())
        # Each account receives its own prior state plus an aggregate of historical
        # counterpart states: the causal, temporal GraphSAGE neighbourhood signal.
        self.message_projection = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.memory_update = nn.GRUCell(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def initial_state(self, static_features: Tensor) -> TemporalState:
        memory = torch.tanh(self.static_projection(static_features))
        return TemporalState(memory=memory, last_seen=torch.zeros(len(static_features), device=memory.device))

    def score_and_update(self, state: TemporalState, events: Events) -> tuple[Tensor, TemporalState]:
        src_state, dst_state = state.memory[events.src], state.memory[events.dst]
        src_delta = torch.log1p(torch.clamp(events.timestamp - state.last_seen[events.src], min=0))
        dst_delta = torch.log1p(torch.clamp(events.timestamp - state.last_seen[events.dst], min=0))
        edge = self.edge_projection(events.edge_attr)
        time = self.time_projection(torch.stack((src_delta, dst_delta), dim=1))
        logits = self.decoder(torch.cat((src_state, dst_state, torch.abs(src_state - dst_state), edge), dim=1)).squeeze(1)

        # Every message uses the pre-batch state. Thus later events cannot influence
        # earlier scores, and events sharing a timestamp cannot influence each other.
        src_message = self.message_projection(torch.cat((src_state, dst_state, edge, time), dim=1))
        dst_message = self.message_projection(torch.cat((dst_state, src_state, edge, time), dim=1))
        endpoints = torch.cat((events.src, events.dst))
        messages = torch.cat((src_message, dst_message))
        sums = torch.zeros_like(state.memory).index_add(0, endpoints, messages)
        counts = torch.zeros((len(state.memory), 1), device=state.memory.device).index_add(
            0, endpoints, torch.ones((len(endpoints), 1), device=state.memory.device)
        )
        changed = torch.where(counts.squeeze(1) > 0)[0]
        aggregate = sums[changed] / counts[changed].clamp_min(1)
        next_memory = state.memory.clone()
        next_memory[changed] = self.memory_update(aggregate, state.memory[changed])
        next_seen = state.last_seen.clone()
        latest = next_seen.clone()
        latest.scatter_reduce_(0, endpoints, torch.cat((events.timestamp, events.timestamp)), reduce="amax", include_self=True)
        next_seen[changed] = latest[changed]
        return logits, TemporalState(next_memory, next_seen)


def batches(events: Events, batch_size: int):
    for start in range(0, len(events), batch_size):
        end = min(start + batch_size, len(events))
        yield Events(events.src[start:end], events.dst[start:end], events.edge_attr[start:end], events.timestamp[start:end], events.labels[start:end])


def epoch_sample_mask(labels: Tensor, negative_ratio: int, generator: torch.Generator) -> Tensor:
    """Select positives and a uniform stream-wide negative sample for one epoch.

    Sampling inside each temporal micro-batch silently drops every all-negative
    batch.  With rare AML labels that makes the negative distribution depend on
    where positives happen to occur.  This mask samples from the complete task
    while the caller still streams every event causally to update account state.
    """
    positives = torch.where(labels == 1)[0]
    negatives = torch.where(labels == 0)[0]
    mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    mask[positives] = True
    if negative_ratio <= 0:
        mask[negatives] = True
    elif len(positives):
        count = min(len(negatives), len(positives) * negative_ratio)
        chosen = negatives[torch.randperm(len(negatives), device=labels.device, generator=generator)[:count]]
        mask[chosen] = True
    return mask


def masked_loss(logits: Tensor, labels: Tensor, mask: Tensor, pos_weight: float = 1.0) -> Tensor | None:
    """BCE for a preselected epoch sample within one chronological batch."""
    if not bool(mask.any()):
        return None
    return F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=torch.tensor(pos_weight, device=labels.device),
    )


def train_temporal_epoch(
    model: CausalTemporalGraphSAGE, static: Tensor, events: Events, optimizer: torch.optim.Optimizer,
    sample_mask: Tensor, batch_size: int, tbptt_steps: int, pos_weight: float = 1.0,
) -> tuple[float, int]:
    """Train chronologically while retaining gradients across a bounded temporal window."""
    model.train()
    state = model.initial_state(static)
    offset = 0
    total_weighted_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    pending_loss = None
    pending_examples = 0
    optimizer.zero_grad()
    event_batches = list(batches(events, batch_size))
    for batch_index, event_batch in enumerate(event_batches, start=1):
        logits, state = model.score_and_update(state, event_batch)
        batch_mask = sample_mask[offset:offset + len(event_batch)]
        offset += len(event_batch)
        loss = masked_loss(logits, event_batch.labels, batch_mask, pos_weight)
        if loss is not None:
            selected = int(batch_mask.sum().item())
            weighted_loss = loss * selected
            pending_loss = weighted_loss if pending_loss is None else pending_loss + weighted_loss
            pending_examples += selected
            total_weighted_loss += float(loss.detach().item()) * selected
            total_examples += selected

        window_end = batch_index % tbptt_steps == 0 or batch_index == len(event_batches)
        if window_end and pending_examples:
            (pending_loss / pending_examples).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            pending_loss = None
            pending_examples = 0
        if window_end:
            state = state.detached()
    return total_weighted_loss / max(total_examples, 1), optimizer_steps


@torch.no_grad()
def score_stream(model: CausalTemporalGraphSAGE, state: TemporalState, events: Events, batch_size: int) -> tuple[np.ndarray, np.ndarray, TemporalState]:
    model.eval()
    scores, labels = [], []
    for event_batch in batches(events, batch_size):
        logits, state = model.score_and_update(state, event_batch)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(event_batch.labels.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels), state.detached()


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probabilities, np.linspace(0, 1, 501)))
    return float(candidates[int(np.argmax([f1_score(labels, probabilities >= value, zero_division=0) for value in candidates]))])


def metric_block(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def alert_budget_metrics(labels: np.ndarray, probabilities: np.ndarray, budgets: list[int]) -> dict[str, dict[str, float | int]]:
    order = np.argsort(-probabilities, kind="stable")
    positives = int(labels.sum())
    results: dict[str, dict[str, float | int]] = {}
    for budget in budgets:
        used = min(budget, len(labels))
        hits = int(labels[order[:used]].sum())
        results[str(budget)] = {
            "alerts": used, "true_positives": hits, "precision_at_k": hits / used if used else 0.0,
            "recall_at_k": hits / positives if positives else 0.0,
        }
    return results


def platt_calibrate(validation_labels: np.ndarray, validation_probabilities: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Calibrate only from validation labels; no test labels enter fitting."""
    clipped_validation = np.clip(validation_probabilities, 1e-6, 1 - 1e-6)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    validation_logits = np.log(clipped_validation / (1 - clipped_validation)).reshape(-1, 1)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(class_weight="balanced", random_state=0, max_iter=1000)
    calibrator.fit(validation_logits, validation_labels)
    return calibrator.predict_proba(logits)[:, 1]


def train_bank(
    args: argparse.Namespace, bank: str, device: torch.device,
    node_encoder: FeatureEncoder | None = None, edge_encoder: FeatureEncoder | None = None,
) -> dict[str, object]:
    bank_seed = args.seed + BANKS.index(bank) * 10_000
    set_seed(bank_seed)
    development_splits = ("training", "validation") if args.validation_only else SPLITS
    static, event_sets, node_columns, edge_columns = build_bank_data(
        args.dataset_dir, bank, node_encoder, edge_encoder, development_splits,
    )
    static = static.to(device)
    event_sets = {split: event_set.to(device) for split, event_set in event_sets.items()}
    model = CausalTemporalGraphSAGE(static.shape[1], event_sets["training"].edge_attr.shape[1], args.hidden_channels, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    labels = event_sets["training"].labels
    positives = int(labels.sum().item())
    negatives = len(labels) - positives
    ratio = 0 if args.full_training_weighted_bce else args.negative_ratio
    weight = negatives / positives if args.full_training_weighted_bce else args.loss_pos_weight
    sampled_negatives = negatives if ratio == 0 else min(negatives, positives * ratio)
    generator = torch.Generator(device=device).manual_seed(bank_seed)
    best_state: dict[str, Tensor] | None = None
    best_score, best_epoch, stale, stop_epoch = -np.inf, None, 0, None
    test_count = len(event_sets["testing"]) if "testing" in event_sets else "not loaded"
    print(f"\n{bank}: train={len(labels):,}, positives={positives:,}, validation={len(event_sets['validation']):,}, test={test_count}")
    print(f"  causal batch={args.event_batch_size}, sampled_negatives_per_epoch={sampled_negatives:,}, loss_pos_weight={weight:.6f}")
    for epoch in range(1, args.epochs + 1):
        epoch_mask = epoch_sample_mask(labels, ratio, generator)
        mean_loss, updates = train_temporal_epoch(
            model, static, event_sets["training"], optimizer, epoch_mask,
            args.event_batch_size, args.tbptt_steps, weight,
        )
        # Reconstruct memory with the frozen end-of-epoch parameters.  Reusing the
        # training-loop state would mix memories produced by many intermediate
        # parameter values, so its validation score would not reproduce after the
        # checkpoint is restored.
        selection_state = model.initial_state(static)
        _, _, selection_state = score_stream(
            model, selection_state, event_sets["training"], args.event_batch_size,
        )
        val_probs, val_labels, _ = score_stream(
            model, selection_state, event_sets["validation"], args.event_batch_size,
        )
        val_pr_auc = float(average_precision_score(val_labels, val_probs))
        if val_pr_auc > best_score:
            best_score, best_epoch, stale = val_pr_auc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch:03d} train_loss={mean_loss:.5f} optimizer_steps={updates} validation_pr_auc={val_pr_auc:.5f}")
        if stale >= args.patience:
            stop_epoch = epoch
            print(f"  early stopping at epoch {epoch}; best validation PR-AUC={best_score:.5f}")
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    # Final causal evaluation reconstructs state only from earlier events: train
    # warms validation; train+validation warm testing; no labels update memory.
    state = model.initial_state(static)
    _, _, state = score_stream(model, state, event_sets["training"], args.event_batch_size)
    val_probs, val_labels, state = score_stream(model, state, event_sets["validation"], args.event_batch_size)
    if args.calibration == "platt":
        calibrated_val_probs = platt_calibrate(val_labels, val_probs, val_probs)
    else:
        calibrated_val_probs = val_probs
    alert_budgets = [int(value) for value in args.alert_k.split(",") if value.strip()]
    if not alert_budgets or any(value < 1 for value in alert_budgets):
        raise ValueError("--alert-k must contain positive integers")
    threshold = choose_threshold(val_labels, calibrated_val_probs)
    result = {
        "bank": bank, "seed": bank_seed, "strategy": "full-training weighted BCE" if args.full_training_weighted_bce else "negative-sampled BCE",
        "training_positives": positives, "training_negatives": negatives, "sampled_negatives_per_epoch": sampled_negatives,
        "loss_pos_weight": weight, "best_validation_pr_auc": best_score, "best_validation_epoch": best_epoch,
        "early_stopping_epoch": stop_epoch, "calibration": args.calibration, "threshold": threshold,
        "validation": metric_block(val_labels, calibrated_val_probs, threshold),
        "validation_raw": metric_block(val_labels, val_probs, choose_threshold(val_labels, val_probs)),
        "validation_alert_metrics": alert_budget_metrics(val_labels, calibrated_val_probs, alert_budgets),
        "node_feature_columns": node_columns, "edge_feature_columns": edge_columns,
    }
    if args.validation_only:
        print(f"  selected validation threshold={threshold:.6f}; validation-only run")
    else:
        test_probs, test_labels, _ = score_stream(model, state, event_sets["testing"], args.event_batch_size)
        calibrated_test_probs = (
            platt_calibrate(val_labels, val_probs, test_probs) if args.calibration == "platt" else test_probs
        )
        result["testing"] = metric_block(test_labels, calibrated_test_probs, threshold)
        result["testing_raw_pr_auc"] = float(average_precision_score(test_labels, test_probs))
        result["testing_alert_metrics"] = alert_budget_metrics(test_labels, calibrated_test_probs, alert_budgets)
        print(f"  selected validation threshold={threshold:.6f}; test PR-AUC={result['testing']['pr_auc']:.5f}, "
              f"precision={result['testing']['precision']:.5f}, recall={result['testing']['recall']:.5f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args), "node_columns": node_columns, "edge_columns": edge_columns}, args.output_dir / f"{bank}_causal_temporal_graphsage.pt")
    return result


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.patience < 1 or args.event_batch_size < 1 or args.tbptt_steps < 2 or
            args.negative_ratio < 0 or args.loss_pos_weight <= 0 or args.weight_decay < 0):
        raise SystemExit("Invalid training argument")
    args.dataset_dir = args.dataset_dir.resolve()
    if not args.dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")
    device = torch.device(args.device)
    selected = BANKS if args.bank == "all" else (args.bank,)
    encoders = fit_shared_feature_encoders(args.dataset_dir, BANKS) if args.shared_encoder else (None, None)
    results = [train_bank(args, bank, device, *encoders) for bank in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved checkpoints and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
