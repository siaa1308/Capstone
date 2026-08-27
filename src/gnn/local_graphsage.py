#!/usr/bin/env python3
"""Train local GraphSAGE transaction classifiers for the simulated AML banks.

The label belongs to a transaction, so this is an *edge classification* model:
GraphSAGE produces account embeddings, then an MLP scores each source-account /
destination-account / transaction-feature tuple.  It deliberately reads labels
only from ground_truth.csv.gz and never uses the excluded columns in the dataset
feature manifest.

This is a static graph baseline.  Each split is encoded as its own graph; it does
not use labels from validation or testing.  A future continual/streaming version
must additionally prevent a transaction from seeing later edges in its split.
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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data" / "final_temporal_dataset"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "local_graphsage"
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Citi", "Fifth_Third_Bancorp", "Key_Bank")
SPLITS = ("training", "validation", "testing")
# Raw timestamps are deliberately excluded.  They identify the dataset period and
# are not needed for this static baseline; future temporal models should derive
# causal time features instead of passing a raw timestamp into the classifier.
FORBIDDEN_FEATURES = {
    "node_id", "txn_id", "timestamp", "Transaction_Date", "Transaction_Time",
    "src_id", "dst_id", "src_bank_id", "dst_bank_id", "split", "y",
    "laundering_type", "edge_label", "Is_APP_Fraud", "Is_Cheque_Fraud",
    "APP_Fraudster_ID", "Cheque_Fraudster_ID", "APP_Fraud_Sequence_Number",
    "transaction_type_raw", "transaction_type_model_safe", "From_End_Balance",
    "To_End_Balance", "Controlled_by_Criminal",
}


@dataclass
class RawGraph:
    """A single bank/split graph before feature transforms."""

    bank: str
    split: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    labels: np.ndarray


@dataclass
class GraphTensors:
    """Dense model-ready tensors for one bank/split graph."""

    x: Tensor
    message_edge_index: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> "GraphTensors":
        return GraphTensors(
            x=self.x.to(device),
            message_edge_index=self.message_edge_index.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            labels=self.labels.to(device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bank", choices=("all", *BANKS), default="all")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--negative-ratio", type=int, default=20,
        help="Negatives sampled per positive edge during training; 0 keeps all negatives.",
    )
    parser.add_argument(
        "--loss-pos-weight", type=float, default=1.0,
        help="Positive BCE weight after sampling. Keep 1 when using negative sampling.",
    )
    parser.add_argument(
        "--full-training-weighted-bce", action="store_true",
        help="Use every training edge and set BCE pos_weight to training_negatives / training_positives.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, mps, cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_raw_graph(dataset_dir: Path, bank: str, split: str) -> RawGraph:
    folder = dataset_dir / split / bank
    nodes = pd.read_csv(folder / "node_features.csv.gz")
    edges = pd.read_csv(folder / "edge_list.csv.gz")
    truth = pd.read_csv(folder / "ground_truth.csv.gz")
    required_edge = {"txn_id", "src_node", "dst_node"}
    if not required_edge.issubset(edges.columns):
        raise ValueError(f"{folder}/edge_list.csv.gz misses {required_edge - set(edges.columns)}")
    if "node_id" not in nodes or "txn_id" not in truth or "y" not in truth:
        raise ValueError(f"{folder} does not contain required node/label columns")
    if nodes["node_id"].duplicated().any() or edges["txn_id"].duplicated().any() or truth["txn_id"].duplicated().any():
        raise ValueError(f"{folder} has duplicate node_id or txn_id values")
    if not np.array_equal(nodes["node_id"].to_numpy(), np.arange(len(nodes))):
        raise ValueError(f"{folder} node IDs must be contiguous and begin at zero")
    labels = edges[["txn_id"]].merge(truth[["txn_id", "y"]], on="txn_id", how="left", validate="one_to_one")
    if labels["y"].isna().any() or len(labels) != len(truth):
        raise ValueError(f"{folder} edge/ground-truth join is not one-to-one")
    max_node = int(edges[["src_node", "dst_node"]].to_numpy().max())
    if max_node >= len(nodes) or int(edges[["src_node", "dst_node"]].to_numpy().min()) < 0:
        raise ValueError(f"{folder} edge endpoints fall outside node_features")
    return RawGraph(bank, split, nodes, edges, labels["y"].astype(np.float32).to_numpy())


class FeatureEncoder:
    """Fit categorical/numeric transforms on training data only."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.transformer: ColumnTransformer | None = None

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [c for c in self.columns if c not in numeric]
        parts: list[tuple[str, Pipeline, list[str]]] = []
        if numeric:
            parts.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            parts.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", one_hot_encoder()),
            ]), categorical))
        if not parts:
            raise ValueError("No approved model features found")
        self.transformer = ColumnTransformer(parts, remainder="drop", sparse_threshold=0)
        self.transformer.fit(frame[self.columns])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("FeatureEncoder must be fitted before transform")
        values = self.transformer.transform(frame[self.columns])
        return np.asarray(values, dtype=np.float32)


def approved_columns(dataset_dir: Path, graph: RawGraph) -> tuple[list[str], list[str]]:
    config = json.loads((dataset_dir / "configuration" / "model_feature_columns.json").read_text())
    node = [c for c in config["graph_node_safe_features"] if c in graph.nodes and c not in FORBIDDEN_FEATURES]
    edge = [c for c in config["graph_edge_safe_features"] if c in graph.edges and c not in FORBIDDEN_FEATURES]
    if not node or not edge:
        raise ValueError("Approved node or edge feature list is empty; inspect feature manifest")
    return node, edge


def tensorize(graph: RawGraph, node_encoder: FeatureEncoder, edge_encoder: FeatureEncoder) -> GraphTensors:
    x = torch.from_numpy(node_encoder.transform(graph.nodes))
    edge_attr = torch.from_numpy(edge_encoder.transform(graph.edges))
    edge_index = torch.as_tensor(graph.edges[["src_node", "dst_node"]].to_numpy().T, dtype=torch.long)
    # Bidirectional message passing lets account embeddings use both sent and received relations.
    message_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return GraphTensors(x, message_edge_index, edge_index, edge_attr, torch.from_numpy(graph.labels))


class EdgeGraphSAGE(nn.Module):
    def __init__(self, node_features: int, edge_features: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        classifier_input = hidden_channels * 3 + edge_features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, 1)
        )

    def encode(self, graph: GraphTensors) -> Tensor:
        x = self.conv1(graph.x, graph.message_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, graph.message_edge_index)

    def score_edges(self, embeddings: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = embeddings[edge_index[0]], embeddings[edge_index[1]]
        features = torch.cat((src, dst, torch.abs(src - dst), edge_attr), dim=1)
        return self.classifier(features).squeeze(1)

    def forward(self, graph: GraphTensors) -> Tensor:
        return self.score_edges(self.encode(graph), graph.edge_index, graph.edge_attr)


def sample_training_indices(labels: Tensor, negative_ratio: int, generator: torch.Generator) -> Tensor:
    positives = torch.where(labels == 1)[0]
    negatives = torch.where(labels == 0)[0]
    if len(positives) == 0:
        raise ValueError("Training graph has no positive edges")
    if negative_ratio <= 0:
        return torch.arange(len(labels), device=labels.device)
    count = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = negatives[torch.randperm(len(negatives), generator=generator, device=labels.device)[:count]]
    return torch.cat((positives, sampled_negatives))


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    # Tuning on validation only; a dense grid is stable despite very few positives.
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 501)))
    scores = [f1_score(labels, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


@torch.no_grad()
def predict(model: EdgeGraphSAGE, graph: GraphTensors) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(graph)).detach().cpu().numpy()


def train_one_bank(args: argparse.Namespace, bank: str, device: torch.device) -> dict[str, object]:
    # Make a bank run reproducible whether it is launched alone or as part of
    # `--bank all`.  Without this reset, earlier banks consume the global random
    # state and change Citi's initialization/negative samples in an all-bank run.
    bank_seed = args.seed + BANKS.index(bank) * 10_000
    set_seed(bank_seed)
    raw = {split: read_raw_graph(args.dataset_dir, bank, split) for split in SPLITS}
    node_columns, edge_columns = approved_columns(args.dataset_dir, raw["training"])
    node_encoder = FeatureEncoder(node_columns).fit(raw["training"].nodes)
    edge_encoder = FeatureEncoder(edge_columns).fit(raw["training"].edges)
    graphs = {split: tensorize(raw_graph, node_encoder, edge_encoder).to(device) for split, raw_graph in raw.items()}
    model = EdgeGraphSAGE(graphs["training"].x.shape[1], graphs["training"].edge_attr.shape[1], args.hidden_channels, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(bank_seed)
    best_state: dict[str, Tensor] | None = None
    best_pr_auc, epochs_without_improvement = -np.inf, 0
    best_epoch: int | None = None
    early_stopping_epoch: int | None = None

    train_y = graphs["training"].labels
    training_positives = int(train_y.sum().item())
    training_negatives = int(len(train_y) - training_positives)
    if args.full_training_weighted_bce:
        effective_negative_ratio = 0
        effective_pos_weight = training_negatives / training_positives
        experiment_name = "full-training weighted BCE"
    else:
        effective_negative_ratio = args.negative_ratio
        effective_pos_weight = args.loss_pos_weight
        experiment_name = "negative-sampled BCE"
    sampled_negatives = training_negatives if effective_negative_ratio == 0 else min(
        training_negatives, training_positives * effective_negative_ratio
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(effective_pos_weight, device=device))
    print(f"\n{bank}: train={len(train_y):,}, positives={int(train_y.sum()):,}, "
          f"validation={len(graphs['validation'].labels):,}, test={len(graphs['testing'].labels):,}")
    print(f"  node features={graphs['training'].x.shape[1]}, edge features={graphs['training'].edge_attr.shape[1]}, "
          f"strategy={experiment_name}, sampled_negatives_per_epoch={sampled_negatives:,}, "
          f"loss_pos_weight={effective_pos_weight:.6f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graphs["training"])
        indices = sample_training_indices(train_y, effective_negative_ratio, generator)
        loss = loss_fn(logits[indices], train_y[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_probs = predict(model, graphs["validation"])
        val_pr_auc = average_precision_score(graphs["validation"].labels.cpu().numpy(), val_probs)
        if val_pr_auc > best_pr_auc:
            best_pr_auc = float(val_pr_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch:03d} loss={loss.item():.5f} validation_pr_auc={val_pr_auc:.5f}")
        if epochs_without_improvement >= args.patience:
            early_stopping_epoch = epoch
            print(f"  early stopping at epoch {epoch}; best validation PR-AUC={best_pr_auc:.5f}")
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_probs, test_probs = predict(model, graphs["validation"]), predict(model, graphs["testing"])
    val_y, test_y = graphs["validation"].labels.cpu().numpy(), graphs["testing"].labels.cpu().numpy()
    threshold = choose_threshold(val_y, val_probs)
    result = {
        "bank": bank, "seed": bank_seed, "strategy": experiment_name,
        "training_positives": training_positives, "training_negatives": training_negatives,
        "sampled_negatives_per_epoch": sampled_negatives,
        "loss_pos_weight": effective_pos_weight, "best_validation_pr_auc": best_pr_auc,
        "best_validation_epoch": best_epoch, "early_stopping_epoch": early_stopping_epoch,
        "threshold": threshold,
        "node_feature_columns": node_columns, "edge_feature_columns": edge_columns,
        "validation": metrics(val_y, val_probs, threshold), "testing": metrics(test_y, test_probs, threshold),
    }
    print(f"  selected validation threshold={threshold:.6f}; testing PR-AUC={result['testing']['pr_auc']:.5f}, "
          f"precision={result['testing']['precision']:.5f}, recall={result['testing']['recall']:.5f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "node_columns": node_columns, "edge_columns": edge_columns, "args": vars(args)}, args.output_dir / f"{bank}_graphsage.pt")
    return result


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.negative_ratio < 0 or args.loss_pos_weight <= 0:
        raise SystemExit("epochs/patience/loss-pos-weight must be positive; negative-ratio cannot be negative")
    args.dataset_dir = args.dataset_dir.resolve()
    if not args.dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")
    set_seed(args.seed)
    device = torch.device(args.device)
    selected_banks = BANKS if args.bank == "all" else (args.bank,)
    results = [train_one_bank(args, bank, device) for bank in selected_banks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "metrics.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved checkpoints and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
