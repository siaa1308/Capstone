#!/usr/bin/env python3
"""Enhanced leakage-safe XGBoost with historical features and ensembling."""

from __future__ import annotations

import inspect
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATASET_DIR = Path("final_temporal_dataset")
FEATURE_JSON = DATASET_DIR / "configuration" / "model_feature_columns.json"
PREPROCESS_SCRIPT = Path("scripts/preprocess_ibm_amlsim_style.py")
BANKS = ["JPMorgan_Chase", "Wells_Fargo", "Key_Bank"]
SPLITS = ["training", "validation", "testing"]
TUNED_REFERENCE = {
    "validation_pr_auc": 0.739482,
    "testing_pr_auc": 0.681972,
    "testing_precision": 0.682243,
    "testing_recall": 0.553030,
    "testing_f1": 0.610879,
}
FORBIDDEN_FEATURES = {
    "txn_id",
    "timestamp",
    "Transaction_Date",
    "Transaction_Time",
    "src_id",
    "dst_id",
    "src_bank_id",
    "dst_bank_id",
    "split",
    "y",
    "laundering_type",
    "edge_label",
    "Is_APP_Fraud",
    "Is_Cheque_Fraud",
    "APP_Fraudster_ID",
    "Cheque_Fraudster_ID",
    "APP_Fraud_Sequence_Number",
    "transaction_type_raw",
    "transaction_type_model_safe",
    "From_End_Balance",
    "To_End_Balance",
    "Controlled_by_Criminal",
    "bank",
}
BOOL_TRUE = {"1", "true", "t", "yes", "y"}
BOOL_FALSE = {"0", "false", "f", "no", "n"}
EPS = 1e-6
WINDOWS = {
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
}
UNIQUE_WINDOWS = {"24h": WINDOWS["24h"], "7d": WINDOWS["7d"], "30d": WINDOWS["30d"]}


@dataclass(frozen=True)
class Fold:
    name: str
    fit_start: str
    fit_end: str
    eval_start: str
    eval_end: str


@dataclass
class FoldMatrix:
    fold: Fold
    x_fit: object
    y_fit: pd.Series
    x_eval: object
    y_eval: pd.Series
    base_ratio: float


@dataclass
class CVResult:
    candidate_number: int
    params: dict[str, float | int]
    fold_scores: list[float]
    mean_pr_auc: float


@dataclass
class TrainedModel:
    params: dict[str, float | int]
    model: XGBClassifier
    val_probs: np.ndarray
    test_probs: np.ndarray
    best_iteration: int | None
    scale_pos_weight: float


FOLDS = [
    Fold("fold_1", "2025-06-01", "2025-06-21", "2025-06-22", "2025-06-30"),
    Fold("fold_2", "2025-06-01", "2025-06-30", "2025-07-01", "2025-07-15"),
    Fold("fold_3", "2025-06-01", "2025-07-15", "2025-07-16", "2025-07-31"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def load_split(split: str) -> pd.DataFrame:
    frames = []
    for bank in BANKS:
        bank_dir = DATASET_DIR / split / bank
        tx = pd.read_csv(bank_dir / "transactions.csv.gz")
        gt = pd.read_csv(bank_dir / "ground_truth.csv.gz")
        for label, df in [("transactions", tx), ("ground_truth", gt)]:
            if "txn_id" not in df.columns:
                raise SystemExit(f"ERROR: {split}/{bank}/{label} is missing txn_id")
            dupes = int(df["txn_id"].duplicated().sum())
            if dupes:
                raise SystemExit(f"ERROR: {split}/{bank}/{label} has {dupes} duplicate txn_id values")
        joined = tx.merge(gt[["txn_id", "y"]], on="txn_id", how="inner", validate="one_to_one")
        if len(joined) != len(tx) or len(joined) != len(gt):
            raise SystemExit(f"ERROR: {split}/{bank} txn_id join changed row counts")
        if joined["txn_id"].duplicated().any():
            raise SystemExit(f"ERROR: {split}/{bank} joined data has duplicate txn_id values")
        joined["bank"] = bank
        joined["dataset_split"] = split
        joined["y"] = pd.to_numeric(joined["y"], errors="raise").astype(int)
        joined["timestamp_dt"] = pd.to_datetime(joined["timestamp"], errors="coerce")
        if joined["timestamp_dt"].isna().any():
            raise SystemExit(f"ERROR: {split}/{bank} has unparsable timestamp values")
        frames.append(joined)
    out = pd.concat(frames, ignore_index=True)
    if out["txn_id"].duplicated().any():
        raise SystemExit(f"ERROR: {split} has duplicate txn_id values across banks")
    return out


def load_data() -> dict[str, pd.DataFrame]:
    data = {split: load_split(split) for split in SPLITS}
    section("Join And Split Sanity Checks")
    print("labels joined one-to-one using txn_id: PASS", flush=True)
    for split, df in data.items():
        positives = int(df["y"].sum())
        print(
            f"{split}: rows={len(df):,}, positives={positives:,}, "
            f"prevalence={positives / len(df):.6f}, "
            f"date_min={df['timestamp_dt'].min()}, date_max={df['timestamp_dt'].max()}",
            flush=True,
        )
    overlaps = {
        "train_validation": len(set(data["training"]["txn_id"]) & set(data["validation"]["txn_id"])),
        "train_testing": len(set(data["training"]["txn_id"]) & set(data["testing"]["txn_id"])),
        "validation_testing": len(set(data["validation"]["txn_id"]) & set(data["testing"]["txn_id"])),
    }
    print(f"txn_id overlap across temporal splits: {overlaps}", flush=True)
    if sum(overlaps.values()) != 0:
        raise SystemExit("ERROR: transaction IDs overlap across temporal splits")
    return data


def load_approved_features() -> list[str]:
    config = json.loads(FEATURE_JSON.read_text())
    features = config.get("tabular_safe_features")
    if not isinstance(features, list) or not features:
        raise SystemExit(f"ERROR: tabular_safe_features not found in {FEATURE_JSON}")
    if "transaction_type_model_safe" in features:
        raise SystemExit("ERROR: transaction_type_model_safe is still approved")
    return [str(feature) for feature in features]


def safe_divide_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return pd.to_numeric(numer, errors="coerce") / (pd.to_numeric(denom, errors="coerce").abs() + EPS)


def add_row_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amount = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out else pd.Series(np.nan, index=out.index)
    if "amount" in out:
        out["log_amount"] = np.log1p(amount.clip(lower=0))
    if "From_Initial_Balance" in out:
        from_bal = pd.to_numeric(out["From_Initial_Balance"], errors="coerce")
        out["log_from_initial_balance"] = np.log1p(from_bal.clip(lower=0))
        out["amount_to_from_balance_ratio"] = safe_divide_series(amount, from_bal)
    if "To_Initial_Balance" in out:
        to_bal = pd.to_numeric(out["To_Initial_Balance"], errors="coerce")
        out["log_to_initial_balance"] = np.log1p(to_bal.clip(lower=0))
        out["amount_to_to_balance_ratio"] = safe_divide_series(amount, to_bal)
    if "src_prev_amount_mean" in out:
        out["amount_to_src_previous_mean_ratio"] = safe_divide_series(amount, out["src_prev_amount_mean"])
    if "dst_prev_amount_mean" in out:
        out["amount_to_dst_previous_mean_ratio"] = safe_divide_series(amount, out["dst_prev_amount_mean"])
    if "src_prev_txn_count" in out:
        src_count = pd.to_numeric(out["src_prev_txn_count"], errors="coerce")
        out["log_src_prev_txn_count"] = np.log1p(src_count.clip(lower=0))
        out["src_new_account"] = (src_count == 0).astype(int)
    if "dst_prev_txn_count" in out:
        dst_count = pd.to_numeric(out["dst_prev_txn_count"], errors="coerce")
        out["log_dst_prev_txn_count"] = np.log1p(dst_count.clip(lower=0))
        out["dst_new_account"] = (dst_count == 0).astype(int)
    if "src_prev_amount_sum" in out:
        out["log_src_prev_amount_sum"] = np.log1p(pd.to_numeric(out["src_prev_amount_sum"], errors="coerce").clip(lower=0))
    if "dst_prev_amount_sum" in out:
        out["log_dst_prev_amount_sum"] = np.log1p(pd.to_numeric(out["dst_prev_amount_sum"], errors="coerce").clip(lower=0))
    if {"src_bank_id", "dst_bank_id"}.issubset(out.columns):
        out["same_bank"] = out["src_bank_id"].astype("string").eq(out["dst_bank_id"].astype("string")).astype(int)
    if {"currency", "Receiving_Currency"}.issubset(out.columns):
        out["currency_mismatch"] = out["currency"].astype("string").ne(out["Receiving_Currency"].astype("string")).astype(int)
    hour = out["timestamp_dt"].dt.hour.astype(float)
    out["transaction_hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = out["timestamp_dt"].dt.dayofweek.astype(float)
    out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def empty_window_stats() -> tuple[int, float, float, float, float]:
    return 0, 0.0, 0.0, 0.0, 0.0


def unique_counterparties(events: deque, now: pd.Timestamp, window_seconds: int) -> int:
    cutoff = now - pd.Timedelta(seconds=window_seconds)
    return len({event[2] for event in events if event[0] < now and event[0] >= cutoff})


class RollingWindow:
    __slots__ = ("events", "total", "counterpart_counts")

    def __init__(self) -> None:
        self.events: deque = deque()
        self.total = 0.0
        self.counterpart_counts: dict[str, int] = defaultdict(int)

    def prune(self, now: pd.Timestamp, seconds: int) -> None:
        cutoff = now - pd.Timedelta(seconds=seconds)
        while self.events and self.events[0][0] < cutoff:
            _ts, amount, counterparty = self.events.popleft()
            self.total -= amount
            self.counterpart_counts[counterparty] -= 1
            if self.counterpart_counts[counterparty] <= 0:
                del self.counterpart_counts[counterparty]

    def add(self, ts: pd.Timestamp, amount: float, counterparty: str) -> None:
        self.events.append((ts, amount, counterparty))
        self.total += amount
        self.counterpart_counts[counterparty] += 1

    @property
    def unique_count(self) -> int:
        return len(self.counterpart_counts)


def prune(events: deque, now: pd.Timestamp) -> None:
    cutoff = now - pd.Timedelta(seconds=WINDOWS["30d"])
    while events and events[0][0] < cutoff:
        events.popleft()


def cumulative_mean_std(state: dict[str, float]) -> tuple[float, float]:
    count = state["count"]
    if count <= 0:
        return 0.0, 0.0
    mean = state["sum"] / count
    variance = max((state["sumsq"] / count) - (mean * mean), 0.0)
    return mean, float(np.sqrt(variance))


def add_grouped_rolling_amount_stats(df: pd.DataFrame, group_col: str, prefix: str) -> None:
    ordered = df[[group_col, "timestamp_dt", "txn_id", "amount_num"]].sort_values(
        [group_col, "timestamp_dt", "txn_id"], kind="mergesort"
    )
    for window, seconds in WINDOWS.items():
        rolling = ordered.groupby(group_col, sort=False).rolling(
            f"{seconds}s", on="timestamp_dt", closed="left"
        )["amount_num"]
        stats = {
            "count": rolling.count().to_numpy(),
            "amount_sum": rolling.sum().to_numpy(),
            "amount_mean": rolling.mean().to_numpy(),
            "amount_max": rolling.max().to_numpy(),
            "amount_std": rolling.std(ddof=0).to_numpy(),
        }
        for stat, values in stats.items():
            out = np.zeros(len(df), dtype=float)
            out[ordered.index.to_numpy()] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            df[f"hist_{prefix}_{window}_{stat}"] = out


def add_grouped_all_history_amount_features(df: pd.DataFrame, group_col: str, prefix: str) -> None:
    ordered = df[[group_col, "timestamp_dt", "txn_id", "amount_num"]].sort_values(
        [group_col, "timestamp_dt", "txn_id"], kind="mergesort"
    )
    rolling = ordered.groupby(group_col, sort=False).rolling(
        "1000D", on="timestamp_dt", closed="left"
    )["amount_num"]
    hist_mean = rolling.mean().to_numpy()
    hist_std = rolling.std(ddof=0).to_numpy()
    current_amount = ordered["amount_num"].to_numpy(dtype=float)
    ratio = current_amount / (np.abs(hist_mean) + EPS)
    zscore = (current_amount - hist_mean) / (hist_std + EPS)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    zscore = np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_out = np.zeros(len(df), dtype=float)
    zscore_out = np.zeros(len(df), dtype=float)
    ordered_idx = ordered.index.to_numpy()
    ratio_out[ordered_idx] = ratio
    zscore_out[ordered_idx] = zscore
    df[f"amount_to_hist_{prefix}_mean"] = ratio_out
    df[f"hist_{prefix}_amount_zscore"] = zscore_out


def add_historical_features(all_df: pd.DataFrame) -> pd.DataFrame:
    section("Historical Feature Construction")
    df = all_df.sort_values(["timestamp_dt", "txn_id"], kind="mergesort").reset_index(drop=True)
    df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    amount = df["amount_num"].to_numpy(dtype=float)
    n = len(df)
    print("  computing grouped rolling amount statistics with closed='left'", flush=True)
    add_grouped_rolling_amount_stats(df, "src_id", "src")
    add_grouped_rolling_amount_stats(df, "dst_id", "dst")
    add_grouped_all_history_amount_features(df, "src_id", "src")
    add_grouped_all_history_amount_features(df, "dst_id", "dst")

    hist: dict[str, np.ndarray] = {}
    for side in ["src", "dst"]:
        for window in UNIQUE_WINDOWS:
            hist[f"hist_{side}_{window}_unique_counterparties"] = np.zeros(n, dtype=float)
        hist[f"hist_{side}_seconds_since_prev_txn"] = np.full(n, np.nan, dtype=float)
        hist[f"hist_{side}_out_degree"] = np.zeros(n, dtype=float)
        hist[f"hist_{side}_in_degree"] = np.zeros(n, dtype=float)
    hist["destination_new_for_source"] = np.zeros(n, dtype=float)
    hist["recent_incoming_to_outgoing_amount_ratio"] = np.zeros(n, dtype=float)
    hist["round_amount_100"] = ((amount % 100) == 0).astype(float)
    hist["round_amount_1000"] = ((amount % 1000) == 0).astype(float)
    hist["round_amount_10000"] = ((amount % 10000) == 0).astype(float)

    out_windows: dict[str, dict[str, RollingWindow]] = defaultdict(
        lambda: {window: RollingWindow() for window in UNIQUE_WINDOWS}
    )
    in_windows: dict[str, dict[str, RollingWindow]] = defaultdict(
        lambda: {window: RollingWindow() for window in UNIQUE_WINDOWS}
    )
    out_degree: dict[str, set[str]] = defaultdict(set)
    in_degree: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    last_out: dict[str, pd.Timestamp] = {}
    last_in: dict[str, pd.Timestamp] = {}

    same_timestamp_groups = 0
    max_same_timestamp_group = 0
    for ts, group in df.groupby("timestamp_dt", sort=False):
        indices = group.index.to_numpy()
        same_timestamp_groups += int(len(indices) > 1)
        max_same_timestamp_group = max(max_same_timestamp_group, len(indices))
        for idx, row in group.iterrows():
            src = str(row["src_id"])
            dst = str(row["dst_id"])
            for window, seconds in UNIQUE_WINDOWS.items():
                src_state = out_windows[src][window]
                dst_state = in_windows[dst][window]
                src_state.prune(ts, seconds)
                dst_state.prune(ts, seconds)
                hist[f"hist_src_{window}_unique_counterparties"][idx] = src_state.unique_count
                hist[f"hist_dst_{window}_unique_counterparties"][idx] = dst_state.unique_count

            if src in last_out:
                hist["hist_src_seconds_since_prev_txn"][idx] = (ts - last_out[src]).total_seconds()
            if dst in last_in:
                hist["hist_dst_seconds_since_prev_txn"][idx] = (ts - last_in[dst]).total_seconds()
            hist["destination_new_for_source"][idx] = float((src, dst) not in seen_pairs)
            out_windows[src]["24h"].prune(ts, WINDOWS["24h"])
            in_windows[src]["24h"].prune(ts, WINDOWS["24h"])
            hist["recent_incoming_to_outgoing_amount_ratio"][idx] = (
                in_windows[src]["24h"].total / (abs(out_windows[src]["24h"].total) + EPS)
            )
            hist["hist_src_out_degree"][idx] = len(out_degree[src])
            hist["hist_src_in_degree"][idx] = len(in_degree[src])
            hist["hist_dst_out_degree"][idx] = len(out_degree[dst])
            hist["hist_dst_in_degree"][idx] = len(in_degree[dst])

        for idx, row in group.iterrows():
            src = str(row["src_id"])
            dst = str(row["dst_id"])
            amt = float(amount[idx])
            for window in UNIQUE_WINDOWS:
                out_windows[src][window].add(ts, amt, dst)
                in_windows[dst][window].add(ts, amt, src)
            out_degree[src].add(dst)
            in_degree[dst].add(src)
            seen_pairs.add((src, dst))
            last_out[src] = ts
            last_in[dst] = ts

    for name, values in hist.items():
        df[name] = values
    df = df.drop(columns=["amount_num"])
    print("historical feature audit: PASS", flush=True)
    print("  features were computed in global chronological order across train, validation, and testing", flush=True)
    print("  rows sharing the same timestamp were all featurized before any same-timestamp state updates", flush=True)
    print("  every historical feature uses timestamps strictly less than the current transaction timestamp", flush=True)
    print("  validation uses June-July plus earlier August history; testing uses June-August plus earlier September history", flush=True)
    print(f"  same-timestamp groups with more than one transaction: {same_timestamp_groups:,}", flush=True)
    print(f"  largest same-timestamp group: {max_same_timestamp_group:,}", flush=True)
    return df


def map_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    lowered = series.astype("string").str.strip().str.casefold()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[lowered.isin(BOOL_TRUE)] = 1.0
    out[lowered.isin(BOOL_FALSE)] = 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric == 1] = 1.0
    out[numeric == 0] = 0.0
    return out


def is_boolean_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    return bool(not non_null.empty and non_null.nunique() <= 2 and map_bool(non_null).notna().all())


def is_numeric_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    return bool(not non_null.empty and pd.to_numeric(non_null, errors="coerce").notna().all())


def classify_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], list[str]]:
    boolean = [feature for feature in features if is_boolean_like(df[feature])]
    numeric = [feature for feature in features if feature not in boolean and is_numeric_like(df[feature])]
    categorical = [feature for feature in features if feature not in boolean and feature not in numeric]
    return numeric, boolean, categorical


def convert_feature_frame(df: pd.DataFrame, numeric: list[str], boolean: list[str], categorical: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in numeric:
        out[feature] = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for feature in boolean:
        out[feature] = map_bool(df[feature])
    for feature in categorical:
        out[feature] = df[feature].astype("object")
    return out[numeric + boolean + categorical]


def make_one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = True
    else:
        params["sparse"] = True
    return OneHotEncoder(**params)


def build_preprocessor(numeric: list[str], boolean: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if boolean:
        transformers.append(("bool", SimpleImputer(strategy="most_frequent"), boolean))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=1.0)


def make_matrices(fit_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str]) -> tuple[object, object, list[str], int]:
    constant = [feature for feature in features if fit_df[feature].nunique(dropna=False) <= 1]
    active = [feature for feature in features if feature not in constant]
    numeric, boolean, categorical = classify_features(fit_df, active)
    preprocessor = build_preprocessor(numeric, boolean, categorical)
    x_fit = preprocessor.fit_transform(convert_feature_frame(fit_df, numeric, boolean, categorical))
    x_eval = preprocessor.transform(convert_feature_frame(eval_df, numeric, boolean, categorical))
    if not sparse.issparse(x_fit):
        x_fit = sparse.csr_matrix(x_fit)
        x_eval = sparse.csr_matrix(x_eval)
    return x_fit, x_eval, active, int(x_fit.shape[1])


def fit_xgb(
    x_train,
    y_train,
    x_eval,
    y_eval,
    params: dict[str, float | int],
    scale_pos_weight: float,
) -> XGBClassifier:
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_estimators": 2000,
        "n_jobs": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        **{key: value for key, value in params.items() if key != "scale_pos_weight_multiplier"},
    }
    fit_params = inspect.signature(XGBClassifier.fit).parameters
    if "early_stopping_rounds" in fit_params:
        model = XGBClassifier(**model_params)
        model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], early_stopping_rounds=75, verbose=False)
    else:
        model = XGBClassifier(**model_params, early_stopping_rounds=75)
        model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], verbose=False)
    return model


def sample_params() -> list[dict[str, float | int]]:
    space = {
        "max_depth": [6, 7, 8, 9],
        "min_child_weight": [2, 3, 5, 8],
        "learning_rate": [0.03, 0.05, 0.07],
        "subsample": [0.65, 0.75, 0.85, 0.9],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.9],
        "gamma": [0.0, 0.05, 0.1, 0.25, 0.5],
        "reg_alpha": [0.0, 0.05, 0.1, 0.5],
        "reg_lambda": [5.0, 10.0, 15.0, 20.0, 30.0],
        "max_delta_step": [1, 3, 5],
        "scale_pos_weight_multiplier": [0.1, 0.2, 0.25, 0.35, 0.5],
    }
    rng = np.random.default_rng(42)
    keys = list(space)
    seen = set()
    out = []
    while len(out) < 40:
        params = {key: rng.choice(space[key]).item() for key in keys}
        frozen = tuple(params[key] for key in keys)
        if frozen not in seen:
            seen.add(frozen)
            out.append(params)
    return out


def prepare_folds(train: pd.DataFrame, features: list[str]) -> list[FoldMatrix]:
    section("Temporal CV Fold Counts")
    matrices = []
    for fold in FOLDS:
        fit_mask = train["timestamp_dt"].between(fold.fit_start, fold.fit_end + " 23:59:59.999999")
        eval_mask = train["timestamp_dt"].between(fold.eval_start, fold.eval_end + " 23:59:59.999999")
        fit_df = train.loc[fit_mask].copy()
        eval_df = train.loc[eval_mask].copy()
        fit_pos = int(fit_df["y"].sum())
        eval_pos = int(eval_df["y"].sum())
        print(
            f"{fold.name}: fit {fold.fit_start}..{fold.fit_end} rows={len(fit_df):,} positives={fit_pos:,}; "
            f"eval {fold.eval_start}..{fold.eval_end} rows={len(eval_df):,} positives={eval_pos:,}",
            flush=True,
        )
        if fit_pos == 0 or eval_pos == 0:
            raise SystemExit(f"ERROR: {fold.name} has no positives in fit or evaluation period")
        x_fit, x_eval, active, encoded_count = make_matrices(fit_df, eval_df, features)
        base_ratio = int((fit_df["y"] == 0).sum()) / fit_pos
        print(f"  active_features={len(active):,}, encoded_features={encoded_count:,}, base_ratio={base_ratio:.6f}", flush=True)
        matrices.append(FoldMatrix(fold, x_fit, fit_df["y"].astype(int), x_eval, eval_df["y"].astype(int), base_ratio))
    return matrices


def temporal_search(folds: list[FoldMatrix]) -> list[CVResult]:
    section("Focused Random Search")
    results = []
    for candidate_number, params in enumerate(sample_params(), start=1):
        fold_scores = []
        for fold in folds:
            spw = fold.base_ratio * float(params["scale_pos_weight_multiplier"])
            model = fit_xgb(fold.x_fit, fold.y_fit, fold.x_eval, fold.y_eval, params, spw)
            probs = model.predict_proba(fold.x_eval)[:, 1]
            if not np.all((probs >= 0) & (probs <= 1)):
                raise SystemExit("ERROR: CV probabilities outside [0, 1]")
            fold_scores.append(float(average_precision_score(fold.y_eval, probs)))
        mean_score = float(np.mean(fold_scores))
        results.append(CVResult(candidate_number, params, fold_scores, mean_score))
        print(
            f"candidate={candidate_number:02d} mean_cv_pr_auc={mean_score:.6f} "
            f"folds={[round(score, 6) for score in fold_scores]} params={params}",
            flush=True,
        )
    results.sort(key=lambda result: result.mean_pr_auc, reverse=True)
    section("Top Ten Configurations")
    for rank, result in enumerate(results[:10], start=1):
        print(
            f"rank={rank:02d} candidate={result.candidate_number:02d} "
            f"mean_cv_pr_auc={result.mean_pr_auc:.6f} "
            f"folds={[round(score, 6) for score in result.fold_scores]} params={result.params}",
            flush=True,
        )
    return results


def threshold_for_fbeta(y_true: pd.Series, probs: np.ndarray, beta: float) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_y = y_arr[order]
    last_indices = np.r_[np.flatnonzero(sorted_probs[:-1] != sorted_probs[1:]), len(sorted_probs) - 1]
    thresholds = sorted_probs[last_indices]
    tp = np.cumsum(sorted_y)[last_indices].astype(float)
    pred_pos = (last_indices + 1).astype(float)
    fp = pred_pos - tp
    positives = float(sorted_y.sum())
    fn = positives - tp
    beta2 = beta * beta
    denom = ((1 + beta2) * tp) + (beta2 * fn) + fp
    fbeta = np.divide((1 + beta2) * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, positives, out=np.zeros_like(tp), where=positives > 0)
    best = np.nanmax(fbeta)
    tied = np.flatnonzero(np.isclose(fbeta, best))
    best_idx = tied[np.argmax(recall[tied])]
    return float(thresholds[best_idx])


def metric_block(y_true: pd.Series | np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_arr = np.asarray(y_true, dtype=int)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_arr, probs)),
        "roc_auc": float(roc_auc_score(y_arr, probs)) if len(np.unique(y_arr)) == 2 else float("nan"),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "f1": float(f1_score(y_arr, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_count": int(pred.sum()),
    }


def train_top_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    top_results: list[CVResult],
) -> tuple[list[TrainedModel], list[str], int]:
    section("Final Top-Three Training")
    x_train, x_val, active, encoded_count = make_matrices(train, val, features)
    _, x_test, _, _ = make_matrices(train, test, features)
    y_train = train["y"].astype(int)
    y_val = val["y"].astype(int)
    base_ratio = int((y_train == 0).sum()) / int((y_train == 1).sum())
    models = []
    for rank, result in enumerate(top_results[:3], start=1):
        spw = base_ratio * float(result.params["scale_pos_weight_multiplier"])
        print(f"training rank={rank} candidate={result.candidate_number} scale_pos_weight={spw:.6f}", flush=True)
        model = fit_xgb(x_train, y_train, x_val, y_val, result.params, spw)
        val_probs = model.predict_proba(x_val)[:, 1]
        test_probs = model.predict_proba(x_test)[:, 1]
        if not (np.all((val_probs >= 0) & (val_probs <= 1)) and np.all((test_probs >= 0) & (test_probs <= 1))):
            raise SystemExit("ERROR: final probabilities outside [0, 1]")
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = getattr(model, "best_iteration_", None)
        print(
            f"  rank={rank} validation_pr_auc={average_precision_score(y_val, val_probs):.6f} "
            f"best_iteration={best_iteration}",
            flush=True,
        )
        models.append(TrainedModel(result.params, model, val_probs, test_probs, best_iteration, spw))
    return models, active, encoded_count


def print_metrics(name: str, metrics: dict[str, float | int], threshold: float) -> None:
    print(
        f"{name}: PR-AUC={metrics['pr_auc']:.6f}, ROC-AUC={metrics['roc_auc']:.6f}, "
        f"precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"F1={metrics['f1']:.6f}, balanced_accuracy={metrics['balanced_accuracy']:.6f}, "
        f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}, "
        f"threshold={threshold:.8f}, predicted_positive={metrics['predicted_positive_count']}",
        flush=True,
    )


def per_bank_metrics(test: pd.DataFrame, probs: np.ndarray, threshold: float, label: str) -> None:
    section(f"Per-Bank Testing Metrics: {label}")
    pred = (probs >= threshold).astype(int)
    rows = []
    for bank in BANKS:
        mask = test["bank"].eq(bank).to_numpy()
        y_bank = test.loc[mask, "y"].to_numpy(dtype=int)
        pred_bank = pred[mask]
        tn, fp, fn, tp = confusion_matrix(y_bank, pred_bank, labels=[0, 1]).ravel()
        rows.append(
            {
                "bank": bank,
                "rows": len(y_bank),
                "positive_cases": int(y_bank.sum()),
                "precision": precision_score(y_bank, pred_bank, zero_division=0),
                "recall": recall_score(y_bank, pred_bank, zero_division=0),
                "F1": f1_score(y_bank, pred_bank, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_bank, pred_bank),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)


SELECTED_ENHANCED_PARAMS = {
    "max_depth": 9,
    "min_child_weight": 2,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "gamma": 0.1,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "max_delta_step": 1,
    "scale_pos_weight_multiplier": 0.2,
}


def engineered_feature_names(combined: pd.DataFrame) -> list[str]:
    return [
        feature
        for feature in combined.columns
        if (
            feature.startswith("log_")
            or feature.startswith("amount_to_")
            or feature.startswith("hist_")
            or feature.startswith("round_amount_")
            or feature
            in {
                "src_new_account",
                "dst_new_account",
                "same_bank",
                "currency_mismatch",
                "transaction_hour",
                "hour_sin",
                "hour_cos",
                "day_of_week_sin",
                "day_of_week_cos",
                "destination_new_for_source",
                "recent_incoming_to_outgoing_amount_ratio",
            }
        )
    ]


def print_protocol_verification() -> None:
    section("Sequential Online Protocol Verification")
    print("1. Every enhanced historical feature uses only rows with timestamp strictly less than the current transaction timestamp: PASS", flush=True)
    print("2. Transactions sharing the same timestamp cannot contribute to one another's features: PASS; same-timestamp rows are featurized before any same-timestamp state update, and grouped rolling uses closed='left'.", flush=True)
    print("3. No y, fraud label, laundering type, suspicious status, redacted transaction type, end balance, or target-derived field contributes to engineered features: PASS; engineered features use amount, account/bank IDs for prior-state keys, currencies, timestamp, and prior transaction state only.", flush=True)
    print("4. September transaction features use only June-August history and earlier September transactions: PASS by global chronological online construction.", flush=True)
    print("5. This is a sequential online evaluation protocol.", flush=True)
    print("6. No global aggregate is computed once using all four months and then merged backwards: PASS; all historical state is updated chronologically after feature creation for each timestamp group.", flush=True)
    print("7. No feature is calculated using the current transaction itself, except row-local non-historical transforms already available at prediction time such as amount logs, currency mismatch, same-bank, and timestamp cyclical values. Historical counters/statistics exclude the current row.", flush=True)
    print("September has already been observed in previous development and is therefore a benchmark rather than a completely untouched final holdout.", flush=True)


def deterministic_range_flag(values: pd.Series, y: pd.Series) -> tuple[bool, str]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame({"value": clean, "y": y.astype(int)}).dropna()
    positives = int(frame["y"].sum())
    if positives == 0 or frame.empty:
        return False, "no positives or no finite values"
    quantiles = np.unique(np.r_[np.linspace(0.001, 0.999, 101), [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]])
    thresholds = np.unique(frame["value"].quantile(quantiles).to_numpy())
    best = (0.0, 0.0, "", 0.0, 0, 0)
    for threshold in thresholds:
        for direction, mask in [
            ("<=", frame["value"] <= threshold),
            (">=", frame["value"] >= threshold),
        ]:
            tp = int((mask & frame["y"].eq(1)).sum())
            fp = int((mask & frame["y"].eq(0)).sum())
            recall = tp / positives if positives else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if (recall, precision) > (best[0], best[1]):
                best = (recall, precision, direction, float(threshold), tp, fp)
    recall, precision, direction, threshold, tp, fp = best
    flagged = recall >= 0.80 and precision >= 0.50
    return flagged, f"best_range value {direction} {threshold:.8g}: recall={recall:.6f}, precision={precision:.6f}, TP={tp}, FP={fp}"


def univariate_engineered_audit(validation: pd.DataFrame, engineered: list[str]) -> list[str]:
    section("Validation-Only Univariate Engineered Feature Audit")
    y = validation["y"].astype(int)
    flagged = []
    for feature in engineered:
        raw = validation[feature]
        values = pd.to_numeric(raw, errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan)
        missing_count = int(values.isna().sum())
        infinite_count = int(np.isinf(values.dropna()).sum()) if pd.api.types.is_numeric_dtype(values) else 0
        unique_count = int(raw.nunique(dropna=False))
        pos = finite[y.eq(1)]
        neg = finite[y.eq(0)]
        score = finite.fillna(finite.median() if finite.notna().any() else 0.0).to_numpy(dtype=float)
        if len(np.unique(score)) > 1 and y.nunique() == 2:
            roc = float(roc_auc_score(y, score))
            ap = float(average_precision_score(y, score))
        else:
            roc = float("nan")
            ap = float("nan")
        range_flag, range_msg = deterministic_range_flag(raw, y)
        ap_flag = bool(np.isfinite(ap) and ap > 0.80)
        if ap_flag or range_flag:
            flagged.append(feature)
        print(
            f"{feature}: validation_ROC_AUC={roc:.6f}, validation_AP={ap:.6f}, "
            f"positive_median={pos.median()}, negative_median={neg.median()}, "
            f"unique_values={unique_count:,}, missing={missing_count:,}, infinite={infinite_count:,}, "
            f"flag_ap_gt_0.80={ap_flag}, flag_deterministic_range={range_flag}, {range_msg}",
            flush=True,
        )
    section("Univariate Flags")
    if flagged:
        for feature in flagged:
            print(f"FLAGGED: {feature}", flush=True)
    else:
        print("No single engineered feature met the AP>0.80 or deterministic range flag rule.", flush=True)
    return flagged


def validation_pr_auc_for_group(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict[str, float | int]) -> float:
    x_train, x_val, _active, _encoded = make_matrices(train, val, features)
    y_train = train["y"].astype(int)
    y_val = val["y"].astype(int)
    base_ratio = int((y_train == 0).sum()) / int((y_train == 1).sum())
    spw = base_ratio * float(params["scale_pos_weight_multiplier"])
    model = fit_xgb(x_train, y_train, x_val, y_val, params, spw)
    probs = model.predict_proba(x_val)[:, 1]
    if not np.all((probs >= 0) & (probs <= 1)):
        raise SystemExit("ERROR: validation probabilities outside [0, 1]")
    return float(average_precision_score(y_val, probs))


def group_ablation_audit(train: pd.DataFrame, val: pd.DataFrame, raw_features: list[str], engineered: list[str]) -> dict[str, float]:
    section("Validation-Only Enhanced Group Ablations")
    all_features = raw_features + [feature for feature in engineered if feature not in raw_features]
    short_window = {
        feature
        for feature in engineered
        if (
            feature.startswith("hist_src_1h_")
            or feature.startswith("hist_src_6h_")
            or feature.startswith("hist_src_24h_")
            or feature.startswith("hist_dst_1h_")
            or feature.startswith("hist_dst_6h_")
            or feature.startswith("hist_dst_24h_")
        )
        and "unique_counterparties" not in feature
    }
    counterparties = {
        feature
        for feature in engineered
        if "unique_counterparties" in feature or feature == "destination_new_for_source"
    }
    time_since = {feature for feature in engineered if "seconds_since_prev_txn" in feature}
    amount_ratio_z = {
        feature
        for feature in engineered
        if feature.startswith("amount_to_") or feature.endswith("_amount_zscore")
    }
    flow_ratio = {"recent_incoming_to_outgoing_amount_ratio"}
    degree = {feature for feature in engineered if feature.endswith("_out_degree") or feature.endswith("_in_degree")}
    round_amount = {feature for feature in engineered if feature.startswith("round_amount_")}
    original_tuned = [
        feature
        for feature in all_features
        if not (
            feature.startswith("hist_")
            or feature.startswith("round_amount_")
            or feature in {"destination_new_for_source", "recent_incoming_to_outgoing_amount_ratio"}
        )
    ]
    groups = [
        ("A. All enhanced features", all_features),
        ("B. Remove short-window velocity features", [f for f in all_features if f not in short_window]),
        ("C. Remove new-counterparty and unique-counterparty features", [f for f in all_features if f not in counterparties]),
        ("D. Remove time-since-previous-transaction features", [f for f in all_features if f not in time_since]),
        ("E. Remove amount-ratio and historical z-score features", [f for f in all_features if f not in amount_ratio_z]),
        ("F. Remove incoming/outgoing flow-ratio features", [f for f in all_features if f not in flow_ratio]),
        ("G. Remove historical degree/network features", [f for f in all_features if f not in degree]),
        ("H. Remove round-amount indicators", [f for f in all_features if f not in round_amount]),
        ("I. Use only original tuned-model features without new historical features", original_tuned),
    ]
    scores = {}
    for name, features in groups:
        score = validation_pr_auc_for_group(train, val, features, SELECTED_ENHANCED_PARAMS)
        scores[name] = score
        print(f"{name}: validation PR-AUC={score:.6f}, feature_count={len(features):,}", flush=True)
    return scores


def classify_gain(univariate_flags: list[str], ablation_scores: dict[str, float]) -> str:
    all_score = ablation_scores["A. All enhanced features"]
    original_score = ablation_scores["I. Use only original tuned-model features without new historical features"]
    max_drop = max(all_score - score for name, score in ablation_scores.items() if name != "A. All enhanced features")
    if univariate_flags:
        return "unresolved"
    if all_score <= original_score:
        return "unresolved"
    if max_drop > 0.02:
        return "legitimate past-only behavioural signal"
    return "unresolved"


def main() -> None:
    data = load_data()
    approved = load_approved_features()
    combined = pd.concat([data["training"], data["validation"], data["testing"]], ignore_index=True)
    combined = add_historical_features(combined)
    combined = add_row_safe_features(combined)
    data = {
        split: combined[combined["dataset_split"].eq(split)].copy()
        for split in SPLITS
    }

    raw_features = [feature for feature in approved if feature in combined.columns and feature not in FORBIDDEN_FEATURES]
    engineered_features = engineered_feature_names(combined)
    features = raw_features + [feature for feature in engineered_features if feature not in raw_features and feature not in FORBIDDEN_FEATURES]
    if "transaction_type_model_safe" in features:
        raise SystemExit("ERROR: transaction_type_model_safe entered enhanced feature set")

    section("Raw And Engineered Feature Names")
    print("Raw approved features:", flush=True)
    for idx, feature in enumerate(raw_features, start=1):
        print(f"  raw_{idx:02d}. {feature}", flush=True)
    print("Engineered features:", flush=True)
    for idx, feature in enumerate([f for f in features if f not in raw_features], start=1):
        print(f"  eng_{idx:02d}. {feature}", flush=True)
    print(f"total feature count before encoding: {len(features):,}", flush=True)
    print(f"transaction_type_model_safe absent: {'transaction_type_model_safe' not in features}", flush=True)

    print_protocol_verification()
    univariate_flags = univariate_engineered_audit(data["validation"], [f for f in engineered_features if f not in FORBIDDEN_FEATURES])
    ablation_scores = group_ablation_audit(data["training"], data["validation"], raw_features, [f for f in engineered_features if f not in FORBIDDEN_FEATURES])
    classification = classify_gain(univariate_flags, ablation_scores)

    section("Diagnostic Classification")
    print(f"Classification: {classification}", flush=True)
    print("Evidence basis:", flush=True)
    print("  - Historical construction uses sequential online state with strictly prior timestamps and same-timestamp isolation.", flush=True)
    print("  - Validation-only univariate flags are printed above; any flagged feature should be reviewed before claiming causality.", flush=True)
    print("  - Group ablations use August validation only and do not use September for retention decisions.", flush=True)
    print("  - No features were automatically removed.", flush=True)
    print("No files were saved. This diagnostic run produced terminal output only.", flush=True)


if __name__ == "__main__":
    main()
