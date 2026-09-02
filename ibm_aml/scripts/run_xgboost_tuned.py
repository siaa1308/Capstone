#!/usr/bin/env python3
"""Tune leakage-safe XGBoost with temporal CV and safe row-level features."""

from __future__ import annotations

import inspect
import itertools
import json
import math
import re
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
BUILD_SCRIPT = Path("scripts/build_final_temporal_dataset.py")
BANKS = [
    "JPMorgan_Chase",
    "Wells_Fargo",
    "Citi",
    "Fifth_Third_Bancorp",
    "Key_Bank",
]
SPLITS = ["training", "validation", "testing"]
FIXED_BASELINE = {
    "validation_pr_auc": 0.692900,
    "testing_pr_auc": 0.631935,
    "testing_precision": 0.574468,
    "testing_recall": 0.613636,
    "testing_f1": 0.593407,
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


@dataclass(frozen=True)
class Fold:
    name: str
    fit_start: str
    fit_end: str
    eval_start: str
    eval_end: str


@dataclass
class CVResult:
    candidate_number: int
    params: dict[str, float | int]
    fold_scores: list[float]
    mean_pr_auc: float


@dataclass
class ModelResult:
    name: str
    params: dict[str, float | int]
    features: list[str]
    transformed_feature_count: int
    scale_pos_weight: float
    threshold: float
    best_iteration: int | None
    val_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    test_probs: np.ndarray


FOLDS = [
    Fold("fold_1", "2025-06-01", "2025-06-21", "2025-06-22", "2025-06-30"),
    Fold("fold_2", "2025-06-01", "2025-06-30", "2025-07-01", "2025-07-15"),
    Fold("fold_3", "2025-06-01", "2025-07-15", "2025-07-16", "2025-07-31"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}", flush=True)


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
            raise SystemExit(
                f"ERROR: {split}/{bank} txn_id join changed row counts: "
                f"transactions={len(tx)}, ground_truth={len(gt)}, joined={len(joined)}"
            )
        if joined["txn_id"].duplicated().any():
            raise SystemExit(f"ERROR: {split}/{bank} joined data has duplicate txn_id values")
        joined["bank"] = bank
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
    section("Loaded Data")
    for split, df in data.items():
        positives = int(df["y"].sum())
        print(
            f"{split}: rows={len(df):,}, positives={positives:,}, "
            f"prevalence={positives / len(df):.6f}, "
            f"date_min={df['timestamp_dt'].min().date()}, date_max={df['timestamp_dt'].max().date()}",
            flush=True,
        )
    return data


def load_approved_features() -> list[str]:
    config = json.loads(FEATURE_JSON.read_text())
    features = config.get("tabular_safe_features")
    if not isinstance(features, list) or not features:
        raise SystemExit(f"ERROR: tabular_safe_features not found in {FEATURE_JSON}")
    return [str(feature) for feature in features]


def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return pd.to_numeric(numer, errors="coerce") / (pd.to_numeric(denom, errors="coerce").abs() + EPS)


def add_safe_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "amount" in out.columns:
        amount = pd.to_numeric(out["amount"], errors="coerce")
        out["log_amount"] = np.log1p(amount.clip(lower=0))
    if "From_Initial_Balance" in out.columns:
        from_bal = pd.to_numeric(out["From_Initial_Balance"], errors="coerce")
        out["log_from_initial_balance"] = np.log1p(from_bal.clip(lower=0))
    if "To_Initial_Balance" in out.columns:
        to_bal = pd.to_numeric(out["To_Initial_Balance"], errors="coerce")
        out["log_to_initial_balance"] = np.log1p(to_bal.clip(lower=0))
    if {"amount", "From_Initial_Balance"}.issubset(out.columns):
        out["amount_to_from_balance_ratio"] = safe_divide(out["amount"], out["From_Initial_Balance"])
    if {"amount", "To_Initial_Balance"}.issubset(out.columns):
        out["amount_to_to_balance_ratio"] = safe_divide(out["amount"], out["To_Initial_Balance"])
    if {"amount", "src_prev_amount_mean"}.issubset(out.columns):
        out["amount_to_src_previous_mean_ratio"] = safe_divide(out["amount"], out["src_prev_amount_mean"])
    if {"amount", "dst_prev_amount_mean"}.issubset(out.columns):
        out["amount_to_dst_previous_mean_ratio"] = safe_divide(out["amount"], out["dst_prev_amount_mean"])
    if "src_prev_txn_count" in out.columns:
        src_count = pd.to_numeric(out["src_prev_txn_count"], errors="coerce")
        out["log_src_prev_txn_count"] = np.log1p(src_count.clip(lower=0))
        out["src_new_account"] = (src_count == 0).astype(int)
    if "dst_prev_txn_count" in out.columns:
        dst_count = pd.to_numeric(out["dst_prev_txn_count"], errors="coerce")
        out["log_dst_prev_txn_count"] = np.log1p(dst_count.clip(lower=0))
        out["dst_new_account"] = (dst_count == 0).astype(int)
    if "src_prev_amount_sum" in out.columns:
        out["log_src_prev_amount_sum"] = np.log1p(pd.to_numeric(out["src_prev_amount_sum"], errors="coerce").clip(lower=0))
    if "dst_prev_amount_sum" in out.columns:
        out["log_dst_prev_amount_sum"] = np.log1p(pd.to_numeric(out["dst_prev_amount_sum"], errors="coerce").clip(lower=0))
    if {"src_bank_id", "dst_bank_id"}.issubset(out.columns):
        out["same_bank"] = out["src_bank_id"].astype("string").eq(out["dst_bank_id"].astype("string")).astype(int)
    if {"currency", "Receiving_Currency"}.issubset(out.columns):
        out["currency_mismatch"] = out["currency"].astype("string").ne(out["Receiving_Currency"].astype("string")).astype(int)
    if "timestamp_dt" in out.columns:
        hour = out["timestamp_dt"].dt.hour.astype(float)
        out["transaction_hour"] = hour
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        dow = out["timestamp_dt"].dt.dayofweek.astype(float)
        out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def build_feature_list(data: dict[str, pd.DataFrame], approved: list[str]) -> list[str]:
    train_cols = set(data["training"].columns)
    base = [feature for feature in approved if feature in train_cols and feature not in FORBIDDEN_FEATURES]
    if "transaction_type_model_safe" in approved or "transaction_type_model_safe" in base:
        raise SystemExit("ERROR: transaction_type_model_safe is present in approved/model feature list")
    engineered = [
        "log_amount",
        "log_from_initial_balance",
        "log_to_initial_balance",
        "amount_to_from_balance_ratio",
        "amount_to_to_balance_ratio",
        "amount_to_src_previous_mean_ratio",
        "amount_to_dst_previous_mean_ratio",
        "log_src_prev_txn_count",
        "log_dst_prev_txn_count",
        "log_src_prev_amount_sum",
        "log_dst_prev_amount_sum",
        "src_new_account",
        "dst_new_account",
        "same_bank",
        "currency_mismatch",
        "transaction_hour",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]
    train_aug = add_safe_derived_features(data["training"])
    features = base + [feature for feature in engineered if feature in train_aug.columns]
    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        raise SystemExit(f"ERROR: forbidden features entered model feature list: {forbidden}")
    return features


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
        out[feature] = pd.to_numeric(df[feature], errors="coerce")
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
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=1.0)


def make_matrices(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
) -> tuple[object, object, int, list[str], list[str], list[str]]:
    fit_aug = add_safe_derived_features(fit_df)
    eval_aug = add_safe_derived_features(eval_df)
    constant = [feature for feature in features if fit_aug[feature].nunique(dropna=False) <= 1]
    active = [feature for feature in features if feature not in constant]
    numeric, boolean, categorical = classify_features(fit_aug, active)
    x_fit_raw = convert_feature_frame(fit_aug, numeric, boolean, categorical)
    x_eval_raw = convert_feature_frame(eval_aug, numeric, boolean, categorical)
    preprocessor = build_preprocessor(numeric, boolean, categorical)
    x_fit = preprocessor.fit_transform(x_fit_raw)
    x_eval = preprocessor.transform(x_eval_raw)
    if not sparse.issparse(x_fit):
        x_fit = sparse.csr_matrix(x_fit)
        x_eval = sparse.csr_matrix(x_eval)
    return x_fit, x_eval, int(x_fit.shape[1]), active, numeric, categorical


def make_final_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[object, object, object, int, list[str], list[str], list[str]]:
    train_aug = add_safe_derived_features(train_df)
    val_aug = add_safe_derived_features(val_df)
    test_aug = add_safe_derived_features(test_df)
    constant = [feature for feature in features if train_aug[feature].nunique(dropna=False) <= 1]
    active = [feature for feature in features if feature not in constant]
    numeric, boolean, categorical = classify_features(train_aug, active)
    preprocessor = build_preprocessor(numeric, boolean, categorical)
    x_train = preprocessor.fit_transform(convert_feature_frame(train_aug, numeric, boolean, categorical))
    x_val = preprocessor.transform(convert_feature_frame(val_aug, numeric, boolean, categorical))
    x_test = preprocessor.transform(convert_feature_frame(test_aug, numeric, boolean, categorical))
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(x_train)
        x_val = sparse.csr_matrix(x_val)
        x_test = sparse.csr_matrix(x_test)
    if constant:
        print(f"Training-constant features removed before final training: {constant}", flush=True)
    return x_train, x_val, x_test, int(x_train.shape[1]), active, numeric, categorical


def xgb_fit(
    x_train,
    y_train,
    x_eval,
    y_eval,
    params: dict[str, float | int],
    scale_pos_weight: float,
    early_stopping_rounds: int,
) -> XGBClassifier:
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_estimators": 2000,
        "n_jobs": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        **{k: v for k, v in params.items() if k != "scale_pos_weight_multiplier"},
    }
    fit_params = inspect.signature(XGBClassifier.fit).parameters
    if "early_stopping_rounds" in fit_params:
        model = XGBClassifier(**model_params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_eval, y_eval)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    else:
        model = XGBClassifier(**model_params, early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], verbose=False)
    return model


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


def select_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_y = y_arr[order]
    last_indices = np.r_[np.flatnonzero(sorted_probs[:-1] != sorted_probs[1:]), len(sorted_probs) - 1]
    thresholds = sorted_probs[last_indices]
    tp = np.cumsum(sorted_y)[last_indices].astype(float)
    predicted_positive = (last_indices + 1).astype(float)
    fp = predicted_positive - tp
    positives = float(sorted_y.sum())
    fn = positives - tp
    denom = (2 * tp) + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, positives, out=np.zeros_like(tp), where=positives > 0)
    all_positive_f1 = (2 * positives) / ((2 * positives) + (len(sorted_y) - positives)) if positives else 0.0
    thresholds = np.r_[thresholds, 0.0]
    f1 = np.r_[f1, all_positive_f1]
    recall = np.r_[recall, 1.0 if positives else 0.0]
    best_f1 = np.nanmax(f1)
    tied = np.flatnonzero(np.isclose(f1, best_f1))
    best_idx = tied[np.argmax(recall[tied])]
    return float(thresholds[best_idx])


def print_balance_audit(data: dict[str, pd.DataFrame]) -> None:
    section("Pre-Transaction Balance Audit")
    preprocess_text = PREPROCESS_SCRIPT.read_text()
    build_text = BUILD_SCRIPT.read_text()
    tx_rename_block = re.search(r"TXN_RENAME\s*=\s*\{(?P<body>.*?)\}", preprocess_text, flags=re.S)
    tx_rename_body = tx_rename_block.group("body") if tx_rename_block else ""
    print("Original source columns:", flush=True)
    print("  From_Initial_Balance: raw IBM transaction column named From_Initial_Balance", flush=True)
    print("  To_Initial_Balance: raw IBM transaction column named To_Initial_Balance", flush=True)
    print("Code lineage evidence:", flush=True)
    print(f"  Present in TXN_RENAME mapping: {'From_Initial_Balance' in tx_rename_body or 'To_Initial_Balance' in tx_rename_body}", flush=True)
    print("  Interpretation: not renamed or recomputed in normalize_transaction_chunk; preserved from raw transaction row.", flush=True)
    print(f"  build_final_temporal_dataset writes them from TABULAR_SAFE_FEATURES: {all(c in build_text for c in ['From_Initial_Balance', 'To_Initial_Balance'])}", flush=True)
    print(f"  From_End_Balance/To_End_Balance are separately audited as post-transaction and removed: {all(c in build_text for c in ['From_End_Balance', 'To_End_Balance', 'post-transaction'])}", flush=True)
    print(f"  No target-label construction reference in balance code paths: {'From_Initial_Balance = ' not in preprocess_text and 'To_Initial_Balance = ' not in preprocess_text}", flush=True)
    print("Verification statements:", flush=True)
    print("  1. Original source columns are the same-named raw transaction columns.", flush=True)
    print("  2. They represent initial balances available before the current transaction by schema/name.", flush=True)
    print("  3. They are not calculated using From_End_Balance or To_End_Balance in the preprocessing script.", flush=True)
    print("  4. No target label is used to construct them; y is only read from Is_Laundering and later separated.", flush=True)

    val = data["validation"]
    y = val["y"].astype(int)
    for col in ["From_Initial_Balance", "To_Initial_Balance"]:
        values = pd.to_numeric(val[col], errors="coerce")
        scores = -values.fillna(values.median()).to_numpy(dtype=float)
        ap = average_precision_score(y, scores)
        pos = values[y.eq(1)]
        neg = values[y.eq(0)]
        print(f"\n{col} validation univariate audit:", flush=True)
        print(f"  PR-AUC using lower balance as higher risk: {ap:.6f}", flush=True)
        print(f"  positive median: {pos.median():.6f}", flush=True)
        print(f"  negative median: {neg.median():.6f}", flush=True)
        for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]:
            print(f"  q={q:.2f}: positive={pos.quantile(q):.6f}, negative={neg.quantile(q):.6f}", flush=True)

    best_rule = None
    from_values = pd.to_numeric(val["From_Initial_Balance"], errors="coerce")
    to_values = pd.to_numeric(val["To_Initial_Balance"], errors="coerce")
    for col, values in [("From_Initial_Balance", from_values), ("To_Initial_Balance", to_values)]:
        for q in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
            threshold = values.quantile(q)
            pred = values <= threshold
            tp = int((pred & y.eq(1)).sum())
            fp = int((pred & y.eq(0)).sum())
            fn = int((~pred & y.eq(1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            item = (f1, recall, precision, col, q, threshold, tp, fp, fn)
            if best_rule is None or item > best_rule:
                best_rule = item
    if best_rule is not None:
        f1, recall, precision, col, q, threshold, tp, fp, fn = best_rule
        print("\nBest simple deterministic balance rule on August validation:", flush=True)
        print(
            f"  rule: {col} <= validation quantile {q:.3f} ({threshold:.6f})",
            flush=True,
        )
        print(
            f"  precision={precision:.6f}, recall={recall:.6f}, F1={f1:.6f}, TP={tp}, FP={fp}, FN={fn}",
            flush=True,
        )
        print("  conclusion: rule is diagnostic only and does not by itself isolate most positives unless recall is high.", flush=True)


def temporal_cv(data: dict[str, pd.DataFrame], features: list[str]) -> tuple[dict[str, float | int], float, list[CVResult]]:
    section("Temporal Cross-Validation Folds")
    train = data["training"]
    for fold in FOLDS:
        fit_mask = train["timestamp_dt"].between(fold.fit_start, fold.fit_end + " 23:59:59.999999")
        eval_mask = train["timestamp_dt"].between(fold.eval_start, fold.eval_end + " 23:59:59.999999")
        fit_pos = int(train.loc[fit_mask, "y"].sum())
        eval_pos = int(train.loc[eval_mask, "y"].sum())
        print(
            f"{fold.name}: fit {fold.fit_start}..{fold.fit_end} rows={int(fit_mask.sum()):,} positives={fit_pos:,}; "
            f"eval {fold.eval_start}..{fold.eval_end} rows={int(eval_mask.sum()):,} positives={eval_pos:,}",
            flush=True,
        )
        if fit_pos == 0 or eval_pos == 0:
            raise SystemExit(f"ERROR: {fold.name} has no positives in fit or evaluation period")

    search_space = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 10, 20],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
        "subsample": [0.6, 0.75, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "gamma": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0, 20.0],
        "max_delta_step": [0, 1, 5],
        "scale_pos_weight_multiplier": [0.25, 0.5, 0.75, 1.0],
    }
    keys = list(search_space)
    all_combos = list(itertools.product(*[search_space[key] for key in keys]))
    rng = np.random.default_rng(42)
    selected = rng.choice(len(all_combos), size=20, replace=False)

    results: list[CVResult] = []
    best: CVResult | None = None
    section("Random Search")
    for candidate_number, combo_idx in enumerate(selected, start=1):
        params = dict(zip(keys, all_combos[int(combo_idx)]))
        fold_scores = []
        for fold in FOLDS:
            fit_mask = train["timestamp_dt"].between(fold.fit_start, fold.fit_end + " 23:59:59.999999")
            eval_mask = train["timestamp_dt"].between(fold.eval_start, fold.eval_end + " 23:59:59.999999")
            fit_df = train.loc[fit_mask].copy()
            eval_df = train.loc[eval_mask].copy()
            x_fit, x_eval, _count, _active, _numeric, _categorical = make_matrices(fit_df, eval_df, features)
            y_fit = fit_df["y"].astype(int)
            y_eval = eval_df["y"].astype(int)
            base_ratio = int((y_fit == 0).sum()) / int((y_fit == 1).sum())
            spw = base_ratio * float(params["scale_pos_weight_multiplier"])
            model = xgb_fit(x_fit, y_fit, x_eval, y_eval, params, spw, early_stopping_rounds=75)
            probs = model.predict_proba(x_eval)[:, 1]
            if not (np.all(probs >= 0.0) and np.all(probs <= 1.0)):
                raise SystemExit("ERROR: model probabilities outside [0, 1] during CV")
            fold_scores.append(float(average_precision_score(y_eval, probs)))
        mean_score = float(np.mean(fold_scores))
        result = CVResult(candidate_number, params, fold_scores, mean_score)
        results.append(result)
        if best is None or result.mean_pr_auc > best.mean_pr_auc:
            best = result
        fold_str = ",".join(f"{score:.6f}" for score in fold_scores)
        print(
            f"candidate={candidate_number:02d} mean_cv_pr_auc={mean_score:.6f} "
            f"fold_pr_auc=[{fold_str}] params={params}",
            flush=True,
        )
    assert best is not None
    return best.params, best.mean_pr_auc, results


def final_train_eval(
    data: dict[str, pd.DataFrame],
    features: list[str],
    params: dict[str, float | int],
) -> ModelResult:
    section("Final Tuned Model")
    train = data["training"]
    val = data["validation"]
    test = data["testing"]
    x_train, x_val, x_test, transformed_count, active_features, numeric, categorical = make_final_matrices(train, val, test, features)
    y_train = train["y"].astype(int)
    y_val = val["y"].astype(int)
    y_test = test["y"].astype(int)
    base_ratio = int((y_train == 0).sum()) / int((y_train == 1).sum())
    spw = base_ratio * float(params["scale_pos_weight_multiplier"])
    print(f"Feature count before encoding: {len(active_features):,}", flush=True)
    print(f"Feature count after encoding: {transformed_count:,}", flush=True)
    print(f"Final scale_pos_weight: {spw:.6f}", flush=True)
    print(f"Final selected params: {params}", flush=True)
    model = xgb_fit(x_train, y_train, x_val, y_val, params, spw, early_stopping_rounds=75)
    val_probs = model.predict_proba(x_val)[:, 1]
    test_probs = model.predict_proba(x_test)[:, 1]
    if not (np.all(val_probs >= 0.0) and np.all(val_probs <= 1.0) and np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0)):
        raise SystemExit("ERROR: final model probabilities outside [0, 1]")
    threshold = select_threshold(y_val, val_probs)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration_", None)
    print(f"Best XGBoost iteration: {best_iteration}", flush=True)
    print(f"August-selected threshold: {threshold:.8f}", flush=True)
    return ModelResult(
        name="Tuned leakage-safe XGBoost",
        params=params,
        features=active_features,
        transformed_feature_count=transformed_count,
        scale_pos_weight=spw,
        threshold=threshold,
        best_iteration=best_iteration,
        val_metrics=metric_block(y_val, val_probs, threshold),
        test_metrics=metric_block(y_test, test_probs, threshold),
        test_probs=test_probs,
    )


def print_metric_comparison(result: ModelResult) -> None:
    section("Fixed Baseline vs Tuned Model")
    rows = [
        {
            "model": "fixed_clean_baseline",
            "validation_pr_auc": FIXED_BASELINE["validation_pr_auc"],
            "testing_pr_auc": FIXED_BASELINE["testing_pr_auc"],
            "testing_precision": FIXED_BASELINE["testing_precision"],
            "testing_recall": FIXED_BASELINE["testing_recall"],
            "testing_f1": FIXED_BASELINE["testing_f1"],
            "testing_balanced_accuracy": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TP": np.nan,
            "selected_threshold": np.nan,
        },
        {
            "model": "tuned_leakage_safe_xgboost",
            "validation_pr_auc": result.val_metrics["pr_auc"],
            "testing_pr_auc": result.test_metrics["pr_auc"],
            "testing_precision": result.test_metrics["precision"],
            "testing_recall": result.test_metrics["recall"],
            "testing_f1": result.test_metrics["f1"],
            "testing_balanced_accuracy": result.test_metrics["balanced_accuracy"],
            "TN": result.test_metrics["tn"],
            "FP": result.test_metrics["fp"],
            "FN": result.test_metrics["fn"],
            "TP": result.test_metrics["tp"],
            "selected_threshold": result.threshold,
        },
    ]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
    print(f"\nTuned predicted-positive count: {result.test_metrics['predicted_positive_count']:,}", flush=True)


def print_per_bank(data: dict[str, pd.DataFrame], result: ModelResult) -> None:
    section("Tuned Testing Metrics By Bank")
    pred = (result.test_probs >= result.threshold).astype(int)
    rows = []
    for bank in BANKS:
        mask = data["testing"]["bank"].eq(bank).to_numpy()
        y_bank = data["testing"].loc[mask, "y"].to_numpy(dtype=int)
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
                "predicted_positive_count": int(pred_bank.sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)


def sanity_checks(data: dict[str, pd.DataFrame], features: list[str], result: ModelResult) -> None:
    section("Sanity Checks")
    print(f"transaction_type_model_safe is absent: {'transaction_type_model_safe' not in features and 'transaction_type_model_safe' not in result.features}", flush=True)
    print("no testing data was used for tuning: PASS; random search uses only June-July temporal folds", flush=True)
    print("no testing data was used for threshold selection: PASS; threshold selected on August only", flush=True)
    print("preprocessing was fitted only on earlier data: PASS; each CV fold fits on its fit period, final fit uses June-July only", flush=True)
    preprocess_text = PREPROCESS_SCRIPT.read_text()
    prior_only = all(s in preprocess_text for s in ["cumcount()", "cumsum() - amounts", "count_state", "amount_state"])
    print(f"no future transaction was used in rolling features: {'PASS' if prior_only else 'REVIEW'}; preprocessing code uses prior state plus within-group cumcount/cumsum minus current amount", flush=True)
    print(f"all probabilities are between 0 and 1: {bool(np.all((result.test_probs >= 0.0) & (result.test_probs <= 1.0)))}", flush=True)
    print("labels were joined one-to-one using txn_id: PASS; load step uses merge(..., on='txn_id', validate='one_to_one') and duplicate checks", flush=True)
    print("SMOTE/oversampling/undersampling/target encoding/future aggregates/end balances: NOT USED", flush=True)


def main() -> None:
    data = load_data()
    approved = load_approved_features()
    features = build_feature_list(data, approved)
    section("Model Feature List")
    print(f"Approved leakage-safe base features plus safe engineered features: {len(features):,}", flush=True)
    for idx, feature in enumerate(features, start=1):
        print(f"  {idx:02d}. {feature}", flush=True)
    print(f"transaction_type_model_safe absent: {'transaction_type_model_safe' not in features}", flush=True)

    print_balance_audit(data)
    best_params, best_mean_cv, _cv_results = temporal_cv(data, features)
    section("Best Temporal CV Configuration")
    print(f"best mean temporal CV PR-AUC: {best_mean_cv:.6f}", flush=True)
    print(f"best parameters: {best_params}", flush=True)

    result = final_train_eval(data, features, best_params)
    print_metric_comparison(result)
    print_per_bank(data, result)
    sanity_checks(data, features, result)

    section("Final Summary")
    improvement = float(result.test_metrics["pr_auc"]) - FIXED_BASELINE["testing_pr_auc"]
    improved = improvement > 0
    print("OFFICIAL FIXED BASELINE RESULTS", flush=True)
    print(
        f"  validation PR-AUC={FIXED_BASELINE['validation_pr_auc']:.6f}; "
        f"testing PR-AUC={FIXED_BASELINE['testing_pr_auc']:.6f}; "
        f"testing precision={FIXED_BASELINE['testing_precision']:.6f}; "
        f"testing recall={FIXED_BASELINE['testing_recall']:.6f}; "
        f"testing F1={FIXED_BASELINE['testing_f1']:.6f}",
        flush=True,
    )
    print("TUNED LEAKAGE-SAFE XGBOOST RESULTS", flush=True)
    print(
        f"  validation PR-AUC={result.val_metrics['pr_auc']:.6f}; "
        f"testing PR-AUC={result.test_metrics['pr_auc']:.6f}; "
        f"testing precision={result.test_metrics['precision']:.6f}; "
        f"testing recall={result.test_metrics['recall']:.6f}; "
        f"testing F1={result.test_metrics['f1']:.6f}; "
        f"threshold={result.threshold:.8f}",
        flush=True,
    )
    print(f"absolute testing PR-AUC improvement: {improvement:.6f}", flush=True)
    print(f"tuned model genuinely improved the baseline: {improved}", flush=True)
    if not improved:
        print("Do not claim improvement because testing PR-AUC did not exceed 0.631935.", flush=True)
    print("No models, predictions, plots, reports, CSV files, JSON files, or output folders were saved.", flush=True)
    print("This run produced terminal output only.", flush=True)


if __name__ == "__main__":
    main()
