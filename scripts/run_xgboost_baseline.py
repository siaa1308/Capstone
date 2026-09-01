#!/usr/bin/env python3
"""Run the cleaned pooled-bank XGBoost tabular AML baseline."""

from __future__ import annotations

import inspect
import json
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
BANKS = [
    "JPMorgan_Chase",
    "Wells_Fargo",
    "Key_Bank",
]
SPLITS = ["training", "validation", "testing"]
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


@dataclass
class RunResult:
    name: str
    features: list[str]
    removed_requested: list[str]
    removed_constant: list[str]
    val_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    threshold: float
    best_iteration: int | None
    scale_pos_weight: float
    transformed_feature_count: int
    test_probs: np.ndarray


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def load_approved_features() -> list[str]:
    config = json.loads(FEATURE_JSON.read_text())
    features = config.get("tabular_safe_features")
    if not isinstance(features, list) or not features:
        raise SystemExit(f"ERROR: tabular_safe_features not found in {FEATURE_JSON}")
    return [str(feature) for feature in features]


def load_split(split: str) -> pd.DataFrame:
    frames = []
    for bank in BANKS:
        bank_dir = DATASET_DIR / split / bank
        tx_path = bank_dir / "transactions.csv.gz"
        gt_path = bank_dir / "ground_truth.csv.gz"
        tx = pd.read_csv(tx_path)
        gt = pd.read_csv(gt_path)
        for label, df, path in [("transactions", tx, tx_path), ("ground_truth", gt, gt_path)]:
            if "txn_id" not in df.columns:
                raise SystemExit(f"ERROR: {path} is missing txn_id")
            dupes = int(df["txn_id"].duplicated().sum())
            if dupes:
                raise SystemExit(f"ERROR: {split}/{bank} {label} has {dupes} duplicate txn_id values")

        joined = tx.merge(gt[["txn_id", "y"]], on="txn_id", how="inner", validate="one_to_one")
        if len(joined) != len(tx) or len(joined) != len(gt):
            raise SystemExit(
                f"ERROR: {split}/{bank} txn_id join changed row counts: "
                f"transactions={len(tx)}, ground_truth={len(gt)}, joined={len(joined)}"
            )
        if joined["txn_id"].duplicated().any():
            raise SystemExit(f"ERROR: {split}/{bank} joined rows contain duplicate txn_id values")
        joined["bank"] = bank
        joined["y"] = pd.to_numeric(joined["y"], errors="raise").astype(int)
        frames.append(joined)

    out = pd.concat(frames, ignore_index=True)
    if out["txn_id"].duplicated().any():
        raise SystemExit(f"ERROR: {split} has duplicate txn_id values across banks after joining")
    return out


def print_split_counts(data: dict[str, pd.DataFrame]) -> None:
    section("Split Counts")
    for split in SPLITS:
        df = data[split]
        y = df["y"].astype(int)
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        print(
            f"{split}: rows={len(df):,} y=0={negatives:,} y=1={positives:,} "
            f"prevalence={positives / len(df):.6f}"
        )
        for bank, rows in df.groupby("bank", sort=False).size().items():
            print(f"  {bank}: {rows:,}")


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
    if non_null.empty:
        return False
    return bool(non_null.nunique() <= 2 and map_bool(non_null).notna().all())


def is_numeric_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(pd.to_numeric(non_null, errors="coerce").notna().all())


def classify_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], list[str]]:
    boolean = [feature for feature in features if is_boolean_like(df[feature])]
    numeric = [feature for feature in features if feature not in boolean and is_numeric_like(df[feature])]
    categorical = [feature for feature in features if feature not in boolean and feature not in numeric]
    return numeric, boolean, categorical


def convert_feature_frame(
    df: pd.DataFrame,
    numeric: list[str],
    boolean: list[str],
    categorical: list[str],
) -> pd.DataFrame:
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


def prepare_matrices(
    data: dict[str, pd.DataFrame],
    features: list[str],
) -> tuple[object, object, object, int, tuple[list[str], list[str], list[str]]]:
    numeric, boolean, categorical = classify_features(data["training"], features)
    x_train_raw = convert_feature_frame(data["training"], numeric, boolean, categorical)
    x_val_raw = convert_feature_frame(data["validation"], numeric, boolean, categorical)
    x_test_raw = convert_feature_frame(data["testing"], numeric, boolean, categorical)
    preprocessor = build_preprocessor(numeric, boolean, categorical)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)
    x_test = preprocessor.transform(x_test_raw)
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(x_train)
        x_val = sparse.csr_matrix(x_val)
        x_test = sparse.csr_matrix(x_test)
    return x_train, x_val, x_test, int(x_train.shape[1]), (numeric, boolean, categorical)


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
        "rows": int(len(y_arr)),
        "positive_cases": int(y_arr.sum()),
    }


def fit_model(x_train, y_train: pd.Series, x_val, y_val: pd.Series, scale_pos_weight: float, name: str) -> XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
    }
    print(f"\nTraining {name} with XGBoost {xgb.__version__}")
    fit_params = inspect.signature(XGBClassifier.fit).parameters
    if "early_stopping_rounds" in fit_params:
        model = XGBClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            early_stopping_rounds=100,
            verbose=False,
        )
    else:
        model = XGBClassifier(**params, early_stopping_rounds=100)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def run_xgboost(
    name: str,
    data: dict[str, pd.DataFrame],
    candidate_features: list[str],
    remove_features: set[str],
) -> RunResult:
    requested_removed = sorted(feature for feature in remove_features if feature in candidate_features)
    base_features = [feature for feature in candidate_features if feature not in remove_features]
    constant = [feature for feature in base_features if data["training"][feature].nunique(dropna=False) <= 1]
    features = [feature for feature in base_features if feature not in constant]
    if not features:
        raise SystemExit(f"ERROR: {name} has no remaining non-constant features")

    x_train, x_val, x_test, transformed_count, feature_types = prepare_matrices(data, features)
    numeric, boolean, categorical = feature_types
    y_train = data["training"]["y"].astype(int)
    y_val = data["validation"]["y"].astype(int)
    y_test = data["testing"]["y"].astype(int)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise SystemExit("ERROR: training labels contain no positive cases")
    scale_pos_weight = neg / pos

    print(
        f"{name}: raw_features={len(features):,}, transformed_features={transformed_count:,}, "
        f"numeric={len(numeric):,}, boolean={len(boolean):,}, categorical={len(categorical):,}, "
        f"scale_pos_weight={scale_pos_weight:.6f}"
    )
    if constant:
        print(f"{name}: training-constant features removed: {constant}")

    model = fit_model(x_train, y_train, x_val, y_val, scale_pos_weight, name)
    val_probs = model.predict_proba(x_val)[:, 1]
    test_probs = model.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_val, val_probs)
    val_metrics = metric_block(y_val, val_probs, threshold)
    test_metrics = metric_block(y_test, test_probs, threshold)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration_", None)

    return RunResult(
        name=name,
        features=features,
        removed_requested=requested_removed,
        removed_constant=constant,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        threshold=threshold,
        best_iteration=best_iteration,
        scale_pos_weight=scale_pos_weight,
        transformed_feature_count=transformed_count,
        test_probs=test_probs,
    )


def print_full_metrics(label: str, metrics: dict[str, float | int], threshold: float) -> None:
    print(f"\n{label} metrics at selected threshold {threshold:.8f}")
    print(f"  rows: {metrics['rows']:,}")
    print(f"  positive cases: {metrics['positive_cases']:,}")
    print(f"  PR-AUC / Average Precision: {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"  precision: {metrics['precision']:.6f}")
    print(f"  recall: {metrics['recall']:.6f}")
    print(f"  F1: {metrics['f1']:.6f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"  TN: {metrics['tn']:,}")
    print(f"  FP: {metrics['fp']:,}")
    print(f"  FN: {metrics['fn']:,}")
    print(f"  TP: {metrics['tp']:,}")
    print(f"  predicted-positive count: {metrics['predicted_positive_count']:,}")


def print_per_bank_metrics(data: dict[str, pd.DataFrame], result: RunResult) -> None:
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
    print("\nTesting metrics by bank using the single global validation-selected threshold")
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            formatters={
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "F1": "{:.6f}".format,
                "balanced_accuracy": "{:.6f}".format,
            },
        )
    )


def print_ablation_summary(result: RunResult) -> None:
    print(f"\n{result.name}")
    print(f"  requested features removed: {result.removed_requested if result.removed_requested else 'none'}")
    print(f"  remaining non-constant feature count: {len(result.features):,}")
    print(f"  validation PR-AUC: {result.val_metrics['pr_auc']:.6f}")
    print(f"  testing PR-AUC: {result.test_metrics['pr_auc']:.6f}")


def main() -> None:
    section("Loading Data")
    approved = load_approved_features()
    data = {split: load_split(split) for split in SPLITS}
    print_split_counts(data)

    section("Final Clean Feature Selection")
    if "transaction_type_model_safe" in approved:
        raise SystemExit("ERROR: transaction_type_model_safe is still present in tabular_safe_features")
    print("transaction_type_model_safe absent from tabular_safe_features: PASS")

    train_cols = set(data["training"].columns)
    missing = [feature for feature in approved if feature not in train_cols]
    candidate_features = [
        feature
        for feature in approved
        if feature in train_cols and feature not in FORBIDDEN_FEATURES
    ]
    if missing:
        print("Approved features missing from transactions and skipped:")
        for feature in missing:
            print(f"  {feature}")
    if "transaction_type_model_safe" in candidate_features:
        raise SystemExit("ERROR: transaction_type_model_safe entered candidate feature set")

    constant = [feature for feature in candidate_features if data["training"][feature].nunique(dropna=False) <= 1]
    final_features = [feature for feature in candidate_features if feature not in constant]
    if constant:
        print("Training-constant features excluded from modeling:")
        for feature in constant:
            print(f"  {feature}")

    print("\nExact final feature list:")
    for idx, feature in enumerate(final_features, start=1):
        print(f"  {idx:02d}. {feature}")
    print(f"Final raw feature count: {len(final_features):,}")
    print(f"transaction_type_model_safe absent from final feature list: {'PASS' if 'transaction_type_model_safe' not in final_features else 'FAIL'}")

    section("Final Clean XGBoost Baseline")
    clean = run_xgboost("Clean baseline without transaction_type_model_safe", data, candidate_features, set())
    print(f"\nBest XGBoost iteration: {clean.best_iteration}")
    print(f"scale_pos_weight from training labels only: {clean.scale_pos_weight:.6f}")
    print(f"transformed feature count: {clean.transformed_feature_count:,}")
    print(f"selected threshold from validation only: {clean.threshold:.8f}")
    print_full_metrics("Validation", clean.val_metrics, clean.threshold)
    print_full_metrics("Testing", clean.test_metrics, clean.threshold)
    print_per_bank_metrics(data, clean)

    section("Clean-Feature Ablations")
    ablations = [
        ("1. Clean baseline without transaction_type_model_safe", set()),
        ("2. Clean baseline minus Is_Hold, Sufficient_Funds, Overdraft_Okay, and Is_All_Cash", {"Is_Hold", "Sufficient_Funds", "Overdraft_Okay", "Is_All_Cash"}),
        ("3. Clean baseline minus Amount_Received and Receiving_Currency", {"Amount_Received", "Receiving_Currency"}),
        ("4. Clean baseline minus From_Initial_Balance and To_Initial_Balance", {"From_Initial_Balance", "To_Initial_Balance"}),
        ("5. Clean baseline minus all src_prev_* and dst_prev_* features", {
            "src_prev_txn_count",
            "src_prev_amount_sum",
            "src_prev_amount_mean",
            "dst_prev_txn_count",
            "dst_prev_amount_sum",
            "dst_prev_amount_mean",
        }),
    ]
    ablation_results: list[RunResult] = []
    for name, removals in ablations:
        result = clean if not removals else run_xgboost(name, data, candidate_features, removals)
        if not removals:
            result.name = name
        ablation_results.append(result)
        print_ablation_summary(result)

    section("Feature-Group Interpretation")
    print("transaction_type_model_safe was permanently excluded because the prior audit identified it as a synthetic target shortcut.")
    print("No additional feature group is automatically removed by this run.")
    print(
        "Flagging rule used here: another group would be flagged only with evidence that it is "
        "target-derived, post-outcome, unavailable at prediction time, or another synthetic label-generation shortcut."
    )
    print("The ablations below are diagnostic only:")
    for result in ablation_results:
        print(f"  {result.name}: validation PR-AUC={result.val_metrics['pr_auc']:.6f}, testing PR-AUC={result.test_metrics['pr_auc']:.6f}")

    section("Official Result")
    print("Report this as the official leakage-safe XGBoost baseline:")
    print("  Clean baseline without transaction_type_model_safe")
    print(f"  validation PR-AUC: {clean.val_metrics['pr_auc']:.6f}")
    print(f"  testing PR-AUC: {clean.test_metrics['pr_auc']:.6f}")
    print(f"  testing precision: {clean.test_metrics['precision']:.6f}")
    print(f"  testing recall: {clean.test_metrics['recall']:.6f}")
    print(f"  testing F1: {clean.test_metrics['f1']:.6f}")
    print(
        "  testing confusion matrix [[TN, FP], [FN, TP]]: "
        f"[[{clean.test_metrics['tn']}, {clean.test_metrics['fp']}], "
        f"[{clean.test_metrics['fn']}, {clean.test_metrics['tp']}]]"
    )
    print("\nNo model, predictions, plots, reports, metrics files, or output folders were saved.")
    print("This run produced terminal output only.")


if __name__ == "__main__":
    main()
