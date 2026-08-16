# IBM AML XGBoost Baseline


This folder contains the leakage-safe XGBoost baseline script for the IBM synthetic AML dataset.


## Script


```text
scripts/run_xgboost_baseline.py
Purpose

This script trains a tabular XGBoost baseline using the final temporally split IBM AML dataset.

The dataset is expected to be organised as:

final_temporal_dataset/
├── training/
├── validation/
├── testing/
└── configuration/model_feature_columns.json

The script uses:

transactions.csv.gz
ground_truth.csv.gz

It joins labels using txn_id.

Important leakage handling

The feature transaction_type_model_safe was excluded because it behaved like a synthetic label-generation shortcut.

Known leakage fields such as y, laundering_type, edge_label, fraud flags, fraudster IDs, end balances, and criminal-control indicators are not used as model features.

Current clean baseline result
Validation PR-AUC: 0.692900
Testing PR-AUC: 0.631935
Testing precision: 0.574468
Testing recall: 0.613636
Testing F1: 0.593407
Confusion matrix: [[133722, 60], [51, 81]]
Selected threshold: 0.70878369
How to run

From the IBM dataset project folder:

python scripts/run_xgboost_baseline.py --dataset-dir final_temporal_dataset

