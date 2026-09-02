# IBM AML XGBoost Baseline


This folder contains the leakage-safe XGBoost baseline script for the IBM synthetic AML dataset,
along with the source scripts that generate and audit the committed dataset under `data/`.


## Pipeline

The final dataset that training reads is `data/final_temporal_dataset/`. It is produced by, in order:

1. `scripts/preprocess_ibm_amlsim_style.py` — reads the raw IBM release zip and writes a
   prepared, model-facing master transaction table (`prepared_data/transactions_master.csv.gz`
   and `prepared_data/model_feature_columns.json`). Not distributed in this repo (raw source
   data and its intermediate output are large and are not committed); this script is included
   so the pipeline is reproducible from the raw IBM dataset.
2. `scripts/build_final_temporal_dataset.py` — reads `prepared_data/`, applies the temporal
   split (training/validation/testing) and the five-bank split, and writes
   `data/final_temporal_dataset/` including `configuration/model_feature_columns.json`. This is
   the authoritative generator for the dataset committed in this repository.
3. `scripts/audit_final_temporal_dataset.py` — validates the generated dataset (schema,
   temporal boundaries, graph consistency, leakage columns) against `prepared_data/`.
4. `scripts/generate_data_quality_report.py` and `scripts/generate_eda_report.py` — regenerate
   the reports under `data/final_temporal_dataset/reports/`.
5. `scripts/verify_no_leaked_transaction_type.py` — a lightweight, dependency-free regression
   check (config schema + CSV headers + training-script guards) that `transaction_type_model_safe`
   specifically can never re-enter the pipeline. Run it after any regeneration.

Steps 1-2 require the raw IBM dataset zip, which is not part of this repository.


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

transaction_type_model_safe was a redacted "safe" replacement for the raw transaction-type
column, but the redaction flag itself still correlates strongly with the laundering label and
was confirmed to be leakage. It is not merely excluded at training time: as of the current
`scripts/preprocess_ibm_amlsim_style.py` and `scripts/build_final_temporal_dataset.py`, it is
never created, never enters `prepared_data/model_feature_columns.json`'s safe-feature list, and
is never written to `transactions.csv.gz` or `edge_list.csv.gz`. Both generator scripts also
assert at build time that it cannot appear in `tabular_safe_features` or
`graph_edge_safe_features`. This training script's `FORBIDDEN_FEATURES` check remains as an
additional, defense-in-depth guard, not the only line of defense.

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

