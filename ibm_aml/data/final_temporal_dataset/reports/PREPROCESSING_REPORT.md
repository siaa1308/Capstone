# Preprocessing Report

Original files used: `prepared_data/transactions_master.csv.gz` and `prepared_data/accounts_clean.csv.gz`.

Timestamps were read from the prepared `timestamp` column and parsed with pandas. Transactions were assigned to `training` for 2025-06-01 through 2025-07-31, `validation` for 2025-08-01 through 2025-08-31, and `testing` for 2025-09-01 through 2025-09-30.

Selected bank matches:

{
  "JPMorgan Chase": "JPMorgan Chase",
  "Wells Fargo": "Wells Fargo",
  "Citi": "Citi",
  "Fifth Third Bancorp": "Fifth Third Bancorp",
  "Key Bank": "Key Bank"
}

Transactions were assigned to banks using `src_bank_id` only. Unselected source banks were excluded.

Rolling transaction features (`src_prev_*` and `dst_prev_*`) came from `transactions_master.csv.gz` and were audited to ensure they match strictly earlier transactions for sampled and deterministic checks. Split-specific node features were regenerated instead of copying `prepared_data/account_node_features.csv.gz`. Node aggregate features use only transactions before the split start: training has no prior in-period history, validation uses June-July history, and testing uses June-August history. This graph protocol prevents training features from seeing validation/testing transactions and prevents testing node features from using September transactions.

Graph edges are one edge per transaction. Local graph node IDs are rebuilt independently for every split/bank folder and start at 0.

Confirmed leakage columns removed from model-facing files: y, laundering_type, edge_label, Is_APP_Fraud, Is_Cheque_Fraud, APP_Fraudster_ID, Cheque_Fraudster_ID, APP_Fraud_Sequence_Number, `transaction_type_raw`, `From_End_Balance`, `To_End_Balance`, and `Controlled_by_Criminal`.

Additional uncertain leakage columns requiring manual review: none currently flagged. High-cardinality identifiers were excluded from feature lists, not treated as target leakage.
