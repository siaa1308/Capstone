# File Usage Guide

Use `transactions.csv.gz` for tabular ML features and join labels from `ground_truth.csv.gz` using `txn_id`.

Use `edge_list.csv.gz`, `node_map.csv.gz`, and `node_features.csv.gz` for graph ML. `src_node` and `dst_node` are local node IDs that start at 0 inside each split/bank folder. `node_map.csv.gz` maps those local IDs back to original account IDs.

Required modeling files are the five core files in every split/bank folder plus `configuration/model_feature_columns.json`. The files under `reports/` are supporting audit, quality, and EDA documentation.
