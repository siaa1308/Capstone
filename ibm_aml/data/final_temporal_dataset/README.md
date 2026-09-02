# Final Temporal IBM AML Dataset

This dataset is organized by temporal split first and source bank second. Only transactions whose `src_bank_id` matched the five selected banks are included.

`transactions.csv.gz` = tabular model input.
`ground_truth.csv.gz` = labels for supervised learning and evaluation.
`edge_list.csv.gz` = graph connections and safe transaction edge attributes.
`node_map.csv.gz` = original account ID to local numeric node ID mapping.
`node_features.csv.gz` = safe features describing graph nodes/accounts.
`model_feature_columns.json` = approved feature and exclusion lists.

The label `y` is not deleted. It is separated from model inputs and stored in `ground_truth.csv.gz`, joined by `txn_id`.
