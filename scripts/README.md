# Repository utilities

The active scripts prepare and audit data; they do not define the final model architecture.

- `preprocess_ibm_amlsim_style.py`: preprocess the original IBM AMLSim-style archive.
- `build_final_temporal_dataset.py`: construct the fixed temporal dataset.
- `audit_final_temporal_dataset.py`: verify prepared files and split integrity.
- `verify_no_leaked_transaction_type.py`: audit forbidden/leaked features.
- `audit_federated_heterogeneity.py`: quantify bank-level non-IID differences.
- `generate_data_quality_report.py`: generate dataset quality summaries.
- `generate_eda_report.py`: generate exploratory summaries.

`historical/` contains superseded baseline and reporting utilities. Final model entry points are in `src/gnn/`.

