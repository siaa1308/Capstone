# Repository Manifest

This manifest records the final repository surface after the capstone cleanup. Model logic and completed evaluation outputs were not changed. Files are classified as **final**, **supporting**, or **historical** so readers can distinguish the paper pipeline from audit evidence.

## Final directory tree

```text
ibm_aml/
├── README.md
├── requirements.txt
├── configs/final_pipeline.json
├── data/
│   ├── README.md
│   └── final_temporal_dataset/
├── src/gnn/
│   ├── causal_temporal_graphsage.py
│   ├── continual_temporal_graphsage.py
│   ├── federated_causal_temporal_graphsage.py
│   ├── evaluate_frozen_temporal.py
│   ├── run_temporal_multiseed.py
│   └── legacy/
├── scripts/
│   ├── active data/audit utilities
│   └── historical/
├── tests/test_temporal_gnn_protocol.py
├── docs/
│   ├── FINAL_EVALUATION_REPORT.md
│   ├── FEDAVG_COLLAPSE_INVESTIGATION.md
│   ├── PAPER_EVALUATION_PROTOCOL.md
│   ├── REPOSITORY_MANIFEST.md
│   └── historical/
└── artifacts/
    ├── final_evaluation/{local,cl,corrected_fedavg}/
    └── historical/
```

## Final pipeline

| Stage | Active code | Frozen artifacts |
|---|---|---|
| Local Temporal GNN | `src/gnn/causal_temporal_graphsage.py` | `artifacts/final_evaluation/local/seed_{42,52,62}` |
| Continual Learning + replay | `src/gnn/continual_temporal_graphsage.py` | `artifacts/final_evaluation/cl/seed_{42,52,62}` |
| FedAvg + local CL/replay | `src/gnn/federated_causal_temporal_graphsage.py` | `artifacts/final_evaluation/corrected_fedavg/seed_{42,52,62}` |

The immutable consolidated settings are in `configs/final_pipeline.json`. The original corrected-FedAvg freeze record remains at `artifacts/final_evaluation/CORRECTED_FEDAVG_FROZEN_CONFIG.json`.

## Inventory summary

- Python: 21 files across active source/utilities/tests and explicitly marked legacy code.
- Notebooks: none.
- Tests: one protocol test module containing seven tests.
- Prepared data: 110 files under `data/final_temporal_dataset/` (approximately 84 MB).
- Final evaluation: 32 files—21 checkpoints and 11 JSON configuration/metric/summary records.
- Historical archive: 385 files, including 236 checkpoints across five-bank evidence, the original FedAvg result, and development forensics.
- Documentation: seven Markdown files under `docs/`, plus directory-specific READMEs.
- Temporary logs, editor files, bytecode caches, and virtual environments: none retained.

## Final source and support files

- `src/gnn/evaluate_frozen_temporal.py`: checkpoint evaluator.
- `src/gnn/run_temporal_multiseed.py`: Local multi-seed orchestration.
- `tests/test_temporal_gnn_protocol.py`: temporal-gradient, causal-state, replay, split, and protocol checks.
- `scripts/preprocess_ibm_amlsim_style.py`: source-data preprocessing.
- `scripts/build_final_temporal_dataset.py`: fixed temporal dataset construction.
- `scripts/audit_final_temporal_dataset.py`: prepared-data audit.
- `scripts/verify_no_leaked_transaction_type.py`: feature leakage audit.
- `scripts/audit_federated_heterogeneity.py`: non-IID client evidence.
- `scripts/generate_data_quality_report.py` and `scripts/generate_eda_report.py`: report generators.

## Authoritative documentation

- `README.md`: project entry point.
- `docs/FINAL_EVALUATION_REPORT.md`: authoritative final protocol, results, and conclusion.
- `docs/FEDAVG_COLLAPSE_INVESTIGATION.md`: original FedAvg failure analysis.
- `docs/PAPER_EVALUATION_PROTOCOL.md`: test-set and reporting policy.
- `data/final_temporal_dataset/README.md`: prepared dataset specification.
- `data/final_temporal_dataset/reports/`: data quality, EDA, preprocessing, and leakage reports.

## Data preservation

`data/final_temporal_dataset/` remains unchanged and contains the fixed June–September temporal data. New modeling uses JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp data remains preserved in the same dataset for historical auditability and must not be removed.

## Historical archive

| Location | Contents |
|---|---|
| `artifacts/historical/original_fedavg_final/` | Original poor FedAvg final checkpoints, metrics, and initial freeze record |
| `artifacts/historical/legacy_five_bank/` | Five-bank Local/temporal/continual artifacts, including Citi and Fifth Third checkpoints |
| `artifacts/historical/legacy_federated/` | Superseded early federated experiment artifacts |
| `artifacts/historical/validation_search/` | Earlier validation searches, including FedProx diagnostics |
| `artifacts/historical/corrected_development/` | Corrected TBPTT, Local/CL, and FedAvg validation-only development evidence |
| `docs/historical/` | Superseded optimization/gate reports |
| `scripts/historical/` | Superseded XGBoost and summary utilities |
| `src/gnn/legacy/` | Superseded model runners |

Historical records are evidence, not current recommendations. In particular, the original FedAvg result of September PR-AUC `0.228 ± 0.035` is intentionally retained and explained in the final report; it was superseded by corrected FedAvg + local CL/replay, not hidden or overwritten.

## Cleanup classification

### Kept

- Final code, tests, frozen configurations, final checkpoints, and final metrics.
- Dataset, preprocessing code, reports, and all excluded-bank historical evidence.
- Validation summaries and experiment metadata needed to trace model selection.

### Moved

- All non-final artifact families from the repository root into `artifacts/historical/` categories.
- The original FedAvg final run into `artifacts/historical/original_fedavg_final/`.
- Superseded reports, scripts, and model runners into explicit historical/legacy directories.

### Deleted

- Local `.venv/` environment.
- Python `__pycache__/` directories and compiled bytecode.
- macOS `.DS_Store` files.
- Untracked duplicate `README 2.md`, which repeated the contradictory pre-final root README.

No bank data, final results, original FedAvg evidence, or excluded-bank artifacts were deleted or overwritten.

Development checkpoint families under `validation_search/` and `corrected_development/` were conservatively preserved because some carry round-level and seed-level forensic provenance. They are isolated from final results and documented as historical rather than treated as active artifacts.

## Reproducibility boundary

The September evaluation is complete and frozen. Repository verification must use unit tests, syntax checks, imports, JSON validation, and CLI `--help` checks only. Any future modeling phase should create a new experiment directory, select only on development/validation data, and reserve a later untouched temporal holdout.

## Frozen reproduction commands

Create an environment and install the minimal active dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The following commands record the frozen pipeline. Substitute each `SEED` with 42, 52, and 62. They include September scoring and therefore must only be used for an intentional clean-room reproduction—not for further tuning or repeated evaluation.

```bash
python -B src/gnn/causal_temporal_graphsage.py --bank all --shared-encoder --hidden-channels 128 --dropout 0.25 --learning-rate 0.001 --weight-decay 0.0001 --event-batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --epochs 25 --patience 5 --seed SEED --output-dir artifacts/final_evaluation/local/seed_SEED

python -B src/gnn/continual_temporal_graphsage.py --bank all --hidden-channels 128 --dropout 0.25 --learning-rate 0.001 --weight-decay 0.0001 --batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --epochs-per-task 20 --replay-size 2000 --seed SEED --output-dir artifacts/final_evaluation/cl/seed_SEED

python -B src/gnn/federated_causal_temporal_graphsage.py --algorithm fedavg --local-training continual_replay --replay-size 2000 --rounds 16 --patience 0 --local-epochs 3 --client-weighting uniform --server-learning-rate 1.0 --hidden-channels 128 --dropout 0.25 --batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --learning-rate 0.001 --weight-decay 0.0001 --optimizer adamw --seed SEED --output-dir artifacts/final_evaluation/corrected_fedavg/seed_SEED
```

Non-training verification:

```bash
python -m unittest discover -s tests -v
python src/gnn/causal_temporal_graphsage.py --help
python src/gnn/continual_temporal_graphsage.py --help
python src/gnn/federated_causal_temporal_graphsage.py --help
python src/gnn/evaluate_frozen_temporal.py --help
```
