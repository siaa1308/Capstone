# Federated Continual Learning for AML Detection

This capstone evaluates privacy-preserving anti-money-laundering detection on a chronological, highly imbalanced transaction graph. The final pipeline progresses from a bank-local temporal GNN, to replay-based continual learning, to federated continual learning with FedAvg. PR-AUC is the primary metric; Recall@K and Precision@K measure performance under fixed analyst alert budgets.

## Final result

| Frozen method | August validation PR-AUC | September PR-AUC |
|---|---:|---:|
| Local Temporal GNN | 0.404 ± 0.063 | 0.442 ± 0.029 |
| Continual Learning + replay | 0.599 ± 0.038 | 0.581 ± 0.016 |
| FedAvg + local CL/replay | **0.708 ± 0.094** | **0.759 ± 0.036** |

The corrected FedAvg pipeline is the strongest final model. It improves September PR-AUC by 0.318 absolute (71.9% relative) over Local and by 0.178 absolute (30.5% relative) over CL. The earlier FedAvg result of 0.228 ± 0.035 is retained as a diagnostic negative result under `artifacts/historical/original_fedavg_final/`; it used a mismatched no-replay, sample-weighted procedure.

See [`docs/FINAL_EVALUATION_REPORT.md`](docs/FINAL_EVALUATION_REPORT.md) for the authoritative paper-ready results and limitations.

## Cohort and temporal protocol

New modeling uses exactly three clients:

- JPMorgan Chase
- Wells Fargo
- Key Bank

Citi and Fifth Third Bancorp were excluded from new runs after historical development analysis. Their data and historical artifacts remain preserved for auditability.

| Split | Period | Purpose |
|---|---|---|
| Training | June–July 2025 | Learning and replay construction |
| Validation | August 2025 | Configuration, checkpoint, round, and threshold selection |
| Test | September 2025 | One frozen final evaluation |

The corrected temporal implementation processes transactions chronologically in batches of 1,024 and uses two-step truncated backpropagation through time (TBPTT). September data is not used for training or model selection.

## Final architecture

1. `causal_temporal_graphsage.py`: bank-local causal Temporal GraphSAGE with GRU state.
2. `continual_temporal_graphsage.py`: the same model trained on June, then July with a 2,000-event replay buffer.
3. `federated_causal_temporal_graphsage.py`: three bank-local CL/replay clients, three local epochs per task, 16 FedAvg rounds, and equal client weights.

FedProx and superseded static/legacy runners are not part of the final pipeline. Exact immutable settings are in [`configs/final_pipeline.json`](configs/final_pipeline.json).

## Repository layout

```text
configs/                   Frozen final configuration
data/final_temporal_dataset/
                           Chronological prepared data and data reports
src/gnn/                   Final model, training, and evaluation code
scripts/                   Dataset preparation and audit utilities
tests/                     Temporal-integrity and protocol regression tests
artifacts/final_evaluation/
                           Final Local, CL, and corrected-FedAvg results
artifacts/historical/      Preserved superseded experiments and diagnostics
docs/FINAL_EVALUATION_REPORT.md
                           Authoritative research report
docs/REPOSITORY_MANIFEST.md
                           File classification and reproducibility map
```

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The final runnable pipeline depends only on NumPy, pandas, scikit-learn, and PyTorch. Historical static GraphSAGE code may require additional packages and is retained for audit, not as an active entry point.

## Reproducibility and safe use

Run protocol checks without training:

```bash
python -m unittest discover -s tests -v
python src/gnn/causal_temporal_graphsage.py --help
python src/gnn/continual_temporal_graphsage.py --help
python src/gnn/federated_causal_temporal_graphsage.py --help
```

The exact commands used for the completed runs are recorded in the final report. Do not rerun or tune against September as part of normal repository use. Any new research cycle should use validation-only development and a genuinely untouched later temporal holdout.

## Key documentation

- [`docs/FINAL_EVALUATION_REPORT.md`](docs/FINAL_EVALUATION_REPORT.md): final results, exact protocol, operational metrics, and scientific conclusion.
- [`docs/FEDAVG_COLLAPSE_INVESTIGATION.md`](docs/FEDAVG_COLLAPSE_INVESTIGATION.md): why the original FedAvg setup failed and how validation-only diagnosis corrected it.
- [`docs/PAPER_EVALUATION_PROTOCOL.md`](docs/PAPER_EVALUATION_PROTOCOL.md): evaluation policy and test-set safeguards.
- [`data/final_temporal_dataset/README.md`](data/final_temporal_dataset/README.md): prepared dataset layout and split details.
- [`docs/REPOSITORY_MANIFEST.md`](docs/REPOSITORY_MANIFEST.md): final versus historical file map.

## Scientific caveats

The data is simulated/experimental, client distributions are non-IID, only three institutions are active, and the cohort was selected post hoc from earlier development evidence. The corrected FedAvg result was selected strictly from August validation and scored once on September, but the project contains older September results; confirmation on a later untouched period is still required before making broad generalization claims.
