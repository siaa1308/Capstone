# Final Temporal GNN Pipeline

This directory contains the active model implementation for the frozen three-bank experiment. The authoritative configuration is `../../configs/final_pipeline.json`; paper-ready metrics are in `../../docs/FINAL_EVALUATION_REPORT.md`.

## Active modules

| Module | Role |
|---|---|
| `causal_temporal_graphsage.py` | Local chronological Temporal GraphSAGE with GRU account memory and corrected TBPTT |
| `continual_temporal_graphsage.py` | Local June → July continual training with historical replay |
| `federated_causal_temporal_graphsage.py` | FedAvg orchestration with bank-local CL/replay, diagnostics, and validation round selection |
| `evaluate_frozen_temporal.py` | Explicit checkpoint evaluation utility |
| `run_temporal_multiseed.py` | Multi-seed Local runner |

The active cohort constant is `("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")`. Citi and Fifth Third are deliberately excluded from new modeling, but their data and historical artifacts remain intact.

## Frozen settings

Common settings are hidden size 128, batch size 1,024, TBPTT 2, dropout 0.25, negative ratio 20, AdamW learning rate `1e-3`, weight decay `1e-4`, and seeds 42/52/62.

- Local: maximum 25 epochs, patience 5.
- CL: 20 epochs per task, replay capacity 2,000.
- FedAvg: local CL/replay every round, three local epochs per task, 16 rounds, equal client weighting.

FedProx is not a final method. Its CLI option remains in the shared federated research runner so old experiments stay reproducible; do not present it as part of the final pipeline.

## Safety boundary

August validation PR-AUC selected all configurations and checkpoints. September is the completed final holdout and must not be used for further tuning. The commands used for the frozen evaluations are documented in `../../docs/FINAL_EVALUATION_REPORT.md`; they are records, not instructions to rerun the completed test.

## Non-training checks

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/gnn/causal_temporal_graphsage.py --help
python src/gnn/continual_temporal_graphsage.py --help
python src/gnn/federated_causal_temporal_graphsage.py --help
python src/gnn/evaluate_frozen_temporal.py --help
```

Superseded static and early federated runners are under `legacy/`. They are retained for auditability and are not maintained entry points.
