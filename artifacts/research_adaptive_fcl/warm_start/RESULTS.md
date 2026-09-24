# Adaptive FCL validation results

Development evidence only. No September scores are included.

| Method | Seeds | Macro August PR-AUC | Paired gain over matched baseline |
|---|---|---:|---:|
| federated/baseline | 42, 52, 62 | 0.7673 ± 0.1216 | — |
| federated/combined | 42 | 0.8227 (one seed) | -0.0372 |
| federated/context_harmonized | 42, 52, 62 | 0.8114 ± 0.1432 | +0.0441 |
| federated/context_replay | 42 | 0.8968 (one seed) | +0.0370 |
| federated/context_retention | 42 | 0.7317 (one seed) | -0.1282 |
| federated/temporal_aggregation | 42 | 0.9034 (one seed) | +0.0435 |

Each seed first averages PR-AUC equally over banks. ± is sample SD across seeds.
Single-seed screen rows are not comparable in certainty to three-seed confirmation.
The historical final FedAvg validation reference is 0.7083 ± 0.0937; its September reference is 0.7590 ± 0.0360.
Do not compare a new August score to the historical September score as an improvement estimate.

The matched new baseline uses the same corrected calendar boundary, runtime, RNG policy, and compute budget as the candidates.
