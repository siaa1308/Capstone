# Adaptive FCL validation results

Development evidence only. No September scores are included.

| Method | Seeds | Macro August PR-AUC | Paired gain over matched baseline |
|---|---|---:|---:|
| federated/baseline | 42, 52, 62 | 0.7492 ± 0.1118 | — |
| federated/combined | 42 | 0.4122 (one seed) | -0.3741 |
| federated/context_replay | 42, 52, 62 | 0.8331 ± 0.0704 | +0.0839 |
| federated/context_retention | 42 | 0.4129 (one seed) | -0.3734 |
| federated/temporal_aggregation | 42 | 0.7407 (one seed) | -0.0456 |
| local/baseline | 42, 52, 62 | 0.5214 ± 0.0552 | — |
| local/context_replay | 42, 52, 62 | 0.6326 ± 0.0446 | +0.1112 |
| local/context_retention | 42, 52, 62 | 0.5208 ± 0.0199 | -0.0007 |

Each seed first averages PR-AUC equally over banks. ± is sample SD across seeds.
Single-seed screen rows are not comparable in certainty to three-seed confirmation.
The historical final FedAvg validation reference is 0.7083 ± 0.0937; its September reference is 0.7590 ± 0.0360.
Do not compare a new August score to the historical September score as an improvement estimate.

The matched new baseline uses the same corrected calendar boundary, runtime, RNG policy, and compute budget as the candidates.
