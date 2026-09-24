# Adaptive FCL validation results

Development evidence only. No September scores are included.

| Method | Seeds | Macro August PR-AUC | Paired gain over matched baseline |
|---|---|---:|---:|
| federated/baseline | 42 | 0.4869 (one seed) | — |
| federated/context_replay | 42 | 0.4883 (one seed) | +0.0014 |

Each seed first averages PR-AUC equally over banks. ± is sample SD across seeds.
Single-seed screen rows are not comparable in certainty to three-seed confirmation.
The historical final FedAvg validation reference is 0.7083 ± 0.0937; its September reference is 0.7590 ± 0.0360.
Do not compare a new August score to the historical September score as an improvement estimate.

The matched new baseline uses the same corrected calendar boundary, runtime, RNG policy, and compute budget as the candidates.
