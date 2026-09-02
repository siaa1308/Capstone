# Corrected Local Temporal GNN: controlled validation gate

## Protocol

This gate used only training and validation data. September was neither loaded nor scored. All runs used the fixed temporal split, active three-bank cohort, shared training-only encoder, chronological event batches of 1,024, two-batch TBPTT, AdamW, weight decay `1e-4`, dropout `0.25`, negative ratio 20 unless it was the controlled variable, at most 25 epochs, patience 5, and validation PR-AUC selection.

The temporal-gradient correction was applied to Local and CL and is now mandatory (`tbptt_steps >= 2`). A regression test verifies that the Local training path updates the GRU temporal-memory parameters.

## Seed-42 one-factor screen

| Candidate | Only change from control | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---|---:|---:|---:|---:|
| Corrected control | none | 0.2792 | 0.2174 | 0.2516 | 0.3686 |
| Lower LR | `1e-3 → 5e-4` | 0.1103 | 0.1451 | 0.1206 | 0.0652 |
| Hidden 64 | `32 → 64` | **0.3216** | 0.3900 | 0.1273 | 0.4477 |
| More negatives | `20:1 → 50:1` | 0.2338 | 0.2263 | 0.1219 | 0.3533 |

Hidden size was the only beneficial change and was therefore the only candidate advanced to the fixed-seed check.

## Hidden-64 stability

| Seed | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---:|---:|---:|---:|---:|
| 42 | 0.3216 | 0.3900 | 0.1273 | 0.4477 |
| 52 | 0.2923 | 0.3614 | 0.1463 | 0.3692 |
| 62 | 0.3345 | 0.3995 | 0.1320 | 0.4721 |
| Mean ± sample SD | **0.3162 ± 0.0216** | 0.3836 ± 0.0198 | 0.1352 ± 0.0099 | 0.4297 ± 0.0537 |

## Gate decision

Hidden-64 is the strongest corrected Local candidate in this controlled phase and is reasonably stable, but it does not exceed the historical uncorrected Local validation result (`0.329 ± 0.019`). This is not evidence that the temporal-gradient defect should be restored. The corrected model is the scientifically valid foundation for subsequent work.

The next controlled phase should build CL on hidden-64 and compare replay capacity/ratio or replay sampling one factor at a time. No Local, CL, or FedAvg configuration is frozen for final September evaluation yet, and no September evaluation was performed in this gate.
