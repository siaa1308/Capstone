# Paper Evaluation Protocol

## Objective

The primary metric is test PR-AUC.  Accuracy is not reported as a model-quality
metric because the positive class represents roughly 0.1% of transactions and a
trivial all-negative classifier would obtain misleadingly high accuracy.

## Temporal protocol

| Role | Period | Permitted use |
|---|---|---|
| Training | June--July 2025 | Fit model parameters and replay buffers |
| Validation | August 2025 | Select hyperparameters, early-stopping epoch, calibration, and threshold |
| Test | September 2025 | One frozen final evaluation only |

No test labels may be used for feature engineering, choosing a method, selecting
an alert threshold, or choosing a checkpoint.  Results already inspected during
development are labelled development-test diagnostics; final paper numbers must
come from a frozen configuration and, where possible, a fresh temporal holdout.

## Current holdout status (2026-09-02)

September results already exist in legacy artifacts and the three-bank cohort
was chosen after historical performance inspection. The optimized runs selected
every setting on August validation only and evaluated frozen checkpoints on
September, but those values are frozen confirmatory diagnostics—not a genuinely
untouched test estimate. The repository contains no later raw period. A
post-September period is required for an unbiased final generalization claim.

## Required comparisons

1. Leakage-safe tabular XGBoost baseline.
2. Static GraphSAGE baseline.
3. Local causal temporal GraphSAGE (one model per bank).
4. Federated causal temporal GraphSAGE using FedAvg.
5. Federated continual causal temporal GraphSAGE without replay.
6. Federated continual causal temporal GraphSAGE with replay.

Every GNN comparison must use the same safe feature schema, hidden size, negative
sampling policy, seed list, and validation-only selection rules unless the
experiment explicitly studies one of those variables.

## Metrics

For every bank and seed, report:

- PR-AUC (primary, threshold independent)
- ROC-AUC (supporting only)
- precision, recall, and F1 at an August-selected threshold
- precision@K and recall@K for K = 10, 25, and 50 alerts

The main table reports bank-level values plus macro mean and standard deviation
over banks and fixed seeds (42, 52, 62).  Do not pool transactions as the sole
summary because JPMorgan Chase has substantially more events than the other
simulated clients.

## Continual-learning retention

For each chronological update, report old-task PR-AUC before and after the update:

```
forgetting = PR-AUC(before update) - PR-AUC(after update)
```

Retention diagnostics must be labelled separately from the future held-out test
result; they are not a replacement for it.
