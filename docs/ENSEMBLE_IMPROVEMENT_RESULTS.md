# Improvement beyond 87.1%: frozen three-model ensemble

Completed 23 October 2026 on branch `sia`. Equal-weight probability averaging of
the three existing context-replay/temporal-harmonization models achieves
**0.906778 September macro average precision (90.68%)**. The previous reported
individual-model mean is **0.8710 ± 0.0611 (87.10%)**. The descriptive difference
is **+3.58 percentage points**.

This is one three-member ensemble, not a new three-seed mean or an architecture
improvement. It requires three forward passes and three account-memory states.
The historical best individual seed already scored 0.9316; this ensemble does
not exceed that seed. The comparison is with the reported mean across all three
individual models, and no seed was removed.

## Measured results

| System | August validation macro AP | September macro AP |
|---|---:|---:|
| Existing individual models, mean ± SD | 0.8114 ± 0.1432 | 0.8710 ± 0.0611 |
| Fixed equal-probability ensemble | **0.885944** | **0.906778** |

| Bank | Ensemble validation AP | Ensemble September AP |
|---|---:|---:|
| JPMorgan Chase | 0.830082 | 0.990053 |
| Wells Fargo | 0.962283 | 0.959906 |
| Key Bank | 0.865468 | 0.770376 |

At 25 alerts per bank over September, ensemble macro precision is 0.866667 and
macro recall is 0.823148. Threshold-based classification is a separate question:
JPMorgan's validation-selected threshold yields September precision 0.20 and
recall 1.0, despite strong ranking. This is not a calibrated production alert policy.

## Evaluation protocol

The equal-weight rule and members (seeds 42, 52, 62) were written to `DESIGN.json`
before ensemble scoring. No ensemble weights were fitted. The validation phase
loaded June–July and August only, reconstructed causal state separately for each
member, and selected bank thresholds from August. Validation performance exceeded
the historical mean, so the fixed ensemble was evaluated once on September.

`FREEZE.json` records checkpoint hashes, evaluator source hash, development-data
fingerprints, thresholds, and August results. The evaluation verifies these
before accessing September, refuses to overwrite an existing evaluation, and
uses the frozen thresholds. Reloaded August macro AP exactly matches the freeze.
There was no model/weight/threshold change after viewing September.

**September is exploratory:** it was inspected repeatedly in the project's
history. This is a measured result on that period, not an unbiased new test
estimate. It has no multi-ensemble confidence interval. A later untouched period
and independent data are still needed for generalization claims.

## What this means for novelty

Ensembling is established practice and is not the novel contribution. A separate
implemented candidate, the causal bank/task tail-ranking guard, was compared
against matched continuation in six runs. It was too restrictive and did not
beat the control. Its code, diagnostics, prior-art comparison and negative result
are preserved in [CAUSAL_TAIL_GUARD_STUDY.md](CAUSAL_TAIL_GUARD_STUDY.md).

The literature already includes [federated continual graph learning for AML
(FC2L)](https://link.springer.com/article/10.1007/s10618-026-01192-0),
[federated gradient correction (FedAGC)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_FedAGC_Federated_Continual_Learning_with_Asymmetric_Gradient_Correction_ICCV_2025_paper.pdf),
and [federated line search (FedNL)](https://proceedings.mlr.press/v162/safaryan22a.html).
Neither this search nor the implementation establishes an unprecedented method.
The performance target has been exceeded in this exploratory ensemble comparison;
the requested established novelty and guaranteed publication acceptance have not.

## Artifacts and reproduction

- Entry point: `scripts/evaluate_seed_ensemble.py` (`freeze` and `evaluate`).
- Fixed design and validation evidence: `artifacts/research_seed_ensemble/DESIGN.json`
  and `FREEZE.json`.
- Measured result: `artifacts/research_seed_ensemble/exploratory_september/metrics.json`.
- Existing members: `artifacts/research_adaptive_fcl/warm_start/federated/context_harmonized/seed_{42,52,62}/global_model.pt`.
- Guard comparison: `artifacts/research_tail_guard/summary.json` and all per-round logs.

Preserve the completed evaluation; do not repeatedly rescore September to search
for a better number. For reproduction on new development data, use a new output
directory and reserve a genuinely unseen test period.

Validation completed: 33 relevant tests pass (6 guard, 3 ensemble, 17 existing
adaptive-FCL, 7 existing temporal protocol). All six guard-study jobs and the
separate frozen ensemble evaluation completed successfully. No original model
checkpoint, dataset or previous result was overwritten. The simulator/evaluator
is implemented; Kafka integration has not been changed.
