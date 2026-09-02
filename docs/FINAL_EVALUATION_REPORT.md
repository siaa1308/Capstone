# Final Evaluation Report

## 1. Final Experimental Objective

This experiment compares three frozen, causally trained AML pipelines: corrected Local Temporal GNN, corrected continual learning (CL) with replay, and federated temporal GNN with FedAvg. Configurations were fixed from validation evidence before the final September scoring. The primary outcome is PR-AUC because fraud labels are severely imbalanced; analyst-budget and conventional classification metrics provide operational context.

## 2. Dataset and Cohort

The active cohort is JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp were excluded from new modeling runs after weaker historical development PR-AUC; their data, metrics, checkpoints, and other historical artifacts remain preserved.

The fixed chronological split is:

| Split | Period | Use |
|---|---|---|
| Training | June–July 2025 | Parameter learning and replay construction |
| Validation | August 2025 | Configuration/checkpoint/round selection and threshold fitting |
| Test | September 2025 | One final frozen evaluation |

No split boundary was changed. Development runs did not load September. The initial frozen evaluation ran Local, CL, and the original FedAvg once for each seed. A later validation-only forensic audit showed that original FedAvg had not used CL/replay and was dominated by sample weighting. The corrected FedAvg configuration was separately frozen in `artifacts/final_evaluation/CORRECTED_FEDAVG_FROZEN_CONFIG.json` and then evaluated once per seed. Local and CL were not rerun. No configuration was changed after its September result.

## 3. Experimental Protocol

All methods use the same approved causal node/edge features and a shared feature encoder fitted only on the three clients' training rows. Transactions are processed chronologically in batches of 1,024. Each event is scored from account state available before its batch updates memory. Two-batch truncated backpropagation through time (TBPTT) is mandatory; this preserves gradients into time projection, message projection, and GRU memory-update modules while bounding the graph. A regression test verifies that the GRU parameters update.

Common optimization settings are AdamW, learning rate `1e-3`, dropout `0.25`, weight decay `1e-4`, hidden size 128, and a 20:1 stream-wide negative sample that retains every positive. Every transaction still updates causal memory, including events not selected for the loss. Seeds are 42, 52, and 62; no seed was removed. PR-AUC selects Local checkpoints and the FedAvg communication round. Classification thresholds are selected from August labels only. September labels never affect training, replay, early stopping, round selection, or thresholds.

## 4. Final Model Configurations

### Local Temporal GNN

- Hidden size 128; current one-layer temporal GraphSAGE-style mean neighbor aggregation and GRU memory update.
- Batch 1,024; TBPTT 2; negative ratio 20.
- AdamW `lr=1e-3`, weight decay `1e-4`; dropout `0.25`.
- Maximum 25 epochs; patience 5; per-bank checkpoint selected by validation PR-AUC.

### Continual Learning + Replay

- Same corrected hidden-128 backbone, optimizer, batching, and TBPTT settings.
- Two chronological tasks: June, then July plus replay.
- 20 epochs per task; replay capacity 2,000.
- Replay includes all available historical positives, then uniformly sampled historical negatives up to capacity.

### FedAvg

- Same corrected hidden-128 backbone, optimizer, batching, and TBPTT settings.
- Three independent bank clients; only parameters are aggregated.
- Every client performs corrected local continual training: June, then July plus a 2,000-event June replay sample.
- Three local epochs per task per round; 16 communication rounds.
- Uniform client weighting; full server averaging.
- The global checkpoint is selected by the highest unweighted macro bank validation PR-AUC.

The historically frozen FedAvg used two full-stream local epochs, at most 12 rounds, patience 3, sample-count weighting, and **no replay**. It is preserved below as “Original FedAvg (diagnostic)” and is superseded for the final method comparison by the corrected federated continual-learning configuration.

## 5. Validation Results

These values come from the frozen final runs and are macro averages over banks, summarized across seeds with sample standard deviation.

| Method | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---:|---:|---:|---:|
| Corrected Local | 0.404 ± 0.063 | 0.386 ± 0.099 | 0.291 ± 0.075 | 0.536 ± 0.037 |
| Corrected CL + replay | 0.599 ± 0.038 | 0.724 ± 0.125 | 0.526 ± 0.035 | 0.546 ± 0.039 |
| Original FedAvg (diagnostic) | 0.252 ± 0.044 | 0.315 ± 0.079 | 0.078 ± 0.032 | 0.361 ± 0.044 |
| **Corrected FedAvg + local CL/replay** | **0.708 ± 0.094** | 0.714 ± 0.057 | 0.693 ± 0.211 | 0.719 ± 0.016 |

The Local hidden-128 configuration was frozen from its preceding validation-only confirmation (`0.431 ± 0.062`). Corrected FedAvg was selected from a validation-only estimate of `0.736 ± 0.094`; its independently executed frozen final runs reproduced `0.708 ± 0.094` validation. This difference was retained without reruns or post-test changes.

## 6. Final September Results

All values are macro over banks followed by mean ± sample SD across seeds.

| Method | PR-AUC | Recall@10 | Precision@10 | Recall@25 | Precision@25 | Recall@50 | Precision@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected Local | 0.442 ± 0.029 | 0.224 ± 0.023 | 0.656 ± 0.051 | 0.397 ± 0.023 | 0.484 ± 0.020 | 0.580 ± 0.042 | 0.344 ± 0.014 |
| Corrected CL + replay | 0.581 ± 0.016 | 0.289 ± 0.018 | 0.744 ± 0.019 | 0.533 ± 0.050 | 0.578 ± 0.041 | 0.715 ± 0.029 | 0.409 ± 0.015 |
| Original FedAvg (diagnostic) | 0.228 ± 0.035 | 0.202 ± 0.013 | 0.589 ± 0.038 | 0.306 ± 0.014 | 0.347 ± 0.013 | 0.333 ± 0.008 | 0.184 ± 0.004 |
| **Corrected FedAvg + local CL/replay** | **0.759 ± 0.036** | **0.367 ± 0.025** | **0.956 ± 0.051** | **0.672 ± 0.033** | **0.716 ± 0.028** | **0.797 ± 0.011** | **0.427 ± 0.013** |

| Method | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Corrected Local | 0.451 ± 0.093 | 0.393 ± 0.040 | 0.363 ± 0.061 | 0.969 ± 0.012 |
| Corrected CL + replay | 0.666 ± 0.015 | 0.428 ± 0.019 | 0.445 ± 0.006 | 0.958 ± 0.008 |
| Original FedAvg (diagnostic) | 0.126 ± 0.021 | 0.444 ± 0.079 | 0.184 ± 0.031 | 0.949 ± 0.006 |
| **Corrected FedAvg + local CL/replay** | **0.724 ± 0.057** | **0.606 ± 0.037** | **0.551 ± 0.015** | **0.984 ± 0.009** |

CL improves on Local by **0.140 absolute PR-AUC**, or **31.7% relative**. Corrected FedAvg improves on Local by **0.318 absolute** (**71.9% relative**) and on CL by **0.178 absolute** (**30.5% relative**). It improves over original FedAvg by **0.531 absolute** (**232.3% relative**).

## 7. Per-Bank Results

September PR-AUC, mean ± sample SD across all three seeds:

| Method | JPMorgan Chase | Wells Fargo | Key Bank |
|---|---:|---:|---:|
| Corrected Local | 0.275 ± 0.059 | 0.446 ± 0.075 | 0.604 ± 0.029 |
| Corrected CL + replay | 0.563 ± 0.156 | 0.613 ± 0.030 | 0.568 ± 0.087 |
| Original FedAvg (diagnostic) | 0.254 ± 0.085 | 0.133 ± 0.016 | 0.298 ± 0.010 |
| **Corrected FedAvg + local CL/replay** | **0.841 ± 0.096** | **0.827 ± 0.055** | **0.608 ± 0.028** |

Corrected FedAvg is strongest at JPMorgan and Wells Fargo and narrowly exceeds Local at Key Bank. Equal weighting prevents JPMorgan's larger sample count from erasing Wells Fargo's locally useful solution.

## 8. Multi-Seed Stability

| Method | Seed 42 | Seed 52 | Seed 62 | September mean ± SD |
|---|---:|---:|---:|---:|
| Corrected Local | 0.417 | 0.434 | 0.473 | 0.442 ± 0.029 |
| Corrected CL + replay | 0.596 | 0.584 | 0.565 | 0.581 ± 0.016 |
| Original FedAvg (diagnostic) | 0.266 | 0.222 | 0.198 | 0.228 ± 0.035 |
| **Corrected FedAvg + local CL/replay** | **0.752** | **0.798** | **0.727** | **0.759 ± 0.036** |

Corrected FedAvg is strongest at every seed and has moderate macro variance. CL remains the most stable (`SD=0.016`), while corrected FedAvg has `SD=0.036`. Corrected FedAvg's per-bank variance is largest at JPMorgan (`SD=0.096`) but is far below the instability seen during validation at Wells Fargo.

## 9. Local → CL → FedAvg Analysis

The final evidence supports the full **Local → CL → corrected FedAvg** progression. Replay-based sequential training improves temporal generalization from 0.442 to 0.581 macro September PR-AUC. Corrected federated continual learning then improves it to 0.759.

The original 0.228 FedAvg result remains scientifically important: plain repeated full-stream client training, sample weighting, and no replay caused a severe collapse under non-IID client updates. The forensic audit measured a one-round Wells Fargo drop from 0.657 locally to 0.076 after sample-weighted aggregation. The corrected method uses the intended bank-local CL/replay procedure, equal weights, three local epochs per task, and enough rounds for collaboration to emerge. This supports federated collaboration under the corrected protocol; it does not imply generic FedAvg is robust to arbitrary non-IID training schedules.

## 10. Operational Significance

Recall@K is the fraction of all fraud events found when analysts can investigate only the top K alerts; Precision@K is the fraction of those K alerts that are truly fraudulent. With a 25-alert budget, corrected FedAvg finds 67.2% of fraud events per bank on average and 71.6% of investigated alerts are true positives. CL finds 53.3% with 57.8% precision, and Local finds 39.7% with 48.4% precision. At 50 alerts, corrected FedAvg reaches 79.7% recall and 42.7% precision, the strongest operational ranking quality.

## 11. Limitations

- Fraud labels are extremely rare; August contains few positives, making PR-AUC and checkpoint selection noisy.
- Banks are strongly non-IID in volume, fraud prevalence, graph structure, and feature distributions.
- The institutions and transactions are simulated/experimental and may not reproduce production AML behavior or operational constraints.
- The three-bank cohort was selected after historical performance inspection. This post-hoc selection limits unbiased generalization claims.
- Only three institutions are active, limiting federated diversity and statistical power.
- Legacy repository artifacts already contain September results. Although this final cycle froze configurations and did not optimize against September, September is not a historically pristine holdout for the project as a whole. A post-September temporal period is required for a fully unbiased external confirmation.
- The final Local validation figures vary somewhat from the immediately preceding validation-only confirmation, consistent with seed/thread-level numerical sensitivity; all final values are retained without reruns.
- Corrected FedAvg was developed after the original September FedAvg result exposed a failure. Its configuration was selected using validation only and evaluated once afterward, but the broader research sequence is adaptive; an untouched later period is still needed to confirm the corrected gain without project-level post-hoc influence.
- Corrected FedAvg validation remains noisy (`0.708 ± 0.094` in the final runs), especially for Wells Fargo, despite much stronger September stability.
- The simulation shares a training-only feature transformation for tensor compatibility. A production federation would require a pre-agreed schema or privacy-preserving preprocessing statistics.

## 12. Reproducibility

The consolidated final configuration is in `configs/final_pipeline.json`. The initial freeze record, which includes the original diagnostic FedAvg, is archived at `artifacts/historical/original_fedavg_final/INITIAL_FROZEN_CONFIGS.json`; corrected FedAvg was independently frozen in `artifacts/final_evaluation/CORRECTED_FEDAVG_FROZEN_CONFIG.json`. Final checkpoints and metrics are under:

- `artifacts/final_evaluation/local/seed_{42,52,62}`
- `artifacts/final_evaluation/cl/seed_{42,52,62}`
- `artifacts/historical/original_fedavg_final/fedavg/seed_{42,52,62}` (original diagnostic)
- `artifacts/final_evaluation/corrected_fedavg/seed_{42,52,62}` (final corrected method)

Commands below were executed once for each `SEED` in `42 52 62`:

```bash
.venv/bin/python -B src/gnn/causal_temporal_graphsage.py --bank all --shared-encoder --hidden-channels 128 --dropout 0.25 --learning-rate 0.001 --weight-decay 0.0001 --event-batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --epochs 25 --patience 5 --seed SEED --output-dir artifacts/final_evaluation/local/seed_SEED

.venv/bin/python -B src/gnn/continual_temporal_graphsage.py --bank all --hidden-channels 128 --dropout 0.25 --learning-rate 0.001 --weight-decay 0.0001 --batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --epochs-per-task 20 --replay-size 2000 --seed SEED --output-dir artifacts/final_evaluation/cl/seed_SEED

.venv/bin/python -B src/gnn/federated_causal_temporal_graphsage.py --algorithm fedavg --rounds 12 --patience 3 --local-epochs 2 --client-weighting samples --hidden-channels 128 --dropout 0.25 --batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --learning-rate 0.001 --weight-decay 0.0001 --optimizer adamw --seed SEED --output-dir artifacts/final_evaluation/fedavg/seed_SEED

.venv/bin/python -B src/gnn/federated_causal_temporal_graphsage.py --algorithm fedavg --local-training continual_replay --replay-size 2000 --rounds 16 --patience 0 --local-epochs 3 --client-weighting uniform --server-learning-rate 1.0 --hidden-channels 128 --dropout 0.25 --batch-size 1024 --tbptt-steps 2 --negative-ratio 20 --learning-rate 0.001 --weight-decay 0.0001 --optimizer adamw --seed SEED --output-dir artifacts/final_evaluation/corrected_fedavg/seed_SEED
```

These commands record the paths used when the experiments ran. During final repository cleanup, the original diagnostic FedAvg output was moved from `artifacts/final_evaluation/fedavg` to `artifacts/historical/original_fedavg_final/fedavg`; its contents were not overwritten.

Before corrected FedAvg evaluation, seven unit/protocol tests passed. Python syntax compilation, frozen-JSON validation, and `--help` CLI/import checks passed. The validation-only sanity audit confirmed replay capacity 2,000 in every client at every round and reproduced validation selection across all seeds. Validation warms testing causally, but neither validation nor testing labels update account memory. Thresholds are frozen from validation. Local and CL were not rerun.

## 13. Final Conclusion

Corrected federated continual learning with FedAvg is the strongest final method: `0.759 ± 0.036` September PR-AUC, a 71.9% relative gain over Local and 30.5% over CL, with the best Recall@K and Precision@K at every analyst budget. CL remains a strong, stable intermediate stage at `0.581 ± 0.016`, above Local at `0.442 ± 0.029`. The original `0.228 ± 0.035` FedAvg result is preserved as a diagnostic negative result caused by a mismatched no-replay procedure and dominant sample weighting. Under the corrected frozen protocol, the capstone evidence supports the hypothesized **Local → CL → FedAvg** progression, subject to confirmation on a genuinely untouched later temporal holdout.
