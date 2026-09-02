# Temporal GNN optimization report

## Scope and evaluation status

The active cohort is JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp data and historical artifacts were left intact and were not used in new modeling runs. All configuration decisions in this optimization were made from August validation PR-AUC. September was scored only after configurations and checkpoints were frozen.

September cannot honestly be called a genuinely untouched final holdout: legacy repository artifacts already contain repeated September results, and the three-bank cohort was chosen after historical performance inspection. The repository contains no post-September raw period. Consequently, the numbers below are **frozen confirmatory test diagnostics**, not unbiased untouched-holdout estimates. A later temporal period is required for a defensible final paper claim.

## A. Historical baseline

The historical active-cohort reference is approximately 0.383 macro test PR-AUC for the three-seed local causal temporal model, 0.295 for ten-round FedAvg, and 0.263 for federated continual learning with replay. These runs used earlier code/protocols and are retained for auditability, not treated as directly comparable optimized results.

## B. Problems discovered

- Temporal micro-batches with no positive label were skipped, coupling negative sampling to positive-event locations and discarding most ordinary transactions from the loss.
- Local checkpoint selection used an account-memory state generated while weights changed during the epoch; restored checkpoints did not reproduce the recorded validation PR-AUC.
- Exploratory local runs had no validation-only mode and therefore always inspected September.
- Continual and older federated scripts did not consistently select checkpoints/rounds using macro validation PR-AUC.
- Client volume differs by 5.64× and fraud rate by 8.68×. JPMorgan also has the largest mean FedAvg update norm; sample-count weighting gives it about 63% of aggregation mass.
- The single-process simulation centrally constructs a common training-only feature transform before federated training. No raw rows are exchanged during rounds, but a production privacy claim requires a pre-agreed schema or privacy-preserving aggregation of preprocessing statistics.
- September has already been inspected historically, so a truly untouched final estimate is unavailable.

## C. Optimizations performed

- Replaced batch-conditioned sampling with one seeded, stream-wide negative sample per epoch while preserving causal updates from every event.
- Rebuilt causal training memory under frozen end-of-epoch weights before validation checkpoint selection.
- Added validation-only execution, frozen-checkpoint evaluation, Recall/Precision@10/25/50, shared training-only encoders, client-weighting alternatives, FedProx, update-norm diagnostics, and separate non-overwriting experiment directories.
- Tuned only against validation PR-AUC. Final candidates used seeds 42, 52, and 62.

## D. Validation candidates (seed 42 development sweep)

| Stage | Candidate | Macro validation PR-AUC |
|---|---|---:|
| local | h64_lr1e3_n20_d25 | 0.3340 |
| local | h32_lr1e3_n20_d25 | 0.3288 |
| local | h32_lr3e4_n20_d25 | 0.2293 |
| local | h32_lr1e3_n50_d25 | 0.2290 |
| continual | e20_r2000 | 0.3160 |
| continual | e15_r2000 | 0.2959 |
| continual | e20_r5000 | 0.2907 |
| continual | e20_r500 | 0.2759 |
| continual | e10_r0 | 0.2668 |
| continual | e20_r0 | 0.2597 |
| continual | e10_r2000 | 0.2429 |
| continual | e30_r2000 | 0.2376 |
| continual | e10_r500 | 0.2335 |
| fedavg | e2_lr1e3_samples | 0.3576 |
| fedavg | e2_lr7e4_samples | 0.3549 |
| fedavg | e2_lr5e4_samples | 0.3452 |
| fedavg | e2_lr1e3_sqrt | 0.3364 |
| fedavg | e1_lr1e3_uniform | 0.2980 |
| fedavg | e1_lr1e3_sqrt | 0.2977 |
| fedavg | e1_lr1e3_samples | 0.2951 |
| fedavg | e1_lr3e4_samples | 0.2304 |
| fedprox | mu_0 (FedAvg equivalence control) | 0.3572 |
| fedprox | e1_mu_0p0001 | 0.2955 |
| fedprox | mu_0p0001 | 0.2945 |
| fedprox | mu_0p00001 | 0.2945 |
| fedprox | e1_mu_0p001 | 0.2937 |
| fedprox | mu_0p000001 | 0.2918 |
| fedprox | mu_0p001 | 0.2896 |
| fedprox | mu_0p01 | 0.2890 |
| fedprox | e1_mu_0p01 | 0.2866 |
| fedprox | mu_0p1 | 0.2725 |

## E. Frozen configurations

- Local: hidden 32, dropout 0.25, AdamW learning rate 1e-3, weight decay 1e-4, negative ratio 20, batch 1024, maximum 25 epochs, patience 5.
- CL: same backbone/optimizer, 20 epochs per June/July task, replay capacity 2,000.
- FedAvg: same backbone, 2 local epochs, at most 12 rounds, round patience 3, sample-count weighting.
- FedProx: same backbone, 1 local epoch, at most 12 rounds, round patience 3, sample-count weighting, μ=1e-4.

The μ=0 FedProx control exactly reproduced FedAvg, confirming the implementation boundary. Every positive μ candidate was worse on validation. The nonzero μ=1e-4/one-local-epoch setting was frozen because it was the strongest actual FedProx candidate at seed 42; multi-seed evaluation then exposed its instability.

## F. Multi-seed validation stability

| Method | Macro validation PR-AUC |
|---|---:|
| Local Temporal GNN | 0.329 ± 0.019 |
| + Continual Learning | 0.318 ± 0.044 |
| + FedAvg | 0.338 ± 0.022 |
| + FedProx | 0.241 ± 0.091 |

## G. Frozen confirmatory September results

| Method | Macro PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---:|---:|---:|---:|
| Local Temporal GNN | 0.237 ± 0.025 | 0.292 | 0.129 | 0.291 |
| + Continual Learning | 0.324 ± 0.024 | 0.386 | 0.247 | 0.339 |
| + FedAvg | 0.273 ± 0.036 | 0.293 | 0.172 | 0.354 |
| + FedProx | 0.202 ± 0.016 | 0.179 | 0.124 | 0.303 |

| Method | PR-AUC | Recall@10 | Precision@10 | Recall@25 | Precision@25 | Recall@50 | Precision@50 | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local Temporal GNN | 0.237 ± 0.025 | 0.198 ± 0.023 | 0.556 ± 0.038 | 0.281 ± 0.045 | 0.329 ± 0.041 | 0.371 ± 0.033 | 0.213 ± 0.012 | 0.244 ± 0.062 | 0.458 ± 0.193 | 0.250 ± 0.021 | 0.923 ± 0.013 |
| + Continual Learning | 0.324 ± 0.024 | 0.252 ± 0.021 | 0.689 ± 0.038 | 0.345 ± 0.036 | 0.391 ± 0.041 | 0.443 ± 0.061 | 0.247 ± 0.024 | 0.270 ± 0.123 | 0.584 ± 0.358 | 0.234 ± 0.062 | 0.923 ± 0.016 |
| + FedAvg | 0.273 ± 0.036 | 0.209 ± 0.042 | 0.600 ± 0.088 | 0.335 ± 0.046 | 0.391 ± 0.043 | 0.402 ± 0.058 | 0.231 ± 0.025 | 0.272 ± 0.085 | 0.524 ± 0.246 | 0.303 ± 0.018 | 0.928 ± 0.010 |
| + FedProx | 0.202 ± 0.016 | 0.173 ± 0.005 | 0.522 ± 0.019 | 0.225 ± 0.042 | 0.280 ± 0.040 | 0.270 ± 0.059 | 0.164 ± 0.027 | 0.168 ± 0.090 | 0.381 ± 0.167 | 0.195 ± 0.060 | 0.882 ± 0.007 |

All entries are macro-over-bank values summarized as mean ± sample standard deviation over seeds, except the per-bank PR-AUC columns, which are seed means.

## H. Progression analysis

Validation does not show a reliable monotonic progression: Local 0.329 ± 0.019, CL 0.318 ± 0.044, FedAvg 0.338 ± 0.022, FedProx 0.241 ± 0.091. FedAvg's small validation gain over local overlaps seed variation and masks weak Wells Fargo performance.

Replay retention is heterogeneous. Mean June forgetting (positive means degradation) is -0.014 ± 0.053 macro across seeds: JPMorgan 0.004 ± 0.210, Wells Fargo 0.027 ± 0.072, and Key Bank -0.073 ± 0.029. Replay reliably preserves/improves Key Bank's older task, but JPMorgan and Wells Fargo are not stable enough to claim general forgetting prevention.

On the frozen September diagnostic, CL is strongest, FedAvg improves over local but not CL, and FedProx is worst. Replay therefore appears useful for later-period adaptation despite weaker/noisier August selection, whereas the chosen proximal constraint over-regularizes local learning and does not solve the observed client heterogeneity.

## I. Scientific conclusion

The evidence does **not** support Local < CL < FedAvg < FedProx. It supports a narrower claim: replay-based continual learning improved the already-inspected later-period diagnostic, and FedAvg produced a modest gain over local, but FedProx did not improve stability or PR-AUC. These conclusions require confirmation on a genuinely untouched post-September temporal holdout before being presented as generalization results.
