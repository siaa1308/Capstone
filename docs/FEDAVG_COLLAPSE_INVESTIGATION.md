# FedAvg Collapse Investigation (Validation Only)

## Scope and protocol guard

This investigation used only June–July training data and August validation data. Every run passed `--validation-only`, so `build_bank_data` loaded only `training` and `validation`. September was not loaded or scored, and `artifacts/final_evaluation` was not modified. The completed outputs are archived under `artifacts/historical/corrected_development/fedavg_collapse_investigation`.

## Implementation audit: final FedAvg versus corrected CL

| Component | Corrected local CL | FedAvg used in final evaluation | Match? |
|---|---|---|---|
| Hidden size | 128 | 128 | Yes |
| TBPTT | 2 | 2 | Yes |
| Replay | Capacity 2,000 | None | **No** |
| Local procedure | June, then July + June replay | Full June–July stream repeated each round | **No** |
| Features/preprocessing | Shared training-only encoder and approved causal features | Same | Yes |
| Temporal batching | Chronological, batch 1,024 | Same | Yes |
| Loss | Negative-sampled BCE | Same | Yes |
| Negative sampling | Stream-wide 20:1 within each task | Stream-wide 20:1 over full training stream | Partly |
| Optimizer | AdamW, `lr=1e-3`, `wd=1e-4`; state retained from June to July+replay | New AdamW per client per round; no task boundary | **No** |
| Checkpoint selection | Fixed task schedule; validation threshold only | Best global round by macro validation PR-AUC | Correct for FedAvg |
| Test isolation during development | Validation-only loader | Validation-only loader in this audit | Yes |

The answer to the central audit question is therefore **no**: the completed FedAvg pipeline was not bank-local continual learning followed by FedAvg. It was ordinary repeated full-stream local training followed by FedAvg. Replay was never active.

The separate historical `federated_continual_temporal_graphsage.py` was not a valid substitute: it used one-batch state detachment (no temporal/GRU credit assignment), loaded/evaluated testing unconditionally, lacked validation-selected checkpoints, and did not expose the frozen hidden/dropout/TBPTT protocol. It was not used in this investigation.

## Diagnostic evidence

### Exact local CL before one aggregation

Each client received the same initial global model and ran the exact 20-epoch-per-task June → July+replay procedure. Replay was confirmed at 2,000 local historical events per bank, and temporal parameters changed at every client.

| Stage | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---:|---:|---:|---:|
| Bank-local CL before aggregation | 0.441 | 0.214 | **0.657** | 0.451 |
| Global model after one sample-weighted average | 0.275 | 0.317 | **0.076** | 0.432 |

One aggregation destroys Wells Fargo ranking quality. The aggregation weights were JPMorgan 63.47%, Wells Fargo 25.28%, and Key Bank 11.25% because training volumes are 126,423, 50,346, and 22,412 respectively.

Client drift was severe. After exact local CL, total update L2 norms were 19.63 (JPMorgan), 12.53 (Wells Fargo), and 10.01 (Key Bank). Temporal/GRU update norms were 15.00, 8.39, and 5.76; static/edge/decoder norms were 12.66, 9.31, and 8.19. Pairwise temporal distances were 16.89 (JPM–Wells), 15.99 (JPM–Key), and 9.42 (Wells–Key). The largest client therefore contributed the largest and most divergent update while receiving nearly two-thirds of aggregation mass.

### Optimizer and checkpoint handling

The plain FedAvg runner correctly creates a fresh client optimizer each round; optimizer state is not accidentally aggregated or shared between clients. Resetting client optimizer state each round is standard FedAvg behavior, not a code bug, but it differs from CL, where one optimizer is retained across June and July+replay. The corrected federated-CL path now retains one optimizer across those two local tasks and resets it only when a new global round begins.

The global checkpoint logic is correct: every round evaluates the aggregated global model on all three validation streams, computes the unweighted bank macro PR-AUC, and saves the state only when that macro improves. Restoring the saved state reproduces the reported per-bank macro. Threshold selection is validation-only and does not affect PR-AUC.

## Controlled seed-42 experiments

All candidates use hidden 128, TBPTT 2, replay 2,000 where noted, batch 1,024, AdamW `1e-3`, weight decay `1e-4`, dropout `0.25`, negative ratio 20, and validation-only model selection.

| Candidate | Best round | Macro PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---|---:|---:|---:|---:|---:|
| Plain full-stream, 2 epochs, 12 rounds, sample weights | 5 | 0.283 | 0.354 | 0.084 | 0.412 |
| Exact CL, 20 epochs/task, one sample-weighted aggregation | 1 | 0.275 | 0.317 | 0.076 | 0.432 |
| CL replay, 1 epoch/task, 12 rounds, sample weights | 2 | 0.201 | 0.293 | 0.098 | 0.213 |
| CL replay, 2 epochs/task, 12 rounds, sample weights | 9 | 0.349 | 0.440 | 0.104 | 0.504 |
| CL replay, 3 epochs/task, 12 rounds, sample weights | 11 | 0.457 | 0.585 | 0.141 | 0.647 |
| CL replay, 3 epochs/task, 12 rounds, **uniform weights** | 12 | 0.624 | 0.659 | 0.531 | 0.682 |
| CL replay, 3 epochs/task, 16 rounds, sample weights | 15 | 0.686 | 0.695 | 0.707 | 0.657 |
| Uniform, 12 rounds, all-parameter server LR 0.5 | 9 | 0.426 | 0.574 | 0.112 | 0.590 |
| Uniform, 12 rounds, temporal-only server LR 0.5 | 12 | 0.560 | 0.698 | 0.265 | 0.717 |
| CL replay, 3 epochs/task, **16 rounds, uniform weights** | 16 | **0.772** | 0.700 | 0.878 | 0.739 |

Local epochs show a controlled progression: one epoch underfits, two improves, and three is materially stronger. An eight-round cap would have selected only 0.365 in the matched sample-weighted trajectory; useful collaboration emerges later. Conservative interpolation did not solve the client imbalance: it delayed learning and especially hurt Wells Fargo. Equal weighting directly addresses domination and is the stronger intervention.

## Round-by-round global validation PR-AUC

| Candidate/seed | Round sequence |
|---|---|
| Plain final-style seed 42 | 0.276, 0.227, 0.278, 0.269, **0.283**, 0.216, 0.268, 0.277 |
| CL e1 sample seed 42 | 0.084, **0.201**, 0.130, 0.113, 0.137 |
| CL e2 sample seed 42 | 0.305, 0.307, 0.312, 0.326, 0.319, 0.236, 0.337, 0.349, **0.349**, 0.289, 0.206, 0.328 |
| CL e3 sample seed 42 | 0.299, 0.250, 0.308, 0.268, 0.303, 0.365, 0.263, 0.238, 0.371, 0.424, **0.457**, 0.326 |
| Best, seed 42 | 0.271, 0.386, 0.421, 0.375, 0.352, 0.363, 0.380, 0.443, 0.420, 0.440, 0.432, 0.605, 0.644, 0.654, 0.756, **0.772** |
| Best, seed 52 | 0.329, 0.426, 0.461, 0.405, 0.480, 0.466, 0.515, 0.575, 0.609, 0.706, 0.764, 0.594, 0.741, **0.806**, 0.769, 0.786 |
| Best, seed 62 | 0.277, 0.396, 0.418, 0.444, 0.508, 0.483, 0.494, 0.442, 0.326, 0.395, 0.520, 0.529, 0.589, 0.377, 0.488, **0.628** |

The curves are not monotonic, especially at seed 62. Validation checkpointing is therefore necessary, and the maximum round budget cannot be interpreted as the selected round.

## Best multi-seed validation result

The selected development configuration is bank-local June → July+replay training, replay capacity 2,000, three local epochs per task per round, 16 maximum rounds, uniform client weighting, and full server averaging (`server LR=1`).

| Seed | Best round | Macro PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---:|---:|---:|---:|---:|---:|
| 42 | 16 | 0.772 | 0.700 | 0.878 | 0.739 |
| 52 | 14 | 0.806 | 0.733 | 0.939 | 0.746 |
| 62 | 16 | 0.628 | 0.633 | 0.569 | 0.683 |
| Mean ± sample SD | — | **0.736 ± 0.094** | 0.689 ± 0.051 | 0.795 ± 0.198 | 0.723 ± 0.034 |

This is an absolute validation gain of **0.484** over the final FedAvg reference (`0.252`), or 191.9% relative. It also exceeds the corrected Local and CL validation references, but that comparison is development-only and is not a new September result.

At each seed's best round, bank-local pre-aggregation macro PR-AUC is `0.728 ± 0.075`; the aggregated global result is `0.736 ± 0.094`. Aggregation is no longer globally destructive, but its effect remains heterogeneous:

| Bank | Local before aggregation | Global after aggregation | Mean change |
|---|---:|---:|---:|
| JPMorgan | 0.587 ± 0.087 | 0.689 ± 0.051 | +0.102 |
| Wells Fargo | 0.897 ± 0.033 | 0.795 ± 0.198 | −0.102 |
| Key Bank | 0.701 ± 0.300 | 0.723 ± 0.034 | +0.022 |

Temporal/GRU updates remain larger than static/edge/decoder updates at the best rounds (mean group L2 2.84 versus 2.22). However, damping only temporal parameters reduced validation performance to 0.560 at seed 42, while full equal-weight averaging achieved 0.772. Temporal averaging is therefore a source of sensitivity, not the sole failure mechanism.

## Root cause and decision

The collapse has four interacting causes:

1. **Procedure mismatch:** final FedAvg did not use CL or replay at all.
2. **Client dominance:** sample-count weighting assigned 63.5% to JPMorgan despite highly non-IID data.
3. **Client drift:** local parameter trajectories were far apart, with particularly large temporal/GRU divergence; averaging erased Wells Fargo's locally useful solution.
4. **Insufficient schedule for federated CL:** one local epoch underfit, and useful equal-weight collaboration emerged mainly after rounds 10–16. Early stopping with patience 3 can terminate before that regime.

The best validation-only configuration is materially better across every fixed seed, so the fix is not a single lucky-seed result. Stability is nevertheless incomplete: macro SD is 0.094, Wells Fargo SD is 0.198, and round curves remain volatile.

Another development phase is scientifically justified only for reproducibility/stability work—deterministic execution, stronger validation resampling or an additional untouched development period—not for further tuning against August and never against September. The completed September final evaluation must remain unchanged. Without a new untouched temporal holdout, this result should be reported as a post-final validation diagnosis rather than substituted for the paper's frozen test result.
