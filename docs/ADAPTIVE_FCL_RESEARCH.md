# Adaptive continual and federated AML research

This is a separate research cycle. The frozen paper implementation, configuration,
checkpoints, and September results are preserved. The training entry point loads
June–July training and August validation only. It cannot score September.

## Hypotheses and prior work

1. **Risk and account-context replay.** Retaining isolated positives loses the
   account history that produced their temporal state. Retain rare positives,
   high-scoring historical negatives, and strictly earlier transactions involving
   their endpoints, with random coverage filling the remaining memory.
2. **Replay logit retention.** Match June teacher logits on the stored June events
   while learning July, with equal weight for positive and negative replay classes
   within a TBPTT window. This tests whether retention improves stability under
   extremely rare labels.
3. **Temporal update harmonization.** Handle disagreement in the time/message/GRU
   parameters separately from the static projection and transaction decoder.
   Clip temporal update norms to twice their client median, partially project out
   conflicting peer directions, and keep equal-weight averaging for the decoder.

These are proposed contributions for this project, **not established claims of
first-in-literature novelty**. Closely related work includes:

- [MIR, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html): selects replay examples by predicted interference. Our buffer uses historical risk and earlier account context, not MIR's virtual-update criterion.
- [Dark Experience Replay, NeurIPS 2020](https://arxiv.org/abs/2004.07211): combines rehearsal with past-logit consistency. Our retention targets come from a completed June model and its full causal June stream; this is an adaptation, not the original DER algorithm.
- [FedGH, 2023](https://arxiv.org/abs/2309.06692): addresses heterogeneous-client update conflict. Our projection is restricted to the temporal parameter group, uses original peer directions, and averages corrections to avoid client-order dependence.
- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295): establishes server-side adaptive optimization as prior art; changing FedAvg to an adaptive optimizer alone would not constitute novelty.
- [History Repeats, 2023](https://arxiv.org/abs/2305.18675): studies replay and temporal regularization for continual temporal knowledge graphs. A temporal-graph replay claim must distinguish this prior work.

A defensible project claim requires ablations and multi-seed evidence for the
specific combination on temporal AML. It does not establish a generally superior
continual/federated method or production privacy guarantee.

## Replay construction

The capacity remains 2,000 events, matching the baseline. Keep all June positives
unless they alone exceed capacity (then select a seeded uniform subset). Allocate
25% of the remaining slots to the highest-scoring June negatives, 25% to earlier
account context, and the rest to uniform coverage. Distribute context slots round
robin across positive/hard-negative anchors. Exclude same-time and future context.
Use unused context slots for coverage. Sort the finished replay chronologically
before appending July. Selection never uses validation labels or predictions.

For retention, record the June model's causal score at each selected event while
streaming the complete June history. Convert clipped probabilities to logits.
July learning minimizes sampled BCE plus `0.05 * class_balanced_replay_logit_MSE`.
Only replay positions have teacher targets; July positions do not contribute to
the retention loss. The same two-batch TBPTT and gradient clipping are retained.
Only the retained events/logits need survive the task boundary; this experiment
still uses an offline June–July dataset and is not a task-free online learner.

## Aggregation

Let `delta_i` be client i's temporal update from the round's common global model.
Clip its norm at `2 * median_j ||delta_j||`. For every peer j with a negative dot
product, subtract half the conflicting projection, averaging the corrections
over peers. All projections use the original clipped peer updates, not sequentially
mutated ones. Average the resulting temporal deltas uniformly and add them to the
global temporal parameters. Average all other parameters uniformly as usual.

Only model updates are required at the server. No replay records or teacher
targets are transmitted. The simulation still fits a shared training-only feature
encoder, just like the frozen research pipeline; this is not private preprocessing.

## Controlled protocol

The design is recorded before screening in `configs/adaptive_fcl_study.json`.

| Variable | Setting |
|---|---|
| Cohort | JPMorgan Chase, Wells Fargo, Key Bank |
| Model | Existing CausalTemporalGraphSAGE, hidden 128, dropout 0.25 |
| Tasks | Calendar June, then July plus June replay |
| Loss sampling | All positives, 20:1 stream-wide negatives |
| Temporal batching | 1,024 events, TBPTT 2 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Federated budget | 16 rounds, 3 epochs per local task per round |
| Standalone CL | 20 epochs per task |
| Replay capacity | 2,000 events |
| Selection | Highest macro-bank August PR-AUC |
| Seeds | Screen 42; confirm selected variant and baseline on 52 and 62 |
| Runtime | CPU, one Torch thread, deterministic algorithms |

All variants use the corrected calendar boundary. Unlike the frozen source, which
counts 30 days from each bank's first event, this runner anchors the task boundary
to July 1 midnight. One JPMorgan transaction changes task assignment. The initial
screen used explicit per-client dropout reseeding and produced a weak seed-42
baseline (0.4869). It is retained as an RNG-sensitivity diagnostic. The main study
in `configs/adaptive_fcl_global_rng.json` restores the original global RNG policy;
a regression test verifies identical uniform-replay client parameter updates under
matched task boundaries. Thus the newly rerun baseline, not the historical single
run, is the controlled comparison. Historical results remain useful references.

The screen includes baseline, context replay, context replay with retention,
temporal aggregation alone, and the combined method. All screen outcomes are
retained. Confirmation uses the selected configuration unchanged on two further
seeds. Report bank-macro means and sample standard deviations, per-bank PR-AUC,
analyst-budget metrics, and June retention before/after July.

August contains only 59 positives across the active banks. These comparisons are
development evidence and can overfit the validation period. The existing September
results were already inspected throughout the project. No new result on September
can be called an untouched confirmatory test; a later temporal holdout is needed.

## Run

From the repository root:

```bash
python3 -B tests/test_adaptive_fcl.py -v
python3 -B scripts/run_adaptive_fcl_study.py --phase screen --jobs 2 \
  --study-config configs/adaptive_fcl_global_rng.json \
  --output-root artifacts/research_adaptive_fcl/global_rng
python3 -B scripts/summarize_adaptive_fcl_study.py --freeze-selection \
  --study-config configs/adaptive_fcl_global_rng.json \
  --root artifacts/research_adaptive_fcl/global_rng
python3 -B scripts/run_adaptive_fcl_study.py --phase confirm --winner VARIANT --jobs 2 \
  --study-config configs/adaptive_fcl_global_rng.json \
  --output-root artifacts/research_adaptive_fcl/global_rng
python3 -B scripts/run_adaptive_fcl_study.py --phase local --jobs 2 \
  --study-config configs/adaptive_fcl_global_rng.json \
  --output-root artifacts/research_adaptive_fcl/global_rng
```

Each run saves a development-data content fingerprint, environment versions,
configuration, source hash/snapshot, round-by-round metrics, retention diagnostics,
and the validation-selected checkpoint. Outputs live under
`artifacts/research_adaptive_fcl/`; nonempty run directories are never overwritten.
The launcher skips completed runs, but deliberately does not resume partial ones.

## Follow-up: improve the existing saved model

The scratch experiment exposed substantial runtime/RNG sensitivity. A separate
warm-start study therefore initializes each seed from its actual frozen final
checkpoint, which reproduces the saved August score (seed 42: 0.714167). Both
ordinary FedAvg continuation and every candidate get exactly four additional
rounds at the same local-training budget. The starting model is eligible as round
zero, preventing an unsuccessful continuation from displacing it.

The original warm-start design is `configs/adaptive_fcl_warm_start.json`. After
the five seed-42 variants completed, retention was observed to be harmful. An
explicit adaptive follow-up, `configs/adaptive_fcl_warm_refined.json`, adds one
ablation combining context replay and temporal aggregation without distillation.
The complete screen is retained in `artifacts/research_adaptive_fcl/warm_start/`.
This additional candidate was specified before its run and before confirmation
on seeds 52/62. It must not be presented as part of an entirely preregistered study.

Seed-42 validation selected the no-distillation combination at 0.923388 versus
0.859876 for ordinary continuation. `selection.json` records the frozen choice and
hashes of all screen results. Three-seed confirmation is complete: mean August PR-AUC is 0.8114 versus
0.7673 for matched continuation, with positive paired gains on all three seeds.
See [the completed results](ADAPTIVE_FCL_RESULTS.md) for the separate exploratory
September evaluation and limitations.

The standalone CL follow-up in `configs/adaptive_cl_context.json` isolates context
replay without distillation. It reuses the identical completed local baselines and
preserves the initial negative retention comparison.

## Separate exploratory September evaluation

After the three-seed validation comparison is complete, freeze both the selected
candidate and its matched continuation control with checkpoint and metric hashes:

```bash
python3 -B scripts/evaluate_adaptive_candidate.py freeze \
  --study-root artifacts/research_adaptive_fcl/warm_start \
  --candidate context_harmonized \
  --output artifacts/research_adaptive_fcl/warm_start/EXPLORATORY_FREEZE.json
python3 -B scripts/evaluate_adaptive_candidate.py evaluate \
  --freeze artifacts/research_adaptive_fcl/warm_start/EXPLORATORY_FREEZE.json \
  --output artifacts/research_adaptive_fcl/warm_start/exploratory_september \
  --acknowledge-exploratory-september
```

The evaluator refuses changed checkpoints/selection evidence or an existing
evaluation directory. Training does not load September; this separate explicit
action does. Preserve all seed results and do not tune after viewing them. The
historical 0.759 September PR-AUC can be compared descriptively, but these results
cannot substitute for a genuinely untouched later-period evaluation.

The new methods are currently implemented in the research simulator. The older
Kafka worker is not yet an implementation of this study and retains the integration
issues documented during repository review. No live distributed run is implied.
