# Causal tail-ranking guard: development experiment

The reference on branch `sia` is **0.8710 ± 0.0611 September macro average
precision**, across seeds 42, 52, and 62. The corresponding August validation
mean is 0.8114. The percentages are not accuracy. September has already been
inspected and is not an untouched test set.

## Research question

Does checking the actual causal ranking behavior of a proposed global update,
separately for each bank and past training month, protect rare-fraud knowledge
more effectively than harmonizing temporal parameter directions alone?

The implementation adds an optional **causal tail-ranking guard** to the existing
context replay + temporal harmonization method. It does not replace the encoder
or change the classifier architecture. A candidate model rebuilds account memory
from the full chronological training stream. June and July share this causal
memory; each event is scored before its batch updates it. No evaluation labels
enter memory. Each bank/month returns one risk value:

`R = mean softplus(z_negative - z_positive)`

The mean covers every positive and the highest-scoring
`min(number_of_negatives, 20 * number_of_positives)` negative events in that
training month. The use of logits avoids sigmoid saturation. All candidates
recompute their hardest negatives. A group without both classes supplies no
constraint; no artificial zero risk is assigned.

Let `d` be the proposed harmonized update. Try `theta + a*d` for
`a = 1, 0.5, 0.25`, in that order. Accept the first candidate whose risk rises by
at most 5% relative to the pre-round model in **every** supported bank/month.
If none passes, keep the previous model exactly. This empirical constraint says
nothing by itself about future AP. In particular it can be too conservative and
prevent useful adaptation; rejected updates and failures must be reported.

## Prior art and novelty boundary

Checked 23 October 2026:

| Primary source | Established overlap | Distinction to investigate |
|---|---|---|
| [FC2L, Data Mining and Knowledge Discovery 2026](https://link.springer.com/article/10.1007/s10618-026-01192-0) | Federated continual graph learning already exists for AML; uses orthogonal projection and contrastive objectives | This experiment evaluates candidate updates using causal bank/month tail-ranking risks |
| [FedAGC, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_FedAGC_Federated_Continual_Learning_with_Asymmetric_Gradient_Correction_ICCV_2025_paper.pdf) | Class-level gradient correction protects old/new knowledge in federated continual learning | This guard checks realized nonlinear model risks after memory reconstruction, without transmitting class gradients |
| [FedNL, ICML 2022](https://proceedings.mlr.press/v162/safaryan22a.html) | Federated line search and scalar local function evaluations already exist | The constraint is on every bank/month's hard-negative ranking risk, applied to the existing temporal replay model |

Ranking losses, backtracking, replay, and client protection are not new on their
own. This targeted search does **not** establish that the exact combination has
never been published or implemented. A lack of public code also does not establish
scientific novelty. Any paper should present this as a candidate method until
systematic prior-art comparison and convincing experiments support a stronger claim.

## Controlled experiment

The design was saved in `configs/causal_tail_guard_study.json` before execution.
Both variants start from the same seed-specific 87.1%-study checkpoint and receive
four more rounds, three epochs per task, replay capacity 2,000, and the same
global RNG policy. The control retains context replay and harmonization and
disables only the guard. Round zero remains eligible. All three seeds are retained.
August macro AP selects checkpoints. Training loads June–July and August only;
September is unavailable through the development loader.

The comparison matches local training budgets, not compute or communication.
The guard incurs up to four additional full causal training-stream evaluations
per round (one reference and three candidates). A distributed implementation
would also broadcast candidate parameters and collect bank/task risk scalars.
Those scalars are not differentially private. This implementation runs in the
research simulator and requires the full completed training history; it is not
a bounded-memory or Kafka deployment claim.

Run `python3 -B scripts/run_tail_guard_study.py`. Artifacts and logs go to
`artifacts/research_tail_guard`; partial runs are not silently resumed and completed
evidence is preserved. The runner records training configuration, data fingerprint,
source snapshots, checkpoint selection, all candidate decisions, and elapsed time.

## Publication evidence still needed

An unseen post-September period or independent dataset is required for a fresh
test claim. A stronger paper also needs more banks/tasks, complete ablations,
matched line-search and gradient-protection baselines, runtime/communication
costs, and temporal uncertainty estimates. Three random seeds on one simulated
dataset are insufficient to guarantee acceptance or universal superiority.

## Completed result: do not promote the guard

All six four-round runs completed. Validation macro AP:

| Seed | Same-budget continuation | Tail guard | Guard minus control |
|---|---:|---:|---:|
| 42 | 0.923388 | 0.923388 | 0.000000 |
| 52 | 0.860740 | 0.860740 | 0.000000 |
| 62 | 0.794603 | 0.686662 | -0.107941 |
| Mean ± sample SD | 0.859577 ± 0.064400 | 0.823597 ± 0.122656 | -0.035980 |

Seeds 42/52 retain round zero in both arms. For seed 62, both arms select
round one. The guard rejects all 12 full-step proposals; only the first
seed-62 round accepts a quarter step, and all other rounds keep the old model.
It therefore impedes useful adaptation. Total measured run time is about
205 seconds for the guard and 155 seconds for continuation, under two concurrent
CPU workers; these are not isolated hardware benchmarks.

No September evaluation was performed for either arm. This candidate does not
support a positive novelty/performance claim. The implementation is opt-in and
the original checkpoints remain unchanged. The separate equal-weight ensemble
study is recorded in [ENSEMBLE_IMPROVEMENT_RESULTS.md](ENSEMBLE_IMPROVEMENT_RESULTS.md).
