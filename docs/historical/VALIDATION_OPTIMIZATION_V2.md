# Validation-only optimization follow-up

## Decision

No new federated configuration is promoted. The existing FedAvg configuration remains the strongest federated validation result (`0.338 ± 0.022`). Every positive FedProx coefficient tested under a directly matched configuration was worse than its `mu=0` FedAvg control. Consequently, no new September evaluation was run.

The replay capacity of 2,000 remains the continual-learning choice: it improved mean validation PR-AUC over no replay, but the forgetting diagnostics do not support a general claim that replay prevents forgetting at every bank.

All runs in this follow-up used only the training and validation splits. Validation-only data loading now excludes the testing directory entirely rather than merely declining to score it.

## FedProx diagnosis

The implemented objective is correct:

`BCE + (mu / 2) * sum(||local_parameter - frozen_global_parameter||^2)`

The global parameters are cloned before each client's local training. A `mu=0` control exactly reproduces FedAvg. A plain-SGD control also made matched FedAvg and FedProx trajectories nearly identical for a small proximal contribution, further confirming the implementation boundary.

The old FedAvg/FedProx comparison was not controlled: frozen FedAvg used two local epochs while frozen FedProx used one. More importantly, the temporal state was detached after every event batch. Since each batch is scored before its state update, the time projection, message projection, and GRU memory-update modules received no gradient. Only the static/edge projections and decoder learned, while randomly initialized temporal dynamics still affected later predictions. This seed-dependent untrained path is the most plausible cause of the unusually high FedProx variance.

The client distributions also amplify instability. Training volumes are 126,423 for JPMorgan Chase, 50,346 for Wells Fargo, and 22,412 for Key Bank; sample weighting therefore gives JPMorgan about 63% of aggregation mass. In the matched diagnostic, JPMorgan's update norm was also materially larger than Key Bank's. August validation has only 59 positive examples across the cohort, making round selection noisy.

## Controlled correction

The federated trainer now supports truncated backpropagation through time (`--tbptt-steps`). Losses are accumulated across the requested number of temporal batches before state detachment. Training diagnostics record optimizer steps, steps with nonzero temporal-module gradients, sampled data loss, and proximal squared L2. A regression test confirms that two-batch TBPTT reaches the temporal modules.

Optimizer choice, SGD momentum, and Adam epsilon are explicit parameters. Future metrics files also include the complete run configuration. These controls were added for diagnosis; validation evidence did not justify promoting them.

Training remains mini-batch based, not full-dataset gradient descent. The matched runs used chronological event batches of 1,024 and preserved causal state across batches; `tbptt_steps=2` controlled the gradient horizon.

## Matched FedProx coefficient check

All rows below use seed 42, hidden size 32, dropout 0.25, AdamW learning rate `1e-3`, weight decay `1e-4`, batch size 1,024, negative ratio 20, two local epochs, at most 12 rounds, patience 3, sample-count aggregation, and two-batch TBPTT.

| mu | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---:|---:|---:|---:|---:|
| 0 (FedAvg control) | 0.3216 | 0.4792 | 0.1357 | 0.3499 |
| 1e-6 | 0.2271 | 0.4272 | 0.1119 | 0.1422 |
| 1e-5 | 0.2341 | 0.4277 | 0.1312 | 0.1434 |
| 1e-4 | 0.2571 | 0.3545 | 0.1327 | 0.2841 |
| 1e-3 | 0.2672 | 0.3425 | 0.1326 | 0.3266 |

No positive `mu` beats the matched FedAvg control. Increasing Adam epsilon to `1e-4` made FedProx more balanced across banks but still reduced macro validation PR-AUC (`0.2790` versus matched FedAvg `0.3059`). Plain SGD underfit (`~0.104` for both methods). A larger sweep is therefore not scientifically warranted.

## FedAvg temporal-gradient check

| Seed | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---:|---:|---:|---:|---:|
| 42 | 0.3216 | 0.4792 | 0.1357 | 0.3499 |
| 52 | 0.2328 | 0.2941 | 0.0639 | 0.3404 |
| 62 | 0.3388 | 0.4744 | 0.1046 | 0.4374 |
| Mean ± sample SD | **0.2977 ± 0.0569** | 0.4159 ± 0.1055 | 0.1014 ± 0.0360 | 0.3759 ± 0.0535 |

Two-batch TBPTT is algorithmically more faithful, but it is weaker and less stable on validation than the existing FedAvg result (`0.338 ± 0.022`). Four-batch TBPTT, one local epoch, and a lower learning rate also failed to improve the seed-42 control. The current data do not justify replacing the frozen FedAvg candidate.

The historical validation-to-September drop cannot be optimized away using September labels. Validation evidence shows marked bank imbalance and noisy round selection; temporal distribution shift is a plausible explanation, but a genuinely untouched later holdout is needed to test it.

## Continual replay check

The matched replay comparison uses seeds 42, 52, and 62, 20 epochs per task, and otherwise the existing local configuration.

| Replay capacity | Macro validation PR-AUC | JPMorgan | Wells Fargo | Key Bank |
|---:|---:|---:|---:|---:|
| 0 | 0.2834 ± 0.0432 | 0.4943 ± 0.0328 | 0.0391 ± 0.0283 | 0.3168 ± 0.1552 |
| 2,000 | **0.3181 ± 0.0440** | 0.5898 ± 0.0337 | 0.0842 ± 0.0092 | 0.2804 ± 0.1197 |

Replay improves macro validation by about 0.035, driven by JPMorgan and Wells Fargo, while Key Bank declines. Mean older-period PR-AUC drop (positive means forgetting) changes from `-0.0042, -0.0086, -0.0744` without replay to `0.0044, 0.0268, -0.0733` with replay for JPMorgan, Wells Fargo, and Key Bank respectively. Replay therefore improves adaptation validation but does not demonstrate consistent forgetting prevention. No replay-size expansion was launched.

## Scientific conclusion

The FedProx loss is implemented correctly, but the current proximal method is not supported by validation evidence. Correcting temporal credit assignment exposed a real modeling defect but did not improve validation stability. The appropriate outcome is to retain the stronger existing FedAvg baseline, retain replay capacity 2,000 for CL with a qualified adaptation claim, and report FedProx as a negative result rather than force the desired ordering.

September remains untouched in this follow-up. No final-test result was selected, inspected, or generated during these experiments.
