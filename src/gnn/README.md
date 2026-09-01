# Local GraphSAGE baseline

`local_graphsage.py` trains one leakage-safe transaction (edge) classifier per simulated bank. It reads the final temporal dataset only, joins transaction edges to labels through `txn_id`, and uses the feature allow-lists in `configuration/model_feature_columns.json`.

Run all three active local models (JPMorgan Chase, Wells Fargo, and Key Bank):

```bash
python3 src/gnn/local_graphsage.py --epochs 80 --negative-ratio 20
```

Or run a single bank:

```bash
python3 src/gnn/local_graphsage.py --bank Key_Bank --epochs 80 --negative-ratio 20
```

The training procedure samples negatives only in the training loss. Validation and testing retain their original distributions. With the default `--negative-ratio 20`, retain the default `--loss-pos-weight 1`; applying a large positive loss weight on top of sampling can overemphasize the rare class.

Checkpoints and `metrics.json` are saved in `artifacts/local_graphsage/`. The validation split selects an F1 threshold; testing metrics are reported at that untouched validation-selected threshold.

Each bank derives its own fixed seed from `--seed`, so running `--bank Key_Bank` gives the same result as Key Bank within `--bank all` (on the same device and software versions).

This is a static GNN baseline. Although no target labels from validation or testing are used in message passing, the complete graph for each evaluation split is available at once. A later temporal/continual version should score transactions chronologically, exposing only edges that occurred before the transaction being scored.

## Causal Temporal GraphSAGE

`causal_temporal_graphsage.py` is the next baseline. It retains GraphSAGE's account-neighbour message-passing idea but processes transaction edges chronologically. A transaction is scored before its micro-batch updates account state, so it never sees later transaction edges. Node identities are remapped through `account_id` to keep each bank's account state persistent across training, validation, and testing.

```bash
python3 src/gnn/causal_temporal_graphsage.py --bank Key_Bank --epochs 50 --negative-ratio 20
```

The evaluation state is causal: training events warm validation; training plus validation events warm testing. Labels are never incorporated into account memory. As with the static baseline, negative sampling occurs only in the training loss and validation/testing retain their original distributions.

## Follow-on experiments

The completed validation-driven three-bank optimization is documented in
`docs/OPTIMIZATION_REPORT.md`. New development runs should pass
`--validation-only`; September appears in legacy artifacts and is no longer a
genuinely untouched holdout.

The frozen local backbone uses hidden size 32, batch size 1024, negative ratio
20, learning rate 0.001, and a shared active-cohort training-only feature
encoder. Negative examples are sampled uniformly across the complete epoch
stream while every event still updates causal account memory.

Run three fixed seeds (with validation-only Platt calibration and analyst-budget metrics):

```bash
python3 src/gnn/run_temporal_multiseed.py --seeds 42,52,62 --calibration platt
```

Run local June→July continual learning. Compare `--replay-size 0` against a fixed replay buffer:

```bash
python3 src/gnn/continual_temporal_graphsage.py --replay-size 2000 --output-dir artifacts/continual_temporal_optimized
```

Run validation-selected three-client FedAvg with a shared training-only feature schema:

```bash
python3 src/gnn/federated_causal_temporal_graphsage.py --algorithm fedavg --rounds 10 --local-epochs 2 --output-dir artifacts/fedavg_optimized
```

Run FedProx with the same protocol. Tune `--prox-mu` using macro validation
PR-AUC only; useful initial candidates are `0.001`, `0.01`, and `0.1`:

```bash
python3 src/gnn/federated_causal_temporal_graphsage.py --algorithm fedprox --prox-mu 0.01 --rounds 10 --local-epochs 2 --output-dir artifacts/fedprox_optimized
```

Both federated variants select the communication round using August macro
PR-AUC and evaluate September only after restoring that frozen checkpoint.

## Federated Continual Temporal GraphSAGE

`federated_continual_temporal_graphsage.py` combines the two experiments above.
Each bank receives the current global temporal GNN, trains it on its next
chronological local window plus a replay buffer sampled solely from earlier local
windows, and returns only updated model weights for data-size-weighted FedAvg.
Raw transactions, labels, and replay examples never leave their local client.

The default 31-day task window divides the June--July training span into two
federated continual-learning rounds:

```bash
python3 src/gnn/federated_continual_temporal_graphsage.py --task-days 31 --replay-size 2000 --local-epochs 2 --fed-rounds-per-task 3
```

The output includes each client's current-task size, positive count, replay size,
and FedAvg aggregation weight for every round. Validation chooses thresholds only
after final aggregation; testing remains untouched until the final report.

`--fed-rounds-per-task` is intentionally separate from chronological tasks: it
allows the global model to converge on a task before the next task arrives. With
the supplied June--July training span, the command above runs three FedAvg rounds
on June followed by three FedAvg rounds on July plus replay.
