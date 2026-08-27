# Local GraphSAGE baseline

`local_graphsage.py` trains one leakage-safe transaction (edge) classifier per simulated bank. It reads the final temporal dataset only, joins transaction edges to labels through `txn_id`, and uses the feature allow-lists in `configuration/model_feature_columns.json`.

Run all five local models:

```bash
python3 src/gnn/local_graphsage.py --epochs 80 --negative-ratio 20
```

Or run a single bank:

```bash
python3 src/gnn/local_graphsage.py --bank Citi --epochs 80 --negative-ratio 20
```

The training procedure samples negatives only in the training loss. Validation and testing retain their original distributions. With the default `--negative-ratio 20`, retain the default `--loss-pos-weight 1`; applying a large positive loss weight on top of sampling can overemphasize the rare class.

Checkpoints and `metrics.json` are saved in `artifacts/local_graphsage/`. The validation split selects an F1 threshold; testing metrics are reported at that untouched validation-selected threshold.

Each bank derives its own fixed seed from `--seed`, so running `--bank Citi` gives the same result as Citi within `--bank all` (on the same device and software versions).

This is a static GNN baseline. Although no target labels from validation or testing are used in message passing, the complete graph for each evaluation split is available at once. A later temporal/continual version should score transactions chronologically, exposing only edges that occurred before the transaction being scored.

## Causal Temporal GraphSAGE

`causal_temporal_graphsage.py` is the next baseline. It retains GraphSAGE's account-neighbour message-passing idea but processes transaction edges chronologically. A transaction is scored before its micro-batch updates account state, so it never sees later transaction edges. Node identities are remapped through `account_id` to keep each bank's account state persistent across training, validation, and testing.

```bash
python3 src/gnn/causal_temporal_graphsage.py --bank Citi --epochs 50 --negative-ratio 20
```

The evaluation state is causal: training events warm validation; training plus validation events warm testing. Labels are never incorporated into account memory. As with the static baseline, negative sampling occurs only in the training loss and validation/testing retain their original distributions.

## Follow-on experiments

Run three fixed seeds (with validation-only Platt calibration and analyst-budget metrics):

```bash
python3 src/gnn/run_temporal_multiseed.py --seeds 42,52,62 --calibration platt
```

Run local June→July continual learning. Compare `--replay-size 0` against a fixed replay buffer:

```bash
python3 src/gnn/continual_temporal_graphsage.py --replay-size 2000
```

Run simulated five-client FedAvg with a shared training-only feature schema:

```bash
python3 src/gnn/federated_causal_temporal_graphsage.py --rounds 3 --local-epochs 1
```
