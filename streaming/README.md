# Training dashboard backend — first implementation

This directory provides a process controller and persistent status for the existing
federation pipeline. It does not yet provide a browser dashboard, Docker packaging,
platform launchers, remote readiness/control messages, or automatic training resume.

## Implemented

- `RunController.start` launches only the central or bank entry point in a child
  Python process with an allowlisted configuration snapshot. Secrets remain in the
  environment; they are not copied into configuration snapshots or SQLite events.
- An OS lock held by the child prevents simultaneous supervised jobs sharing a
  state directory, including when the parent launcher exits. Use one fixed state
  directory per laptop. Existing direct CLI processes do not participate in this lock.
- SQLite records role, configuration, run outcome, and ordered events. Status
  queries mark orphaned Running records Interrupted only after obtaining the lock.
- Existing aggregator and worker CLIs delegate to callable `run` functions.
  Structured events report initialization, model loading, epoch completion, update
  publication/validation, aggregation, and checkpoint completion. FedAvg and local
  optimizer logic are unchanged.
- Checkpoints use temporary writes, file flush/fsync, atomic replacement, and a
  hash-verified commit marker written after both weights and manifest. Existing or
  partially written rounds cannot be silently overwritten. Filesystem/power-loss
  guarantees still depend on the host; this is not a database/filesystem transaction.

`update_published` means broker delivery, not central acceptance. `update_validated`
currently describes validation in the live aggregator; accepted update payloads are
not yet persisted for resume. A bank's Completed process means it published its
configured updates; only central `round_completed` means an aggregate was saved.

## Developer usage

Use the existing federation dependencies and a valid configuration/dataset. Supply
the same secret environment variables as the existing CLI. The default state folder
is `streaming/.local/`, ignored by Git. Configuration snapshots contain local paths
and other non-secret settings; treat this folder as local application data.

From the repository root:

```powershell
python -m streaming.app.controller start --role central --config distributed_federation/config.json
python -m streaming.app.controller start --role bank --client-id bank-1 --config distributed_federation/config.json
python -m streaming.app.controller status
python -m streaming.app.controller events --run 1 --after 0
```

Run central and bank commands on their respective laptops. These are temporary
developer entry points; the planned launchers/UI will remove routine terminal use.
The `start` command waits for its child so failures retain a useful exit code;
Python API callers receive a `Popen` handle immediately. Starting a process is not
a readiness confirmation. Events return up to 500 entries; use the last event ID
as `--after` to read subsequent entries. The run argument is the local SQLite record
ID, distinct from the federation configuration's `run_id`.

An already used `(run_id, actor)` cannot be started again in the same state store.
Use a new run ID for a new experiment. Do not erase state to simulate recovery or
reuse output directories. Snapshotting configuration does not freeze dataset or
initial-checkpoint contents: keep them unchanged during a run.

## Validation

```powershell
python -m unittest discover -s tests/streaming -v
```

The standard-library test suite covers cross-process locks, config snapshotting,
durable events, process outcomes, checkpoint corruption and partial-write handling,
and the real aggregator/worker loops with three banks over two rounds. That round
trip uses signed/chunked messages, duplicated deliveries, and the existing FedAvg
function with deterministic scalar doubles and an in-memory broker. Unequal counts
verify weighting; a missing-bank test verifies aggregation does not proceed.

These tests do not execute PyTorch training, tensor compatibility checks, actual
Kafka, or multi-laptop networking. A real dependency-backed integration run remains
an acceptance gate, along with preprocessing equivalence and platform checks.

## Next implementation steps

1. Persist frozen initial models and bank updates before committing Kafka offsets;
   implement receipts/outbox and checkpoint reconciliation.
2. Implement explicit resume with original round/seed/configuration identity,
   authenticated readiness, and cooperative stop/cancel handling. Current entry
   points must not be advertised as resumable.
3. Add the local API/dashboard over the controller and event store, then package
   ML dependencies and validate Windows/macOS launchers and real Kafka transport.
