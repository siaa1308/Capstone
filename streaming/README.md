# Model-weight transfer dashboard

Current work focuses on moving weight files, not running model training. The existing
training supervisor remains available separately; the dashboard never calls it.

## Launch locally

From the repository root in PowerShell, with your virtual environment active:

```powershell
python -m streaming.app.dashboard
```

Open http://127.0.0.1:8765. The dashboard and history work using only Python's standard
library. No package download is necessary to inspect the UI. Keep the terminal running.
Use `--port 8766 --state-dir streaming/.local/other` for a separate local instance.
Only one dashboard may own a given state directory.

## Enable actual transfers later

```powershell
python -m pip install -r streaming/requirements.txt
$env:FCL_TRANSFER_SECRET = '<shared random secret of at least 32 characters>'
python -m streaming.app.dashboard
```

Use the same securely shared secret on both machines; do not commit it. PyTorch,
safetensors and training datasets are not required. Provision the Kafka topic
`fcl.weight-files.v1` on a reachable broker, with retention suitable for your files.
Both machines must reach its advertised listeners. Broker provisioning and network
readiness checks are not automated by this increment.

Choose send or receive, enter the same transfer ID, sender and recipient on both
sides, and select a file on the sender. Use a unique transfer ID per file. Start the
receiver and sender within two minutes of each other. Published means Kafka delivery;
Verified means the receiver checked HMAC and SHA-256 and saved the file. No automatic
receipt is sent back to the sender yet. Downloaded files use a generic `.weights`
extension; preserve the original format/extension separately when needed.

Files are opaque bytes, limited to 100 MB. They are never deserialized or trained.
Signatures establish shared-secret membership, not independent sender identity or
encryption. Recipient IDs filter delivery logically; they are not Kafka ACLs.

History and files persist under ignored `streaming/.local/transfers`. Uploads are
retained locally, including uploads from failed starts; automatic storage cleanup is
pending. Refreshing the browser does not restart transfers. Interrupted jobs are
marked after application restart; automatic resume is not implemented. Cancel is
local, retains prior evidence, and cannot retract broker messages. Receivers use a
fresh group and replay retained messages to find the requested ID. Late or replayed
files are possible when IDs are reused. No exactly-once delivery claim is made.

The loopback API enforces Host checks and a session token plus same-origin checks
for mutations. It is a local prototype, not an internet-facing service. One local
transfer runs at a time. Receiving is bounded to two minutes; the sender also has a
two-minute deadline. The checklist reports local prerequisites; it does not claim
broker or peer connectivity has been tested.

## Validation

```powershell
python -m unittest discover -s tests/streaming -v
```

Transfer tests use Kafka doubles for signed, duplicate and out-of-order chunks,
wrong signatures, cancellation, durable history and HTTP session protection. Actual
Kafka, multi-laptop connectivity and large-file performance remain unverified.

---

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

## Local stop and cancel controls

```powershell
python -m streaming.app.controller stop --run 1
python -m streaming.app.controller cancel --run 1
```

Use the same `--state-dir` as the running process (a global option before the
command). The run number is the local SQLite record ID. Requests are persistent
and idempotent; cancel supersedes stop. Terminal or orphaned runs reject controls.
A request is not an acknowledgment: status stays Running until the worker reaches
a safe boundary and records Stopped or Cancelled. A run that finishes before
observing a request can still be Completed.

Stop finishes the current local round and prevents the next round. Central saves
the aggregate; a bank finishes publishing its update, which does not prove central
acceptance. Cancel is checked while waiting for messages, between training batches,
before aggregation/checkpoint writing, and between publication attempts. Checkpoint
writes already in progress finish atomically at their existing boundaries. Loading,
a batch, aggregation, and the existing broker flush (up to 30 seconds) can delay
acknowledgment. Already queued/published messages cannot be retracted.

These controls affect only the selected supervised process on this laptop. They do
not broadcast commands to other machines. Other participants can keep running or
time out; never display them as stopped based on a local acknowledgment. Authenticated
remote coordination and explicit recovery are still pending. Completed checkpoints
are retained; a stopped/cancelled run cannot simply be restarted with the same ID.

The tests also cover durable request precedence, terminal rejection, acknowledged
outcomes and lock release, publication backpressure cancellation, cancellation while
waiting without aggregation, and three banks stopping after one saved round.

## Next implementation steps

1. Persist frozen initial models and bank updates before committing Kafka offsets;
   implement receipts/outbox and checkpoint reconciliation.
2. Implement explicit resume with original round/seed/configuration identity,
   authenticated readiness, and cooperative stop/cancel handling. Current entry
   points must not be advertised as resumable.
3. Add the local API/dashboard over the controller and event store, then package
   ML dependencies and validate Windows/macOS launchers and real Kafka transport.
