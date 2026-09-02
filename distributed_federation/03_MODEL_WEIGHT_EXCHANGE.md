# Federated Model-Weight Exchange Workflow

This document describes the protocol implemented by `central/aggregator.py`,
`client/bank_worker.py`, and the modules under `common/`. Do not run it until all
VMs pass the network, Kafka, secret, dependency, protocol, and model checks in
the preflight tool.

## 1. Roles

### Central aggregator

- Owns the authoritative global model.
- Starts each federated round by publishing the current global weights.
- Receives exactly one valid update from each participating bank.
- Validates update metadata and tensor structure.
- Computes sample-weighted FedAvg.
- Saves the aggregated global weights and a JSON audit manifest after each round.

### Bank client

- Consumes the global model for the requested round.
- Verifies its hash, architecture, feature schema, and round number.
- Loads only its private local bank data.
- Trains locally for the configured number of epochs.
- Publishes model weights and sample count, never raw transactions.

## 2. Decide the number of logical banks before coding

The current dataset contains five simulated banks, while the physical test layout has one central VM and three bank VMs. Choose one of these modes and record it in the experiment configuration:

### Mode A - three-client demonstration

The confirmed demonstration uses one logical worker per bank VM:

| Client ID | ZeroTier member | ZeroTier IP | Dataset key |
|---|---|---|---|
| `bank-1` | `KeyBank` | `10.170.231.168` | `Key_Bank` |
| `bank-2` | `FifthThirdBancorp` | `10.170.231.115` | `Fifth_Third_Bancorp` |
| `bank-3` | `JPMorganChase` | `10.170.231.174` | `JPMorgan_Chase` |

This is the configured classroom run. Its FedAvg output must not be presented
as equivalent to the existing five-client experiment.

### Mode B - five-client reproduction

Run five logical bank workers across the three bank VMs, for example two workers on Bank VM 1, two on Bank VM 2, and one on Bank VM 3. Every worker needs a unique `client_id`, bank partition, Kafka consumer group, local output directory, and HMAC secret. Resource use will be higher.

Do not combine two bank datasets into one update merely because they share a physical VM. They remain separate logical clients and produce separate updates. The central server must obtain the configured client roster from one experiment configuration rather than assuming three or five clients in code.

## 3. Compatibility requirement

Do not federate the existing independently encoded per-bank checkpoints. Their input dimensions differ between banks.

Use the shared feature schema already fitted by `federated_causal_temporal_graphsage.py`. The current federated global checkpoint has:

- 62 encoded node inputs
- 35 encoded edge inputs
- 64 hidden units
- 64,449 trainable parameters

Every client must load the same model class, ordered feature schema, parameter names, tensor shapes, and starting global checkpoint.

## 4. Kafka topics used by the implementation

| Topic | Producer | Consumer | Key | Purpose |
|---|---|---|---|---|
| `fcl.global-model` | Central | Banks | `run_id` | Latest global model for a run |
| `fcl.client-updates` | Banks | Central | `client_id` | Local model updates |

Each bank uses its own consumer group for `fcl.global-model`, so every bank gets
its own copy. The central aggregator has one run-specific consumer group for
`fcl.client-updates`. The signed envelopes carry round and training metadata;
there is currently no separate control or metrics topic.

## 5. Round protocol

```text
Central                                      Bank 1 / Bank 2 / Bank 3
   |                                                     |
   |-- global-model: round 1 weights ------------------->|
   |                                                     | verify and train locally
   |<---------------- client-updates: round 1 -----------|
   | validate all three updates                          |
   | sample-weighted FedAvg                              |
   | save global checkpoint for round 2                  |
   |-- global-model: round 2 weights ------------------->|
```

If a client fails, the central process times out and exits without aggregating an
incomplete round. It does not reuse an earlier update. To retry, stop every
remaining process, choose a fresh `run_id`, rerun preflight, start all workers,
and then start the aggregator. See `04_RESTART_AND_RECOVERY.md`.

## 6. Message envelope

Each model message should carry metadata similar to:

```json
{
  "protocol_version": 1,
  "message_type": "client_update",
  "run_id": "demo-2026-09-01-01",
  "round_id": 1,
  "client_id": "bank-1",
  "model_name": "CausalTemporalGraphSAGE",
  "schema_sha256": "computed-at-runtime",
  "base_model_sha256": "computed-at-runtime",
  "payload_sha256": "computed-at-runtime",
  "message_hmac_sha256": "computed-at-runtime",
  "training_examples": 126423,
  "local_epochs": 1,
  "chunk_index": 0,
  "chunk_count": 1,
  "created_at_utc": "2026-09-01T04:30:00Z"
}
```

Required validation before aggregation:

1. `run_id` and `round_id` match the active round.
2. `client_id` is one of the authorized banks and has not already submitted.
3. `base_model_sha256` matches the global model published for the round.
4. `schema_sha256` matches the central schema.
5. Every tensor name, shape, and dtype matches the global state.
6. All tensor values are finite.
7. The reconstructed payload hash matches `payload_sha256`.
8. `training_examples` is positive and within the expected range.
9. A per-client HMAC validates the claimed `client_id` and message contents.

## 7. Serialization

Prefer `safetensors` for weight transport. It is designed for tensor data and avoids loading arbitrary pickled Python objects from remote clients.

Do not send:

- the complete Python model object;
- optimizer objects unless the protocol explicitly requires them;
- local data, labels, account identifiers, or paths;
- unrestricted pickle payloads received from another computer.

SHA-256 detects accidental or malicious payload changes but does not authenticate who sent the payload. Assign each bank a separate random HMAC secret and configure the matching secret on the central server. Keep these secrets in VM-local environment files with mode `600`; never place them in Git or Kafka messages.

Suggested serialization flow:

```python
from hashlib import sha256
from safetensors.torch import save

payload = save({name: tensor.detach().cpu().contiguous()
                for name, tensor in model.state_dict().items()})
payload_hash = sha256(payload).hexdigest()
```

Suggested safe loading flow:

```python
from hashlib import sha256
from safetensors.torch import load

assert sha256(payload).hexdigest() == expected_payload_hash
received_state = load(payload)

expected_state = model.state_dict()
assert received_state.keys() == expected_state.keys()
for name, tensor in received_state.items():
    assert tensor.shape == expected_state[name].shape
    assert tensor.dtype == expected_state[name].dtype
    assert tensor.isfinite().all()

model.load_state_dict(received_state, strict=True)
```

Install the transport dependencies inside each VM's virtual environment:

```bash
python -m pip install -r distributed_federation/requirements-distributed.txt
```

The runnable implementation pins the transport dependencies in
`requirements-distributed.txt`; the model dependencies are installed with the
versioned commands in `README.md`. Record the exact resolved versions from the
first successful four-VM run.

## 8. Kafka payload strategy

The current federated model contains approximately 64,449 parameters, or roughly 258 KB as float32 before metadata. It should fit below the configured 2 MB Kafka limit.

Still implement chunking so later models remain supported:

- Recommended chunk size: 512 KiB.
- Use the same Kafka key for every chunk of one client update.
- Include `chunk_index`, `chunk_count`, and full-payload SHA-256.
- Reassemble all chunks before deserialization.
- Reject duplicate or conflicting chunks.
- Expire incomplete uploads after the round timeout.

Configure producers with acknowledgements and idempotence enabled. With `confluent-kafka`, the minimum relevant settings are:

```python
producer_config = {
    "bootstrap.servers": broker,
    "acks": "all",
    "enable.idempotence": True,
    "compression.type": "zstd",
    "message.timeout.ms": 60000,
}
```

Kafka values are bytes, so a binary envelope is preferable. If JSON is used during the first prototype, Base64 increases payload size by about one third and must be included in size calculations.

## 9. FedAvg calculation

For clients `1..K`, weights `W_k`, and local training-example counts `n_k`:

```text
W_global = sum(n_k * W_k) / sum(n_k)
```

Reference implementation:

```python
import torch


def fedavg(client_states, example_counts):
    if len(client_states) != len(example_counts) or not client_states:
        raise ValueError("A non-empty example count is required for every state")
    total = sum(example_counts)
    if total <= 0:
        raise ValueError("Total example count must be positive")

    names = tuple(client_states[0].keys())
    for state in client_states[1:]:
        if tuple(state.keys()) != names:
            raise ValueError("Client state dictionaries have different keys or ordering")
    averaged = {}
    for name in names:
        reference = client_states[0][name]
        if not reference.is_floating_point():
            if any(not torch.equal(state[name], reference) for state in client_states[1:]):
                raise ValueError(f"Non-floating buffer differs between clients: {name}")
            averaged[name] = reference.clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for state, count in zip(client_states, example_counts):
            tensor = state[name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f"Incompatible tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite tensor: {name}")
            accumulator.add_(tensor.to(torch.float64), alpha=count / total)
        averaged[name] = accumulator.to(reference.dtype)
    return averaged
```

The current repository's federated simulation weights each bank by its number of training events. The distributed implementation should preserve that behavior.

## 10. Responsibilities during a test

### Central operator

1. Confirm all three bank VMs respond over ZeroTier.
2. Confirm Kafka is healthy and topics exist.
3. Record the Git commit and environment-lock files.
4. Choose a unique `run_id`.
5. Confirm all bank workers are waiting, then start the aggregator.
6. Monitor received updates by client ID.
7. Verify the `.safetensors` checkpoint and JSON manifest saved for every round
   as `global_round_001.*`, `global_round_002.*`, and so on.
8. Preserve the final console log and run artifacts for the experiment record.

### Each bank operator

1. Pull the same Git commit.
2. Activate the VM-local virtual environment.
3. Confirm the assigned bank partition and client ID.
4. Confirm the Kafka broker is `10.170.231.39:9092`.
5. Start the bank worker before the central operator begins the round.
6. Never substitute another bank's dataset or reuse an old update.

## 11. Recommended first test sequence

Do not begin with expensive training.

1. **Connectivity test:** pass preflight's TCP and Kafka metadata checks.
2. **Tracked-weight transfer test:** send
   `artifacts/federated_causal_temporal_graphsage/global_model.pt` with
   `tools.weight_smoke_test` and verify the same SHA-256 on all three banks.
3. **One-client dry run:** central plus one bank, one local batch, one round.
4. **Three-client dry run:** all banks, one local batch, one round.
5. **Aggregation equivalence:** compare distributed FedAvg with the existing same-machine simulation using the same initial state and client updates.
6. **Failure test:** stop one bank and confirm the round times out without aggregation.
7. **Stale-update test:** submit a prior-round update and confirm rejection.
8. **Full experiment:** only after all previous checks pass.

The runnable central service, worker, signed/chunked protocol, configuration,
and preflight checker are now present. Follow `README.md` and do not proceed to
the full experiment until every VM passes preflight and the smaller tests above.

## 12. Reproducibility record

For every run save:

- Git commit hash
- `run_id` and timestamps
- Python and dependency versions
- VM architecture (`x86_64` or `aarch64`)
- client-to-bank mapping
- feature-schema hash
- initial and final model hashes
- random seeds
- local epochs and learning rate
- sample counts
- per-bank metrics
- aggregation duration and Kafka transfer sizes
- rejected or missing updates

## 13. Limitations to state in the report

- The banks are simulations produced by partitioning a synthetic IBM AML dataset.
- A single Kafka broker is a demonstration component and a single point of failure.
- ZeroTier secures the overlay path, but application-level Kafka authentication should be added for a deployment beyond the classroom test.
- FedAvg does not itself provide secure aggregation or differential privacy.
- Sending model weights instead of raw data improves data locality but does not eliminate model-update privacy attacks.
- ARM64 and x86-64 clients can exchange standard float tensors, but they must use compatible model and library behavior and must not exchange virtual environments or native binaries.

## 14. Implemented files

The implementation is isolated in this folder:

```text
distributed_federation/
├── 01_VM_SETUP.md
├── 02_ZEROTIER_KAFKA_SETUP.md
├── 03_MODEL_WEIGHT_EXCHANGE.md
├── 04_RESTART_AND_RECOVERY.md
├── README.md
├── config.example.json
├── .env.example
├── requirements-distributed.txt
├── common/
│   ├── config.py
│   ├── kafka_io.py
│   ├── model_runtime.py
│   ├── protocol.py
├── central/
│   └── aggregator.py
├── client/
│   └── bank_worker.py
└── tools/
    └── preflight.py
```

Do not commit a real `.env`, ZeroTier credentials, HMAC secrets, SSH passwords,
or private keys. The classroom network ID and stable central managed IP are
documented in `02_ZEROTIER_KAFKA_SETUP.md`; they are routing identifiers, not
authentication secrets.
