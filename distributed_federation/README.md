# Distributed federation components

For the planned browser-operated training workflow, start with the
[federated training and streaming blueprint](../docs/STREAMING_BLUEPRINT.md).
For the implemented process controller, persistent events, developer commands,
tests, and current limitations, see the [backend guide](../streaming/README.md).
**The browser dashboard and platform launchers are not implemented yet.**

Central broadcasts the same model to every bank. Banks train locally and return
weights and example counts; central waits for all banks, applies sample-weighted
FedAvg, saves the result, and begins the next round. The application supervises this
existing learning workflow and adds structured status and persistence.

## Components

- `common/protocol.py`: signed chunk envelopes and hash-verified reconstruction.
- `central/aggregator.py`: broadcast, update validation, aggregation, and checkpoints.
- `client/bank_worker.py`: model receipt, local training, and update publication.
- `common/model_runtime.py`: preprocessing, model validation, training, and FedAvg.
- `common/events.py`: optional structured events for supervised execution.
- `common/checkpoints.py`: checkpoint writes and hash-verified commit markers.
- `common/kafka_io.py`: Kafka producer/consumer helpers.
- `tools/weight_smoke_test.py`: existing central-to-bank byte-transfer test without
  model loading. It verifies bytes in memory and prints success; it does not save
  the received file or return a completion receipt to central.
- `kafka/compose.yaml`: existing Kafka deployment, used as an implementation reference.

The backend integration lives in `streaming/`. Remote readiness/receipts, durable
update recovery, browser controls, and platform packaging remain planned work.
See the blueprint for the full scope and acceptance criteria.

## Earlier operational documentation

The seven migration-only setup guides, quickstart, teammate guide, and central
checklist have been removed. Their substantive earlier contents remain in
[Git history](https://github.com/siaa1308/Capstone/tree/1294588780ddef3512910accce3b0fc48ee655f4/distributed_federation).
Those documents mixed VM administration, weight transfer, and model training.

For the existing project context, see the [project README](../README.md).
Research and dataset documentation remain separate from this application work.
