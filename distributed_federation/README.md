# Distributed federation components

For the planned container-based, browser-operated file-transfer workflow, start
with the [model-weight streaming blueprint](../docs/STREAMING_BLUEPRINT.md).
**The dashboard and launchers are planned, not implemented.**

This folder still contains the existing federation implementation. The streaming
phase leaves model training, aggregation, configuration, and evaluation unchanged.
No training command is part of the proposed streaming workflow.

## Components relevant to streaming

- `common/protocol.py`: signed chunk envelopes and hash-verified reconstruction.
- `common/kafka_io.py`: Kafka producer/consumer helpers.
- `tools/weight_smoke_test.py`: existing central-to-bank byte-transfer test without
  model loading. It verifies bytes in memory and prints success; it does not save
  the received file or return a completion receipt to central.
- `kafka/compose.yaml`: existing Kafka deployment, used as an implementation reference.

The planned application adds a separate streaming service, dedicated topics,
persistent received files, authenticated receipts, status reporting, and platform
launchers. See the blueprint for scope, phases, and acceptance criteria.

## Earlier operational documentation

The numbered setup guides, quickstart, teammate guide, and central checklist now
point to the blueprint. Their old contents remain in
[Git history](https://github.com/siaa1308/Capstone/tree/1294588780ddef3512910accce3b0fc48ee655f4/distributed_federation).
Those documents mixed VM administration, weight transfer, and model training.

For the existing project context, see the [project README](../README.md).
Research reports available on newer branches remain outside this streaming change.
