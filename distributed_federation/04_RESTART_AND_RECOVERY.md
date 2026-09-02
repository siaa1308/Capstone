# Restart and Recovery Runbook

> First-time setup now starts in [`QUICKSTART.md`](QUICKSTART.md). This page is for recovery after that setup has already worked.

Use this checklist after a VM reboot, Docker/Kafka restart, terminal closure, or
failed federation attempt. Commands assume the repository is `~/Capstone`, the
tracked Kafka setup is under `distributed_federation/kafka`, and the central
ZeroTier IP is `10.170.231.39`.

## Important retry rule

The pipeline does not resume a partially completed round. After an interrupted
attempt, stop the aggregator and every bank worker, choose a new `run_id` in the
same `config.json` on all four VMs, and restart from round 1. Do not delete prior
artifacts; they are the audit record for the earlier attempt.

## 1. Stop leftover pipeline processes

In each terminal running a worker or aggregator, press `Ctrl+C`. On every VM,
check that no federation process remains:

```bash
pgrep -af 'distributed_federation.(central.aggregator|client.bank_worker)'
```

If the command prints a process that belongs to the failed attempt, stop that
exact PID gracefully with `kill PID`, wait a few seconds, and run the check again.

## 2. Restore ZeroTier on every VM

```bash
sudo systemctl enable --now zerotier-one
sudo zerotier-cli info
sudo zerotier-cli listnetworks
ip -brief address | grep -E '(^| )zt'
```

Required result: ZeroTier reports `ONLINE`, network `166359304edeba91` reports
`OK`, and the VM has its assigned managed IP. If it shows `ACCESS_DENIED`, the
network owner must authorize that node. Do not repeatedly leave and rejoin the
network, because that can create a new node identity requiring authorization.

After Kafka starts, use the client preflight in Step 6. It checks the actual
Kafka TCP connection and broker metadata, so a separate `nc` test is unnecessary.
Ping is optional; filtered ICMP is not a pipeline problem when preflight passes.

## 3. Restore Docker and Kafka on the central VM

Wait for the central ZeroTier address before starting Kafka because Docker binds
port 9092 specifically to `10.170.231.39`:

```bash
ip -o -4 addr show | grep '10.170.231.39/'
sudo systemctl enable --now docker
cd ~/Capstone/distributed_federation/kafka
grep '^CENTRAL_ZT_IP=10.170.231.39$' .env
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=100 kafka
sudo ss -lntp | grep ':9092'
```

Confirm broker health:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-broker-api-versions.sh \
  --bootstrap-server fcl-kafka:19092
```

The broker now uses a named Docker volume, and the `kafka-init` service creates
the two topics automatically. If either topic is missing, recreate it safely:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --create --if-not-exists \
  --topic fcl.global-model --partitions 1 --replication-factor 1 \
  --config retention.ms=604800000

sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --create --if-not-exists \
  --topic fcl.client-updates --partitions 3 --replication-factor 1 \
  --config retention.ms=86400000

sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --list
```

## 4. Synchronize repository and configuration

On every VM:

```bash
cd ~/Capstone
git status --short --branch
git pull --ff-only origin Anshul-feat/modelSharing
git rev-parse HEAD
source .venv/bin/activate
```

All four VMs must show the same commit. Do not pull over uncommitted work; stop
and reconcile it first. Confirm every VM has the same client mapping, model
settings, broker, topics, and newly chosen `run_id`:

```bash
python -m json.tool distributed_federation/config.json
```

Only `--client-id` and the worker's local secret differ between bank commands.
The ZeroTier display name does not choose the bank dataset; `clients` in
`config.json` does.

## 5. Reload secrets

On the central VM:

```bash
cd ~/Capstone
set -a
source distributed_federation/.env
set +a
```

On each bank VM, reload the central secret and only that bank's secret from the
team's private secret-storage method. If entering them interactively:

```bash
read -rsp 'Central HMAC secret: ' FCL_CENTRAL_SECRET; echo
export FCL_CENTRAL_SECRET
read -rsp 'This bank client secret: ' FCL_CLIENT_SECRET; echo
export FCL_CLIENT_SECRET
```

Never paste secrets into Git, screenshots, issue trackers, or this runbook.

## 6. Run preflight on all four VMs

Central:

```bash
cd ~/Capstone
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role central --deep-model-check
```

Each bank, using its assigned ID:

```bash
cd ~/Capstone
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role client \
  --client-id bank-1 --deep-model-check
```

Do not start until every VM prints `Preflight passed.` The preflight verifies
imports, signed chunk assembly, TCP access, both Kafka topics, required secrets,
and—when requested—the real model and shared schema.

## 7. Send the tracked test weights without training

For the first setup, or after material Kafka changes, verify the exact tracked
weight file can travel through the pipeline before starting expensive training.
Use a unique smoke-test ID on all four VMs; it is separate from the experiment
`run_id` in `config.json`.

Start one receiver on each bank VM first:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json --role receive \
  --client-id bank-1 --test-id weights-2026-09-02-01
```

Use `bank-2` and `bank-3` on the corresponding VMs. Then send from the central
VM:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json --role send \
  --test-id weights-2026-09-02-01 \
  --weights artifacts/federated_causal_temporal_graphsage/global_model.pt
```

The sender and all three receivers must print the same SHA-256 and byte count.
This test does not deserialize the pickle checkpoint, load bank data, update a
model, or aggregate anything. It exercises the real Kafka producer/consumer,
per-bank consumer groups, HMAC verification, chunk reassembly, and payload hash.

Use a new `--test-id` for every attempt so committed Kafka offsets from an older
test cannot hide a delivery problem.

## 8. Restart the training pipeline in the correct order

Start the worker first on each bank VM:

```bash
python -m distributed_federation.client.bank_worker \
  --config distributed_federation/config.json --client-id bank-1
```

Use `bank-2` and `bank-3` on the assigned VMs. Once all workers are waiting,
start the central aggregator:

```bash
python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

Successful rounds create `.safetensors` weights and JSON manifests under:

```text
artifacts/distributed_federation/<run_id>/
```

## 9. Fast failure diagnosis

| Symptom | Check |
|---|---|
| ZeroTier `ACCESS_DENIED` | Authorize the exact VM node in ZeroTier Central. |
| Client preflight cannot reach Kafka | Check ZeroTier status, Kafka container, port binding, and central UFW rule. |
| Ping fails but client preflight succeeds | Continue; the real Kafka connection is authoritative. |
| Broker metadata timeout | Check advertised listener and `CENTRAL_ZT_IP` in `~/Capstone/distributed_federation/kafka/.env`. |
| Missing Kafka topics | Run the idempotent topic-creation commands in Step 3. |
| Missing/short secret | Reload the correct environment variables in the current terminal. |
| Worker waits forever | Confirm identical `run_id`, topic names, broker, schema settings, and Git commit. |
| Central reports a missing client | Verify the assigned `client_id`; stop all processes and retry with a new `run_id`. |
| Schema mismatch | Confirm identical code, configuration, schema bank list, datasets, and dependency versions. |
| Stale/wrong base model | Stop the full run and restart all nodes with a new `run_id`. |
| Kafka fails after reboot | Wait for `10.170.231.39`, then rerun `docker compose up -d`. |

## 10. Intentional shutdown after a successful run

Stop the Python processes with `Ctrl+C` if they are still open. Kafka can remain
running. To stop it without deleting its current container:

```bash
cd ~/Capstone/distributed_federation/kafka
sudo docker compose stop
```

Restart later with `sudo docker compose up -d`. Avoid `docker compose down -v`:
the demo broker is disposable, but that command makes accidental data loss and
topic recreation more likely.
