# Simple four-computer setup

Use this page first. It tells you how to check whether each step is already complete and what to do only when it is missing.

## The one-minute picture

| Computer | What it runs |
|---|---|
| Central VM | ZeroTier, Docker, one Kafka broker, and the aggregator |
| Bank 1 VM | ZeroTier and the `bank-1` worker |
| Bank 2 VM | ZeroTier and the `bank-2` worker |
| Bank 3 VM | ZeroTier and the `bank-3` worker |

Kafka carries signed, chunked model weights. It does not carry the banks' raw transaction rows.

Run commands from `~/Capstone` unless a step says otherwise. Finish each numbered checkpoint on all relevant VMs before moving forward.

## 1. Is the Ubuntu VM ready?

Run on all four VMs:

```bash
hostname
python3 --version
git --version
```

Ready means all three commands work. If they do, skip installation.

If not ready, install Ubuntu Server 24.04 in the VM and follow `01_VM_SETUP.md`. Do not troubleshoot Kafka until the VM itself passes this check.

## 2. Is ZeroTier connected?

Run on all four VMs:

```bash
sudo zerotier-cli info
sudo zerotier-cli listnetworks
```

Ready means the service says `ONLINE`, network `166359304edeba91` says `OK`, and the VM has its assigned `10.170.231.x` address.

If `zerotier-cli` is missing:

```bash
curl -sSf https://install.zerotier.com | sudo bash
sudo systemctl enable --now zerotier-one
sudo zerotier-cli join 166359304edeba91
```

The network owner must authorize the new node in ZeroTier Central. If it says `ACCESS_DENIED`, authorization—not reinstallation—is the missing step.

Confirmed addresses:

| Role | Address |
|---|---|
| Central | `10.170.231.39` |
| Bank 1 | `10.170.231.168` |
| Bank 2 | `10.170.231.115` |
| Bank 3 | `10.170.231.174` |

## 3. Is the repository and Python environment ready?

Run on all four VMs:

```bash
cd ~/Capstone
git status --short --branch
test -x .venv/bin/python && echo "venv exists"
```

If the repository is missing:

```bash
git clone --branch Anshul-feat/modelSharing https://github.com/siaa1308/Capstone.git ~/Capstone
cd ~/Capstone
```

If `.venv` is missing, or imports fail:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip librdkafka-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r distributed_federation/requirements-distributed.txt
```

Verify instead of guessing:

```bash
source .venv/bin/activate
python -c "import torch, safetensors, confluent_kafka; print('Python dependencies OK')"
```

## 4. Is Kafka running? (central VM only)

Check Docker first:

```bash
sudo docker version
sudo docker compose version
```

If either command is missing, use the Docker installation section in `02_ZEROTIER_KAFKA_SETUP.md`.

Start the tracked broker configuration:

```bash
cd ~/Capstone/distributed_federation/kafka
test -f .env || cp .env.example .env
grep '^CENTRAL_ZT_IP=10.170.231.39$' .env
sudo docker compose config --quiet
sudo docker compose up -d
sudo docker compose ps
```

Ready means `fcl-kafka` becomes `healthy`. The one-time `kafka-volume-init` and
`kafka-init` containers should finish successfully; the first prepares storage
permissions and the second automatically creates both topics.

Verify the topics:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --list
```

The list must include `fcl.global-model` and `fcl.client-updates`.

## 5. Is the shared federation configuration ready?

On one VM, create it if it does not exist:

```bash
cd ~/Capstone
test -f distributed_federation/config.json || \
  cp distributed_federation/config.example.json distributed_federation/config.json
sed -i 's/CENTRAL_ZT_IP/10.170.231.39/' distributed_federation/config.json
python -m json.tool distributed_federation/config.json >/dev/null && echo "Config OK"
```

Copy the same `config.json` to all four VMs. Before every run, confirm all four copies have the same `run_id`, broker, clients, topics, and model settings. Use a new `run_id` after every failed or completed attempt.

## 6. Are the secrets loaded?

Central needs `FCL_CENTRAL_SECRET` and all three `FCL_CLIENT_SECRET_BANK_N` values. Each bank needs `FCL_CENTRAL_SECRET` and only its own value renamed to `FCL_CLIENT_SECRET`.

Check without printing secrets:

```bash
python - <<'PY'
import os
for name in ("FCL_CENTRAL_SECRET", "FCL_CLIENT_SECRET"):
    print(name, "OK" if len(os.getenv(name, "")) >= 32 else "MISSING")
PY
```

On central, use the variable names in `distributed_federation/.env.example`; the central preflight checks all four. Never put real secrets in Git or group chat.

## 7. Does the complete setup pass?

Central VM:

```bash
cd ~/Capstone
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role central --deep-model-check
```

Each bank VM, changing the ID:

```bash
cd ~/Capstone
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role client \
  --client-id bank-1 --deep-model-check
```

This replaces `nc`. It checks the Python packages, signed chunk protocol, real Kafka connection, required topics, secrets, dataset/schema, and model. Do not start the experiment until all four computers print `Preflight passed.`

## 8. Can a real weight file travel through Kafka?

Use a new test ID. Start the receiver first on every bank:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json --role receive \
  --client-id bank-1 --test-id weights-test-001
```

Change the client ID on Bank 2 and Bank 3. Then send from central:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json --role send \
  --test-id weights-test-001 \
  --weights artifacts/federated_causal_temporal_graphsage/global_model.pt
```

Success means the sender and all three receivers print the same SHA-256 and byte count. If this passes, Kafka model-weight delivery works independently of training.

## 9. Run the actual federation

Start one worker on each bank first:

```bash
python -m distributed_federation.client.bank_worker \
  --config distributed_federation/config.json --client-id bank-1
```

After all three workers are waiting, start central:

```bash
python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

Results appear under `artifacts/distributed_federation/<run_id>/`.

## If anything fails

Do not reinstall everything. Record which numbered checkpoint failed and its exact output. Use `04_RESTART_AND_RECOVERY.md` for restarts and its diagnosis table for errors.

The most common causes are:

- ZeroTier node has not been authorized.
- Kafka started before the central ZeroTier address appeared.
- `config.json` differs between computers.
- Secrets were not loaded in the current terminal.
- A reused `run_id` or smoke-test ID consumed old Kafka offsets.
- Workers were assigned the wrong `--client-id`.
