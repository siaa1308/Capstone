# Plug-and-play model-weight federation

> **Start with [`QUICKSTART.md`](QUICKSTART.md).** Send
> [`TEAMMATE_WEIGHT_STREAMING.md`](TEAMMATE_WEIGHT_STREAMING.md) to each bank
> teammate. The central owner should use
> [`CENTRAL_PRE_RUN_CHECKLIST.md`](CENTRAL_PRE_RUN_CHECKLIST.md) after every VM
> restart and before every experiment. The numbered documents remain detailed
> reference material.

This folder runs the repository's existing `CausalTemporalGraphSAGE` model across one central VM and three bank VMs. Raw rows stay inside each worker process; Kafka carries only signed, chunked `safetensors` model states and small metadata records.

Confirmed infrastructure for the classroom run:

- ZeroTier network: `capstonePhase3` (`166359304edeba91`)
- Central member: `CentralServer(Broker)`
- Central ZeroTier IP and Kafka broker: `10.170.231.39:9092`
- Central ZeroTier interface: `ztyewypcw7`
- Bank 1: `KeyBank` at `10.170.231.168` (`Key_Bank` dataset)
- Bank 2: `FifthThirdBancorp` at `10.170.231.115` (`Fifth_Third_Bancorp` dataset)
- Bank 3: `JPMorganChase` at `10.170.231.174` (`JPMorgan_Chase` dataset)

Read `01_VM_SETUP.md` and `02_ZEROTIER_KAFKA_SETUP.md` first. Use
`04_RESTART_AND_RECOVERY.md` whenever the VMs or pipeline have been stopped or
rebooted. Run every command below from the repository root inside an Ubuntu VM.

## 1. Install the Python runtime on every VM

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip librdkafka-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.* --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r distributed_federation/requirements-distributed.txt
```

## 2. Make one shared run configuration

On the central VM:

```bash
cp distributed_federation/config.example.json distributed_federation/config.json
nano distributed_federation/config.json
```

Set the confirmed central broker address on every VM:

```bash
sed -i 's/CENTRAL_ZT_IP/10.170.231.39/' distributed_federation/config.json
python -m json.tool distributed_federation/config.json >/dev/null && echo "Config JSON is valid"
```

The same client mapping, model settings, and `run_id` must be used on all four
VMs. Use a new `run_id` for every attempt, including after a failed or interrupted
attempt, so Kafka cannot mix runs. The current implementation restarts a run from
round 1; it does not resume a partially completed Kafka round. The ZeroTier
display name `KeyBank` does not automatically select the `Key_Bank` dataset;
the assignment is controlled only by the `clients` entries in `config.json`.

The default starts a newly seeded global model. To resume the repository's trusted checkpoint, set `initial_checkpoint` to `artifacts/federated_causal_temporal_graphsage/global_model.pt`. The code refuses to load pickle checkpoints from outside this repository.

## 3. Set secrets

Generate and load four values on the central VM. The resulting `.env` is
ignored by Git and must stay private:

```bash
umask 077
{
  printf 'FCL_CENTRAL_SECRET=%s\n' "$(openssl rand -hex 32)"
  printf 'FCL_CLIENT_SECRET_BANK_1=%s\n' "$(openssl rand -hex 32)"
  printf 'FCL_CLIENT_SECRET_BANK_2=%s\n' "$(openssl rand -hex 32)"
  printf 'FCL_CLIENT_SECRET_BANK_3=%s\n' "$(openssl rand -hex 32)"
} > distributed_federation/.env
set -a
source distributed_federation/.env
set +a
```

Give every worker the central secret and only its matching bank secret through a
private channel. Do not put secrets in Git, screenshots, or group chat.

On each bank VM, enter the two values without displaying them on screen:

```bash
read -rsp 'Central HMAC secret: ' FCL_CENTRAL_SECRET; echo
export FCL_CENTRAL_SECRET
read -rsp 'This bank client secret: ' FCL_CLIENT_SECRET; echo
export FCL_CLIENT_SECRET
```

After opening a new central terminal, reload its saved values with:

```bash
set -a
source distributed_federation/.env
set +a
```

## 4. Preflight every VM

Central:

```bash
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role central --deep-model-check
```

Bank 1 (change the ID on the other VMs):

```bash
source .venv/bin/activate
python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json --role client --client-id bank-1 --deep-model-check
```

Do not start the experiment until all four preflights say `Preflight passed.`

Before the first training run, use the tracked
`artifacts/federated_causal_temporal_graphsage/global_model.pt` checkpoint for
the no-training Kafka transfer test in `04_RESTART_AND_RECOVERY.md`. This proves
that the real weight file can be signed, chunked, delivered to all three worker
consumer groups, reconstructed, and hash-verified without model loading or
local training.

If the central VM was rebooted, first confirm ZeroTier shows `OK`, then restore
the broker before running preflight:

```bash
cd ~/Capstone/distributed_federation/kafka
sudo docker compose up -d
sudo docker compose ps
```

## 5. Run

Start one assigned bank worker on each bank VM first. They will wait for the
first global model:

```bash
python -m distributed_federation.client.bank_worker \
  --config distributed_federation/config.json --client-id bank-1
```

Use `bank-2` and `bank-3` on the other two VMs. After all three workers are
waiting, start the central process:

```bash
python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

The central process waits for every configured client and stops the round on
timeout; it never silently aggregates an incomplete round. With the example
configuration, completed global models and JSON audit manifests appear under
`artifacts/distributed_federation/demo-001/`.

## Important simulation boundary

The current project fits one shared categorical encoder from all five repository datasets so tensor shapes match on every VM. Therefore every clone currently contains all five datasets even though each worker trains only its assigned bank. This accurately tests distributed weight exchange, Kafka, ZeroTier, and FedAvg, but it is not yet a production privacy boundary. A later production version should distribute a frozen encoder/schema without distributing other banks' raw files.
