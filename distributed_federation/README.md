# Plug-and-play model-weight federation

This folder runs the repository's existing `CausalTemporalGraphSAGE` model across one central VM and three bank VMs. Raw rows stay inside each worker process; Kafka carries only signed, chunked `safetensors` model states and small metadata records.

Read `01_VM_SETUP.md` and `02_ZEROTIER_KAFKA_SETUP.md` first. Run every command below from the repository root inside an Ubuntu VM.

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

Replace `CENTRAL_ZT_IP` with the central VM's ZeroTier IP. Give the same `config.json` to every bank VM. Do not change it independently on different machines. Use a new `run_id` for every attempt so Kafka cannot mix runs.

The default starts a newly seeded global model. To resume the repository's trusted checkpoint, set `initial_checkpoint` to `artifacts/federated_causal_temporal_graphsage/global_model.pt`. The code refuses to load pickle checkpoints from outside this repository.

## 3. Set secrets

Generate four values on the central VM:

```bash
openssl rand -hex 32
openssl rand -hex 32
openssl rand -hex 32
openssl rand -hex 32
```

Set the first as `FCL_CENTRAL_SECRET`. Set one remaining value for each central-side `FCL_CLIENT_SECRET_BANK_N`. Give every worker the central secret and only its own client secret. Environment variables last for the current shell only.

Central example:

```bash
export FCL_CENTRAL_SECRET='paste-central-secret'
export FCL_CLIENT_SECRET_BANK_1='paste-bank-1-secret'
export FCL_CLIENT_SECRET_BANK_2='paste-bank-2-secret'
export FCL_CLIENT_SECRET_BANK_3='paste-bank-3-secret'
```

Bank 1 example (use its assigned value):

```bash
export FCL_CENTRAL_SECRET='paste-central-secret'
export FCL_CLIENT_SECRET='paste-bank-1-secret'
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

## 5. Run

Start the central process first:

```bash
python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

Then run exactly one assigned worker per bank VM:

```bash
python -m distributed_federation.client.bank_worker \
  --config distributed_federation/config.json --client-id bank-1
```

Use `bank-2` and `bank-3` on the other two VMs. The central process waits for every configured client and stops the round on timeout; it never silently aggregates an incomplete round. Completed global models and JSON audit manifests appear under `artifacts/distributed_federation/<run_id>/`.

## Important simulation boundary

The current project fits one shared categorical encoder from all five repository datasets so tensor shapes match on every VM. Therefore every clone currently contains all five datasets even though each worker trains only its assigned bank. This accurately tests distributed weight exchange, Kafka, ZeroTier, and FedAvg, but it is not yet a production privacy boundary. A later production version should distribute a frozen encoder/schema without distributing other banks' raw files.
