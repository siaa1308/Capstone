# Teammate guide: receive and send model weights

Send this page to every teammate. It covers the setup on a bank VM and the exact commands used during a run.

## What you are responsible for

Each teammate runs one bank worker:

| Assignment | Client ID | Local dataset |
|---|---|---|
| Bank 1 | `bank-1` | `Key_Bank` |
| Bank 2 | `bank-2` | `Fifth_Third_Bancorp` |
| Bank 3 | `bank-3` | `JPMorgan_Chase` |

The program handles weight transfer automatically:

```text
Central aggregator
  → sends signed global weights through Kafka
Bank worker
  → verifies and loads the global weights
  → trains on its assigned local bank data
  → sends signed updated weights through Kafka
Central aggregator
  → verifies all bank updates and performs FedAvg
```

Do not send raw transaction rows, datasets, `.venv`, or model files through chat. During a real run, do not manually upload weights—the worker and aggregator do that for you.

## Information the central owner must give you privately

Before starting, obtain:

1. Your assigned client ID: `bank-1`, `bank-2`, or `bank-3`.
2. The exact Git branch and commit.
3. The shared `distributed_federation/config.json` file.
4. `FCL_CENTRAL_SECRET`.
5. Your own bank secret only.

The Kafka broker is currently:

```text
10.170.231.39:9092
```

Never post secrets in GitHub, screenshots, issues, or group chat.

## 1. Check ZeroTier

Run inside your Ubuntu VM:

```bash
sudo zerotier-cli info
sudo zerotier-cli listnetworks
```

Ready means ZeroTier says `ONLINE` and network `166359304edeba91` says `OK`.

If ZeroTier is missing:

```bash
curl -sSf https://install.zerotier.com | sudo bash
sudo systemctl enable --now zerotier-one
sudo zerotier-cli join 166359304edeba91
```

Tell the network owner your new node needs authorization. If the network says `ACCESS_DENIED`, wait for authorization; reinstalling will not fix it.

## 2. Get the matching project code

If the repository is not cloned:

```bash
git clone --branch Anshul-feat/modelSharing \
  https://github.com/siaa1308/Capstone.git ~/Capstone
```

If it already exists:

```bash
cd ~/Capstone
git status --short
git pull --ff-only origin Anshul-feat/modelSharing
```

If `git status --short` shows tracked changes, stop and ask before pulling. Local `.venv` and `environment-lock.txt` entries disappear after the current `.gitignore` is pulled.

Confirm your commit:

```bash
git rev-parse HEAD
```

It must match the commit supplied by the central owner.

## 3. Prepare Python

Check whether the environment already exists:

```bash
cd ~/Capstone
test -x .venv/bin/python && echo "VENV EXISTS" || echo "VENV MISSING"
```

If missing:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip librdkafka-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.11.*" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r distributed_federation/requirements-distributed.txt
```

If it already exists, activate it:

```bash
source .venv/bin/activate
```

Verify:

```bash
python -c "import torch, safetensors, confluent_kafka; print('DEPENDENCIES OK')"
```

## 4. Install the shared configuration

Place the exact file supplied by central at:

```text
~/Capstone/distributed_federation/config.json
```

Check it:

```bash
cd ~/Capstone
python -m json.tool distributed_federation/config.json >/dev/null && echo "CONFIG OK"
grep '"broker"' distributed_federation/config.json
grep '"run_id"' distributed_federation/config.json
```

The broker must be `10.170.231.39:9092`. All four computers must use the same configuration and `run_id`.

Do not commit `config.json`; it is a per-run local file ignored by Git.

### Update the run ID before a new attempt

Central chooses one new run ID and tells all three teammates. For example:

```text
team-run-20260902-01
```

Update your local copy by replacing the example value below with the value
central supplied:

```bash
cd ~/Capstone
NEW_RUN_ID='team-run-20260902-01'
sed -i -E "s/(\"run_id\"[[:space:]]*:[[:space:]]*)\"[^\"]*\"/\1\"${NEW_RUN_ID}\"/" \
  distributed_federation/config.json
python -m json.tool distributed_federation/config.json >/dev/null && echo "CONFIG OK"
grep '"run_id"' distributed_federation/config.json
```

The printed value must match central and the other two banks exactly. Use a
new run ID after every completed, stopped, timed-out, or failed attempt. The run
ID is different from the smoke-test `--test-id`.

## 5. Load your two secrets

Enter the values privately in every new terminal used for the worker. Nothing is displayed while typing:

```bash
read -rsp 'Central secret: ' FCL_CENTRAL_SECRET; echo
export FCL_CENTRAL_SECRET

read -rsp 'My bank secret: ' FCL_CLIENT_SECRET; echo
export FCL_CLIENT_SECRET
```

Check without printing them:

```bash
python - <<'PY'
import os
for name in ("FCL_CENTRAL_SECRET", "FCL_CLIENT_SECRET"):
    print(name, "OK" if len(os.getenv(name, "")) >= 32 else "MISSING")
PY
```

Both must say `OK`.

### When central rotates the security keys

Normally, enter the same two existing values again after opening a new terminal
or rebooting. Do not generate your own replacements.

If central explicitly says the keys were rotated, discard the old values and
repeat the two `read` commands using the new central secret and your new bank
secret. Environment assignments overwrite the old terminal values. Verify both
again with the safe check above.

Central must rotate all related keys as one coordinated change. If even one
computer uses an older value, signatures will be rejected.

## 6. Run your preflight

Replace `bank-1` with your assignment:

```bash
cd ~/Capstone
source .venv/bin/activate

python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json \
  --role client \
  --client-id bank-1 \
  --deep-model-check
```

Do not begin the shared run unless the final line is:

```text
Preflight passed.
```

Send central only the result and your Git commit—not your secret values.

## 7. One-time model-weight delivery test

This test proves you can receive a complete weight file from central without training.

Central will give everyone a unique test ID such as `team-test-001`. Start your receiver first:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json \
  --role receive \
  --client-id bank-1 \
  --test-id team-test-001
```

Keep it running. After all three receivers are ready, central sends the file. Success looks like:

```text
WEIGHT_SMOKE_TEST_RECEIVED client=bank-1 sha256=... bytes=... file=...
```

All three banks and central must report the same SHA-256 and byte count. Use a new test ID for every retry.

## 8. Run the real bank worker

Start all three bank workers before central starts the aggregator. Replace the client ID with your assignment:

```bash
cd ~/Capstone
source .venv/bin/activate

python -m distributed_federation.client.bank_worker \
  --config distributed_federation/config.json \
  --client-id bank-1
```

Leave the terminal open. The worker will automatically:

1. Wait for the global weights from central.
2. Verify their signature, run ID, round, schema, and tensor shapes.
3. Train locally on its assigned dataset.
4. Send its signed update back to central.
5. Repeat for every configured round.

Normal progress resembles:

```text
[bank-1] Round 1: training locally
[bank-1] Round 1: sent update ...
```

Completion resembles:

```text
[bank-1] Completed all 3 rounds
```

## Commands for the central owner

### Send the one-time smoke-test weights

Start this only after all three bank receivers are waiting:

```bash
cd ~/Capstone
source .venv/bin/activate
set -a
source distributed_federation/.env
set +a

python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json \
  --role send \
  --test-id team-test-001 \
  --weights artifacts/federated_causal_temporal_graphsage/global_model.pt
```

### Start the real federation

Start only after all three workers are waiting:

```bash
cd ~/Capstone
source .venv/bin/activate
set -a
source distributed_federation/.env
set +a

python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

The aggregator automatically sends each global model and receives all three bank updates. It refuses to aggregate an incomplete round. Results are saved under:

```text
artifacts/distributed_federation/<run_id>/
```

## Stop, retry, and recover

- Press `Ctrl+C` to stop a worker or aggregator.
- After an interrupted attempt, stop all four processes.
- Choose a new `run_id` in the same `config.json` on all four computers.
- Use a new smoke-test ID for every smoke-test retry.
- Restart workers first, then the aggregator.
- Do not delete Kafka topics merely because a run failed; run IDs isolate attempts.

For detailed recovery, use `distributed_federation/04_RESTART_AND_RECOVERY.md`.

## What to report when something fails

Send central:

```text
Assigned client ID:
Git commit:
ZeroTier status (ONLINE/other):
Preflight final output:
Exact error:
```

Never include secret values.
