# Central VM checklist before every test run

Use this checklist after starting or rebooting the central VM. Run commands in
order. Do not start the aggregator until all checkpoints pass.

## 1. Restore and verify ZeroTier

```bash
sudo systemctl enable --now zerotier-one
sudo zerotier-cli info
sudo zerotier-cli listnetworks
ip -o -4 addr show | grep '10.170.231.39/'
```

Continue only when ZeroTier says `ONLINE`, network `166359304edeba91` says
`OK`, and the last command shows `10.170.231.39`.

## 2. Restore Docker and Kafka

```bash
sudo systemctl enable --now docker
sudo systemctl is-active docker

cd ~/Capstone/distributed_federation/kafka
grep '^CENTRAL_ZT_IP=10.170.231.39$' .env
sudo docker compose up -d
sudo docker compose ps -a
```

Wait until `fcl-kafka` is `healthy`. The one-time `kafka-volume-init` and
`kafka-init` services may show `Exited (0)`; that is successful.

Confirm both topics:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --list
```

Required:

```text
fcl.client-updates
fcl.global-model
```

Do not use `docker compose down -v` during a normal restart.

## 3. Synchronize the repository

```bash
cd ~/Capstone
git status --short --branch
git pull --ff-only origin Anshul-feat/modelSharing
git rev-parse HEAD
```

If tracked changes appear, stop and reconcile them before pulling. Give the
final commit hash to all three teammates; their hashes must match.

## 4. Restore the Python environment

```bash
cd ~/Capstone
source .venv/bin/activate
python -c "import torch, safetensors, confluent_kafka; print('DEPENDENCIES OK')"
```

If `.venv` is missing, follow the installation section in `QUICKSTART.md`.

## 5. Load the existing central security keys

The saved local file is `distributed_federation/.env`. A reboot clears shell
variables but does not delete this file.

```bash
test -f distributed_federation/.env && echo "SECRET FILE EXISTS"
set -a
source distributed_federation/.env
set +a
```

Verify without printing values:

```bash
python - <<'PY'
import os
names = [
    "FCL_CENTRAL_SECRET",
    "FCL_CLIENT_SECRET_BANK_1",
    "FCL_CLIENT_SECRET_BANK_2",
    "FCL_CLIENT_SECRET_BANK_3",
]
for name in names:
    print(name, "OK" if len(os.getenv(name, "")) >= 32 else "MISSING")
PY
```

All four must say `OK`.

### Rotate keys only when intentional

Do not generate new keys merely because the VM rebooted. Rotate them only if a
key may have leaked or the team intentionally wants new keys:

```bash
cd ~/Capstone
cp distributed_federation/.env distributed_federation/.env.backup
chmod 600 distributed_federation/.env.backup
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

After rotation, privately give every teammate the new central secret and only
their new bank secret. Do not begin until all four computers use the new set.

## 6. Choose and apply a new run ID

Use a new ID for every completed, stopped, timed-out, or failed attempt. Example:

```bash
cd ~/Capstone
NEW_RUN_ID='team-run-20260902-01'
sed -i -E "s/(\"run_id\"[[:space:]]*:[[:space:]]*)\"[^\"]*\"/\1\"${NEW_RUN_ID}\"/" \
  distributed_federation/config.json
python -m json.tool distributed_federation/config.json >/dev/null && echo "CONFIG OK"
grep '"broker"' distributed_federation/config.json
grep '"run_id"' distributed_federation/config.json
```

Required broker:

```text
10.170.231.39:9092
```

Send the exact updated `config.json` to all three teammates, or send the new run
ID and have them use the documented update command. Confirm everyone reports
the same value before starting.

Do not commit `config.json`.

## 7. Run central preflight

```bash
cd ~/Capstone
source .venv/bin/activate

python -m distributed_federation.tools.preflight \
  --config distributed_federation/config.json \
  --role central \
  --deep-model-check
```

Continue only when it prints `Preflight passed.`

Ask each teammate to confirm all of the following:

```text
Preflight passed
Correct assigned client ID
Same Git commit
Same run ID
Worker ready to start
```

## 8. Optional weight smoke test

Run this after infrastructure or Kafka changes, when onboarding a new teammate,
or before an important demonstration. It is optional before every routine run.

Choose a test ID that has never been used, for example:

```text
team-test-20260902-01
```

Have all three teammates start the receive command in
`TEAMMATE_WEIGHT_STREAMING.md`. When all are waiting, send:

```bash
python -m distributed_federation.tools.weight_smoke_test \
  --config distributed_federation/config.json \
  --role send \
  --test-id team-test-20260902-01 \
  --weights artifacts/federated_causal_temporal_graphsage/global_model.pt
```

Continue only if central and all three banks report the same SHA-256 and byte
count. Use a new test ID for every retry.

## 9. Start the real run

Teammates start their three bank workers first. Wait for all three to confirm
their worker is running and waiting. Then start central:

```bash
cd ~/Capstone
source .venv/bin/activate
set -a
source distributed_federation/.env
set +a

python -m distributed_federation.central.aggregator \
  --config distributed_federation/config.json
```

Keep this terminal open. For every round, verify central prints an accepted
update from `bank-1`, `bank-2`, and `bank-3`, followed by `aggregated and saved`.

## 10. Confirm completion

Central should finish with:

```text
[central] Run <run_id> complete (3 rounds)
```

Check the saved results:

```bash
find "artifacts/distributed_federation/${NEW_RUN_ID}" -maxdepth 1 -type f -printf '%f\n' | sort
```

Do not delete prior run directories; they are the experiment audit trail.

## If the attempt fails

1. Stop the aggregator and every bank worker with `Ctrl+C`.
2. Keep Kafka running.
3. Keep the partial artifacts.
4. Choose a new run ID on all four computers.
5. Rerun all preflights.
6. Start the workers first and the aggregator last.

Use `04_RESTART_AND_RECOVERY.md` for detailed diagnosis.
