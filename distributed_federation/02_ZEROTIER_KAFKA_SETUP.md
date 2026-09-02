# ZeroTier Network and Kafka Broker Setup

This guide connects all Ubuntu VMs through one private ZeroTier network and runs one Kafka broker on the central VM.

```text
Bank VMs -- ZeroTier private network --> Central VM:9092 --> Kafka
```

The first test uses one Kafka broker because this is a four-computer capstone simulation, not a production Kafka cluster. Kafka is a transport and coordination layer; it is not the federated aggregator itself.

## Confirmed classroom values

| Setting | Value |
|---|---|
| ZeroTier network | `capstonePhase3` |
| Network ID | `166359304edeba91` |
| Central member | `CentralServer(Broker)` |
| Central ZeroTier IP | `10.170.231.39` |
| Central ZeroTier interface | `ztyewypcw7` |
| Bank 1 | `KeyBank` (`10.170.231.168`) |
| Bank 2 | `FifthThirdBancorp` (`10.170.231.115`) |
| Bank 3 | `JPMorganChase` (`10.170.231.174`) |
| Kafka broker | `10.170.231.39:9092` |

The 16-character network ID is used to join the network and is not a password.
Never commit a ZeroTier API token, account credential, SSH password, or HMAC
secret.

## Relationship to the earlier Kafka prototype

The earlier [`AnshulBanda/Capstone`](https://github.com/AnshulBanda/Capstone) repository is accessible and contains a useful proof of concept:

- `fraud_detection/Data streaming/docker-compose.yml` starts ZooKeeper and a Confluent Kafka 7.5.0 broker.
- `producer.py` reads `cleaned_elliptic.csv` and publishes JSON rows to `bank_1_transactions`.
- `test_kafka.py` sends a small connectivity message.
- `commands.md` records topic creation and consumer commands.

The producer loop, JSON serialization idea, topic testing sequence, and manual verification approach can be adapted. Do not copy the old configuration unchanged because:

- it hard-codes `172.22.19.172` in multiple files;
- it opens port 9092 without restricting it to a ZeroTier interface;
- its broker advertises a machine-specific address;
- it uses the older ZooKeeper architecture, while Kafka 4.3.1 uses KRaft;
- it streams raw transaction rows, whereas the federated protocol must transfer model weights and metrics without sharing bank datasets.

The new setup therefore preserves the tested Kafka concept while replacing the address, security boundary, broker configuration, topics, and message format.

## 1. Use the private ZeroTier network

The network owner should:

1. Sign in to ZeroTier Central and open `capstonePhase3`.
2. Confirm the network ID is `166359304edeba91`.
3. Keep automatic IPv4 assignment enabled.
4. Authorize only the four known VM node IDs.

Use this naming convention when authorizing members:

- `fcl-central`
- `fcl-bank-1`
- `fcl-bank-2`
- `fcl-bank-3`

## 2. Install ZeroTier inside every Ubuntu VM

ZeroTier must run inside the VM, not only on the Windows or macOS host.

Review the installer at `https://install.zerotier.com` before running it, then install:

```bash
curl -sSf https://install.zerotier.com | sudo bash
sudo systemctl enable --now zerotier-one
sudo zerotier-cli info
```

Join the confirmed team network:

```bash
sudo zerotier-cli join 166359304edeba91
```

The network owner must authorize each of the four nodes in ZeroTier Central. Then verify:

```bash
sudo zerotier-cli listnetworks
ip -brief address
```

Record the managed ZeroTier addresses in a private team note:

| Role | Hostname | ZeroTier IP |
|---|---|---|
| Central | `CentralServer(Broker)` | `10.170.231.39` |
| Bank 1 | `KeyBank` | `10.170.231.168` |
| Bank 2 | `FifthThirdBancorp` | `10.170.231.115` |
| Bank 3 | `JPMorganChase` | `10.170.231.174` |

Test from every bank VM:

```bash
ping -c 3 10.170.231.39
```

SSH is optional; Kafka and model transfer do not use it. If remote
administration is wanted, test SSH only after confirming the username and IP:

```bash
ssh anshul-banda@10.170.231.39
```

## 3. Restrict the Ubuntu firewall to ZeroTier

On each VM, find the ZeroTier interface name, normally beginning with `zt`:

```bash
sudo zerotier-cli listnetworks
ip -brief link
```

On the confirmed central VM, verify that `10.170.231.39` maps to
`ztyewypcw7`:

```bash
ip -o -4 addr show | awk '$4 ~ /^10\.170\.231\.39\// {print $2}'
```

Then run on the central VM:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 9993/udp
sudo ufw allow in on ztyewypcw7 to any port 22 proto tcp
sudo ufw allow in on ztyewypcw7 to any port 9092 proto tcp
sudo ufw enable
sudo ufw status verbose
```

On each bank VM, port 9092 does not need to be opened because banks initiate the Kafka connection:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 9993/udp
sudo ufw enable
sudo ufw status verbose
```

On a bank VM, inbound port 22 is optional. If SSH is wanted, use the exact
interface shown in that VM's `dev` column. Bank VMs must not open inbound port
9092 because they initiate outbound connections to the central broker.

Do not expose Kafka through the host router or public internet.

## 4. Install Docker on the central VM

Use Docker's official Ubuntu repository. These commands support both Ubuntu 24.04 `amd64` and `arm64`:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify the installation:

```bash
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
sudo docker run --rm hello-world
```

For the first test, use `sudo docker` rather than adding every user to the privileged `docker` group.

## 5. Configure the single Kafka broker

Apache Kafka 4.3.1 uses KRaft mode and the broker requires Java 17 when installed natively. The official JVM Docker image contains the broker runtime.

On the central VM, confirm its ZeroTier address before creating the broker configuration:

```bash
sudo zerotier-cli listnetworks
ip -brief address
```

Then create the Kafka directory:

```bash
mkdir -p ~/fcl-kafka
cd ~/fcl-kafka
```

Create a local `.env` file with the confirmed central ZeroTier IPv4 address:

```bash
printf 'CENTRAL_ZT_IP=%s\n' '10.170.231.39' > .env
chmod 600 .env
```

Do not commit this `.env` file. Create `compose.yaml` with the following content:

```yaml
services:
  kafka:
    image: apache/kafka:4.3.1
    container_name: fcl-kafka
    hostname: fcl-kafka
    restart: unless-stopped
    ports:
      - "${CENTRAL_ZT_IP}:9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: "broker,controller"
      KAFKA_CONTROLLER_QUORUM_VOTERS: "1@fcl-kafka:9093"
      KAFKA_CONTROLLER_LISTENER_NAMES: "CONTROLLER"
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: "CONTROLLER:PLAINTEXT,INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT"
      KAFKA_LISTENERS: "INTERNAL://0.0.0.0:19092,EXTERNAL://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093"
      KAFKA_ADVERTISED_LISTENERS: "INTERNAL://fcl-kafka:19092,EXTERNAL://${CENTRAL_ZT_IP}:9092"
      KAFKA_INTER_BROKER_LISTENER_NAME: "INTERNAL"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"
      KAFKA_MESSAGE_MAX_BYTES: 2097152
      KAFKA_REPLICA_FETCH_MAX_BYTES: 2097152
      KAFKA_LOG_DIRS: "/tmp/kraft-combined-logs"
      CLUSTER_ID: "4L6g3nShT-eMCtK--X86sw"
```

This first broker intentionally uses ephemeral container storage, matching Apache's single-node example. Authoritative round checkpoints must be saved by the central aggregator outside Kafka. Recreating the Kafka container will require recreating the topics.

Start and inspect Kafka:

```bash
if grep -Eq '^CENTRAL_ZT_IP=([0-9]{1,3}\.){3}[0-9]{1,3}$' .env; then
  echo "CENTRAL_ZT_IP is valid"
else
  echo "Invalid CENTRAL_ZT_IP in .env; fix it before continuing"
fi
cat .env
sudo docker compose config >/dev/null
sudo docker compose pull
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=100 kafka
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server fcl-kafka:19092
```

The host-side port is bound specifically to the ZeroTier IP. This is important because Docker-published ports can bypass normal UFW processing. The advertised listener must also contain the central VM's ZeroTier IP—not `localhost`, the Windows host address, or the Wi-Fi address.

## 6. Create the federation topics

The runnable pipeline currently uses exactly two topics. Round metadata travels
inside the signed model envelopes, so separate control and metrics topics are
not required by the current implementation.

Run on the central VM:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --create --if-not-exists \
  --topic fcl.global-model --partitions 1 --replication-factor 1 \
  --config retention.ms=604800000

sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --create --if-not-exists \
  --topic fcl.client-updates --partitions 3 --replication-factor 1 \
  --config retention.ms=86400000

```

List topics:

```bash
sudo docker exec fcl-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server fcl-kafka:19092 --list
```

## 7. Verify access from every bank VM

On each bank VM:

```bash
nc -vz 10.170.231.39 9092
```

If `nc` is unavailable:

```bash
sudo apt install -y netcat-openbsd
```

Verify the Python Kafka client installed during VM setup:

```bash
cd ~/Capstone
source .venv/bin/activate
python -c "import confluent_kafka; print('confluent-kafka is installed')"
```

If that reports `ModuleNotFoundError`, install the complete tested dependency
set with `python -m pip install -r distributed_federation/requirements-distributed.txt`.

Test metadata access:

```bash
python - <<'PY'
from confluent_kafka.admin import AdminClient

broker = "10.170.231.39:9092"
metadata = AdminClient({"bootstrap.servers": broker}).list_topics(timeout=10)
expected = {"fcl.global-model", "fcl.client-updates"}
missing = expected - set(metadata.topics)
assert not missing, f"Missing topics: {sorted(missing)}"
print("KAFKA_METADATA_TEST_OK", sorted(expected))
PY
```

## 8. Security boundary for the demonstration

The initial broker uses plaintext Kafka inside the encrypted ZeroTier overlay. That is acceptable for a controlled classroom prototype if:

- the ZeroTier network remains private;
- only the four named VMs are authorized;
- Kafka port 9092 is permitted only on the ZeroTier interface;
- no router port forwarding is enabled;
- no credentials or network tokens are committed.

The official `apache/kafka:4.3.1` image supports both `linux/amd64` and `linux/arm64`, so the central role can later move between a Windows-hosted x86 VM and an Apple-Silicon-hosted ARM VM. Copy application checkpoints, not Docker data directories, when moving that role.

For deployment beyond the demonstration, configure Kafka TLS/SASL and certificate-based client authentication.

## 9. Troubleshooting

### A validation command closes the terminal

An older guide used `... || { echo ...; exit 1; }`. In an interactive shell,
`exit 1` closes the terminal when `.env` is invalid. Reopen the terminal, run
`cd ~/fcl-kafka`, write the exact `.env` shown in Step 5, and use the non-exiting
`if` validation block in this version.

### After reboot, Kafka port 9092 is unavailable

ZeroTier may restore its IP after Docker first tries to bind to it. On the
central VM, wait until `sudo zerotier-cli listnetworks` shows `OK`, then run:

```bash
cd ~/fcl-kafka
sudo docker compose up -d
sudo docker compose ps
sudo ss -lntp | grep ':9092'
```

Then retry `nc -vz 10.170.231.39 9092` from the bank VM.

### Ping reports zero received but `nc` succeeds

The successful TCP test is authoritative for Kafka. ICMP ping can be filtered
without breaking port 9092. Do not disable UFW when `nc` and the Python metadata
test both succeed.

### SSH asks for a password

SSH is optional. It expects the remote Ubuntu account password, not a ZeroTier
or GitHub password. The remote username is the exact, case-sensitive output of
`whoami`. Password characters are intentionally not displayed while typing.

### ZeroTier shows `ACCESS_DENIED`

Authorize that exact node in ZeroTier Central.

### Ping works but Kafka does not

Check:

```bash
sudo ufw status verbose
sudo ss -lntp | grep 9092
sudo docker compose -f ~/fcl-kafka/compose.yaml logs --tail=100 kafka
```

Confirm `KAFKA_ADVERTISED_LISTENERS` contains the central ZeroTier IP.

### The central ZeroTier IP changed

Stop Kafka, update `.env`, and recreate the container:

```bash
cd ~/fcl-kafka
sudo docker compose down
sudo docker compose up -d
```

Prefer assigning stable managed IPs to the four members in ZeroTier Central.

## 10. End-to-end infrastructure gate

Do not start a distributed model run until all checks below pass:

| Check | Run on | Required result |
|---|---|---|
| `sudo zerotier-cli info` | Every VM | `ONLINE` |
| `sudo zerotier-cli listnetworks` | Every VM | Network status `OK` and a managed IP |
| `ping -c 3 10.170.231.39` | Every bank | Replies helpful; Kafka port test below is authoritative |
| `sudo docker compose ps` | Central | Kafka state is running |
| `kafka-broker-api-versions.sh` command above | Central | Broker information, no timeout |
| `nc -vz 10.170.231.39 9092` | Every bank | Connection succeeded |
| Python metadata test above | Every bank | `KAFKA_METADATA_TEST_OK` |

If any check fails, stop at that row. Do not compensate by exposing Kafka on the Wi-Fi address or disabling the firewall globally.

## Official references

- [ZeroTier quickstart](https://docs.zerotier.com/quickstart/)
- [ZeroTier CLI](https://docs.zerotier.com/cli/)
- [Apache Kafka 4.3.1 downloads](https://kafka.apache.org/community/downloads/)
- [Apache Kafka Docker guide](https://kafka.apache.org/43/getting-started/docker/)
- [Apache Kafka Docker examples](https://github.com/apache/kafka/tree/trunk/docker/examples)
- [Docker Engine installation on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker firewall behavior](https://docs.docker.com/engine/network/packet-filtering-firewalls/)
