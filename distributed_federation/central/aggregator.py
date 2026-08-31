from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime

from distributed_federation.common.config import load_config, secret_env_name
from distributed_federation.common.kafka_io import make_consumer, make_producer, publish_payload
from distributed_federation.common.model_runtime import (
    deserialize_state,
    fedavg,
    load_initial_checkpoint,
    prepare_runtime,
    save_state_and_manifest,
    serialize_state,
    validate_state,
)
from distributed_federation.common.protocol import ChunkAssembler, peek_sender, sha256_bytes, verify_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the central federated model aggregator")
    parser.add_argument("--config", required=True, help="Path to federation JSON configuration")
    parser.add_argument("--broker", help="Override broker host:port from the JSON file")
    return parser.parse_args()


def required_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) < 32:
        raise RuntimeError(f"Set {name} to a random secret containing at least 32 characters")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.broker)
    central_secret = required_secret("FCL_CENTRAL_SECRET")
    client_secrets = {client.client_id: required_secret(secret_env_name(client.client_id)) for client in config.clients}

    print(f"[central] Loading shared schema from {config.dataset_dir}", flush=True)
    runtime = prepare_runtime(config, config.clients[0].bank)
    loaded = load_initial_checkpoint(runtime.model, config.initial_checkpoint)
    print(f"[central] Initial checkpoint: {'loaded' if loaded else 'random seeded initialization'}", flush=True)
    print(f"[central] Schema SHA-256: {runtime.schema_sha256}", flush=True)

    producer = make_producer(config.broker)
    consumer = make_consumer(
        config.broker, f"fcl-central-{config.run_id}", [config.update_topic]
    )
    assembler = ChunkAssembler()
    run_output = config.output_dir / config.run_id
    global_state = {k: v.detach().cpu().clone() for k, v in runtime.model.state_dict().items()}

    try:
        for round_id in range(1, config.rounds + 1):
            global_payload = serialize_state(global_state)
            base_hash = sha256_bytes(global_payload)
            metadata = {
                "message_type": "global_model",
                "run_id": config.run_id,
                "round_id": round_id,
                "sender_id": "central",
                "schema_sha256": runtime.schema_sha256,
                "base_model_sha256": base_hash,
            }
            publish_payload(
                producer, config.global_topic, global_payload, metadata,
                central_secret, config.max_chunk_bytes,
            )
            print(f"[central] Round {round_id}: published global model {base_hash[:12]}", flush=True)

            received: dict[str, tuple[dict, int, dict]] = {}
            deadline = time.monotonic() + config.round_timeout_seconds
            while len(received) < len(config.clients):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(client_secrets) - set(received))
                    raise TimeoutError(f"Round {round_id} timed out; missing clients: {', '.join(missing)}")
                message = consumer.poll(min(1.0, remaining))
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError(f"Kafka consumer error: {message.error()}")
                raw = message.value()
                try:
                    sender = peek_sender(raw)
                    secret = client_secrets.get(sender)
                    if secret is None:
                        continue
                    envelope = verify_chunk(raw, secret)
                    if (
                        envelope["message_type"] != "client_update"
                        or envelope["run_id"] != config.run_id
                        or int(envelope["round_id"]) != round_id
                    ):
                        continue
                    assembled = assembler.add(envelope)
                    if assembled is None:
                        continue
                    payload, update_meta = assembled
                    if sender in received:
                        print(f"[central] Ignoring duplicate completed update from {sender}", flush=True)
                        continue
                    if update_meta["schema_sha256"] != runtime.schema_sha256:
                        raise ValueError(f"Schema mismatch from {sender}")
                    if update_meta["base_model_sha256"] != base_hash:
                        raise ValueError(f"Stale/wrong base model from {sender}")
                    examples = int(update_meta.get("training_examples", 0))
                    if examples <= 0:
                        raise ValueError(f"Invalid training_examples from {sender}")
                    state = deserialize_state(payload)
                    validate_state(state, global_state)
                    received[sender] = (state, examples, update_meta)
                    consumer.commit(message=message, asynchronous=False)
                    print(
                        f"[central] Round {round_id}: accepted {sender} "
                        f"({examples} events, loss={update_meta.get('mean_train_loss', 'n/a')})",
                        flush=True,
                    )
                except ValueError as exc:
                    print(f"[central] Rejected Kafka message: {exc}", file=sys.stderr, flush=True)

            ordered = [received[client.client_id] for client in config.clients]
            global_state = fedavg([item[0] for item in ordered], [item[1] for item in ordered])
            runtime.model.load_state_dict(global_state, strict=True)
            final_payload_hash = sha256_bytes(serialize_state(global_state))
            manifest = {
                "run_id": config.run_id,
                "round_id": round_id,
                "created_at": utc_now(),
                "schema_sha256": runtime.schema_sha256,
                "input_global_sha256": base_hash,
                "output_global_sha256": final_payload_hash,
                "clients": [
                    {
                        "client_id": client.client_id,
                        "bank": client.bank,
                        "training_examples": received[client.client_id][1],
                        "mean_train_loss": received[client.client_id][2].get("mean_train_loss"),
                        "update_sha256": received[client.client_id][2]["payload_sha256"],
                    }
                    for client in config.clients
                ],
                "schema": runtime.schema,
            }
            weights_path, _ = save_state_and_manifest(global_state, run_output, round_id, manifest)
            print(f"[central] Round {round_id}: aggregated and saved {weights_path}", flush=True)
    finally:
        consumer.close()

    print(f"[central] Run {config.run_id} complete ({config.rounds} rounds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
