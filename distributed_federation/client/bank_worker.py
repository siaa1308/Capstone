from __future__ import annotations

import argparse
import os
import sys
import time

from distributed_federation.common.config import load_config
from distributed_federation.common.events import emit
from distributed_federation.common.kafka_io import make_consumer, make_producer, publish_payload
from distributed_federation.common.model_runtime import (
    deserialize_state,
    prepare_runtime,
    serialize_state,
    train_local,
    validate_state,
)
from distributed_federation.common.protocol import ChunkAssembler, verify_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bank's federated-learning worker")
    parser.add_argument("--config", required=True, help="Path to federation JSON configuration")
    parser.add_argument("--client-id", required=True, help="Client ID assigned in the configuration")
    parser.add_argument("--broker", help="Override broker host:port from the JSON file")
    return parser.parse_args()


def required_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) < 32:
        raise RuntimeError(f"Set {name} to a random secret containing at least 32 characters")
    return value


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.broker)
    return run(config, args.client_id)


def run(config, client_id) -> int:
    """Callable entry point used by both the CLI and process supervisor."""
    client = config.client(client_id)
    emit('initializing', client_id=client_id)
    central_secret = required_secret("FCL_CENTRAL_SECRET")
    client_secret = required_secret("FCL_CLIENT_SECRET")
    client_index = [c.client_id for c in config.clients].index(client.client_id)

    print(f"[{client.client_id}] Loading local bank {client.bank}", flush=True)
    runtime = prepare_runtime(config, client.bank)
    print(f"[{client.client_id}] Schema SHA-256: {runtime.schema_sha256}", flush=True)
    producer = make_producer(config.broker)
    consumer = make_consumer(
        config.broker,
        f"fcl-worker-{config.run_id}-{client.client_id}",
        [config.global_topic],
    )
    assembler = ChunkAssembler()
    emit('runtime_ready', client_id=client_id, schema_sha256=runtime.schema_sha256,
         training_examples=int(len(runtime.streams['training'])))

    try:
        for expected_round in range(1, config.rounds + 1):
            emit('waiting_for_model', expected_round)
            deadline = time.monotonic() + config.round_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for global model round {expected_round}")
                message = consumer.poll(min(1.0, remaining))
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError(f"Kafka consumer error: {message.error()}")
                try:
                    envelope = verify_chunk(message.value(), central_secret)
                    if (
                        envelope["sender_id"] != "central"
                        or envelope["message_type"] != "global_model"
                        or envelope["run_id"] != config.run_id
                        or int(envelope["round_id"]) != expected_round
                    ):
                        continue
                    assembled = assembler.add(envelope)
                    if assembled is None:
                        continue
                    payload, global_meta = assembled
                    if global_meta["schema_sha256"] != runtime.schema_sha256:
                        raise ValueError("Global model schema does not match this worker")
                    state = deserialize_state(payload)
                    validate_state(state, runtime.model.state_dict())
                    runtime.model.load_state_dict(state, strict=True)
                    emit('model_loaded', expected_round, base_model_sha256=global_meta['payload_sha256'])
                    consumer.commit(message=message, asynchronous=False)
                    break
                except ValueError as exc:
                    print(f"[{client.client_id}] Rejected Kafka message: {exc}", file=sys.stderr, flush=True)

            print(f"[{client.client_id}] Round {expected_round}: training locally", flush=True)
            emit('training', expected_round)
            mean_loss = train_local(runtime, config, expected_round, client_index)
            emit('training_completed', expected_round, mean_train_loss=mean_loss)
            update_payload = serialize_state(runtime.model.state_dict())
            metadata = {
                "message_type": "client_update",
                "run_id": config.run_id,
                "round_id": expected_round,
                "sender_id": client.client_id,
                "schema_sha256": runtime.schema_sha256,
                "base_model_sha256": global_meta["payload_sha256"],
                "training_examples": int(len(runtime.streams["training"])),
                "mean_train_loss": round(mean_loss, 8),
            }
            emit('publishing_update', expected_round, bytes=len(update_payload))
            update_hash = publish_payload(
                producer, config.update_topic, update_payload, metadata,
                client_secret, config.max_chunk_bytes,
            )
            emit('update_published', expected_round, payload_sha256=update_hash,
                 training_examples=metadata['training_examples'])
            print(
                f"[{client.client_id}] Round {expected_round}: sent update "
                f"{update_hash[:12]} (loss={mean_loss:.6f})",
                flush=True,
            )
    finally:
        consumer.close()

    print(f"[{client.client_id}] Completed all {config.rounds} rounds", flush=True)
    emit('worker_completed', rounds=config.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
