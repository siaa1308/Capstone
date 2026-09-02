from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from distributed_federation.common.config import RUN_ID_RE, load_config
from distributed_federation.common.kafka_io import make_consumer, make_producer, publish_payload
from distributed_federation.common.protocol import ChunkAssembler, sha256_bytes, verify_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send or receive a tracked weight file without model loading or training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=("send", "receive"))
    parser.add_argument("--test-id", required=True, help="Unique ID for this smoke-test attempt")
    parser.add_argument("--client-id", help="Configured client ID; required for receive")
    parser.add_argument("--weights", help="Weight file path; required for send")
    parser.add_argument("--broker", help="Override broker host:port from the JSON file")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def required_central_secret() -> str:
    secret = os.environ.get("FCL_CENTRAL_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("Set FCL_CENTRAL_SECRET to at least 32 characters")
    return secret


def send(args: argparse.Namespace, config, secret: str) -> None:
    if not args.weights:
        raise ValueError("--weights is required when --role send")
    path = Path(args.weights).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Weight file not found: {path}")
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"Weight file is empty: {path}")
    digest = sha256_bytes(payload)
    metadata = {
        "message_type": "weight_smoke_test",
        "run_id": args.test_id,
        "round_id": 0,
        "sender_id": "central",
        "schema_sha256": "not-loaded-for-transport-smoke-test",
        "base_model_sha256": digest,
        "filename": path.name,
        "payload_bytes": len(payload),
    }
    producer = make_producer(config.broker)
    published = publish_payload(
        producer, config.global_topic, payload, metadata, secret, config.max_chunk_bytes
    )
    print(f"WEIGHT_SMOKE_TEST_SENT sha256={published} bytes={len(payload)} file={path.name}")


def receive(args: argparse.Namespace, config, secret: str) -> None:
    if not args.client_id:
        raise ValueError("--client-id is required when --role receive")
    config.client(args.client_id)
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    consumer = make_consumer(
        config.broker,
        f"fcl-weight-smoke-{args.test_id}-{args.client_id}",
        [config.global_topic],
    )
    assembler = ChunkAssembler(ttl_seconds=args.timeout_seconds)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while time.monotonic() < deadline:
            message = consumer.poll(min(1.0, deadline - time.monotonic()))
            if message is None:
                continue
            if message.error():
                raise RuntimeError(f"Kafka consumer error: {message.error()}")
            try:
                envelope = verify_chunk(message.value(), secret)
            except ValueError:
                continue
            if (
                envelope["message_type"] != "weight_smoke_test"
                or envelope["run_id"] != args.test_id
                or envelope["sender_id"] != "central"
            ):
                continue
            assembled = assembler.add(envelope)
            if assembled is None:
                continue
            payload, metadata = assembled
            expected_size = int(metadata.get("payload_bytes", -1))
            if len(payload) != expected_size:
                raise ValueError(
                    f"Payload size mismatch: received {len(payload)}, expected {expected_size}"
                )
            digest = sha256_bytes(payload)
            consumer.commit(message=message, asynchronous=False)
            print(
                f"WEIGHT_SMOKE_TEST_RECEIVED client={args.client_id} sha256={digest} "
                f"bytes={len(payload)} file={metadata.get('filename', 'unknown')}"
            )
            return
    finally:
        consumer.close()
    raise TimeoutError(f"Timed out waiting for weight smoke test {args.test_id!r}")


def main() -> int:
    args = parse_args()
    if not RUN_ID_RE.fullmatch(args.test_id):
        raise ValueError("--test-id must be 1-64 letters, numbers, dots, underscores, or hyphens")
    config = load_config(args.config, args.broker)
    secret = required_central_secret()
    if args.role == "send":
        send(args, config, secret)
    else:
        receive(args, config, secret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
