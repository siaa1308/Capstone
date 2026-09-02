from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys

from distributed_federation.common.config import load_config, secret_env_name
from distributed_federation.common.protocol import ChunkAssembler, create_chunks, verify_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a VM before a federation run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=("central", "client"))
    parser.add_argument("--client-id")
    parser.add_argument("--broker")
    parser.add_argument("--deep-model-check", action="store_true", help="Fit encoders and instantiate the model")
    return parser.parse_args()


def check_imports() -> None:
    for name in ("torch", "numpy", "sklearn", "pandas", "safetensors", "confluent_kafka"):
        module = importlib.import_module(name)
        print(f"[ok] {name} {getattr(module, '__version__', '')}")


def check_broker(broker: str) -> None:
    host, port_text = broker.rsplit(":", 1)
    with socket.create_connection((host, int(port_text)), timeout=5):
        pass
    print(f"[ok] TCP connection to Kafka at {broker}")


def check_protocol() -> None:
    secret = "x" * 32
    payload = os.urandom(1_250_000)
    metadata = {
        "message_type": "preflight",
        "run_id": "preflight",
        "round_id": 0,
        "sender_id": "preflight",
        "schema_sha256": "0" * 64,
        "base_model_sha256": "0" * 64,
    }
    assembler = ChunkAssembler(max_payload_bytes=2_000_000)
    result = None
    for raw in reversed(create_chunks(payload, metadata, secret, 524_288)):
        result = assembler.add(verify_chunk(raw, secret)) or result
    if result is None or result[0] != payload:
        raise RuntimeError("Chunk/HMAC round-trip failed")
    print("[ok] Signed, out-of-order chunk round-trip")


def main() -> int:
    args = parse_args()
    if args.role == "client" and not args.client_id:
        raise ValueError("--client-id is required when --role client")
    config = load_config(args.config, args.broker)
    check_imports()
    check_protocol()
    check_broker(config.broker)

    if args.role == "central":
        names = ["FCL_CENTRAL_SECRET", *(secret_env_name(c.client_id) for c in config.clients)]
    else:
        config.client(args.client_id)
        names = ["FCL_CENTRAL_SECRET", "FCL_CLIENT_SECRET"]
    for name in names:
        if len(os.environ.get(name, "")) < 32:
            raise RuntimeError(f"Missing/short secret: {name}")
    print("[ok] Required HMAC secrets")

    if args.deep_model_check:
        from distributed_federation.common.model_runtime import prepare_runtime

        bank = config.clients[0].bank if args.role == "central" else config.client(args.client_id).bank
        runtime = prepare_runtime(config, bank)
        print(
            f"[ok] Model instantiated; schema={runtime.schema_sha256}, "
            f"training_events={len(runtime.streams['training'])}"
        )
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
