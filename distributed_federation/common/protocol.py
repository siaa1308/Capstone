from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = 1
REQUIRED_FIELDS = {
    "protocol_version", "message_type", "run_id", "round_id", "sender_id",
    "schema_sha256", "base_model_sha256", "payload_sha256", "chunk_index",
    "chunk_count", "payload_b64", "created_at",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _signature(message: dict[str, Any], secret: str) -> str:
    unsigned = {k: v for k, v in message.items() if k != "message_hmac_sha256"}
    return hmac.new(secret.encode("utf-8"), canonical_json(unsigned), hashlib.sha256).hexdigest()


def create_chunks(payload: bytes, metadata: dict[str, Any], secret: str, chunk_size: int) -> list[bytes]:
    if not secret or len(secret) < 32:
        raise ValueError("HMAC secret must contain at least 32 characters")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    payload_hash = sha256_bytes(payload)
    chunk_count = max(1, math.ceil(len(payload) / chunk_size))
    messages: list[bytes] = []
    for index in range(chunk_count):
        chunk = payload[index * chunk_size : (index + 1) * chunk_size]
        message = {
            "protocol_version": PROTOCOL_VERSION,
            **metadata,
            "payload_sha256": payload_hash,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "payload_b64": base64.b64encode(chunk).decode("ascii"),
            "created_at": int(time.time()),
        }
        missing = REQUIRED_FIELDS - message.keys()
        if missing:
            raise ValueError(f"Message metadata missing: {', '.join(sorted(missing))}")
        message["message_hmac_sha256"] = _signature(message, secret)
        messages.append(canonical_json(message))
    return messages


def peek_sender(raw: bytes) -> str:
    try:
        return str(json.loads(raw)["sender_id"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Message is not valid protocol JSON") from exc


def verify_chunk(raw: bytes, secret: str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Message is not valid JSON") from exc
    if not isinstance(message, dict):
        raise ValueError("Message must be a JSON object")
    missing = (REQUIRED_FIELDS | {"message_hmac_sha256"}) - message.keys()
    if missing:
        raise ValueError(f"Message fields missing: {', '.join(sorted(missing))}")
    if message["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {message['protocol_version']}")
    expected = _signature(message, secret)
    if not hmac.compare_digest(expected, str(message["message_hmac_sha256"])):
        raise ValueError("Message HMAC is invalid")
    try:
        base64.b64decode(message["payload_b64"], validate=True)
    except Exception as exc:
        raise ValueError("Chunk payload is not valid base64") from exc
    count, index = int(message["chunk_count"]), int(message["chunk_index"])
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("Invalid chunk index/count")
    return message


@dataclass
class _Pending:
    metadata_fingerprint: bytes
    chunks: dict[int, bytes] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.monotonic)


class ChunkAssembler:
    def __init__(self, max_payload_bytes: int = 100_000_000, ttl_seconds: int = 900) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.ttl_seconds = ttl_seconds
        self._pending: dict[tuple[Any, ...], _Pending] = {}

    def add(self, message: dict[str, Any]) -> tuple[bytes, dict[str, Any]] | None:
        self._discard_expired()
        key = (
            message["message_type"], message["run_id"], int(message["round_id"]),
            message["sender_id"], message["payload_sha256"],
        )
        stable = {
            k: v for k, v in message.items()
            if k not in {"payload_b64", "message_hmac_sha256", "chunk_index", "created_at"}
        }
        fingerprint = canonical_json(stable)
        pending = self._pending.setdefault(key, _Pending(fingerprint))
        if not hmac.compare_digest(pending.metadata_fingerprint, fingerprint):
            del self._pending[key]
            raise ValueError("Chunks with the same payload ID have inconsistent metadata")
        chunk = base64.b64decode(message["payload_b64"], validate=True)
        index = int(message["chunk_index"])
        previous = pending.chunks.get(index)
        if previous is not None and previous != chunk:
            del self._pending[key]
            raise ValueError("Conflicting duplicate chunk received")
        pending.chunks[index] = chunk
        pending.updated_at = time.monotonic()
        size = sum(len(value) for value in pending.chunks.values())
        if size > self.max_payload_bytes:
            del self._pending[key]
            raise ValueError("Payload exceeds configured safety limit")
        count = int(message["chunk_count"])
        if len(pending.chunks) != count:
            return None
        payload = b"".join(pending.chunks[i] for i in range(count))
        del self._pending[key]
        if sha256_bytes(payload) != message["payload_sha256"]:
            raise ValueError("Reassembled payload SHA-256 does not match")
        metadata = {k: v for k, v in message.items() if k not in {"payload_b64", "message_hmac_sha256"}}
        return payload, metadata

    def _discard_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        for key in [k for k, value in self._pending.items() if value.updated_at < cutoff]:
            del self._pending[key]
