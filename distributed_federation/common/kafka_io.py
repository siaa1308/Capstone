from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .protocol import create_chunks


def make_producer(broker: str):
    from confluent_kafka import Producer

    return Producer({
        "bootstrap.servers": broker,
        "client.id": "capstone-federation",
        "acks": "all",
        "enable.idempotence": True,
        "compression.type": "zstd",
        "request.timeout.ms": 30000,
    })


def make_consumer(broker: str, group_id: str, topics: Iterable[str]):
    from confluent_kafka import Consumer

    consumer = Consumer({
        "bootstrap.servers": broker,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "session.timeout.ms": 30000,
    })
    consumer.subscribe(list(topics))
    return consumer


def publish_payload(
    producer: Any,
    topic: str,
    payload: bytes,
    metadata: dict[str, Any],
    secret: str,
    chunk_size: int,
) -> str:
    errors: list[str] = []

    def delivered(error: Any, _message: Any) -> None:
        if error is not None:
            errors.append(str(error))

    chunks = create_chunks(payload, metadata, secret, chunk_size)
    payload_hash = ""
    for index, encoded in enumerate(chunks):
        import json

        envelope = json.loads(encoded)
        payload_hash = envelope["payload_sha256"]
        key = f"{envelope['run_id']}:{envelope['round_id']}:{envelope['sender_id']}:{index}"
        while True:
            try:
                producer.produce(topic, key=key.encode(), value=encoded, callback=delivered)
                break
            except BufferError:
                producer.poll(0.25)
        producer.poll(0)
    remaining = producer.flush(30)
    if remaining or errors:
        detail = "; ".join(errors) or f"{remaining} messages not delivered"
        raise RuntimeError(f"Kafka publish failed: {detail}")
    return payload_hash
