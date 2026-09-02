from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_BANKS = {
    "JPMorgan_Chase",
    "Wells_Fargo",
    "Citi",
    "Fifth_Third_Bancorp",
    "Key_Bank",
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ClientConfig:
    client_id: str
    bank: str


@dataclass(frozen=True)
class FederationConfig:
    broker: str
    run_id: str
    clients: tuple[ClientConfig, ...]
    schema_banks: tuple[str, ...]
    dataset_dir: Path
    output_dir: Path
    rounds: int
    hidden_channels: int
    dropout: float
    local_epochs: int
    batch_size: int
    negative_ratio: int
    learning_rate: float
    weight_decay: float
    seed: int
    round_timeout_seconds: int
    max_chunk_bytes: int
    global_topic: str
    update_topic: str
    initial_checkpoint: Path | None

    def client(self, client_id: str) -> ClientConfig:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        choices = ", ".join(c.client_id for c in self.clients)
        raise ValueError(f"Unknown client_id {client_id!r}; choose one of: {choices}")


def _positive_int(raw: dict[str, Any], name: str) -> int:
    value = int(raw[name])
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _repo_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path, broker_override: str | None = None) -> FederationConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "broker", "run_id", "clients", "schema_banks", "dataset_dir", "output_dir",
        "rounds", "hidden_channels", "dropout", "local_epochs", "batch_size",
        "negative_ratio", "learning_rate", "weight_decay", "seed",
        "round_timeout_seconds", "max_chunk_bytes", "topics",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Configuration is missing: {', '.join(missing)}")

    broker = (broker_override or str(raw["broker"])).strip()
    if not broker or ":" not in broker or "CENTRAL_ZT_IP" in broker:
        raise ValueError("Set broker to the central VM ZeroTier IP and port, for example 10.1.2.3:9092")
    run_id = str(raw["run_id"]).strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 letters, numbers, dots, underscores, or hyphens")

    clients = tuple(ClientConfig(str(c["client_id"]).strip(), str(c["bank"]).strip()) for c in raw["clients"])
    if not clients:
        raise ValueError("At least one client is required")
    ids = [c.client_id for c in clients]
    if len(ids) != len(set(ids)) or any(not RUN_ID_RE.fullmatch(x) for x in ids):
        raise ValueError("Client IDs must be unique and use only letters, numbers, dots, underscores, or hyphens")
    invalid_banks = sorted({c.bank for c in clients} - VALID_BANKS)
    if invalid_banks:
        raise ValueError(f"Unknown client banks: {', '.join(invalid_banks)}")

    schema_banks = tuple(str(x) for x in raw["schema_banks"])
    if len(schema_banks) != len(set(schema_banks)):
        raise ValueError("schema_banks must not contain duplicates")
    if set(schema_banks) - VALID_BANKS:
        raise ValueError("schema_banks contains an unknown bank")
    if {c.bank for c in clients} - set(schema_banks):
        raise ValueError("Every client bank must also appear in schema_banks")

    dropout = float(raw["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    learning_rate = float(raw["learning_rate"])
    weight_decay = float(raw["weight_decay"])
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    max_chunk_bytes = _positive_int(raw, "max_chunk_bytes")
    if max_chunk_bytes > 700_000:
        raise ValueError("max_chunk_bytes must be <= 700000 to stay below Kafka's default message limit after base64 encoding")

    topics = raw["topics"]
    global_topic = str(topics.get("global_model", "")).strip()
    update_topic = str(topics.get("client_updates", "")).strip()
    if not global_topic or not update_topic or global_topic == update_topic:
        raise ValueError("topics.global_model and topics.client_updates must be distinct non-empty names")

    dataset_dir = _repo_path(str(raw["dataset_dir"]))
    output_dir = _repo_path(str(raw["output_dir"]))
    assert dataset_dir is not None and output_dir is not None
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    return FederationConfig(
        broker=broker,
        run_id=run_id,
        clients=clients,
        schema_banks=schema_banks,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        rounds=_positive_int(raw, "rounds"),
        hidden_channels=_positive_int(raw, "hidden_channels"),
        dropout=dropout,
        local_epochs=_positive_int(raw, "local_epochs"),
        batch_size=_positive_int(raw, "batch_size"),
        negative_ratio=_positive_int(raw, "negative_ratio"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=int(raw["seed"]),
        round_timeout_seconds=_positive_int(raw, "round_timeout_seconds"),
        max_chunk_bytes=max_chunk_bytes,
        global_topic=global_topic,
        update_topic=update_topic,
        initial_checkpoint=_repo_path(raw.get("initial_checkpoint")),
    )


def secret_env_name(client_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]", "_", client_id).upper()
    return f"FCL_CLIENT_SECRET_{normalized}"
