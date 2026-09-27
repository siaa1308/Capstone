"""Commit a checkpoint pair with a hash-verified marker written last."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(payload, output_dir, round_id, manifest):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f'global_round_{round_id:03d}'
    weights, metadata, marker = [root / (stem + suffix) for suffix in
                                 ('.safetensors', '.json', '.commit.json')]
    if any(path.exists() for path in (weights, metadata, marker)):
        raise FileExistsError('Checkpoint already exists or needs reconciliation; use a new run ID')
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    records = {weights.name: hashlib.sha256(payload).hexdigest(),
               metadata.name: hashlib.sha256(encoded).hexdigest()}
    # Exclusive reservation prevents two writers from replacing the same round.
    # Keep it after a crash: partial outputs require explicit reconciliation.
    reservation = root / (stem + '.reserved')
    with reservation.open('xb'):
        pass
    atomic_write(weights, payload)
    atomic_write(metadata, encoded)
    atomic_write(marker, json.dumps(records, sort_keys=True).encode())
    return weights, metadata


def verify_checkpoint(output_dir, round_id):
    root = Path(output_dir)
    stem = f'global_round_{round_id:03d}'
    records = json.loads((root / (stem + '.commit.json')).read_text())
    expected = {stem + '.safetensors', stem + '.json'}
    if set(records) != expected:
        raise ValueError('Invalid checkpoint commit record')
    for name, digest in records.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Checkpoint integrity check failed')
    return records
