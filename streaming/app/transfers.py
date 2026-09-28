"""Opaque file transfers. No model deserialization or training dependencies."""
from contextlib import contextmanager
import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from distributed_federation.common.protocol import create_chunks, verify_chunk, ChunkAssembler

MAX_BYTES = 100_000_000
TOPIC = 'fcl.weight-files.v1'
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')


class Transfers:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'transfers.sqlite3'
        self.lock = threading.Lock()
        self.active = None
        self.cancelled = threading.Event()
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            for key, raw in db.execute('SELECT id,data FROM jobs').fetchall():
                data = json.loads(raw)
                if data['status'] in ('Queued', 'Sending', 'Receiving'):
                    data.update(status='Interrupted', detail='Application closed before completion.')
                    db.execute('UPDATE jobs SET data=? WHERE id=?', (json.dumps(data), key))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def history(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT data FROM jobs ORDER BY rowid DESC LIMIT 100')]

    def update(self, job, **values):
        job.update(values, updated=time.time())
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                       (job['id'], json.dumps(job)))

    def upload(self, payload):
        if not 0 < len(payload) <= MAX_BYTES:
            raise ValueError('Choose a nonempty file no larger than 100 MB.')
        key = uuid.uuid4().hex
        self.write_file(key + '.upload', payload)
        return {'file_id': key, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def write_file(self, name, payload):
        target = self.root / name
        temp = self.root / (uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('xb') as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    def start(self, settings):
        mode = settings.get('mode')
        if mode not in ('send', 'receive'):
            raise ValueError('Choose send or receive.')
        for key in ('transfer_id', 'sender', 'recipient'):
            if not ID.fullmatch(str(settings.get(key, ''))):
                raise ValueError('Transfer and participant IDs need 1–64 letters, numbers, dots, underscores or hyphens.')
        broker = str(settings.get('broker', '')).strip()
        if not broker or len(broker) > 250 or ':' not in broker or any(c.isspace() for c in broker):
            raise ValueError('Enter a Kafka broker address as host:port.')
        secret = os.environ.get('FCL_TRANSFER_SECRET', '')
        if len(secret) < 32:
            raise ValueError('Set FCL_TRANSFER_SECRET to a shared secret of at least 32 characters before launching.')
        payload = None
        if mode == 'send':
            file_id = str(settings.get('file_id', ''))
            if not re.fullmatch('[a-f0-9]{32}', file_id):
                raise ValueError('Upload a file first.')
            path = self.root / (file_id + '.upload')
            if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
                raise ValueError('Uploaded file is unavailable or too large.')
            payload = path.read_bytes()
        with self.lock:
            if self.active is not None:
                raise ValueError('A local transfer is already active.')
            self.cancelled.clear()
            job = {'id': uuid.uuid4().hex, 'mode': mode, 'transfer_id': settings['transfer_id'],
                   'sender': settings['sender'], 'recipient': settings['recipient'],
                   'status': 'Queued', 'bytes': 0, 'total': len(payload) if payload else None,
                   'detail': 'Waiting for Kafka connection.'}
            self.active = job['id']
            self.update(job)
            threading.Thread(target=self.run, args=(job, broker, secret, payload), daemon=True).start()
        return job['id']

    def cancel(self, key):
        with self.lock:
            if key != self.active:
                raise ValueError('Transfer is not active.')
            self.cancelled.set()

    def run(self, job, broker, secret, payload):
        try:
            if job['mode'] == 'send':
                self.send(job, broker, secret, payload)
            else:
                self.receive(job, broker, secret)
        except InterruptedError:
            self.update(job, status='Cancelled', detail='Local operation stopped; published messages cannot be retracted.')
        except ImportError:
            self.update(job, status='Failed', detail='Kafka client is missing. Install streaming/requirements.txt to enable transfers.')
        except Exception as exc:
            self.update(job, status='Failed', detail='Transfer failed (' + type(exc).__name__ + '). Check broker, topic and credentials.')
        finally:
            with self.lock:
                self.active = None

    def check_cancel(self):
        if self.cancelled.is_set():
            raise InterruptedError()

    def send(self, job, broker, secret, payload):
        from confluent_kafka import Producer
        producer = Producer({'bootstrap.servers': broker, 'enable.idempotence': True,
                             'message.timeout.ms': 15000})
        metadata = {'message_type': 'weight_file', 'run_id': job['transfer_id'], 'round_id': 0,
                    'sender_id': job['sender'], 'recipient_id': job['recipient'],
                    'schema_sha256': 'opaque-file', 'base_model_sha256': '', 'file_size': len(payload)}
        chunks = create_chunks(payload, metadata, secret, 500_000)
        self.update(job, status='Sending', detail='Publishing signed chunks. Progress counts broker acknowledgments.')
        errors = []
        def delivered(size):
            def callback(error, message):
                if error:
                    errors.append(True)
                else:
                    self.update(job, bytes=job['bytes'] + size)
            return callback
        deadline = time.monotonic() + 120
        for index, chunk in enumerate(chunks):
            while True:
                self.check_cancel()
                if time.monotonic() > deadline or errors:
                    raise TimeoutError()
                try:
                    producer.produce(TOPIC, key=job['transfer_id'].encode(), value=chunk,
                                     callback=delivered(min(500_000, len(payload) - index * 500_000)))
                    break
                except BufferError:
                    producer.poll(.2)
            producer.poll(0)
        while len(producer):
            self.check_cancel()
            if time.monotonic() > deadline or errors:
                raise TimeoutError()
            producer.poll(.2)
        if errors:
            raise RuntimeError('delivery failed')
        self.update(job, status='Published', sha256=hashlib.sha256(payload).hexdigest(),
                    detail='Broker acknowledged all chunks. Receiver verification is not yet confirmed.')

    def receive(self, job, broker, secret):
        from confluent_kafka import Consumer
        consumer = Consumer({'bootstrap.servers': broker, 'group.id': 'weight-file-' + job['id'],
                             'auto.offset.reset': 'earliest', 'enable.auto.commit': False,
                             'allow.auto.create.topics': False})
        assembler = ChunkAssembler(MAX_BYTES)
        seen = set()
        selected_hash = None
        deadline = time.monotonic() + 120
        self.update(job, status='Receiving', detail='Waiting up to two minutes for the matching signed transfer.')
        try:
            consumer.subscribe([TOPIC])
            while time.monotonic() < deadline:
                self.check_cancel()
                message = consumer.poll(.25)
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError('broker error')
                raw = message.value()
                if raw is None or len(raw) > 1_000_000:
                    continue
                try:
                    chunk = verify_chunk(raw, secret)
                    if (chunk['message_type'], chunk['run_id'], chunk['sender_id'], chunk.get('recipient_id')) != (
                            'weight_file', job['transfer_id'], job['sender'], job['recipient']):
                        continue
                    size = int(chunk.get('file_size', 0))
                    count = int(chunk['chunk_count'])
                    if not 0 < size <= MAX_BYTES or not 1 <= count <= 200:
                        raise ValueError('invalid transfer size')
                    if selected_hash is not None and selected_hash != chunk['payload_sha256']:
                        raise ValueError('conflicting transfer')
                    selected_hash = chunk['payload_sha256']
                    result = assembler.add(chunk)
                    index = int(chunk['chunk_index'])
                    if index not in seen:
                        seen.add(index)
                        self.update(job, bytes=job['bytes'] + len(base64.b64decode(chunk['payload_b64'])), total=size)
                    if result:
                        data, metadata = result
                        if len(data) != size:
                            raise ValueError('size mismatch')
                        self.check_cancel()
                        self.write_file(job['id'] + '.weights', data)
                        self.update(job, status='Verified', sha256=metadata['payload_sha256'],
                                    detail='Signature and SHA-256 verified; file saved locally. Model contents were not inspected.')
                        return
                except (ValueError, TypeError, KeyError):
                    # Invalid messages never become downloadable results.
                    raise ValueError('invalid signed transfer')
            raise TimeoutError()
        finally:
            consumer.close()
