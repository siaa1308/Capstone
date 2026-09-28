import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from streaming.app.transfers import Transfers
from streaming.app.dashboard import server
from distributed_federation.common.protocol import create_chunks


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.transfers = Transfers(self.tmp.name)

    def test_sender_waits_for_delivery_and_reports_publication_only(self):
        sent = []
        callbacks = []
        class Producer:
            def produce(self, topic, key, value, callback):
                sent.append(value)
                callbacks.append(callback)
            def poll(self, timeout):
                if callbacks:
                    callbacks.pop(0)(None, None)
            def __len__(self):
                return len(callbacks)
        job = dict(id='send', transfer_id='unique', sender='central', recipient='bank-1', bytes=0)
        payload = b'x' * 500001
        with patch.dict('sys.modules', {'confluent_kafka': SimpleNamespace(Producer=lambda config: Producer())}):
            self.transfers.send(job, 'local:9092', 'x' * 32, payload)
        self.assertEqual(len(sent), 2)
        self.assertEqual(job['status'], 'Published')
        self.assertEqual(job['bytes'], len(payload))
        self.assertIn('not yet confirmed', job['detail'])

    def test_receive_signed_out_of_order_duplicates(self):
        payload = b'opaque model weights' * 100
        metadata = dict(message_type='weight_file', run_id='test', round_id=0, sender_id='central',
                        recipient_id='bank-1', schema_sha256='opaque-file', base_model_sha256='', file_size=len(payload))
        chunks = create_chunks(payload, metadata, 'x' * 32, 500)
        messages = iter([chunks[-1], chunks[-1]] + chunks[:-1])
        closed = []
        def poll(timeout):
            raw = next(messages)
            return SimpleNamespace(error=lambda: None, value=lambda: raw)
        consumer = SimpleNamespace(subscribe=lambda topics: None, poll=poll, close=lambda: closed.append(True))
        job = dict(id='a' * 32, mode='receive', transfer_id='test', sender='central', recipient='bank-1', bytes=0)
        with patch.dict('sys.modules', {'confluent_kafka': SimpleNamespace(Consumer=lambda config: consumer)}):
            self.transfers.receive(job, 'local:9092', 'x' * 32)
        self.assertEqual(job['status'], 'Verified')
        self.assertEqual(job['bytes'], len(payload))
        self.assertEqual((Path(self.tmp.name) / (job['id'] + '.weights')).read_bytes(), payload)
        self.assertEqual(closed, [True])

    def test_wrong_signature_never_saves(self):
        metadata = dict(message_type='weight_file', run_id='test', round_id=0, sender_id='central',
                        recipient_id='bank-1', schema_sha256='opaque-file', base_model_sha256='', file_size=1)
        raw = create_chunks(b'x', metadata, 'x' * 32, 500)[0]
        consumer = SimpleNamespace(subscribe=lambda topics: None, close=lambda: None,
                                   poll=lambda timeout: SimpleNamespace(error=lambda: None, value=lambda: raw))
        job = dict(id='b' * 32, mode='receive', transfer_id='test', sender='central', recipient='bank-1', bytes=0)
        with patch.dict('sys.modules', {'confluent_kafka': SimpleNamespace(Consumer=lambda config: consumer)}):
            with self.assertRaises(ValueError):
                self.transfers.receive(job, 'local:9092', 'z' * 32)
        self.assertFalse(list(Path(self.tmp.name).glob('*.weights')))

    def test_upload_and_restart_history(self):
        upload = self.transfers.upload(b'weights')
        self.assertEqual(upload['bytes'], 7)
        with self.assertRaises(ValueError):
            self.transfers.upload(b'')
        self.transfers.update({'id': 'old', 'status': 'Sending'})
        reopened = Transfers(self.tmp.name)
        self.assertEqual(reopened.history()[0]['status'], 'Interrupted')

    def test_cancel_receive_closes_consumer(self):
        closed = []
        consumer = SimpleNamespace(subscribe=lambda topics: None, close=lambda: closed.append(True))
        self.transfers.cancelled.set()
        job = dict(id='cancel', mode='receive', bytes=0)
        with patch.dict('sys.modules', {'confluent_kafka': SimpleNamespace(Consumer=lambda config: consumer)}):
            with self.assertRaises(InterruptedError):
                self.transfers.receive(job, 'local:9092', 'x' * 32)
        self.assertEqual(closed, [True])

    def test_http_session_and_download_boundary(self):
        app = server(Path(self.tmp.name) / 'web', 0)
        thread = threading.Thread(target=app.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:' + str(app.server_port)
        try:
            html = urllib.request.urlopen(base).read().decode()
            token = html.split("const token='")[1].split("'")[0]
            req = urllib.request.Request(base + '/api/upload', data=b'weights', method='POST')
            with self.assertRaises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req)
            self.assertEqual(err.exception.code, 403)
            err.exception.close()
            req.add_header('Origin', base)
            req.add_header('X-Session-Token', token)
            self.assertEqual(json.load(urllib.request.urlopen(req))['bytes'], 7)
            with self.assertRaises(urllib.error.HTTPError) as missing:
                urllib.request.urlopen(base + '/api/download/../../README.md')
            missing.exception.close()
            self.assertEqual(json.load(urllib.request.urlopen(base + '/api/status'))['jobs'], [])
        finally:
            app.shutdown()
            app.server_close()
            thread.join()
