import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from distributed_federation.common.checkpoints import save_checkpoint, verify_checkpoint
from distributed_federation.common.events import emit, event_sink
from streaming.app.locking import RuntimeLock
from streaming.app.state import StateStore
from streaming.app.controller import RunController
from streaming.app.runner import execute

ROOT = Path(__file__).resolve().parents[2]


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def config(self):
        data = json.loads((ROOT / 'distributed_federation/config.example.json').read_text())
        data.update(broker='localhost:9092', dataset_dir=str(self.root),
                    output_dir=str(self.root / 'outputs'), rounds=2)
        path = self.root / 'config.json'
        path.write_text(json.dumps(data))
        return path

    def test_event_persistence_and_cursor(self):
        store = StateStore(self.root / 'state.db')
        run = store.begin('test', 'central', {'rounds': 2})
        store.emit(run, 'broadcasting', 1, bytes=100)
        store.emit(run, 'round_completed', 1)
        store.finish(run, 'Completed')
        reopened = StateStore(store.path)
        self.assertEqual(reopened.runs()[0]['status'], 'Completed')
        first = reopened.events(run)[0]
        self.assertEqual(reopened.events(run, first['id'])[0]['kind'], 'round_completed')
        with self.assertRaises(sqlite3.IntegrityError):
            reopened.begin('test', 'central', {})

    def test_lock_prevents_another_process_and_releases(self):
        path = self.root / 'runtime.lock'
        code = ('from streaming.app.locking import RuntimeLock; import sys; '
                'lock=RuntimeLock(sys.argv[1]); lock.__enter__(); lock.__exit__()')
        with RuntimeLock(path):
            result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_commit_corruption_and_no_overwrite(self):
        save_checkpoint(b'weights', self.root, 1, {'round_id': 1})
        verify_checkpoint(self.root, 1)
        with self.assertRaises(FileExistsError):
            save_checkpoint(b'changed', self.root, 1, {})
        (self.root / 'global_round_001.safetensors').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            verify_checkpoint(self.root, 1)

    def test_partial_checkpoint_is_not_committed(self):
        from distributed_federation.common import checkpoints
        real = checkpoints.atomic_write
        def fail_manifest(path, payload):
            if str(path).endswith('.json'):
                raise OSError('disk full')
            real(path, payload)
        with patch.object(checkpoints, 'atomic_write', side_effect=fail_manifest):
            with self.assertRaises(OSError):
                save_checkpoint(b'weights', self.root, 1, {})
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint(self.root, 1)

    def test_controller_freezes_allowlisted_config(self):
        path = self.config()
        raw = json.loads(path.read_text())
        raw['secret'] = 'must-not-persist'
        path.write_text(json.dumps(raw))
        controller = RunController(self.root / 'state')
        with patch('streaming.app.controller.subprocess.Popen') as popen:
            controller.start(path, 'bank', 'bank-1')
        argv = popen.call_args.args[0]
        saved = Path(argv[argv.index('--config') + 1]).read_text()
        self.assertNotIn('must-not-persist', saved)
        self.assertEqual(json.loads(saved)['rounds'], 2)
        self.assertEqual(argv[-2:], ['--client-id', 'bank-1'])
        with self.assertRaises(ValueError):
            controller.start(path, 'arbitrary-shell')

    def test_runner_persists_failure_without_secret(self):
        state = self.root / 'state'
        with patch('distributed_federation.central.aggregator.run', side_effect=RuntimeError('secret-value')):
            with self.assertRaises(RuntimeError):
                execute(self.config(), 'central', state)
        store = StateStore(state / 'state.sqlite3')
        self.assertEqual(store.runs()[0]['status'], 'Failed')
        self.assertNotIn('secret-value', json.dumps(store.events(1)))

    def test_runner_success_and_event_context_reset(self):
        def train(config):
            emit('round_completed', 1, checkpoint='test')
            return 0
        with patch('distributed_federation.central.aggregator.run', side_effect=train):
            self.assertEqual(execute(self.config(), 'central', self.root / 'state'), 0)
        store = StateStore(self.root / 'state/state.sqlite3')
        self.assertEqual(store.runs()[0]['status'], 'Completed')
        self.assertEqual(store.events(1)[-1]['kind'], 'round_completed')
        emit('outside_run')
        self.assertEqual(len(store.events(1)), 2)

    def test_interrupted_state_is_not_reported_complete(self):
        store = StateStore(self.root / 'state.db')
        store.begin('old', 'central', {})
        store.interrupt_previous()
        self.assertEqual(store.runs()[0]['status'], 'Interrupted')

    def test_status_reconciles_only_when_no_worker_holds_lock(self):
        controller = RunController(self.root / 'state')
        controller.store.begin('old', 'central', {})
        with RuntimeLock(controller.state_dir / 'runtime.lock'):
            self.assertEqual(controller.status()[0]['status'], 'Running')
        self.assertEqual(controller.status()[0]['status'], 'Interrupted')


if __name__ == '__main__':
    unittest.main()
