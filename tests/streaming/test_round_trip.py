"""Exercise real loops/protocol/FedAvg with deterministic tensor and network doubles.

This is not a PyTorch training or real Kafka integration test.
"""
import json
import queue
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from distributed_federation.central import aggregator
from distributed_federation.client import bank_worker
from distributed_federation.common.config import load_config, secret_env_name
from distributed_federation.common.events import event_sink
from distributed_federation.common.checkpoints import save_checkpoint, verify_checkpoint


class Scalar:
    dtype = 'float'

    def __init__(self, value):
        self.value = float(value)

    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Scalar(self.value)
    def is_floating_point(self): return True
    def to(self, **kwargs): return self
    def __mul__(self, other): return Scalar(self.value * other)
    def __add__(self, other): return Scalar(self.value + other.value)
    def isfinite(self): return SimpleNamespace(all=lambda: True)


class Model:
    def __init__(self): self.state = {'weight': Scalar(0)}
    def state_dict(self): return self.state
    def load_state_dict(self, state, strict=True): self.state = state


class Message:
    def __init__(self, value): self.raw = value
    def value(self): return self.raw
    def error(self): return None


class Consumer:
    def __init__(self, messages): self.messages = messages
    def poll(self, timeout):
        try: return self.messages.get(timeout=timeout)
        except queue.Empty: return None
    def commit(self, **kwargs): pass
    def close(self): pass


class RoundTripTests(unittest.TestCase):
    def test_three_banks_two_rounds_weighted_fedavg(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            raw = json.loads((Path(__file__).resolve().parents[2] /
                              'distributed_federation/config.example.json').read_text())
            raw.update(broker='local:9092', dataset_dir=str(root), output_dir=str(root / 'out'),
                       rounds=2, round_timeout_seconds=5, max_chunk_bytes=8)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(raw))
            config = load_config(config_path)
            queues = [queue.Queue() for _ in range(3)]
            upstream = queue.Queue()
            subscribed = threading.Barrier(4)
            counts = [1, 3, 6]
            increments = [1, 3, 5]
            bases = {c.client_id: [] for c in config.clients}
            events = []

            def consumer(broker, group, topics):
                if topics == [config.update_topic]:
                    target = upstream
                else:
                    index = next(i for i, c in enumerate(config.clients) if group.endswith(c.client_id))
                    target = queues[index]
                subscribed.wait(timeout=5)
                return Consumer(target)

            class Producer:
                def produce(self, topic, key, value, callback):
                    targets = queues if topic == config.global_topic else [upstream]
                    for target in targets:
                        target.put(Message(value))
                        # Actual duplicate signed chunks exercise assembler/idempotency.
                        target.put(Message(value))
                    callback(None, None)
                def poll(self, timeout): pass
                def flush(self, timeout): return 0

            def prepare(cfg, bank):
                index = next(i for i, c in enumerate(cfg.clients) if c.bank == bank)
                return SimpleNamespace(model=Model(), schema_sha256='test-schema', schema={},
                                       streams={'training': list(range(counts[index]))})

            def serialize(state):
                return json.dumps({'weight': state['weight'].value}).encode()

            def deserialize(payload):
                return {'weight': Scalar(json.loads(payload)['weight'])}

            def train(runtime, cfg, round_id, client_index):
                value = runtime.model.state['weight'].value
                bases[cfg.clients[client_index].client_id].append(value)
                runtime.model.state = {'weight': Scalar(value + increments[client_index])}
                return 0.25

            def save(state, output, round_id, manifest):
                return save_checkpoint(serialize(state), output, round_id, manifest)

            secrets = {'FCL_CENTRAL_SECRET': 'c' * 32, 'FCL_CLIENT_SECRET': 'b' * 32}
            secrets.update({secret_env_name(c.client_id): 'b' * 32 for c in config.clients})
            stack.enter_context(patch.dict('os.environ', secrets))
            for module in (aggregator, bank_worker):
                for name, replacement in {
                    'make_consumer': consumer, 'make_producer': lambda broker: Producer(),
                    'prepare_runtime': prepare, 'serialize_state': serialize,
                    'deserialize_state': deserialize, 'validate_state': lambda *args: None,
                }.items():
                    stack.enter_context(patch.object(module, name, replacement))
            stack.enter_context(patch.object(aggregator, 'load_initial_checkpoint', return_value=False))
            stack.enter_context(patch.object(aggregator, 'save_state_and_manifest', side_effect=save))
            stack.enter_context(patch.object(bank_worker, 'train_local', side_effect=train))

            def central():
                with event_sink(lambda kind, round_id=None, **data: events.append((kind, round_id, data))):
                    return aggregator.run(config)

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(bank_worker.run, config, c.client_id) for c in config.clients]
                futures.append(pool.submit(central))
                self.assertEqual([f.result(timeout=15) for f in futures], [0, 0, 0, 0])
            # Weighted increment = .1*1 + .3*3 + .6*5 = 4, not equal-weight 3.
            for values in bases.values():
                self.assertEqual(values, [0.0, 4.0])
            output = config.output_dir / config.run_id
            for round_id in (1, 2):
                verify_checkpoint(output, round_id)
            result = json.loads((output / 'global_round_002.safetensors').read_bytes())
            self.assertEqual(result['weight'], 8.0)
            self.assertEqual(sum(e[0] == 'update_validated' for e in events), 6)
            self.assertEqual(sum(e[0] == 'round_completed' for e in events), 2)

    def test_missing_bank_prevents_aggregation(self):
        # The round times out before any aggregate or checkpoint can be produced.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            raw = json.loads((Path(__file__).resolve().parents[2] /
                              'distributed_federation/config.example.json').read_text())
            raw.update(broker='local:9092', dataset_dir=str(root), output_dir=str(root / 'out'))
            path = root / 'config.json'
            path.write_text(json.dumps(raw))
            config = replace(load_config(path), round_timeout_seconds=0)
            secrets = {'FCL_CENTRAL_SECRET': 'c' * 32}
            secrets.update({secret_env_name(c.client_id): 'b' * 32 for c in config.clients})
            stack.enter_context(patch.dict('os.environ', secrets))
            stack.enter_context(patch.object(aggregator, 'prepare_runtime', return_value=
                SimpleNamespace(model=Model(), schema_sha256='schema')))
            for name in ('load_initial_checkpoint', 'make_producer', 'make_consumer', 'publish_payload'):
                stack.enter_context(patch.object(aggregator, name))
            stack.enter_context(patch.object(aggregator, 'serialize_state', return_value=b'weights'))
            average = stack.enter_context(patch.object(aggregator, 'fedavg'))
            with self.assertRaises(TimeoutError):
                aggregator.run(config)
            average.assert_not_called()
            self.assertFalse(config.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
