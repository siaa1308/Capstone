"""The child owns the lock for its entire lifetime, even if its launcher exits."""
import argparse
import json
from pathlib import Path

from distributed_federation.common.config import load_config
from distributed_federation.common.events import event_sink
from distributed_federation.common.control import control_source, check_control, RunCancelled, RunStopped
from .locking import RuntimeLock
from .state import StateStore


def execute(config_path, role, state_dir, client_id=None):
    config = load_config(config_path)
    if role == 'bank':
        config.client(client_id)
    elif role != 'central' or client_id is not None:
        raise ValueError('Invalid role/client selection')
    with RuntimeLock(Path(state_dir) / 'runtime.lock'):
        store = StateStore(Path(state_dir) / 'state.sqlite3')
        store.interrupt_previous()
        actor = 'central' if role == 'central' else 'bank:' + client_id
        # A previously used identity cannot silently replay or overwrite a run.
        run = store.begin(config.run_id, actor, json.loads(Path(config_path).read_text(encoding='utf-8')))
        try:
            with control_source(lambda: store.control(run)), event_sink(lambda kind, round_id=None, **data: store.emit(run, kind, round_id, **data)):
                store.emit(run, 'process_started', role=role, client_id=client_id)
                check_control(round_boundary=True)
                if role == 'central':
                    from distributed_federation.central.aggregator import run as train
                    result = train(config)
                else:
                    from distributed_federation.client.bank_worker import run as train
                    result = train(config, client_id)
            store.finish(run, 'Completed' if result == 0 else 'Failed')
            return result
        except (RunCancelled, RunStopped) as exc:
            status = 'Cancelled' if isinstance(exc, RunCancelled) else 'Stopped'
            store.emit(run, 'process_' + status.lower())
            store.finish(run, status)
            return 0
        except BaseException as exc:
            # Persist error type only: arbitrary exception messages may contain credentials.
            store.emit(run, 'process_failed', error_type=type(exc).__name__)
            store.finish(run, 'Interrupted' if isinstance(exc, KeyboardInterrupt) else 'Failed')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', choices=['central', 'bank'], required=True)
    parser.add_argument('--client-id')
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    return execute(args.config, args.role, args.state_dir, args.client_id)


if __name__ == '__main__':
    raise SystemExit(main())
