from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from distributed_federation.common.config import REPO_ROOT, load_config
from .state import StateStore
from .locking import RuntimeLock

DEFAULT_STATE = REPO_ROOT / 'streaming' / '.local'


class RunController:
    """Starts only known federation roles; credentials stay in the child environment."""

    def __init__(self, state_dir=DEFAULT_STATE, python=sys.executable):
        self.state_dir = Path(state_dir).resolve()
        self.python = str(python)
        self.store = StateStore(self.state_dir / 'state.sqlite3')

    def start(self, config_path, role, client_id=None):
        if role not in {'central', 'bank'}:
            raise ValueError('role must be central or bank')
        config = load_config(config_path)
        if role == 'bank':
            config.client(client_id)
        elif client_id is not None:
            raise ValueError('Central must not specify client_id')
        # Reconstruct an allowlisted snapshot; never copy arbitrary JSON secrets.
        raw = asdict(config)
        raw['topics'] = {'global_model': raw.pop('global_topic'),
                         'client_updates': raw.pop('update_topic')}
        snapshots = self.state_dir / 'configs'
        snapshots.mkdir(exist_ok=True)
        path = snapshots / (uuid.uuid4().hex + '.json')
        path.write_text(json.dumps(raw, default=str, indent=2), encoding='utf-8')
        command = [self.python, '-m', 'streaming.app.runner', '--config', str(path),
                   '--role', role, '--state-dir', str(self.state_dir)]
        if client_id:
            command += ['--client-id', client_id]
        # No shell; no window on Windows. Popen returning is not a readiness claim.
        return subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def stop(self, run, cancel=False):
        self.status()
        return self.store.request_control(run, 'cancel' if cancel else 'stop')

    def status(self):
        try:
            with RuntimeLock(self.state_dir / 'runtime.lock'):
                self.store.interrupt_previous()
        except RuntimeError:
            pass  # A live child owns this state directory; leave its state alone.
        return self.store.runs()


def main():
    parser = argparse.ArgumentParser(description='Developer entry point for supervised federation')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start')
    start.add_argument('--config', required=True)
    start.add_argument('--role', choices=['central', 'bank'], required=True)
    start.add_argument('--client-id')
    commands.add_parser('status')
    for action in ('stop', 'cancel'):
        control = commands.add_parser(action)
        control.add_argument('--run', type=int, required=True)
    events = commands.add_parser('events')
    events.add_argument('--run', type=int, required=True, help='Local integer run record ID')
    events.add_argument('--after', type=int, default=0)
    args = parser.parse_args()
    controller = RunController(args.state_dir)
    if args.command == 'start':
        return controller.start(args.config, args.role, args.client_id).wait()
    if args.command in {'stop', 'cancel'}:
        print(json.dumps({'requested': controller.stop(args.run, args.command == 'cancel')}))
        return 0
    result = (controller.status() if args.command == 'status' else
              controller.store.events(args.run, args.after))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
