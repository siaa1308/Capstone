"""Optional cooperative control for supervised local processes."""
from contextlib import contextmanager
from contextvars import ContextVar
_source = ContextVar('control_source', default=None)

class RunCancelled(Exception):
    pass

class RunStopped(Exception):
    pass

def check_control(round_boundary=False):
    source = _source.get()
    action = source() if source else None
    if action == 'cancel':
        raise RunCancelled()
    if round_boundary and action == 'stop':
        raise RunStopped()

@contextmanager
def control_source(source):
    token = _source.set(source)
    try:
        yield
    finally:
        _source.reset(token)
