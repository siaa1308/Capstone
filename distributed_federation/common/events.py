"""Optional structured events. Existing CLI operation remains unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar('federation_event_sink', default=None)


def emit(kind, round_id=None, **details):
    sink = _sink.get()
    if sink is not None:
        sink(kind, round_id, **details)


@contextmanager
def event_sink(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
