"""Per-solve search context, independent of cached tool instances."""

from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

session_id = ContextVar("parallel_search_session_id", default=None)
cancel_event = ContextVar("parallel_search_cancel_event", default=None)


@contextmanager
def search_run():
    """A new solve owns one opaque ID across its search and fetch calls."""
    token = session_id.set(str(uuid4()))
    try:
        yield
    finally:
        session_id.reset(token)
