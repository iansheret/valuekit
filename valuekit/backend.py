"""Where a batch's work actually runs.

:func:`valuekit.run_all` owns the scheduling -- admission against a worker
limit, deadlines, input-order reassembly, and attributing every failure to
the input that caused it -- none of which cares whether the work happens in
a process on this machine or somewhere else.  This module owns the other
half: starting a unit of work, waiting for one to finish, and killing one.

A backend hands back a *handle* per unit of work.  The handle carries a
three-variant message back, the same union the local worker has always
sent: ``("ok", value)``, ``("err", exc, tb)``, or ``("err_str", type_name,
text, tb)`` when the exception itself could not be sent.  A handle that
yields no message at all died, which the scheduler already knows how to
attribute -- so a lost connection needs no new vocabulary, being exactly
what a killed process already looks like.

Waiting is a backend method rather than a module-level call because
readiness is not portable.  On POSIX ``multiprocessing.connection.wait``
accepts anything exposing ``fileno()``; on Windows it issues an overlapped
zero-length read that only works on a real Win32 handle, so a subprocess
pipe or a socket is not interchangeable there.  Keeping the wait behind the
seam makes that each backend's problem instead of a shared one.
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
from multiprocessing import connection as _mp_connection
from typing import Any, Protocol

__all__ = ["Backend", "Handle", "LocalBackend"]


class Handle(Protocol):
    """One unit of work in flight."""

    def recv(self) -> tuple | None:
        """The worker's message, or None if it died without sending one."""

    def poll(self) -> bool:
        """Whether a message is already waiting."""

    def kill(self) -> None:
        """Stop the work now."""

    def reap(self) -> None:
        """Wait for it to be gone and release what it held."""

    def death(self) -> str:
        """A phrase describing how it died, for the failure message."""


class Backend(Protocol):
    """Somewhere work can run."""

    poll_interval: float | None

    def default_workers(self) -> int: ...
    def start(self, x: Any) -> Handle: ...
    def wait(self, handles: list, timeout: float | None) -> list: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# local processes
# ---------------------------------------------------------------------------


def _child_main(conn, cache_dir: str | None, fn, x) -> None:
    """Runs in the worker process: configure the cache, run one input, send
    one message back: ("ok", value) or ("err", exc, tb) or, when the
    exception or value cannot be pickled, ("err_str", type_name, text, tb).
    """
    try:
        if cache_dir is not None:
            from .pure import set_cache_dir

            set_cache_dir(cache_dir)
        try:
            value = fn(x)
        except BaseException as e:
            tb = traceback.format_exc()
            try:
                conn.send(("err", e, tb))
            except Exception:
                conn.send(("err_str", type(e).__name__, str(e), tb))
            return
        try:
            conn.send(("ok", value))
        except Exception as e:
            conn.send(
                (
                    "err_str",
                    type(e).__name__,
                    f"the result could not be sent back: {e}",
                    traceback.format_exc(),
                )
            )
    finally:
        conn.close()


class _LocalHandle:
    """One spawned process and the pipe it reports on."""

    __slots__ = ("proc", "conn")

    def __init__(self, proc, conn):
        self.proc = proc
        self.conn = conn

    def recv(self) -> tuple | None:
        try:
            return self.conn.recv()
        except EOFError:
            return None  # died without sending

    def poll(self) -> bool:
        return bool(self.conn.poll())

    def kill(self) -> None:
        self.proc.kill()
        self.proc.join()

    def reap(self) -> None:
        self.proc.join()
        self.conn.close()

    def death(self) -> str:
        return (
            f"exit code {self.proc.exitcode}; a segfault or an "
            f"out-of-memory kill?"
        )


class LocalBackend:
    """One spawned process per input, on this machine.

    Isolation is the point: a timeout kills exactly one process and a
    segfault loses exactly one input.  Processes are daemonic, so they are
    cleaned up if the driver exits.
    """

    poll_interval = None  # nothing to hear from; block until something is ready

    def __init__(self, fn, cache_dir: str | None):
        self._fn = fn
        self._cache_dir = cache_dir
        self._ctx = multiprocessing.get_context("spawn")

    def default_workers(self) -> int:
        return os.cpu_count() or 1

    def start(self, x: Any) -> _LocalHandle:
        recv_end, send_end = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_child_main,
            args=(send_end, self._cache_dir, self._fn, x),
            daemon=True,
        )
        proc.start()
        send_end.close()  # keep only the child's handle: EOF then means death
        return _LocalHandle(proc, recv_end)

    def wait(self, handles: list, timeout: float | None) -> list:
        by_conn = {h.conn: h for h in handles}
        ready = _mp_connection.wait(list(by_conn), timeout=timeout)
        return [by_conn[c] for c in ready]

    def close(self) -> None:
        pass
