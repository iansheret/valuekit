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

from .codec import SerializationError
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


# ---------------------------------------------------------------------------
# a worker on the other end of a pipe
# ---------------------------------------------------------------------------
#
# Same machine, same filesystem, no network -- so this proves the framing,
# the value codec, the handshake and the failure mapping without any of
# them depending on ssh being configured.  It is what makes the protocol
# testable in CI, and it is the shape a remote transport will slot into:
# only how the process is launched differs.


class _PipeHandle:
    """One worker subprocess and the framed conversation with it.

    Frames are drained without blocking and parsed out of a buffer, rather
    than read on demand.  A worker speaks several times before it finishes
    -- a greeting, then the result's objects -- so "the pipe is readable"
    does not mean "the answer is here", and reading until it is would sit
    inside a task that has already blown its deadline.
    """

    __slots__ = ("proc", "objects", "_failure", "_buf", "_result", "_eof")

    def __init__(self, proc, failure: str | None = None):
        self.proc = proc
        self.objects: dict[str, bytes] = {}
        self._failure = failure
        self._buf = b""
        self._result: tuple | None = None
        self._eof = False
        if proc is not None:
            os.set_blocking(proc.stdout.fileno(), False)

    def fileno(self) -> int:
        return self.proc.stdout.fileno()

    # -- reading ---------------------------------------------------------

    def drain(self) -> None:
        """Take whatever has arrived and parse any complete frames."""
        from . import wire

        if self._result is not None or self._eof:
            return
        try:
            chunk = os.read(self.fileno(), 1 << 16)
        except BlockingIOError:
            return
        except OSError as e:
            self._result = ("err_str", "OSError", str(e), "")
            return
        if not chunk:
            self._eof = True
            return
        self._buf += chunk
        try:
            self._parse(wire)
        except wire.WireError as e:
            self._result = ("err_str", "WireError", str(e), "")

    def _parse(self, wire) -> None:
        while self._result is None:
            if len(self._buf) < 9:
                return
            n = int.from_bytes(self._buf[1:9], "little")
            if n > wire.MAX_FRAME:
                raise wire.WireError(f"frame claims {n} bytes; refusing")
            if len(self._buf) < 9 + n:
                return
            tag, body = self._buf[:1], self._buf[9 : 9 + n]
            self._buf = self._buf[9 + n :]
            if tag == wire.OBJECT:
                wire.recv_object(body, self.objects)
            elif tag == wire.READY:
                if body:
                    self._result = (
                        "err_str",
                        "RuntimeError",
                        body.decode("utf-8", "replace"),
                        "",
                    )
            elif tag == wire.RESULT:
                if body[:1] == b"o":
                    self._result = ("ok", wire.unpack(body[1:].hex(), self.objects))
                else:
                    kind, text, tb = wire.unstrings(body[1:])
                    self._result = ("err_str", kind, text, tb)
            else:
                raise wire.WireError(f"unexpected frame {tag!r}")

    def settled(self) -> bool:
        """Whether there is an answer, or the worker has gone."""
        return self._failure is not None or self._result is not None or self._eof

    def recv(self) -> tuple | None:
        if self._failure is not None:
            return ("err_str", "RuntimeError", self._failure, "")
        return self._result  # None means it went away without answering

    def poll(self) -> bool:
        self.drain()
        return self._result is not None

    # -- lifecycle -------------------------------------------------------

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def reap(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def death(self) -> str:
        return f"exit code {self.proc.returncode}; a segfault or a broken pipe?"


class PipeBackend:
    """Workers launched as subprocesses of this machine, over stdio.

    The handshake happens per task here because a worker handles one input
    and exits, mirroring the local backend.  Lifting it to once per host is
    what a persistent connection buys, and belongs with the transport that
    needs it.
    """

    poll_interval = None

    def __init__(self, fn, cache_dir: str | None, python: str | None = None):
        import sys

        self._fn = fn
        self._cache_dir = cache_dir or ""
        self._python = python or sys.executable
        # A spawned process inherits sys.path through multiprocessing's
        # preparation data; a plain subprocess does not, so it is handed over
        # explicitly. Once a worker runs against a synced snapshot the path
        # comes from there instead, and this goes away.
        self._env = dict(os.environ)
        self._env["PYTHONPATH"] = os.pathsep.join(
            p for p in sys.path if p and os.path.isdir(p)
        )

        from . import wire
        from .codehash import function_fingerprint
        from .pure import _salt

        self._greeting = wire.strings(
            _salt(),
            getattr(fn, "__module__", "") or "",
            getattr(fn, "__qualname__", "") or "",
            function_fingerprint(fn)[0],
            self._cache_dir,
        )

    def default_workers(self) -> int:
        return os.cpu_count() or 1

    def start(self, x: Any) -> _PipeHandle:
        import subprocess

        from . import wire

        proc = subprocess.Popen(
            [self._python, "-m", "valuekit.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._env,
        )
        handle = _PipeHandle(proc)
        try:
            wire.write_frame(proc.stdin, wire.HELLO, self._greeting)
            root = wire.send_value(proc.stdin, x, set())
            wire.write_frame(proc.stdin, wire.TASK, bytes.fromhex(root))
        except SerializationError as e:
            # A value the wire cannot carry is the caller's problem, not a
            # worker failure: say so against this input rather than letting
            # the worker die of a truncated stream.
            handle.kill()
            handle._failure = str(e)
        except Exception as e:
            handle.kill()
            handle._failure = f"could not send the task: {e}"
        return handle

    def wait(self, handles: list, timeout: float | None) -> list:
        import select

        settled = [h for h in handles if h.settled()]
        if settled:
            return settled  # already answered; do not block on the others
        try:
            readable, _, _ = select.select(handles, [], [], timeout)
        except (OSError, ValueError):
            return [h for h in handles if h.settled()]
        for h in readable:
            h.drain()
        return [h for h in handles if h.settled()]

    def close(self) -> None:
        pass
