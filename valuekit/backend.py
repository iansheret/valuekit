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

import json
import multiprocessing
import os
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
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
        except (EOFError, OSError):
            return None  # died without sending

    def poll(self) -> bool:
        try:
            return bool(self.conn.poll())
        except OSError:
            # Windows reports a dead writer as BrokenPipeError rather than
            # as end-of-file; either way there is nothing to read.
            return False

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


def _apply_queued(inbox: queue.Queue) -> None:
    """Hand every chunk the reader threads have queued to its handle."""
    while True:
        try:
            handle, chunk = inbox.get_nowait()
        except queue.Empty:
            return
        handle.feed(chunk)


class _PipeHandle:
    """One worker subprocess and the framed conversation with it.

    A thread per handle reads the worker's stdout with blocking reads and
    queues each chunk on the backend's inbox; frames are then parsed out of
    a buffer on the scheduler's thread.  A worker speaks several times
    before it finishes -- a greeting, then the result's objects -- so "a
    chunk arrived" does not mean "the answer is here", and reading until it
    is would sit inside a task that has already blown its deadline.

    Threads rather than ``select``: Windows cannot select on a pipe, and a
    blocking read is the one primitive every platform gives a pipe.

    The worker's store is this side's store, so the frames that arrive are
    not only its answer: a trace to keep, a lookup to answer, a run-log
    record to write, or a ``@pure_local`` call to make here.  Lookups are
    answered on the scheduler's thread; a call runs on a pool thread, since
    it may be a download, and replies when it is done.  ``seen`` names every
    object either side has sent, so nothing crosses twice.
    """

    __slots__ = (
        "proc", "objects", "seen", "_failure", "_buf", "_result", "_eof",
        "_inbox", "_backend", "_lock",
    )

    def __init__(self, proc, inbox: queue.Queue, backend=None, failure: str | None = None):
        self.proc = proc
        self.objects: dict[str, bytes] = {}
        self.seen: set[str] = set()
        self._failure = failure
        self._buf = b""
        self._result: tuple | None = None
        self._eof = False
        self._inbox = inbox
        self._backend = backend
        self._lock = threading.Lock()  # stdin is written from two threads
        if proc is not None:
            threading.Thread(target=self._pump, daemon=True).start()

    # -- answering the worker ---------------------------------------------

    def _send(self, tag: bytes, body: bytes = b"") -> None:
        from . import wire

        with self._lock:
            try:
                wire.write_frame(self.proc.stdin, tag, body)
            except (OSError, ValueError):
                pass  # the worker is gone; its result will say so

    def _send_value(self, v, tag: bytes, body: bytes = b"") -> None:
        """Send *v*'s objects and then the frame that names it, atomically
        with respect to the other writer."""
        from . import wire

        with self._lock:
            try:
                root = wire.send_value(self.proc.stdin, v, self.seen)
                wire.write_frame(self.proc.stdin, tag, body or bytes.fromhex(root))
            except (OSError, ValueError):
                pass

    def _store(self):
        return None if self._backend is None else self._backend._store

    def _request(self, wire, tag: bytes, body: bytes) -> None:
        from . import runlog
        from .store import CacheMiss

        store = self._store()
        if tag == wire.TRACE:
            fn_key, doc, *units = wire.unstrings(body)
            if store is not None:
                store.put_trace(fn_key, json.loads(doc), units)
        elif tag == wire.GET_TRACES:
            pairs = [] if store is None else store.get_traces(body.decode())
            self._send(wire.TRACES, json.dumps(pairs).encode())
        elif tag == wire.GET_VALUE:
            try:
                if store is None:
                    raise CacheMiss("the driver has no cache directory")
                v = store.get_value(body.hex())
            except CacheMiss as e:
                self._send(wire.VALUE, str(e).encode())
            else:
                self._send_value(v, wire.VALUE, b"")
        elif tag == wire.EVENT:
            record = json.loads(body)
            runlog.record(store, record.pop("ev", "?"), **record)
        elif tag == wire.CALL:
            module, qualname, root = wire.unstrings(body)
            args, kwargs = wire.unpack(root, self.objects, self._fallback())
            self._backend._submit(self._run_call, wire, module, qualname, args, kwargs)
        else:
            raise wire.WireError(f"unexpected frame {tag!r}")

    def _fallback(self):
        store = self._store()
        return None if store is None else store.get_value

    def _run_call(self, wire, module: str, qualname: str, args, kwargs) -> None:
        """On a pool thread: make the @pure_local call here and reply."""
        from .worker import _resolve

        try:
            fn = _resolve(module, qualname)
            value = fn(*args, **kwargs)
            lookup = getattr(fn, "_valuekit_lookup", None)
            found = lookup(*args, **kwargs) if lookup is not None else None
            h = found[0] if found is not None else ""
        except BaseException as e:
            self._send(
                wire.CALLED,
                b"e" + wire.strings(type(e).__name__, str(e), traceback.format_exc()),
            )
            return
        with self._lock:
            try:
                root = wire.send_value(self.proc.stdin, value, self.seen)
                wire.write_frame(self.proc.stdin, wire.CALLED, b"o" + wire.strings(root, h))
            except (OSError, ValueError):
                pass

    # -- reading ---------------------------------------------------------

    def _pump(self) -> None:
        """Reader thread.  An empty chunk means the pipe closed."""
        fd = self.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            self._inbox.put((self, chunk))
            if not chunk:
                return

    def feed(self, chunk: bytes) -> None:
        """Take one chunk from the reader thread; parse any complete frames."""
        from . import wire

        if self._result is not None or self._eof:
            return
        if not chunk:
            self._eof = True
            return
        self._buf += chunk
        try:
            self._parse(wire)
        except wire.WireError as e:
            self._result = ("err_str", "WireError", str(e), "")

    def drain(self) -> None:
        """Apply whatever has arrived, without blocking."""
        _apply_queued(self._inbox)

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
                self.seen.add(body[:20].hex())
                store = self._store()
                if store is not None:
                    # The worker's results live here and nowhere else.
                    wire.store_object(store, body)
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
                    self._result = (
                        "ok",
                        wire.unpack(body[1:].hex(), self.objects, self._fallback()),
                    )
                else:
                    kind, text, tb = wire.unstrings(body[1:])
                    self._result = ("err_str", kind, text, tb)
            else:
                self._request(wire, tag, body)

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

        from . import sync, wire
        from .codehash import function_fingerprint
        from .pure import _salt

        from .pure import _current_store

        self._fn = fn
        self._cache_dir = cache_dir or ""
        self._python = python or sys.executable
        self._store = _current_store()
        # The worker keeps only the project's source tree, and is told where.
        self._source_root = (
            os.path.join(self._cache_dir, "source") if self._cache_dir else ""
        )
        # A worker gets the variables a process needs to start and nothing
        # of the driver's: no PYTHONPATH (imports must resolve through the
        # source tree, or the check that they did proves nothing), and no
        # credentials, which a @pure_local call keeps on the driver.
        self._env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() in _WORKER_ENV or k.upper().startswith("VALUEKIT_")
        }

        self._inbox: queue.Queue = queue.Queue()
        self._pool: ThreadPoolExecutor | None = None

        self._root = sync.sync_root(fn)
        self._entries = sync.manifest(
            self._root, exclude=[self._cache_dir] if self._cache_dir else []
        )
        self._hash = sync.manifest_hash(self._entries)
        self._roots = sync.import_roots(self._root)
        self._ready = False

        self._ids = (
            _salt(),
            getattr(fn, "__module__", "") or "",
            getattr(fn, "__qualname__", "") or "",
            function_fingerprint(fn)[0],
            self._source_root,
            self._hash,
        )
        self._greeting = wire.strings(*self._ids, *self._roots)

    def default_workers(self) -> int:
        return os.cpu_count() or 1

    def _submit(self, fn, *args) -> None:
        """Run a worker's @pure_local call on a thread of this process."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        self._pool.submit(fn, *args)

    def _spawn(self, extra: list[str], quiet: bool = True):
        import subprocess

        return subprocess.Popen(
            [self._python, "-m", "valuekit.worker", *extra],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
            env=self._env,
            cwd=self._cache_dir or None,
        )

    def ensure_ready(self) -> None:
        """Sync and verify this host once, before any input is dispatched.

        A failure here is a fact about the host, so it is raised as one
        rather than recorded against whichever input happened to go first.
        """
        from . import sync, wire

        if self._ready:
            return
        self._preflight()
        proc = self._spawn(["--ready"], quiet=False)
        try:
            wire.write_frame(proc.stdin, wire.SYNC, self._greeting)
            frame = wire.read_frame(proc.stdout)
            if frame is None or frame[0] != wire.WANT:
                raise RuntimeError(self._why(proc, "the worker said nothing"))
            if frame[1]:
                wire.write_frame(
                    proc.stdin, wire.TREE, sync.pack_tree(self._root, self._entries)
                )
            frame = wire.read_frame(proc.stdout)
            if frame is None or frame[0] != wire.READY:
                raise RuntimeError(self._why(proc, "the worker never reported"))
            if frame[1]:
                raise RuntimeError(frame[1].decode("utf-8", "replace"))
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        self._ready = True

    def _preflight(self) -> None:
        """Refuse a dependency the source tree could never contain.

        Only the outside-the-project case is checked here, because it is the
        only one the worker cannot explain for itself: it would report
        "cannot import X" without being able to say that X lives in a sibling
        checkout the driver never offered to send.  A file inside the project
        but excluded from the manifest already fails loudly at import, which
        is diagnosis enough without a second mechanism.
        """
        from . import sync
        from .codehash import function_fingerprint

        try:
            spans = function_fingerprint(self._fn)[1]
        except Exception:
            return
        root = os.path.realpath(self._root)
        outside = [f for f in sync.user_span_files(spans) if not sync._under(f, root)]
        if outside:
            listed = "\n  ".join(sorted(outside))
            raise RuntimeError(
                f"{getattr(self._fn, '__qualname__', self._fn)} depends on user "
                f"code outside its project at {root}, which cannot be sent to a "
                f"worker:\n  {listed}\n"
                "Move it into the project, or install it as a package so both "
                "machines resolve it the same way."
            )

    @staticmethod
    def _why(proc, fallback: str) -> str:
        """Fold the worker's stderr into the reason it gave none."""
        try:
            err = proc.stderr.read().decode("utf-8", "replace").strip()
        except Exception:
            err = ""
        return f"{fallback}:\n{err}" if err else fallback

    def start(self, x: Any) -> _PipeHandle:
        from . import wire

        proc = self._spawn([])
        handle = _PipeHandle(proc, self._inbox, self)
        try:
            wire.write_frame(proc.stdin, wire.HELLO, self._greeting)
            root = wire.send_value(proc.stdin, x, handle.seen)
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
        settled = [h for h in handles if h.settled()]
        if settled:
            return settled  # already answered; do not block on the others
        try:
            handle, chunk = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return []
        handle.feed(chunk)
        _apply_queued(self._inbox)  # whatever else arrived meanwhile
        return [h for h in handles if h.settled()]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


# What a worker process needs from the environment to start and to find
# its interpreter's own files; everything else stays with the driver.
_WORKER_ENV = frozenset(
    {
        "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
        "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "WINDIR",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "USERNAME", "USER",
        "PYTHONHOME", "PYTHONUTF8", "PYTHONIOENCODING", "VIRTUAL_ENV",
    }
)
