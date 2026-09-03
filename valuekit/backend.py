"""Where a batch's work actually runs.

:func:`valuekit.run_all` owns the scheduling -- admission against each
place's capacity, deadlines, input-order reassembly, and attributing every
failure to the input that caused it -- none of which cares whether the work
happens in a process on this machine or somewhere else.  This module owns
the other half: starting a unit of work, telling the scheduler when it has
something to say, and killing it.

A backend hands back a *handle* per unit of work.  Whenever a handle has
something to feed -- a message, a chunk of bytes, or the news that its
worker is gone -- the backend puts ``(handle, payload)`` on the inbox the
scheduler gave it, and the scheduler calls ``handle.feed(payload)``.  One
inbox for every backend is what lets a batch span machines without the
scheduler waiting on two kinds of thing, and a blocking read on a thread is
the one primitive every platform gives a pipe, which is why there is no
``select`` here.

The handle carries a three-variant message back, the same union the local
worker has always sent: ``("ok", value)``, ``("err", exc, tb)``, or
``("err_str", type_name, text, tb)`` when the exception itself could not be
sent.  A handle that yields no message at all died, which the scheduler
already knows how to attribute.

Two backends.  :class:`LocalBackend` spawns a process per input on this
machine.  :class:`HostBackend` holds one connection to a *host process*
(:mod:`valuekit.host`), on this machine or over ssh, which starts a worker
per task and carries each worker's stream as a numbered channel.  A worker
on a host holds no cache: its store is this process's store, so the frames
that arrive on a channel are not only its answer but a trace to keep, a
lookup to answer, a run-log record to write, or a ``@pure_local`` call to
make here.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from .codec import SerializationError

__all__ = ["Backend", "Handle", "LocalBackend", "HostBackend"]


class Handle(Protocol):
    """One unit of work in flight."""

    def feed(self, payload: Any) -> None:
        """Take what the backend put on the inbox for this handle."""

    def settled(self) -> bool:
        """Whether there is an answer, or the worker has gone."""

    def recv(self) -> tuple | None:
        """The worker's message, or None if it died without sending one."""

    def kill(self) -> None:
        """Stop the work now."""

    def reap(self) -> None:
        """Release what it held once it is finished."""

    def death(self) -> str:
        """A phrase describing how it died, for the failure message."""


class Backend(Protocol):
    """Somewhere work can run."""

    name: str

    def start(self, x: Any) -> Handle: ...
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
    """One spawned process, and a thread waiting for its one message."""

    __slots__ = ("proc", "conn", "_msg", "_settled")

    def __init__(self, proc, conn, inbox: queue.Queue):
        self.proc = proc
        self.conn = conn
        self._msg: tuple | None = None
        self._settled = False
        threading.Thread(target=self._wait, args=(inbox,), daemon=True).start()

    def _wait(self, inbox: queue.Queue) -> None:
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            msg = None  # died without sending; Windows says BrokenPipeError
        inbox.put((self, msg))

    def feed(self, payload: Any) -> None:
        self._msg = payload
        self._settled = True

    def settled(self) -> bool:
        return self._settled

    def recv(self) -> tuple | None:
        return self._msg

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

    name = "local"

    def __init__(self, fn, cache_dir: str | None, inbox: queue.Queue):
        self._fn = fn
        self._cache_dir = cache_dir
        self._inbox = inbox
        self._ctx = multiprocessing.get_context("spawn")

    def start(self, x: Any) -> _LocalHandle:
        recv_end, send_end = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_child_main,
            args=(send_end, self._cache_dir, self._fn, x),
            daemon=True,
        )
        proc.start()
        send_end.close()  # keep only the child's handle: EOF then means death
        return _LocalHandle(proc, recv_end, self._inbox)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# a worker on a channel of a host connection
# ---------------------------------------------------------------------------


class _ChannelWriter:
    """A file-like object whose writes become DATA frames on one channel,
    so :mod:`valuekit.wire` can write to a worker as it would to a pipe."""

    __slots__ = ("_send", "_ch", "_buf")

    def __init__(self, send: Callable[[bytes, bytes], None], ch: int):
        self._send = send
        self._ch = ch
        self._buf = bytearray()

    def write(self, data: bytes) -> int:
        self._buf += data
        return len(data)

    def flush(self) -> None:
        from . import wire

        if self._buf:
            data, self._buf = bytes(self._buf), bytearray()
            self._send(wire.DATA, wire.channelled(self._ch, data))


class _Handle:
    """One worker on a channel, and the framed conversation with it.

    Frames are parsed out of a buffer as bytes arrive.  A worker speaks
    several times before it finishes -- a greeting, store traffic, then the
    result's objects -- so "bytes arrived" does not mean "the answer is
    here", and reading until it is would sit inside a task that has already
    blown its deadline.

    Lookups are answered on the scheduler's thread; a ``@pure_local`` call
    runs on a pool thread, since it may be a download, and replies when it
    is done.  ``seen`` names every object either side has sent, so nothing
    crosses twice.
    """

    __slots__ = (
        "ch", "objects", "seen", "inbox", "out", "_backend", "_failure",
        "_buf", "_result", "_eof", "_exit", "_stderr", "_lock", "_frames", "_raw",
    )

    def __init__(self, backend: HostBackend, ch: int, inbox: queue.Queue, raw: bool = False):
        self.ch = ch
        self.objects: dict[str, bytes] = {}
        self.seen: set[str] = set()
        self.inbox = inbox
        self.out = _ChannelWriter(backend._send, ch)
        self._backend = backend
        self._failure: str | None = None
        self._buf = b""
        self._result: tuple | None = None
        self._eof = False
        self._exit: int | None = None
        self._stderr = b""
        self._lock = threading.Lock()  # the channel is written from two threads
        self._frames: list[tuple[bytes, bytes]] = []  # raw frames, readiness only
        self._raw = raw  # readiness: keep frames as they are, interpret nothing

    # -- what arrives ---------------------------------------------------------

    def feed(self, payload: Any) -> None:
        """A chunk of the worker's stdout, an exit report, or None at EOF."""
        if payload is None:
            self._eof = True
            return
        if isinstance(payload, tuple):
            self._exit, self._stderr = payload
            return
        if self._result is not None:
            return
        self._buf += payload
        from . import wire

        try:
            self._parse(wire)
        except wire.WireError as e:
            self._result = ("err_str", "WireError", str(e), "")

    def _next_frame(self, wire) -> tuple[bytes, bytes] | None:
        if len(self._buf) < 9:
            return None
        n = int.from_bytes(self._buf[1:9], "little")
        if n > wire.MAX_FRAME:
            raise wire.WireError(f"frame claims {n} bytes; refusing")
        if len(self._buf) < 9 + n:
            return None
        tag, body = self._buf[:1], self._buf[9 : 9 + n]
        self._buf = self._buf[9 + n :]
        return tag, body

    def _parse(self, wire) -> None:
        while self._result is None:
            frame = self._next_frame(wire)
            if frame is None:
                return
            tag, body = frame
            if self._raw:
                self._frames.append(frame)
                continue
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

    def wait_frame(self, timeout: float | None) -> tuple[bytes, bytes] | None:
        """Block for the next frame on a raw channel; None if the worker
        went away or *timeout* passed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._frames:
                return self._frames.pop(0)
            if self._eof:
                return None
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                handle, payload = self.inbox.get(timeout=remaining)
            except queue.Empty:
                return None
            handle.feed(payload)

    # -- answering the worker -------------------------------------------------

    def _write_frame(self, tag: bytes, body: bytes = b"") -> None:
        from . import wire

        with self._lock:
            wire.write_frame(self.out, tag, body)

    def _store(self):
        return self._backend._store

    def _fallback(self):
        store = self._store()
        return None if store is None else store.get_value

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
            self._write_frame(wire.TRACES, json.dumps(pairs).encode())
        elif tag == wire.GET_VALUE:
            try:
                if store is None:
                    raise CacheMiss("the driver has no cache directory")
                v = store.get_value(body.hex())
            except CacheMiss as e:
                self._write_frame(wire.VALUE, str(e).encode())
            else:
                with self._lock:
                    wire.send_value(self.out, v, self.seen)
                    wire.write_frame(self.out, wire.VALUE, b"")
        elif tag == wire.EVENT:
            record = json.loads(body)
            record.setdefault("host", self._backend.name)
            runlog.record(store, record.pop("ev", "?"), **record)
        elif tag == wire.CALL:
            module, qualname, root = wire.unstrings(body)
            args, kwargs = wire.unpack(root, self.objects, self._fallback())
            self._backend._submit(self._run_call, wire, module, qualname, args, kwargs)
        else:
            raise wire.WireError(f"unexpected frame {tag!r}")

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
            self._write_frame(
                wire.CALLED,
                b"e" + wire.strings(type(e).__name__, str(e), traceback.format_exc()),
            )
            return
        with self._lock:
            root = wire.send_value(self.out, value, self.seen)
            wire.write_frame(self.out, wire.CALLED, b"o" + wire.strings(root, h))

    # -- the scheduler's view --------------------------------------------------

    def settled(self) -> bool:
        return self._failure is not None or self._result is not None or self._eof

    def recv(self) -> tuple | None:
        if self._failure is not None:
            return ("err_str", "RuntimeError", self._failure, "")
        return self._result  # None means it went away without answering

    def kill(self) -> None:
        from . import wire

        self._backend._send(wire.KILL, wire.channelled(self.ch))

    def reap(self) -> None:
        from . import wire

        self._backend._send(wire.CLOSE, wire.channelled(self.ch))
        self._backend._forget(self.ch)

    def death(self) -> str:
        tail = self._stderr.decode("utf-8", "replace").strip()
        if self._exit is None:
            why = f"the connection to host {self._backend.name!r} closed"
        else:
            why = f"exit code {self._exit} on host {self._backend.name!r}"
        return f"{why}:\n{tail}" if tail else f"{why}; a segfault or a broken pipe?"


# ---------------------------------------------------------------------------
# a host: one connection, many channels
# ---------------------------------------------------------------------------


class HostBackend:
    """Workers on one host, over one connection to its host process.

    *command* launches the host process: for a remote machine an ssh
    invocation, for this machine the interpreter itself.  Either way what
    is at the other end is ``python -m valuekit.host``, and everything after
    the launch is the same.
    """

    def __init__(
        self,
        fn,
        cache_dir: str | None,
        inbox: queue.Queue,
        name: str,
        command: list[str],
        source_root: str,
        workers: int | None = None,
    ):
        from . import sync, wire
        from .codehash import fingerprint_details
        from .pure import _current_store, _salt
        from .worker import EXT_MARK

        self.name = name
        self.capacity = workers
        self.dead = False
        self.failure: str | None = None
        self._fn = fn
        self._cache_dir = cache_dir or ""
        self._inbox = inbox
        self._command = command
        self._store = _current_store()
        self._proc: subprocess.Popen | None = None
        self._out_lock = threading.Lock()
        self._channels: dict[int, _Handle] = {}
        self._next = 1
        self._stderr = bytearray()
        self._pool: ThreadPoolExecutor | None = None
        self._ready = False

        self._root = sync.sync_root(fn)
        self._entries = sync.manifest(
            self._root, exclude=[self._cache_dir] if self._cache_dir else []
        )
        self._hash = sync.manifest_hash(self._entries)
        self._roots = sync.import_roots(self._root)
        fingerprint, _, _, extensions = fingerprint_details(fn)
        self._greeting = wire.strings(
            _salt(),
            getattr(fn, "__module__", "") or "",
            getattr(fn, "__qualname__", "") or "",
            fingerprint,
            source_root,
            self._hash,
            *self._roots,
            EXT_MARK,
            *(part for pair in extensions.items() for part in pair),
        )

    # -- the connection ---------------------------------------------------------

    def _launch(self) -> str:
        """Start the host process and read its greeting; "" or why not."""
        from . import wire
        from .pure import _salt

        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            return f"cannot start {' '.join(self._command)}: {e}"
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        frame = wire.read_frame(self._proc.stdout)
        if frame is None or frame[0] != wire.HOST:
            self._proc.wait()
            return self._why("the host process said nothing")
        salt, cpus = wire.unstrings(frame[1])[:2]
        if salt != _salt():
            self._kill_proc()
            return f"driver is {_salt()}, host {self.name!r} is {salt}"
        if self.capacity is None:
            self.capacity = max(1, int(cpus))
        threading.Thread(target=self._read, daemon=True).start()
        return ""

    def _why(self, fallback: str) -> str:
        tail = bytes(self._stderr).decode("utf-8", "replace").strip()
        code = self._proc.returncode if self._proc is not None else None
        if code == 255:
            fallback = f"ssh to host {self.name!r} failed"
        return f"{fallback}:\n{tail}" if tail else fallback

    def _drain_stderr(self) -> None:
        fd = self._proc.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                return
            self._stderr += chunk
            del self._stderr[: -(64 << 10)]

    def _read(self) -> None:
        """Reader thread: demultiplex the host's frames onto handles."""
        from . import wire

        while True:
            try:
                frame = wire.read_frame(self._proc.stdout)
            except (wire.WireError, OSError, ValueError):
                frame = None
            if frame is None:
                break
            tag, body = frame
            try:
                ch, rest = wire.channel(body)
            except wire.WireError:
                break
            handle = self._channels.get(ch)
            if handle is None:
                continue
            if tag == wire.DATA:
                handle.inbox.put((handle, rest))
            elif tag == wire.EXIT:
                code = int.from_bytes(rest[:4], "little", signed=True)
                handle.inbox.put((handle, (code, rest[4:])))
                handle.inbox.put((handle, None))
        self.dead = True
        for handle in list(self._channels.values()):
            handle.inbox.put((handle, None))

    def _send(self, tag: bytes, body: bytes) -> None:
        from . import wire

        if self._proc is None:
            return
        with self._out_lock:
            try:
                wire.write_frame(self._proc.stdin, tag, body)
            except (OSError, ValueError):
                pass  # the host is gone; the reader thread reports it

    def _forget(self, ch: int) -> None:
        self._channels.pop(ch, None)

    def _open(self, kind: bytes, inbox: queue.Queue) -> _Handle:
        from . import wire

        ch, self._next = self._next, self._next + 1
        handle = _Handle(self, ch, inbox, raw=kind == b"ready")
        self._channels[ch] = handle
        self._send(wire.OPEN, wire.channelled(ch, kind))
        return handle

    # -- readiness ---------------------------------------------------------------

    def ensure_ready(self) -> str:
        """Sync and verify this host once, before any input is dispatched.

        Returns "" when the host can take work, else why not.  A failure
        is a fact about the host, recorded once rather than against every
        input that would have gone there.
        """
        from . import sync, wire

        if self._ready:
            return ""
        if self.failure is not None:
            return self.failure
        reason = self._preflight() or self._launch()
        if reason:
            self.failure = reason
            return reason
        handle = self._open(b"ready", queue.Queue())
        try:
            handle._write_frame(wire.SYNC, self._greeting)
            frame = handle.wait_frame(timeout=120)
            if frame is None or frame[0] != wire.WANT:
                return self._fail(handle, "the worker said nothing")
            if frame[1]:
                handle._write_frame(wire.TREE, sync.pack_tree(self._root, self._entries))
            frame = handle.wait_frame(timeout=600)
            if frame is None or frame[0] != wire.READY:
                return self._fail(handle, "the worker never reported")
            if frame[1]:
                return self._fail(handle, frame[1].decode("utf-8", "replace"))
        finally:
            handle.reap()
        self._ready = True
        return ""

    def _fail(self, handle: _Handle, reason: str) -> str:
        tail = handle._stderr.decode("utf-8", "replace").strip()
        self.failure = f"{reason}:\n{tail}" if tail else reason
        return self.failure

    def _preflight(self) -> str:
        """Refuse a dependency the source tree could never contain.

        Only the outside-the-project case is checked here, because it is the
        only one the worker cannot explain for itself: it would report
        "cannot import X" without being able to say that X lives in a sibling
        checkout the driver never offered to send.
        """
        from . import sync
        from .codehash import function_fingerprint

        try:
            spans = function_fingerprint(self._fn)[1]
        except Exception:
            return ""
        root = os.path.realpath(self._root)
        outside = [f for f in sync.user_span_files(spans) if not sync._under(f, root)]
        if outside:
            listed = "\n  ".join(sorted(outside))
            return (
                f"{getattr(self._fn, '__qualname__', self._fn)} depends on user "
                f"code outside its project at {root}, which cannot be sent to a "
                f"worker:\n  {listed}\n"
                "Move it into the project, or install it as a package so both "
                "machines resolve it the same way."
            )
        return ""

    # -- tasks ---------------------------------------------------------------------

    def _submit(self, fn, *args) -> None:
        """Run a worker's @pure_local call on a thread of this process."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        self._pool.submit(fn, *args)

    def start(self, x: Any) -> _Handle:
        from . import wire

        handle = self._open(b"task", self._inbox)
        try:
            with handle._lock:
                wire.write_frame(handle.out, wire.HELLO, self._greeting)
                root = wire.send_value(handle.out, x, handle.seen)
                wire.write_frame(handle.out, wire.TASK, bytes.fromhex(root))
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

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        self._kill_proc()  # closing stdin is what tells the host to exit
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
