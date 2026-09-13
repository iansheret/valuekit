"""Where a batch's work actually runs.

:func:`valuekit.run_all` owns the scheduling -- admission against each
host's capacity, deadlines, input-order reassembly, and attributing every
failure to the input that caused it -- none of which cares whether the work
happens in a process on this machine or somewhere else.  This module owns
the other half: starting a task, telling the scheduler when it has
something to say, and killing it.

A host returns a *handle* per task.  Whenever a handle has something to
feed -- a message, a chunk of bytes, or the fact that its worker has
exited -- the host puts ``(handle, payload)`` on the completions queue the
scheduler gave it, and the scheduler calls ``handle.feed(payload)``.  One
queue for every host is what lets a batch span hosts without the scheduler
waiting on two kinds of thing, and a blocking read on a thread is the one
primitive every platform gives a pipe, which is why there is no ``select``
here.

A handle answers one question, ``finished()``: nothing yet, or a
:class:`Finished` saying what happened -- a result, an exception, a worker
that exited without answering, or a connection that closed with the input
neither done nor failed.  A worker sends its answer as one of ``("ok",
value)``, ``("err", exc, tb)`` or ``("err_str", type_name, text, tb)`` when
the exception itself could not be sent; the handle turns that into the
``Finished``.

Two hosts.  :class:`LocalHost` spawns a process per input on this
host.  :class:`RemoteHost` holds one connection to a *host process*
(:mod:`valuekit.hostprocess`), on this machine or over ssh, which starts a worker
per task and carries each worker's stream as a numbered channel.  A worker
on a host holds no cache: its store is this process's store, so the messages
that arrive on a channel are not only its answer but a call record to keep, a
lookup to answer, a event to write, or a ``@pure_local`` call to
make here.

The connection is a :class:`Connection`: two byte streams and how they ended,
nothing more.  :class:`ProcessConnection` is a child process's pipes, whether the
child is a Python here or ``ssh`` to one elsewhere; what runs at the far end
is the bootstrap (:mod:`valuekit.bootstrap`), which turns the project into
an environment there and starts the host process in it.  A different
transport later is another ``Connection`` and touches nothing above it.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO, Callable, Protocol

from .codec import SerializationError

__all__ = ["Finished", "Handle", "Connection", "ProcessConnection", "LocalHost", "RemoteHost"]


class Finished:
    """What happened to a task.  ``kind`` is one of:

    * ``"ok"``: ``value`` is the result;
    * ``"error"``: ``exc`` is the exception the worker raised, ``tb`` its
      traceback text;
    * ``"error_text"``: the exception could not be sent; ``text`` names it,
      ``tb`` is its traceback text;
    * ``"exited"``: the worker exited without answering; ``text`` says how;
    * ``"connection_closed"``: the connection to the host closed with the
      input neither done nor failed, so the scheduler may run it elsewhere.
    """

    __slots__ = ("kind", "value", "exc", "text", "tb")

    def __init__(self, kind: str, value: Any = None, exc: BaseException | None = None,
                 text: str = "", tb: str = ""):
        self.kind = kind
        self.value = value
        self.exc = exc
        self.text = text
        self.tb = tb

    @classmethod
    def from_message(cls, msg: tuple) -> "Finished":
        """A worker's answer tuple as a Finished."""
        if msg[0] == "ok":
            return cls("ok", value=msg[1])
        if msg[0] == "err":
            return cls("error", exc=msg[1], tb=msg[2])
        return cls("error_text", text=f"{msg[1]}: {msg[2]}", tb=msg[3])


class Handle(Protocol):
    """One task in flight."""

    def feed(self, payload: Any) -> None:
        """Take what the host put on the completions queue for this handle."""

    def finished(self) -> Finished | None:
        """What happened, or None while the task is still running."""

    def kill(self) -> None:
        """Stop the work now."""

    def release(self) -> None:
        """Let go of what the task held, once it is finished."""


# ---------------------------------------------------------------------------
# local processes
# ---------------------------------------------------------------------------


def _local_worker_main(conn, store_dir: str | None, fn, x) -> None:
    """Runs in the worker process: configure the cache, run one input, send
    one message back: ("ok", value) or ("err", exc, tb) or, when the
    exception or value cannot be pickled, ("err_str", type_name, text, tb).
    What this worker logs goes into the main process's run, named in the
    environment it inherited (see :mod:`valuekit.runlog`).
    """
    try:
        if store_dir is not None:
            from .pure import set_store_dir

            set_store_dir(store_dir)
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

    def __init__(self, proc, conn, completions: queue.Queue):
        self.proc = proc
        self.conn = conn
        self._msg: tuple | None = None
        self._settled = False
        threading.Thread(target=self._wait, args=(completions,), daemon=True).start()

    def _wait(self, completions: queue.Queue) -> None:
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            msg = None  # died without sending; Windows says BrokenPipeError
        completions.put((self, msg))

    def feed(self, payload: Any) -> None:
        self._msg = payload
        self._settled = True

    def finished(self) -> Finished | None:
        if not self._settled:
            return None
        if self._msg is None:
            return Finished(
                "exited",
                text=f"exit code {self.proc.exitcode}; a segfault or an out-of-memory kill?",
            )
        return Finished.from_message(self._msg)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.join()

    def release(self) -> None:
        self.proc.join()
        self.conn.close()


class LocalHost:
    """One spawned process per input, on this machine.

    Isolation is the point: a timeout kills exactly one process and a
    segfault loses exactly one input.  Processes are daemonic, so they are
    cleaned up if the main process exits.
    """

    name = "local"

    def __init__(self, fn, store_dir: str | None, completions: queue.Queue):
        self._fn = fn
        self._store_dir = store_dir
        self._completions = completions
        self._ctx = multiprocessing.get_context("spawn")

    def start(self, x: Any) -> _LocalHandle:
        recv_end, send_end = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_local_worker_main,
            args=(send_end, self._store_dir, self._fn, x),
            daemon=True,
        )
        proc.start()
        send_end.close()  # keep only the child's handle: EOF then means death
        return _LocalHandle(proc, recv_end, self._completions)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# a worker on a channel of a host connection
# ---------------------------------------------------------------------------


class _ChannelWriter:
    """A file-like object whose writes become DATA messages on one channel,
    so :mod:`valuekit.protocol` can write to a worker as it would to a pipe."""

    __slots__ = ("_send", "_ch", "_buf")

    def __init__(self, send: Callable[[bytes, bytes], None], ch: int):
        self._send = send
        self._ch = ch
        self._buf = bytearray()

    def write(self, data: bytes) -> int:
        self._buf += data
        return len(data)

    def flush(self) -> None:
        from . import protocol

        if self._buf:
            data, self._buf = bytes(self._buf), bytearray()
            self._send(protocol.DATA, protocol.channelled(self._ch, data))


class _Handle:
    """One worker on a channel, and the framed conversation with it.

    Frames are parsed out of a buffer as bytes arrive.  A worker speaks
    several times before it finishes -- a HELLO message, store traffic, then the
    result's objects -- so "bytes arrived" does not mean "the answer is
    here", and reading until it is would sit inside a task that has already
    blown its deadline.

    Lookups are answered on the scheduler's thread; a ``@pure_local`` call
    runs on a pool thread, since it may be a download, and replies when it
    is done.  ``seen`` names every object either side has sent, so nothing
    crosses twice.
    """

    __slots__ = (
        "ch", "objects", "seen", "completions", "out", "_host", "_failure",
        "_buf", "_result", "_eof", "_exit", "_stderr", "_lock", "_messages", "_raw",
    )

    def __init__(self, host: RemoteHost, ch: int, completions: queue.Queue, raw: bool = False):
        self.ch = ch
        self.objects: dict[str, bytes] = {}
        self.seen: set[str] = set()
        self.completions = completions
        self.out = _ChannelWriter(host._send, ch)
        self._host = host
        self._failure: str | None = None
        self._buf = b""
        self._result: tuple | None = None
        self._eof = False
        self._exit: int | None = None
        self._stderr = b""
        self._lock = threading.Lock()  # the channel is written from two threads
        self._messages: list[tuple[bytes, bytes]] = []  # raw messages, the check only
        self._raw = raw  # the check: keep messages as they are, interpret nothing

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
        from . import protocol

        try:
            self._parse(protocol)
        except protocol.ProtocolError as e:
            self._result = ("err_str", "ProtocolError", str(e), "")

    def _next_message(self, protocol) -> tuple[bytes, bytes] | None:
        if len(self._buf) < 9:
            return None
        n = int.from_bytes(self._buf[1:9], "little")
        if n > protocol.MAX_MESSAGE:
            raise protocol.ProtocolError(f"message claims {n} bytes; refusing")
        if len(self._buf) < 9 + n:
            return None
        tag, body = self._buf[:1], self._buf[9 : 9 + n]
        self._buf = self._buf[9 + n :]
        return tag, body

    def _parse(self, protocol) -> None:
        while self._result is None:
            message = self._next_message(protocol)
            if message is None:
                return
            tag, body = message
            if self._raw:
                self._messages.append(message)
                continue
            if tag == protocol.OBJECT:
                protocol.recv_object(body, self.objects)
                self.seen.add(body[:20].hex())
                store = self._store()
                if store is not None:
                    # The worker's results live here and nowhere else.
                    protocol.store_object(store, body)
            elif tag == protocol.ACCEPTED:
                if body:
                    self._result = (
                        "err_str",
                        "RuntimeError",
                        body.decode("utf-8", "replace"),
                        "",
                    )
            elif tag == protocol.RESULT:
                if body[:1] == b"o":
                    self._result = (
                        "ok",
                        protocol.unpack(body[1:].hex(), self.objects, self._fallback()),
                    )
                else:
                    kind, text, tb = protocol.unstrings(body[1:])
                    self._result = ("err_str", kind, text, tb)
            else:
                self._request(protocol, tag, body)

    def wait_message(self, timeout: float | None) -> tuple[bytes, bytes] | None:
        """Block for the next message on a raw channel; None if the worker
        went away or *timeout* passed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._messages:
                return self._messages.pop(0)
            if self._eof:
                return None
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                handle, payload = self.completions.get(timeout=remaining)
            except queue.Empty:
                return None
            handle.feed(payload)

    # -- answering the worker -------------------------------------------------

    def _write_message(self, tag: bytes, body: bytes = b"") -> None:
        from . import protocol

        with self._lock:
            protocol.write_message(self.out, tag, body)

    def _store(self):
        return self._host._store

    def _fallback(self):
        store = self._store()
        return None if store is None else store.get_value

    def _request(self, protocol, tag: bytes, body: bytes) -> None:
        from . import events
        from .store import CacheMiss

        store = self._store()
        if tag == protocol.RECORD:
            function_hash, doc = protocol.unstrings(body)
            if store is not None:
                store.put_record(function_hash, json.loads(doc))
        elif tag == protocol.GET_RECORDS:
            pairs = [] if store is None else store.get_records(body.decode())
            self._write_message(protocol.RECORDS, json.dumps(pairs).encode())
        elif tag == protocol.GET_VALUE:
            try:
                if store is None:
                    raise CacheMiss("the main process has no store directory")
                v = store.get_value(body.hex())
            except CacheMiss as e:
                self._write_message(protocol.VALUE, str(e).encode())
            else:
                with self._lock:
                    protocol.send_value(self.out, v, self.seen)
                    protocol.write_message(self.out, protocol.VALUE, b"")
        elif tag == protocol.EVENT:
            record = json.loads(body)
            record.setdefault("host", self._host.name)
            events.record(store, record.pop("ev", "?"), **record)
        elif tag == protocol.LOGGED:
            from . import runlog

            if store is not None:
                runlog.write_line(store, body.decode())
        elif tag == protocol.CALL:
            module, qualname, root = protocol.unstrings(body)
            args, kwargs = protocol.unpack(root, self.objects, self._fallback())
            self._host._submit(self._run_call, protocol, module, qualname, args, kwargs)
        else:
            raise protocol.ProtocolError(f"unexpected message {tag!r}")

    def _run_call(self, protocol, module: str, qualname: str, args, kwargs) -> None:
        """On a pool thread: make the @pure_local call here and reply."""
        from .worker import _resolve

        try:
            fn = _resolve(module, qualname)
            value = fn(*args, **kwargs)
            lookup = getattr(fn, "_valuekit_lookup", None)
            found = lookup(*args, **kwargs) if lookup is not None else None
            h = found[0] if found is not None else ""
        except BaseException as e:
            self._write_message(
                protocol.CALLED,
                b"e" + protocol.strings(type(e).__name__, str(e), traceback.format_exc()),
            )
            return
        with self._lock:
            root = protocol.send_value(self.out, value, self.seen)
            protocol.write_message(self.out, protocol.CALLED, b"o" + protocol.strings(root, h))

    # -- the scheduler's view --------------------------------------------------

    def finished(self) -> Finished | None:
        if self._failure is not None:
            return Finished("error_text", text=f"RuntimeError: {self._failure}")
        if self._result is not None:
            return Finished.from_message(self._result)
        if not self._eof:
            return None
        tail = self._stderr.decode("utf-8", "replace").strip()
        if self._exit is None:
            return Finished("connection_closed", text=f"the connection to host {self._host.name!r} closed")
        why = f"exit code {self._exit} on host {self._host.name!r}"
        return Finished("exited", text=f"{why}:\n{tail}" if tail else f"{why}; a segfault or a broken pipe?")

    def kill(self) -> None:
        from . import protocol

        self._host._send(protocol.KILL, protocol.channelled(self.ch))

    def release(self) -> None:
        from . import protocol

        self._host._send(protocol.CLOSE, protocol.channelled(self.ch))
        self._host._drop_channel(self.ch)


# ---------------------------------------------------------------------------
# a connection: bytes to and from a process somewhere
# ---------------------------------------------------------------------------


class Connection(Protocol):
    """A byte stream each way to a process on a host, and how it ended.

    Everything above this -- channels, the host's first message, tasks -- is
    written against these two streams and nothing else, so what carries
    them (a pipe to a child, an ssh session, later a socket) is the one
    thing a new kind of host has to provide.
    """

    rx: BinaryIO
    tx: BinaryIO

    def close(self) -> None:
        """End the conversation and let the far end go."""

    def failure(self) -> str:
        """What the far end said on stderr and how it exited, or "" if it
        has not ended.  For a failure message; never a reason on its own."""


class ProcessConnection:
    """A process on this machine, or on another through ssh: its stdin and
    stdout are the connection, its stderr is kept for the failure message."""

    def __init__(self, command: list[str]):
        self.command = command
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.rx: BinaryIO = self.proc.stdout  # type: ignore[assignment]
        self.tx: BinaryIO = self.proc.stdin  # type: ignore[assignment]
        self._stderr = bytearray()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stderr(self) -> None:
        fd = self.proc.stderr.fileno()  # type: ignore[union-attr]
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                return
            self._stderr += chunk
            del self._stderr[: -(64 << 10)]

    def failure(self) -> str:
        code = self.proc.poll()
        tail = bytes(self._stderr).decode("utf-8", "replace").strip()
        if code is None:
            return tail
        what = f"exit code {code}"
        if code == 255 and self.command and self.command[0] == "ssh":
            what = "ssh could not connect"
        return f"{what}:\n{tail}" if tail else what

    def close(self) -> None:
        # Closing stdin is what tells a host process to exit; killing is
        # for one that does not.
        for stream in (self.tx, self.rx):
            try:
                stream.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# a host: one connection, many channels
# ---------------------------------------------------------------------------


class RemoteHost:
    """Workers on one host, over one connection to its host process.

    *connect* opens a :class:`Connection` to a Python on the host: for a remote
    host an ssh invocation, for this machine an interpreter here.  The
    bootstrap (:mod:`valuekit.bootstrap`) then turns that into a host
    process running in the project's own environment, and everything after
    that is the same wherever the host is.
    """

    def __init__(
        self,
        project,
        store_dir: str | None,
        completions: queue.Queue,
        name: str,
        connect: Callable[[], Connection],
        source_root: str,
        workers: int | None = None,
    ):
        from . import protocol
        from .functionhash import PYTHON, reachable_set
        from .pure import _current_store

        fn = project.fn
        self.name = name
        self.capacity = workers
        self.dead = False
        self.failure: str | None = None
        self._project = project
        self._store_dir = store_dir or ""
        self._completions = completions
        self._connect = connect
        self._store = _current_store()
        self._connection: Connection | None = None
        self._source_root = source_root
        self.pid: int | None = None  # the host process, once it has sent its first message
        self._out_lock = threading.Lock()
        self._channels: dict[int, _Handle] = {}
        self._next = 1
        self._pool: ThreadPoolExecutor | None = None
        self._ready = False
        self._started = time.strftime("%H:%M:%S")

        reach = reachable_set(fn)
        self._hello = protocol.strings(
            PYTHON,
            getattr(fn, "__module__", "") or "",
            getattr(fn, "__qualname__", "") or "",
            reach.hash,
            project.project_hash,
            json.dumps(reach.extensions, sort_keys=True),
            *project.roots,
        )

    # -- the connection ---------------------------------------------------------

    def _launch(self) -> str:
        """Connect, bring the host up, and read its first message; "" or why not."""
        from . import bootstrap, protocol
        from .functionhash import PYTHON

        try:
            self._connection = self._connect()
        except OSError as e:
            return f"cannot connect to host {self.name!r}: {e}"
        try:
            reason = bootstrap.offer(
                self._connection.rx,
                self._connection.tx,
                self._source_root,
                self._project.name,
                self._project.project_hash,
                PYTHON,
                self._project.entries,
                self._project.pack,
                f"a run of {os.path.basename(sys.argv[0]) or 'python'} on "
                f"{socket.gethostname()} (pid {os.getpid()}, started {self._started})",
            )
        except (OSError, ValueError) as e:
            reason = f"the connection to host {self.name!r} broke: {e}"
        if reason:
            reason = self._why(reason)
            self._connection.close()
            return reason
        try:
            message = protocol.read_message(self._connection.rx)
        except (protocol.ProtocolError, OSError, ValueError):
            message = None
        if message is None or message[0] != protocol.HOST:
            reason = self._why("the host process said nothing")
            self._connection.close()
            return reason
        python, cpus, pid = (protocol.unstrings(message[1]) + ["", "", ""])[:3]
        if python != PYTHON:
            self._connection.close()
            return f"main process runs Python {PYTHON}, host {self.name!r} runs {python}"
        if self.capacity is None:
            self.capacity = max(1, int(cpus))
        self.pid = int(pid) if pid.isdigit() else None
        threading.Thread(target=self._read, daemon=True).start()
        return ""

    def _why(self, fallback: str) -> str:
        detail = self._connection.failure() if self._connection is not None else ""
        return f"{fallback}:\n{detail}" if detail else fallback

    def _read(self) -> None:
        """Reader thread: demultiplex the host's messages onto handles."""
        from . import protocol

        while True:
            try:
                message = protocol.read_message(self._connection.rx)
            except (protocol.ProtocolError, OSError, ValueError):
                message = None
            if message is None:
                break
            tag, body = message
            try:
                ch, rest = protocol.channel(body)
            except protocol.ProtocolError:
                break
            handle = self._channels.get(ch)
            if handle is None:
                continue
            if tag == protocol.DATA:
                handle.completions.put((handle, rest))
            elif tag == protocol.EXIT:
                code = int.from_bytes(rest[:4], "little", signed=True)
                handle.completions.put((handle, (code, rest[4:])))
                handle.completions.put((handle, None))
        self.dead = True
        for handle in list(self._channels.values()):
            handle.completions.put((handle, None))

    def _send(self, tag: bytes, body: bytes) -> None:
        from . import protocol

        if self._connection is None:
            return
        with self._out_lock:
            try:
                protocol.write_message(self._connection.tx, tag, body)
            except (OSError, ValueError):
                pass  # the host is gone; the reader thread reports it

    def _drop_channel(self, ch: int) -> None:
        self._channels.pop(ch, None)

    def _open(self, kind: bytes, completions: queue.Queue) -> _Handle:
        from . import protocol

        ch, self._next = self._next, self._next + 1
        handle = _Handle(self, ch, completions, raw=kind == b"check")
        self._channels[ch] = handle
        self._send(protocol.OPEN, protocol.channelled(ch, kind))
        return handle

    # -- the sync -------------------------------------------------------------------

    def sync(self) -> str:
        """Make this host match the main process, once, before it takes any
        input: update its copy of the project, build the environment, import
        the function (which builds an extension that rebuilds on import) and
        check that what it imported is the main process's function.

        Returns "" when the host can take work, else why not.  A failure
        is a fact about the host, recorded once rather than against every
        input that would have gone there.
        """
        from . import protocol

        if self._ready:
            return ""
        if self.failure is not None:
            return self.failure
        reason = self._project.refusal() or self._launch()
        if reason:
            self.failure = reason
            return reason
        handle = self._open(b"check", queue.Queue())
        try:
            handle._write_message(protocol.HELLO, self._hello)
            message = handle.wait_message(timeout=600)
            if message is None or message[0] != protocol.ACCEPTED:
                return self._fail(handle, "the worker never reported")
            if message[1]:
                return self._fail(handle, message[1].decode("utf-8", "replace"))
        finally:
            handle.release()
        self._ready = True
        return ""

    def _fail(self, handle: _Handle, reason: str) -> str:
        tail = handle._stderr.decode("utf-8", "replace").strip()
        self.failure = f"{reason}:\n{tail}" if tail else reason
        return self.failure

    # -- tasks ---------------------------------------------------------------------

    def _submit(self, fn, *args) -> None:
        """Run a worker's @pure_local call on a thread of this process."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        self._pool.submit(fn, *args)

    def start(self, x: Any) -> _Handle:
        from . import protocol

        handle = self._open(b"task", self._completions)
        try:
            with handle._lock:
                protocol.write_message(handle.out, protocol.HELLO, self._hello)
                root = protocol.send_value(handle.out, x, handle.seen)
                protocol.write_message(handle.out, protocol.TASK, bytes.fromhex(root))
        except SerializationError as e:
            # A value the protocol cannot carry is the caller's problem, not a
            # worker failure: say so against this input rather than letting
            # the worker die of a truncated stream.
            handle.kill()
            handle._failure = str(e)
        except Exception as e:
            handle.kill()
            handle._failure = f"could not send the task: {e}"
        return handle

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
