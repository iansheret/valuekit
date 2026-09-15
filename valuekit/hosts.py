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

A handle has one method, ``finished()``: nothing yet, or a
:class:`Finished` saying what happened -- a result, an exception, a worker
that exited without a result, or a connection that closed with the input
neither done nor failed.  A worker reports a result as the root hash of
its value, or a failure as the exception's type name, message and
traceback text; the handle turns that into the ``Finished``.

A :class:`Host` holds one connection to a *host process*
(:mod:`valuekit.hostprocess`), on this machine or over ssh, which starts a
worker per task and carries each worker's stream as a numbered channel.
A worker's store is this process's store: the messages that arrive on a
channel are not only its result but a call record to store, a lookup to
serve, an event to write, or a ``@pure_local`` call to make here.  A
worker on this machine reads and writes values and call records in the
store directory itself and sends the rest.

The connection is a :class:`Connection`: two byte streams and how they ended,
nothing more.  :class:`ProcessConnection` is a child process's pipes, whether the
child is a Python here or ``ssh`` to one elsewhere; what runs at the far end
is the bootstrap (:mod:`valuekit.bootstrap`), which turns the project into
an environment there and starts the host process in it.  A different
transport later is another ``Connection`` and touches nothing above it.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Protocol

from .codec import SerializationError
from .hostprocess import StderrTail

__all__ = ["Failure", "Finished", "Connection", "ProcessConnection", "Host"]

_POLL = 0.2  # seconds between checks while waiting on a handle's queue
_CHECK_TIMEOUT = 120.0  # seconds a host's check worker may take to import the function and reply


@dataclass(frozen=True)
class Failure:
    """Why an input did not produce a result, as text from wherever it ran.

    ``type`` is the exception's type name, or ``"exit"`` for a worker that
    exited without a result, ``"timeout"`` for one killed at its deadline,
    ``"connection"`` for a host whose connection closed under the input
    twice.  ``traceback`` is the worker's traceback text, or "".
    """

    type: str
    message: str
    traceback: str = ""

    def __str__(self) -> str:
        text = f"{self.type}: {self.message}"
        return f"{text}\n{self.traceback}" if self.traceback else text


class Finished:
    """What happened to a task.  ``kind`` is one of:

    * ``"value"``: ``value`` is the result;
    * ``"failed"``: ``failure`` says why there is none;
    * ``"closed"``: the connection to the host closed with the input
      neither done nor failed, so the scheduler may run it elsewhere.
    """

    __slots__ = ("kind", "value", "failure")

    def __init__(self, kind: str, value: Any = None, failure: Failure | None = None):
        self.kind = kind
        self.value = value
        self.failure = failure


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

    Frames are parsed out of a buffer as bytes arrive.  A worker sends
    several times before it finishes -- a HELLO message, store traffic, then the
    result's objects -- so "bytes arrived" does not mean "the result is
    here", and reading until it is would sit inside a task that has already
    blown its deadline.

    Lookups are served on the scheduler's thread; a ``@pure_local`` call
    runs on a pool thread, since it may be a download, and replies when it
    is done.  ``seen`` names every object either side has sent, so nothing
    crosses twice.
    """

    __slots__ = (
        "ch", "objects", "seen", "completions", "out", "_host", "_failure",
        "_buf", "_result", "_eof", "_exit", "_stderr", "_lock",
    )

    def __init__(self, host: Host, ch: int, completions: queue.Queue):
        self.ch = ch
        self.objects: dict[str, bytes] = {}
        self.seen: set[str] = set()
        self.completions = completions
        self.out = _ChannelWriter(host._send, ch)
        self._host = host
        self._failure: str | None = None
        self._buf = b""
        self._result: Finished | None = None
        self._eof = False
        self._exit: int | None = None
        self._stderr = b""
        self._lock = threading.Lock()  # the channel is written from two threads

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
            self._result = Finished("failed", failure=Failure("ProtocolError", str(e)))

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
            if tag == protocol.OBJECT:
                protocol.recv_object(body, self.objects)
                self.seen.add(body[:20].hex())
                store = self._store()
                if store is not None:
                    # The worker's results live here and nowhere else.
                    protocol.store_object(store, body)
            elif tag == protocol.RESULT:
                message = json.loads(body)
                if "value" in message:
                    root = message["value"]  # None: a check worker's acceptance
                    value = None if root is None else protocol.unpack(root, self.objects, self._fallback())
                    self._result = Finished("value", value=value)
                else:
                    self._result = Finished("failed", failure=Failure(**message["failed"]))
            else:
                self._request(protocol, tag, body)

    def wait(self, timeout: float) -> Finished | None:
        """Feed this handle from its own queue until it has finished or
        *timeout* seconds have passed; the outcome, or None."""
        deadline = time.monotonic() + timeout
        while self.finished() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                handle, payload = self.completions.get(timeout=min(remaining, _POLL))
            except queue.Empty:
                continue
            handle.feed(payload)
        return self.finished()

    # -- serving the worker's requests ----------------------------------------

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
        from .store import CacheMiss

        store = self._store()
        if tag == protocol.RECORD:
            message = json.loads(body)
            if store is not None:
                store.put_record(message["function_hash"], message["record"])
        elif tag == protocol.GET_RECORDS:
            pairs = [] if store is None else store.get_records(body.decode())
            self._write_message(protocol.REPLY, json.dumps(pairs).encode())
        elif tag == protocol.GET_VALUE:
            try:
                if store is None:
                    raise CacheMiss("the main process has no store directory")
                v = store.get_value(body.hex())
            except CacheMiss as e:
                self._write_message(protocol.REPLY, json.dumps({"reason": str(e)}).encode())
            else:
                with self._lock:
                    protocol.send_value(self.out, v, self.seen)
                    protocol.write_message(self.out, protocol.REPLY, b"{}")
        elif tag == protocol.EVENT:
            record = json.loads(body)
            record.setdefault("host", self._host.name)
            if store is not None:
                store.event(record)
        elif tag == protocol.LOGGED:
            if store is not None:
                store.log_line(json.loads(body))
        elif tag == protocol.CALL:
            call = json.loads(body)
            args, kwargs = protocol.unpack(call["root"], self.objects, self._fallback())
            self._host._submit(self._run_call, protocol, call["module"], call["qualname"], args, kwargs)
        else:
            raise protocol.ProtocolError(f"unexpected message {tag!r}")

    def _run_call(self, protocol, module: str, qualname: str, args, kwargs) -> None:
        """On a pool thread: make the @pure_local call here and reply."""
        from .worker import _resolve

        try:
            fn = _resolve(module, qualname)
            value = fn(*args, **kwargs)
            memo = getattr(fn, "_valuekit", None)
            h = (memo.record_hash(*args, **kwargs) if memo is not None else None) or ""
        except BaseException as e:
            failed = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
            self._write_message(protocol.REPLY, json.dumps({"failed": failed}).encode())
            return
        with self._lock:
            root = protocol.send_value(self.out, value, self.seen)
            reply = json.dumps({"root": root, "record_hash": h}).encode()
            protocol.write_message(self.out, protocol.REPLY, reply)

    # -- the scheduler's view --------------------------------------------------

    def finished(self) -> Finished | None:
        if self._failure is not None:
            return Finished("failed", failure=Failure("SerializationError", self._failure))
        if self._result is not None:
            return self._result
        if not self._eof:
            return None
        if self._exit is None:
            return Finished("closed")
        tail = self._stderr.decode("utf-8", "replace").strip()
        why = f"exit code {self._exit} on host {self._host.name!r}"
        return Finished("failed", failure=Failure(
            "exit", f"{why}:\n{tail}" if tail else f"{why}; a segfault or a broken pipe?"
        ))

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
    stdout are the connection, its stderr is read for the failure message."""

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
        self._stderr = StderrTail(self.proc)

    def failure(self) -> str:
        code = self.proc.poll()
        tail = self._stderr.bytes().decode("utf-8", "replace").strip()
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


def hello_message(
    fn, project_hash: str, roots: list[str], store_dir: str | None, path: list[str] = ()
) -> bytes:
    """The HELLO message every worker running *fn* gets.

    *project_hash* and *roots* name the source tree a remote host's worker
    imports from ("" and [] for a worker on this machine, which imports as
    this process does, with *path*, this process's ``sys.path``, ahead of
    its own).  *store_dir* is this process's store directory
    when the worker may read and write values and call records in it
    directly, "" when they go through this process, None when there is no
    store.  The extension markers are every one this process has computed,
    which include the function's, so a worker's function hash is compared
    with this process's (see :mod:`valuekit.functionhash`).
    """
    from .functionhash import PYTHON, extension_markers, reachable_set

    function_hash = reachable_set(fn).hash  # hashes the extensions the function reaches
    return json.dumps(
        {
            "python": PYTHON,
            "module": getattr(fn, "__module__", "") or "",
            "qualname": getattr(fn, "__qualname__", "") or "",
            "function_hash": function_hash,
            "project_hash": project_hash,
            "extensions": extension_markers(),
            "roots": list(roots),
            "store_dir": store_dir,
            "path": list(path),
        },
        sort_keys=True,
    ).encode()


class Host:
    """A machine a batch runs tasks on, through one connection to a host
    process there, which starts one worker per task.

    *connect* opens the connection: for a remote host it runs ssh and the
    bootstrap (:mod:`valuekit.bootstrap`), for this machine it starts a
    host process here.  It returns a :class:`Connection` whose far end is
    a host process about to send its first message, or raises with the
    reason it cannot.  *hello* is the HELLO message every worker gets.
    *capacity* is how many tasks the host runs at once; None means what
    the host process reports.  With *check*, the host runs the function
    once on a check channel before it takes any task (see :meth:`sync`);
    this machine does not, since this process has imported the function.
    """

    def __init__(
        self,
        name: str,
        connect: Callable[[], Connection],
        hello: bytes,
        store,
        completions: queue.Queue,
        capacity: int | None = None,
        check: bool = True,
    ):
        self.name = name
        self.capacity = capacity
        # "new" until sync() is called; "syncing" while it runs; then
        # "ready", or "failed" with the reason in ``failure``, which is also
        # where a host goes when its connection closes.
        self.state = "new"
        self.failure: str | None = None
        self._hello = hello
        self._store = store
        self._completions = completions
        self._connect = connect
        self._check = check
        self._connection: Connection | None = None
        self.pid: int | None = None  # the host process, once it has sent its first message
        self._out_lock = threading.Lock()
        self._channels: dict[int, _Handle] = {}
        self._next = 1
        self._pool: ThreadPoolExecutor | None = None

    # -- the connection ---------------------------------------------------------

    def _launch(self) -> str:
        """Connect and read the host process's first message; "" or why not."""
        from . import protocol
        from .functionhash import PYTHON

        try:
            self._connection = self._connect()
        except Exception as e:
            return f"cannot connect to host {self.name!r}: {e}"
        try:
            message = protocol.read_message(self._connection.rx)
        except (protocol.ProtocolError, OSError, ValueError):
            message = None
        if message is None or message[0] != protocol.HOST:
            reason = self._why("the host process said nothing")
            self._connection.close()
            return reason
        host = json.loads(message[1])
        if host["python"] != PYTHON:
            self._connection.close()
            return f"main process runs Python {PYTHON}, host {self.name!r} runs {host['python']}"
        if self.capacity is None:
            self.capacity = max(1, int(host["cpus"]))
        self.pid = host["pid"]
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
        if self.state != "failed":
            self.state = "failed"
            self.failure = self.failure or f"the connection to host {self.name!r} closed"
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
        handle = _Handle(self, ch, completions)
        self._channels[ch] = handle
        self._send(protocol.OPEN, protocol.channelled(ch, kind))
        return handle

    # -- the sync -------------------------------------------------------------------

    def sync(self) -> str:
        """Decide once, before this host takes any input, whether it can run
        the function: connect (for a remote host, update its copy of the
        project and build the environment), then, with *check*, import the
        function there and compare what was imported with the main
        process's function.

        Returns "" when the host can take work, else why not.  A failure
        is a fact about the host, recorded once rather than against every
        input that would have gone there, and the host has no capacity
        until the decision is made, so no input waits on it.
        """
        from . import protocol

        if self.state == "ready":
            return ""
        if self.state == "failed":
            return self.failure or ""
        self.state = "syncing"
        reason = self._launch()
        if reason:
            return self._fail(None, reason)
        if not self._check:
            self.state = "ready"
            return ""
        handle = self._open(b"check", queue.Queue())
        try:
            handle._write_message(protocol.HELLO, self._hello)
            done = handle.wait(timeout=_CHECK_TIMEOUT)
            if done is None:
                return self._fail(handle, (
                    f"the worker did not reply within {_CHECK_TIMEOUT:.0f} s of being asked to "
                    "import the function; an import that waits on something?"
                ))
            if done.kind == "closed":
                return self._fail(handle, "the worker never reported")
            if done.kind == "failed":
                return self._fail(handle, done.failure.message)
        finally:
            handle.release()
        self.state = "ready"
        return ""

    def _fail(self, handle: _Handle | None, reason: str) -> str:
        tail = handle._stderr.decode("utf-8", "replace").strip() if handle is not None else ""
        self.failure = f"{reason}:\n{tail}" if tail else reason
        self.state = "failed"
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
                # A worker never sends back an object it received, so a
                # value it builds from its input (a result, a logged value's
                # labels) refers to the input's objects: they are kept for
                # this task and written to the store with everything else.
                sent: dict[str, bytes] = {}
                root = protocol.send_value(handle.out, x, handle.seen, sent)
                handle.objects.update(sent)
                if self._store is not None:
                    for h, payload in sent.items():
                        protocol.store_object(self._store, bytes.fromhex(h) + payload)
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
