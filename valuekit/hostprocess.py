"""One process per host per batch: ``python -m valuekit.hostprocess``.

The main process opens one connection to a host -- an ssh session, or a plain
subprocess when the host is this machine -- and this process is what runs
at the other end.  It starts one ``valuekit.worker`` subprocess per task and
carries each worker's stdin and stdout over the connection as a numbered
channel, so a batch of a thousand inputs costs one ssh handshake rather than
a thousand.  (The Windows ssh client has no connection sharing, which is
what rules out a connection per task.)

    host   -> HOST    {python, cpus, pid}
    main   -> OPEN    channel, "check" | "task"     start a worker
    main   -> DATA    channel, bytes                 to that worker's stdin
    host   -> DATA    channel, bytes                 from that worker's stdout
    main   -> CLOSE   channel                        close the worker's stdin
    main   -> KILL    channel                        kill the worker
    host   -> EXIT    channel, exit code, stderr tail

Workers are started with the allowlisted environment, as they are locally;
``VALUEKIT_TREE``, set by the bootstrap that started this process, passes
through it and tells each worker which source tree it is in.
This process exits when its stdin closes, after killing every worker it
started: a dropped connection or a main process that exited leaves nothing running.
Bytes on a channel are passed through untouched; what they mean is between
the main process and the worker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import BinaryIO

from . import protocol
from .functionhash import PYTHON

__all__ = ["main", "serve"]

_STDERR_TAIL = 64 << 10


class StderrTail:
    """The last ``_STDERR_TAIL`` bytes a process has written to stderr,
    read by a thread as they arrive.  ``bytes()`` may be called from any
    thread, at any time."""

    __slots__ = ("_buf", "_lock")

    def __init__(self, proc: subprocess.Popen):
        self._buf = bytearray()
        self._lock = threading.Lock()
        fd = proc.stderr.fileno()  # type: ignore[union-attr]
        threading.Thread(target=self._read, args=(fd,), daemon=True).start()

    def _read(self, fd: int) -> None:
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                return
            with self._lock:
                self._buf += chunk
                del self._buf[:-_STDERR_TAIL]

    def bytes(self) -> bytes:
        with self._lock:
            return bytes(self._buf)


class _Worker:
    __slots__ = ("proc", "stderr")

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.stderr = StderrTail(proc)


# What a process needs from the environment to start and to find its
# interpreter's own files; everything else stays with the main process.  No
# PYTHONPATH (imports must resolve through the source tree, or the check
# that they did proves nothing) and no credentials, which a @pure_local
# call keeps on the main process.
_WORKER_ENV = frozenset(
    {
        "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
        "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "WINDIR",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "USERNAME", "USER",
        "PYTHONHOME", "PYTHONUTF8", "PYTHONIOENCODING", "VIRTUAL_ENV",
    }
)


def worker_env() -> dict[str, str]:
    """The environment a worker is started with: an allowlist."""
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() in _WORKER_ENV or k.upper().startswith("VALUEKIT_")
    }


def serve(rx: BinaryIO, tx: BinaryIO, python: str | None = None, local: bool = False) -> int:
    """Serve the main process at *rx*/*tx* until it closes the connection.
    With *local* this is the main process's own machine: workers get its
    environment unchanged; otherwise the allowlist."""
    python = python or sys.executable
    out_lock = threading.Lock()
    workers: dict[int, _Worker] = {}

    def send(tag: bytes, body: bytes) -> None:
        with out_lock:
            try:
                protocol.write_message(tx, tag, body)
            except (OSError, ValueError):
                pass  # the main process is gone; EOF on stdin follows

    def forward_stdout(ch: int, w: _Worker) -> None:
        """Forward the worker's stdout, then report how it ended."""
        fd = w.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                break
            send(protocol.DATA, protocol.channelled(ch, chunk))
        code = w.proc.wait()
        tail = w.stderr.bytes()
        send(
            protocol.EXIT,
            protocol.channelled(ch, code.to_bytes(4, "little", signed=True) + tail),
        )

    started = {"python": PYTHON, "cpus": os.cpu_count() or 1, "pid": os.getpid()}
    send(protocol.HOST, json.dumps(started).encode())
    env = None if local else worker_env()
    try:
        while True:
            message = protocol.read_message(rx)
            if message is None:
                return 0
            tag, body = message
            ch, rest = protocol.channel(body)
            if tag == protocol.OPEN:
                args = ["--check"] if rest == b"check" else []
                proc = subprocess.Popen(
                    [python, "-m", "valuekit.worker", *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                w = workers[ch] = _Worker(proc)
                threading.Thread(target=forward_stdout, args=(ch, w), daemon=True).start()
            elif tag == protocol.DATA:
                w = workers.get(ch)
                if w is not None:
                    try:
                        w.proc.stdin.write(rest)
                        w.proc.stdin.flush()
                    except (OSError, ValueError):
                        pass  # the worker has exited; its EXIT says so
            elif tag == protocol.CLOSE:
                w = workers.get(ch)
                if w is not None:
                    try:
                        w.proc.stdin.close()
                    except (OSError, ValueError):
                        pass
            elif tag == protocol.KILL:
                w = workers.pop(ch, None)
                if w is not None:
                    _kill(w.proc)
            else:
                raise protocol.ProtocolError(f"unexpected message {tag!r}")
    except protocol.ProtocolError:
        return 2
    finally:
        for w in workers.values():
            _kill(w.proc)


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return serve(sys.stdin.buffer, sys.stdout.buffer, local="--local" in args)


if __name__ == "__main__":
    raise SystemExit(main())
