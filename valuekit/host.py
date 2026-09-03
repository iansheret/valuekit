"""One process per host per batch: ``python -m valuekit.host``.

The driver opens one connection to a host -- an ssh session, or a plain
subprocess when the host is this machine -- and this process is what runs
at the other end.  It starts one ``valuekit.worker`` subprocess per task and
carries each worker's stdin and stdout over the connection as a numbered
channel, so a batch of a thousand inputs costs one ssh handshake rather than
a thousand.  (The Windows ssh client has no connection sharing, which is
what rules out a connection per task.)

    host   -> HOST    salt, CPU count
    driver -> OPEN    channel, "ready" | "task"     start a worker
    driver -> DATA    channel, bytes                 to that worker's stdin
    host   -> DATA    channel, bytes                 from that worker's stdout
    driver -> CLOSE   channel                        close the worker's stdin
    driver -> KILL    channel                        kill the worker
    host   -> EXIT    channel, exit code, stderr tail

Workers are started with the allowlisted environment, as they are locally.
This process exits when its stdin closes, after killing every worker it
started: a dropped connection or a driver that died leaves nothing running.
Bytes on a channel are passed through untouched; what they mean is between
the driver and the worker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import BinaryIO

from . import wire
from .placement import worker_env

__all__ = ["main", "serve"]

_STDERR_TAIL = 64 << 10


class _Worker:
    __slots__ = ("proc", "stderr", "lock")

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.stderr = bytearray()
        self.lock = threading.Lock()


def serve(rx: BinaryIO, tx: BinaryIO, python: str | None = None) -> int:
    python = python or sys.executable
    out_lock = threading.Lock()
    workers: dict[int, _Worker] = {}

    def send(tag: bytes, body: bytes) -> None:
        with out_lock:
            try:
                wire.write_frame(tx, tag, body)
            except (OSError, ValueError):
                pass  # the driver is gone; EOF on stdin follows

    def pump(ch: int, w: _Worker) -> None:
        """Forward the worker's stdout, then report how it ended."""
        fd = w.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                break
            send(wire.DATA, wire.channelled(ch, chunk))
        code = w.proc.wait()
        with w.lock:
            tail = bytes(w.stderr[-_STDERR_TAIL:])
        send(
            wire.EXIT,
            wire.channelled(ch, code.to_bytes(4, "little", signed=True) + tail),
        )

    def drain_stderr(w: _Worker) -> None:
        fd = w.proc.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                return
            with w.lock:
                w.stderr += chunk
                del w.stderr[:-_STDERR_TAIL]

    send(wire.HOST, wire.strings(_salt(), str(os.cpu_count() or 1)))
    env = worker_env()
    try:
        while True:
            frame = wire.read_frame(rx)
            if frame is None:
                return 0
            tag, body = frame
            ch, rest = wire.channel(body)
            if tag == wire.OPEN:
                args = ["--ready"] if rest == b"ready" else []
                proc = subprocess.Popen(
                    [python, "-m", "valuekit.worker", *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                w = workers[ch] = _Worker(proc)
                threading.Thread(target=drain_stderr, args=(w,), daemon=True).start()
                threading.Thread(target=pump, args=(ch, w), daemon=True).start()
            elif tag == wire.DATA:
                w = workers.get(ch)
                if w is not None:
                    try:
                        w.proc.stdin.write(rest)
                        w.proc.stdin.flush()
                    except (OSError, ValueError):
                        pass  # the worker died; its EXIT says so
            elif tag == wire.CLOSE:
                w = workers.get(ch)
                if w is not None:
                    try:
                        w.proc.stdin.close()
                    except (OSError, ValueError):
                        pass
            elif tag == wire.KILL:
                w = workers.pop(ch, None)
                if w is not None:
                    _kill(w.proc)
            else:
                raise wire.WireError(f"unexpected frame {tag!r}")
    except wire.WireError:
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


def _salt() -> str:
    from .pure import _salt

    return _salt()


def main(argv: list[str] | None = None) -> int:
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
