"""A worker that runs one input and reports over a pipe.

``python -m valuekit.worker`` reads framed messages on stdin and writes
them on stdout.  It handles exactly one input and exits, which is what
keeps the isolation guarantee `run_all` already makes: a segfault or a
timeout costs exactly one input and nothing else.

The conversation is short::

    driver -> HELLO   salt, module, qualname, fingerprint, cache dir
    worker -> READY   empty if admitted, otherwise why not
    driver -> OBJECT* the input's object graph
    driver -> TASK    the input's root hash
    worker -> OBJECT* the result's object graph
    worker -> RESULT  ok and a root hash, or a failure

The handshake is the point of the whole design.  A worker recomputes
``function_fingerprint`` for the function it was asked to run and compares:
if the code it would execute is not the code the driver meant, it refuses
rather than returning a plausible answer.  The salt is checked first,
because it carries the Python version and a mismatch there explains an
otherwise opaque difference in the fingerprint -- raw bytecode is part of
it, so two interpreter versions disagree on identical source.

The transport carries values, never code.  What crosses is the fixed set of
storable types, which is a real narrowing of what a local worker accepts
by pickle -- a function passed as an input works locally and cannot cross
here.  That is deliberate: bytes from a peer describe data or they describe
nothing.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any, BinaryIO

from . import events, wire

__all__ = ["main", "serve"]


def _resolve(module: str, qualname: str):
    mod = importlib.import_module(module)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _admit(salt: str, module: str, qualname: str, fingerprint: str) -> str:
    """Return "" if this process should run the function, else the reason."""
    from .codehash import function_fingerprint
    from .pure import _salt

    mine = _salt()
    if mine != salt:
        return f"driver is {salt}, this worker is {mine}"
    if module == "__main__":
        return (
            "the function is defined in __main__; a worker cannot import a "
            "driver script. Move it to a module and import it."
        )
    try:
        fn = _resolve(module, qualname)
    except Exception as e:
        return f"cannot import {module}:{qualname} here ({e})"
    try:
        theirs = function_fingerprint(fn)[0]
    except Exception as e:
        return f"cannot fingerprint {module}:{qualname} here ({e})"
    if theirs != fingerprint:
        return (
            f"{module}:{qualname} differs here: driver has {fingerprint[:12]}, "
            f"this worker has {theirs[:12]}. The code is not in sync."
        )
    return ""


def serve(rx: BinaryIO, tx: BinaryIO) -> int:
    """Run one task off *rx*, reporting on *tx*."""
    frame = wire.read_frame(rx)
    if frame is None:
        return 0  # driver went away before saying anything
    tag, body = frame
    if tag != wire.HELLO:
        wire.write_frame(tx, wire.READY, b"expected a greeting")
        return 1
    salt, module, qualname, fingerprint, cache_dir = wire.unstrings(body)

    reason = _admit(salt, module, qualname, fingerprint)
    wire.write_frame(tx, wire.READY, reason.encode("utf-8"))
    if reason:
        return 1

    if cache_dir:
        from .pure import set_cache_dir

        set_cache_dir(cache_dir)

    objects: dict[str, bytes] = {}
    root = None
    while True:
        frame = wire.read_frame(rx)
        if frame is None:
            return 0  # cancelled before the task arrived
        tag, body = frame
        if tag == wire.OBJECT:
            wire.recv_object(body, objects)
        elif tag == wire.TASK:
            root = body.hex()
            break
        else:
            raise wire.WireError(f"unexpected frame {tag!r}")

    fn = _resolve(module, qualname)
    try:
        x = wire.unpack(root, objects)
    except Exception as e:
        _send_error(tx, type(e).__name__, f"the input could not be read: {e}")
        return 1

    try:
        value = fn(x)
    except BaseException as e:
        _send_error(tx, type(e).__name__, str(e), traceback.format_exc())
        return 1

    try:
        out = wire.send_value(tx, value, set())
    except Exception as e:
        _send_error(tx, type(e).__name__, f"the result could not be sent back: {e}")
        return 1
    wire.write_frame(tx, wire.RESULT, b"o" + bytes.fromhex(out))
    return 0


def _send_error(tx: BinaryIO, kind: str, text: str, tb: str = "") -> None:
    wire.write_frame(tx, wire.RESULT, b"e" + wire.strings(kind, text, tb))


def main(argv: list[str] | None = None) -> int:
    # Declared here rather than in serve(): the role is a fact about this
    # process, and serve() is also called in-process by tests, which must
    # not relabel their own caller.
    events.set_role("worker")
    rx, tx = sys.stdin.buffer, sys.stdout.buffer
    try:
        return serve(rx, tx)
    except wire.WireError:
        return 2
    finally:
        events._flush()


if __name__ == "__main__":
    raise SystemExit(main())
