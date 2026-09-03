"""A worker that runs the driver's code, from a copy of the driver's source.

``python -m valuekit.worker`` reads framed messages on stdin and writes them
on stdout.  It comes in two shapes.  ``--ready`` makes one process per host
that unpacks the project's source tree, imports from it, checks what it
got, and exits; every later process then runs exactly one input against that
finished source tree and exits, which is what keeps the isolation `run_all`
already promises -- a segfault or a timeout costs one input and nothing else.

Readiness is separate for a reason.  A sync failure, a missing dependency or
a compile error is a fact about the *host*, and folding it into the first
task would report it against whichever input happened to go first.  Getting
that wrong would break the one property `run_all` is built around: every
failure recorded against the input that caused it.

    driver -> SYNC    salt, ids, source root, manifest hash, import roots
    worker -> WANT    empty if the source tree is already here, else send it
    driver -> TREE    the tarball, only if wanted
    worker -> READY   empty if admitted, else why not

    driver -> HELLO   salt, module, qualname, fingerprint, source root, source id
    worker -> READY   empty if admitted, else why not
    driver -> OBJECT* the input's object graph
    driver -> TASK    the input's root hash
    ...               store traffic: the worker's cache is the driver's
    worker -> OBJECT* the result's object graph
    worker -> RESULT  ok and a root hash, or a failure

A worker holds no cache.  Its store is a :class:`~valuekit.remotestore.WireStore`,
which sends every value, trace and run-log record to the driver and asks the
driver for every lookup, so a batch's results exist in one place however
many machines ran it.  The only thing written here is the source tree, at
the root the driver named.

Three checks guard the result, and they are deliberately independent.  The
source tree decides what is on ``sys.path``; the *audit* then confirms that what
was actually imported came from there, because a path entry can still lose to
an editable install's meta-path finder; and the fingerprint handshake
confirms the code means what the driver thinks.  The audit matters most: it
is the difference between running the driver's code and running whatever the
worker happened to have.

The transport carries values, never code.  What crosses is the fixed set of
storable types -- a real narrowing of what a local worker accepts by pickle,
since a function passed as an input cannot cross here.  That is deliberate:
bytes from a peer describe data or they describe nothing.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO

from . import runlog, sync, wire
from .remotestore import WireStore

__all__ = ["main", "serve", "serve_ready"]


def _resolve(module: str, qualname: str):
    mod = importlib.import_module(module)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# Greeting fields after the six fixed ones: the import roots, then this
# marker, then the driver's native-extension digests as (module, digest)
# pairs.  See codehash._ext_overrides for why a worker takes them on.
EXT_MARK = "--ext--"


def _tail(parts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split a greeting's trailing fields into import roots and overrides."""
    if EXT_MARK in parts:
        at = parts.index(EXT_MARK)
        roots, pairs = parts[:at], parts[at + 1 :]
    else:
        roots, pairs = parts, []
    overrides = {pairs[i]: pairs[i + 1] for i in range(0, len(pairs) - 1, 2)}
    return [r for r in roots if r], overrides


def _take_overrides(overrides: dict[str, str]) -> None:
    from . import codehash

    codehash._ext_overrides.update(overrides)



def source_dir(source_root: str, manifest_hash: str) -> Path:
    """Where a synced source tree lives, under the root the driver named.

    ``~`` is expanded here, on the machine the tree lives on: the driver
    names the root as configured, without knowing this machine's home.
    """
    return Path(os.path.expanduser(source_root)) / manifest_hash


def _install(source: Path, roots: list[str]) -> None:
    """Put the source tree's import roots ahead of everything else."""
    for rel in reversed(roots):
        entry = str((source / rel).resolve())
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _audit(source: Path) -> str:
    """Confirm every user module actually came from the source tree.

    A source tree on ``sys.path`` is not proof that imports resolved through it:
    a PEP 660 editable install puts a finder on ``sys.meta_path``, which runs
    before any path entry, and namespace packages merge portions across
    entries.  This is the check that is independent of the sync having
    worked -- without it a worker could quietly run the wrong source.
    """
    real = str(source.resolve())
    strays: list[str] = []
    for name, mod in list(sys.modules.items()):
        if name.startswith("valuekit") or name in ("__main__", "__mp_main__"):
            continue  # the harness itself, not the code under test
        fname = getattr(mod, "__file__", None)
        if not fname:
            continue
        try:
            path = os.path.realpath(fname)
        except OSError:
            continue
        if sync.is_environment(path) or sync._under(path, real):
            continue
        strays.append(f"{name} from {path}")
    if strays:
        return (
            "these modules were imported from outside the source tree, so this "
            "worker would not be running the driver's code:\n  "
            + "\n  ".join(sorted(strays))
            + f"\nExpected everything under {real}."
        )
    return ""


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


# ---------------------------------------------------------------------------
# readiness: once per host
# ---------------------------------------------------------------------------


def serve_ready(rx: BinaryIO, tx: BinaryIO) -> int:
    frame = wire.read_frame(rx)
    if frame is None:
        return 0
    tag, body = frame
    if tag != wire.SYNC:
        wire.write_frame(tx, wire.READY, b"expected a sync request")
        return 1
    parts = wire.unstrings(body)
    salt, module, qualname, fingerprint, source_root, mhash = parts[:6]
    roots, overrides = _tail(parts[6:])

    reason = _refuse_early(salt, source_root)
    if reason:
        wire.write_frame(tx, wire.WANT, b"")  # nothing will be sent
        wire.write_frame(tx, wire.READY, reason.encode("utf-8"))
        return 1

    source = source_dir(source_root, mhash)
    complete = source / ".complete"
    wire.write_frame(tx, wire.WANT, b"" if complete.exists() else b"send")
    if not complete.exists():
        frame = wire.read_frame(rx)
        if frame is None or frame[0] != wire.TREE:
            wire.write_frame(tx, wire.READY, b"the project tree never arrived")
            return 1
        try:
            unpack(frame[1], source)
        except Exception as e:
            wire.write_frame(
                tx, wire.READY, f"could not unpack the project: {e}".encode()
            )
            return 1

    _install(source, roots)
    _take_overrides(overrides)

    reason = _admit(salt, module, qualname, fingerprint)
    if not reason:
        # After the import, and before the fingerprint is trusted: a
        # fingerprint that matches the wrong file is still the wrong file.
        reason = _audit(source)
    wire.write_frame(tx, wire.READY, reason.encode("utf-8"))
    return 1 if reason else 0


def _refuse_early(salt: str, source_root: str) -> str:
    from .pure import _salt

    mine = _salt()
    if mine != salt:
        return f"driver is {salt}, this worker is {mine}"
    if not source_root:
        return (
            "this worker was given nowhere to keep a copy of the project. The "
            "driver names the place from its cache directory; configure one "
            "with set_cache_dir()."
        )
    return sync.check_extraction_supported()


def unpack(data: bytes, source: Path) -> None:
    """Unpack into a fresh directory, then move it into place.

    Immutability of the name is not atomicity of the construction: two
    drivers can race on one host, and a half-built tree must never be
    adopted.  Hence a temporary directory, a marker written last, and a
    rename -- with "someone else got there first" treated as success.
    """
    import shutil
    import uuid

    parent = source.parent
    parent.mkdir(parents=True, exist_ok=True)
    prune(parent)
    tmp = parent / f".tmp-{uuid.uuid4().hex}"
    try:
        tmp.mkdir()
        sync.extract_tree(data, str(tmp))
        (tmp / ".complete").write_bytes(b"")
        try:
            os.rename(tmp, source)
        except OSError:
            if (source / ".complete").exists():
                shutil.rmtree(tmp, ignore_errors=True)  # someone else won
            else:
                # Something is in the way without the marker, so it is debris
                # from a crash or a kill: it can never be adopted, and leaving
                # it would block this source tree for good.
                shutil.rmtree(source, ignore_errors=True)
                os.rename(tmp, source)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


_MAX_TREES = 10


def prune(parent: Path) -> None:
    """Drop the oldest source trees. Best effort; a live one is only wasted disk."""
    try:
        kept = sorted(
            (p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    import shutil

    for p in kept[: max(0, len(kept) - _MAX_TREES + 1)]:
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------


def serve(rx: BinaryIO, tx: BinaryIO) -> int:
    """Run one task off *rx*, reporting on *tx*."""
    frame = wire.read_frame(rx)
    if frame is None:
        return 0  # driver went away before saying anything
    tag, body = frame
    if tag != wire.HELLO:
        wire.write_frame(tx, wire.READY, b"expected a greeting")
        return 1
    parts = wire.unstrings(body)
    salt, module, qualname, fingerprint, source_root, mhash = parts[:6]
    roots, overrides = _tail(parts[6:])

    # The source tree has to be on the path before _admit, which imports.
    if mhash:
        source = source_dir(source_root, mhash)
        if not (source / ".complete").exists():
            wire.write_frame(tx, wire.READY, b"the source tree is missing")
            return 1
        _install(source, roots)
    _take_overrides(overrides)

    reason = _admit(salt, module, qualname, fingerprint)
    wire.write_frame(tx, wire.READY, reason.encode("utf-8"))
    if reason:
        return 1

    from .pure import _current_store, set_store

    store = WireStore(rx, tx)
    previous = _current_store()  # serve() runs in-process in tests
    set_store(store)
    try:
        root = None
        while True:
            frame = wire.read_frame(rx)
            if frame is None:
                return 0  # cancelled before the task arrived
            tag, body = frame
            if tag == wire.OBJECT:
                store.receive(body)
            elif tag == wire.TASK:
                root = body.hex()
                break
            else:
                raise wire.WireError(f"unexpected frame {tag!r}")

        fn = _resolve(module, qualname)
        try:
            x = store.unpack(root)
        except Exception as e:
            _send_error(tx, type(e).__name__, f"the input could not be read: {e}")
            return 1

        try:
            value = fn(x)
        except BaseException as e:
            _send_error(tx, type(e).__name__, str(e), traceback.format_exc())
            return 1

        try:
            out = store.put_value(value)
        except Exception as e:
            _send_error(tx, type(e).__name__, f"the result could not be sent back: {e}")
            return 1
        wire.write_frame(tx, wire.RESULT, b"o" + bytes.fromhex(out))
        return 0
    finally:
        set_store(previous)


def _send_error(tx: BinaryIO, kind: str, text: str, tb: str = "") -> None:
    wire.write_frame(tx, wire.RESULT, b"e" + wire.strings(kind, text, tb))


def main(argv: list[str] | None = None) -> int:
    # Declared here rather than in serve(): the role is a fact about this
    # process, and serve() is also called in-process by tests, which must
    # not relabel their own caller.
    runlog.set_role("worker")
    args = sys.argv[1:] if argv is None else argv
    rx, tx = sys.stdin.buffer, sys.stdout.buffer
    try:
        return serve_ready(rx, tx) if "--ready" in args else serve(rx, tx)
    except wire.WireError:
        return 2
    finally:
        runlog._flush()


if __name__ == "__main__":
    raise SystemExit(main())
