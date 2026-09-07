"""A worker that runs the driver's code, from a copy of the driver's source.

``python -m valuekit.worker`` reads framed messages on stdin and writes them
on stdout.  It comes in two shapes.  ``--ready`` makes one process per host
that imports the function from the project's source tree, checks what it
got, and exits; every later process then runs exactly one input against
that same tree and exits, which is what keeps the isolation `run_all`
already promises -- a segfault or a timeout costs one input and nothing else.

Readiness is separate for a reason.  A missing dependency, a build error or
a function that will not import is a fact about the *host*, and folding it
into the first task would report it against whichever input happened to go
first.  Getting that wrong would break the one property `run_all` is built
around: every failure recorded against the input that caused it.

    driver -> HELLO   salt, module, qualname, fingerprint, tree id, import roots
    worker -> READY   empty if admitted, else why not
    driver -> OBJECT* the input's object graph            (task workers only)
    driver -> TASK    the input's root hash
    ...               store traffic: the worker's cache is the driver's
    worker -> OBJECT* the result's object graph
    worker -> RESULT  ok and a root hash, or a failure

The source tree is already here, and so is the environment it needs: the
bootstrap (:mod:`valuekit.bootstrap`) received the tree and built the
environment before this interpreter -- the environment's own -- started,
and named the tree in ``VALUEKIT_TREE``.  The greeting says which tree the
driver meant, and a worker in the wrong one refuses.

A worker holds no cache.  Its store is a :class:`~valuekit.remotestore.WireStore`,
which sends every value, trace and run-log record to the driver and asks the
driver for every lookup, so a batch's results exist in one place however
many machines ran it.

Three checks guard the result, and they are deliberately independent.  The
source tree decides what is on ``sys.path``; the *audit* then confirms that
what was actually imported came from there, because a path entry can still
lose to some other finder on ``sys.meta_path``; and the fingerprint
handshake confirms the code means what the driver thinks.  The audit
matters most: it is the difference between running the driver's code and
running whatever the worker happened to have.

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


def _take_identity(tree_id: str) -> None:
    """A native extension here was built from the tree named *tree_id*, so
    that is its identity, exactly as it is on the driver.  See
    :mod:`valuekit.codehash`."""
    from . import codehash

    codehash._source_id = tree_id or None


def _tree(tree_id: str) -> tuple[Path | None, str]:
    """The source tree this worker runs from, or why it cannot.

    Named by the bootstrap in the environment; the greeting says which tree
    the driver meant, and they must agree.  No tree at all (an empty id and
    nothing in the environment) means the tests' in-process worker, which
    imports as this process does.
    """
    here = os.environ.get("VALUEKIT_TREE", "")
    if not tree_id and not here:
        return None, ""
    if not here:
        return None, "this worker was started outside a source tree"
    if os.path.basename(here) != tree_id:
        return None, (
            f"this worker is in tree {os.path.basename(here)[:12]}, the driver "
            f"meant {tree_id[:12]}"
        )
    return Path(here), ""


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


def _greet(rx: BinaryIO, tx: BinaryIO, audit: bool) -> tuple[bool, str]:
    """Read the greeting, set the process up to run the function, and reply.

    Returns whether to go on and the function's module and qualname joined
    by a colon, for :func:`serve` to resolve.
    """
    frame = wire.read_frame(rx)
    if frame is None:
        return False, ""  # the driver went away before saying anything
    tag, body = frame
    if tag != wire.HELLO:
        wire.write_frame(tx, wire.READY, b"expected a greeting")
        return False, ""
    parts = wire.unstrings(body)
    salt, module, qualname, fingerprint, tree_id = parts[:5]
    roots = [r for r in parts[5:] if r]

    source, reason = _tree(tree_id)
    if not reason:
        # The tree goes on the path before _admit, which imports.
        if source is not None:
            _install(source, roots)
        _take_identity(tree_id)
        reason = _admit(salt, module, qualname, fingerprint)
    if not reason and audit and source is not None:
        # After the import, and before the fingerprint is trusted: a
        # fingerprint that matches the wrong file is still the wrong file.
        reason = _audit(source)
    wire.write_frame(tx, wire.READY, reason.encode("utf-8"))
    return not reason, f"{module}:{qualname}"


def serve_ready(rx: BinaryIO, tx: BinaryIO) -> int:
    """Once per host: import the function from the tree and check it."""
    ok, _ = _greet(rx, tx, audit=True)
    return 0 if ok else 1


def serve(rx: BinaryIO, tx: BinaryIO) -> int:
    """Run one task off *rx*, reporting on *tx*."""
    ok, target = _greet(rx, tx, audit=False)
    if not ok:
        return 0 if not target else 1
    module, qualname = target.split(":", 1)

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
