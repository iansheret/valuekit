"""A worker that runs the main process's code, from a copy of the main process's source.

``python -m valuekit.worker`` reads framed messages on stdin and writes them
on stdout.  It comes in two shapes.  ``--check`` makes one process per host
that imports the function from the project's source tree, checks what it
got, and exits; every later process then runs exactly one input against
that same tree and exits, which is what keeps the isolation `run_all`
already promises -- a segfault or a timeout costs one input and nothing else.

The check is a separate process for three reasons.  It is the host's
first import of the function, and with a build backend that rebuilds on
import it is the build: done once here, rather than by every task worker
at once in one build directory.  It is the moment a host becomes usable,
which the modes need: under ``remote`` this machine waits for it.  And a
missing dependency, a build error or a function that will not import is a
fact about the *host*, reported once, rather than against whichever input
happened to go there first.

    main   -> HELLO   json: python, module, qualname, function_hash, project_hash,
                      extensions, roots
    worker -> ACCEPTED   empty if accepted, else why not
    main   -> OBJECT* the input's object graph            (task workers only)
    main   -> TASK    the input's root hash
    ...               store traffic: the worker's cache is the main process's
    worker -> OBJECT* the result's object graph
    worker -> RESULT  ok and a root hash, or a failure

The source tree is already here, and so is the environment it needs: the
bootstrap (:mod:`valuekit.bootstrap`) received the tree and built the
environment before this interpreter -- the environment's own -- started,
and named the tree in ``VALUEKIT_TREE``.  The HELLO message says which tree the
main process meant, and a worker in the wrong one refuses.

A worker holds no cache.  Its store is a :class:`~valuekit.remotestore.RemoteStore`,
which sends every value, call record and event to the main process and asks the
main process for every lookup, so a batch's results exist in one place however
many machines ran it.

Three checks guard the result, and they are deliberately independent.  The
source tree decides what is on ``sys.path``; the *check_imports* then confirms that
what was actually imported came from there, because a path entry can still
lose to some other finder on ``sys.meta_path``; and the function hash
handshake confirms the code means what the main process thinks.  The check_imports
matters most: it is the difference between running the main process's code and
running whatever the worker happened to have.

The transport carries values, never code.  What crosses is the fixed set of
storable types -- a real narrowing of what a local worker accepts by pickle,
since a function passed as an input cannot cross here.  That is deliberate:
bytes from a peer describe data or they describe nothing.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO

from . import events, project, protocol
from .remotestore import RemoteStore

__all__ = ["main", "serve", "serve_check"]


def _resolve(module: str, qualname: str):
    mod = importlib.import_module(module)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _take_markers(extensions: dict[str, str]) -> None:
    """The main process's markers for the native extensions the function
    reaches, module name -> hash of its binary there.  This worker's binaries
    were built from the same sources on another machine and are never
    hashed; see :mod:`valuekit.functionhash`."""
    from . import functionhash

    functionhash._markers_here = dict(extensions)


def _tree(project_hash: str) -> tuple[Path | None, str]:
    """The source tree this worker runs from, or why it cannot.

    Named by the bootstrap in the environment; the HELLO message says which tree
    the main process meant, and they must agree.  No tree at all (an empty id and
    nothing in the environment) means the tests' in-process worker, which
    imports as this process does.
    """
    here = os.environ.get("VALUEKIT_TREE", "")
    if not project_hash and not here:
        return None, ""
    if not here:
        return None, "this worker was started outside a source tree"
    started_with = os.environ.get("VALUEKIT_PROJECT_HASH", "")
    if started_with != project_hash:
        return None, (
            f"this worker's tree is {started_with[:12]}, the main process meant {project_hash[:12]}"
        )
    return Path(here), ""


def _install(source: Path, roots: list[str]) -> None:
    """Put the source tree's import roots ahead of everything else."""
    for rel in reversed(roots):
        entry = str((source / rel).resolve())
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _check_imports(source: Path) -> str:
    """Confirm every user module actually came from the source tree.

    A source tree on ``sys.path`` is not proof that imports resolved through it:
    a PEP 660 editable install puts a finder on ``sys.meta_path``, which runs
    before any path entry, and namespace packages merge portions across
    entries.  This is the check that is independent of the sync having
    worked -- without it a worker could run the wrong source with no sign of it.
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
        if project.is_environment(path) or project._under(path, real):
            continue
        strays.append(f"{name} from {path}")
    if strays:
        return (
            "these modules were imported from outside the source tree, so this "
            "worker would not be running the main process's code:\n  "
            + "\n  ".join(sorted(strays))
            + f"\nExpected everything under {real}."
        )
    return ""


def _accept(python: str, module: str, qualname: str, function_hash: str) -> str:
    """Return "" if this process should run the function, else the reason."""
    from .functionhash import PYTHON, reachable_set

    if PYTHON != python:
        return f"main process runs Python {python}, this worker runs {PYTHON}"
    if module == "__main__":
        return (
            "the function is defined in __main__; a worker cannot import a "
            "main script. Move it to a module and import it."
        )
    try:
        fn = _resolve(module, qualname)
    except Exception as e:
        return f"cannot import {module}:{qualname} here ({e})"
    try:
        theirs = reachable_set(fn).hash
    except Exception as e:
        return f"cannot hash {module}:{qualname} here ({e})"
    if theirs != function_hash:
        return (
            f"{module}:{qualname} differs here: main process has {function_hash[:12]}, "
            f"this worker has {theirs[:12]}: the code differs."
        )
    return ""


def _handshake(rx: BinaryIO, tx: BinaryIO, check_imports: bool) -> tuple[bool, str]:
    """Read the HELLO message, set the process up to run the function, and reply.

    Returns whether to go on and the function's module and qualname joined
    by a colon, for :func:`serve` to resolve.
    """
    message = protocol.read_message(rx)
    if message is None:
        return False, ""  # the main process went away before saying anything
    tag, body = message
    if tag != protocol.HELLO:
        protocol.write_message(tx, protocol.ACCEPTED, b"expected a HELLO message")
        return False, ""
    try:
        hello = json.loads(body)
        python, module, qualname = hello["python"], hello["module"], hello["qualname"]
        function_hash, project_hash = hello["function_hash"], hello["project_hash"]
        extensions, roots = hello["extensions"], [r for r in hello["roots"] if r]
    except (ValueError, KeyError, TypeError) as e:
        protocol.write_message(tx, protocol.ACCEPTED, f"malformed HELLO message: {e!r}".encode())
        return False, ""

    source, reason = _tree(project_hash)
    if not reason:
        # The tree goes on the path before _accept, which imports.
        if source is not None:
            _install(source, roots)
        _take_markers(extensions)
        reason = _accept(python, module, qualname, function_hash)
    if not reason and check_imports and source is not None:
        # After the import, and before the function hash is compared: a
        # function_hash that matches the wrong file is still the wrong file.
        reason = _check_imports(source)
    protocol.write_message(tx, protocol.ACCEPTED, reason.encode("utf-8"))
    return not reason, f"{module}:{qualname}"


def serve_check(rx: BinaryIO, tx: BinaryIO) -> int:
    """Once per host: import the function from the tree and check it."""
    ok, _ = _handshake(rx, tx, check_imports=True)
    return 0 if ok else 1


def serve(rx: BinaryIO, tx: BinaryIO) -> int:
    """Run one task off *rx*, reporting on *tx*."""
    ok, target = _handshake(rx, tx, check_imports=False)
    if not ok:
        return 0 if not target else 1
    module, qualname = target.split(":", 1)

    from .pure import _current_store, set_store

    store = RemoteStore(rx, tx)
    previous = _current_store()  # serve() runs in-process in tests
    set_store(store)
    try:
        root = None
        while True:
            message = protocol.read_message(rx)
            if message is None:
                return 0  # cancelled before the task arrived
            tag, body = message
            if tag == protocol.OBJECT:
                store.receive(body)
            elif tag == protocol.TASK:
                root = body.hex()
                break
            else:
                raise protocol.ProtocolError(f"unexpected message {tag!r}")

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
        protocol.write_message(tx, protocol.RESULT, b"o" + bytes.fromhex(out))
        return 0
    finally:
        set_store(previous)


def _send_error(tx: BinaryIO, kind: str, text: str, tb: str = "") -> None:
    protocol.write_message(tx, protocol.RESULT, b"e" + protocol.strings(kind, text, tb))


def main(argv: list[str] | None = None) -> int:
    # Declared here rather than in serve(): the role is a fact about this
    # process, and serve() is also called in-process by tests, which must
    # not relabel their own caller.
    events.set_role("worker")
    args = sys.argv[1:] if argv is None else argv
    rx, tx = sys.stdin.buffer, sys.stdout.buffer
    try:
        return serve_check(rx, tx) if "--check" in args else serve(rx, tx)
    except protocol.ProtocolError:
        return 2
    finally:
        events._flush()


if __name__ == "__main__":
    raise SystemExit(main())
