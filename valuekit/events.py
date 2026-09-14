"""The event log: a live record of what a pipeline is doing.

The cache's whole promise is that it can stay on, but nothing about it is
visible from outside: a step that should be hitting and silently is not
looks exactly like one that is.  This module writes what happened -- hits,
misses, forced runs, batch outcomes -- so that another process can watch a
run in progress.  See :mod:`valuekit.monitor` for the reader.

Records go in the configured store directory, under ``events/``.  That is the
whole configuration: the store directory is where valuekit writes, and
nothing is written until one is named, so importing valuekit still has no
effect on its own.  A batch run with no store directory is therefore
unobservable, which is the price of not inventing a second location.

One file per process, ``events/<start>-<pid>.jsonl``, never appended to by
two processes: concurrent appends to a shared file are exactly what does
not work on Windows.  A spawned worker derives its own path from the store
directory it is already given, so there is nothing extra to pass it.

The event log is a diagnostic, never a dependency.  Every failure here is
swallowed: an unwritable directory, a full disk or a serialisation problem
disables the log for the process and changes nothing else.  Writes are
buffered and flushed on an interval rather than per record, so a cache hit
does not cost a syscall; a reader may therefore be a fraction of a second
behind, which is the right trade.

The file is capped.  A pipeline doing millions of hits stops appending
detail once it reaches the cap and counts how many records it dropped,
rather than filling the disk.  Old event files are pruned when a new one is
opened.

The events.  Every record is a JSON object with ``ev``, the event's name,
and ``t``, the time it was written; its other fields are these, and
every field listed is always present.  ``SCHEMA_VERSION`` is the version
of this table.

``process``
    First in every file.  ``v`` (the schema version), ``pid``, ``role``
    (``main`` or ``worker``), ``argv``, ``cwd``.
``hit``, ``miss``, ``forced``, ``error``
    One memoised call, written by :mod:`valuekit.pure`.  ``fn`` is the
    function's qualified name and ``function_hash`` its function hash.
    ``dur`` is the seconds the lookup took, on a hit and on a miss.  A
    miss also has ``exec``, the seconds the body ran, and ``stored``,
    whether a call record was written.  An error has ``exc``, the
    exception's type name.  A forced run has neither duration.
``batch``
    A :func:`valuekit.run_all` call begins.  ``id`` numbers the batch
    within the process, ``fn`` is the function's qualified name, ``name``
    the batch's name, ``n`` the number of inputs, ``mode`` ``parallel`` or
    ``sequential`` (a debugger forced the batch to run in this process).
``start``, ``requeue``, ``outcome``
    One input of a batch: ``id`` the batch, ``i`` the input's index,
    ``host`` where it ran (a host's name, ``local`` for a worker on this
    machine, ``main`` for this process).  ``start`` when the input is
    given to a host; ``requeue`` when its host dropped before it finished,
    so it will start again elsewhere; ``outcome`` when it finished, with
    ``ok`` and, when not ok, ``exc``.
``end``
    The batch ``id`` is over, whether it completed or was interrupted.
``host``
    A remote host's sync finished, or its connection dropped mid-batch:
    ``id`` the batch, ``name`` the host, ``ok``, ``reason`` (None when
    ok), ``capacity``.
``truncated``
    The file reached its cap; ``dropped`` counts the records not written.

An event a remote worker wrote reaches this log through the main
process, which adds ``host``, the host's name, to it.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["record", "events_dir"]

SCHEMA_VERSION = 1

_MAX_BYTES = 32 << 20  # per event file; then detail stops and drops are counted
_MAX_FILES = 50  # event files retained in a directory
_MAX_AGE = 7 * 24 * 3600  # seconds
_FLUSH_INTERVAL = 0.25  # seconds between flushes


_role_override: str | None = None


def set_role(role: str) -> None:
    """Declare this process a "main" or a "worker".

    A worker that was not started by :mod:`multiprocessing` cannot be
    recognised by inspecting the process tree, so one says so instead.  Must
    be called before the first record, since the role is recorded in the
    process header.
    """
    global _role_override
    _role_override = role


def _role() -> str:
    """"main" or "worker".

    A batch runs one worker per input, each recording its own file; without
    this a twelve-input batch reads as thirteen runs.  The multiprocessing
    check covers spawned workers, which do not announce themselves; anything
    else has to call :func:`set_role`.  Imported here rather than at module
    scope to keep this module cheap for :mod:`valuekit.pure`.
    """
    if _role_override is not None:
        return _role_override
    try:
        import multiprocessing

        return "main" if multiprocessing.parent_process() is None else "worker"
    except Exception:
        return "main"


def events_dir(store: Any) -> Path | None:
    """The runs directory of *store*, or None if it has no directory.

    Duck-typed on ``.root`` rather than importing LocalStore: this module is
    imported by :mod:`valuekit.pure` and must not import it back.
    """
    root = getattr(store, "root", None)
    return None if root is None else Path(root) / "events"


class _Writer:
    """Appends records to one file, owned by one process.

    Never raises.  On any failure it disables itself and subsequent writes
    are no-ops.
    """

    __slots__ = ("_fh", "_bytes", "_dropped", "_last_flush", "_disabled")

    def __init__(self, directory: Path):
        self._fh = None
        self._bytes = 0
        self._dropped = 0
        self._last_flush = 0.0
        self._disabled = False
        try:
            directory.mkdir(parents=True, exist_ok=True)
            prune(directory)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            path = directory / f"{stamp}-{os.getpid()}.jsonl"
            self._fh = open(path, "a", encoding="utf-8")
        except Exception:
            self._disabled = True
            return
        self.write(
            "process",
            v=SCHEMA_VERSION,
            pid=os.getpid(),
            role=_role(),
            argv=sys.argv,
            cwd=os.getcwd(),
        )

    def write(self, ev: str, **fields: Any) -> None:
        if self._disabled:
            return
        if self._bytes >= _MAX_BYTES:
            self._dropped += 1
            return
        try:
            record = {"ev": ev, "t": time.time()}
            record.update(fields)
            line = json.dumps(record, default=_unrepresentable) + "\n"
            self._fh.write(line)  # type: ignore[union-attr]
            self._bytes += len(line)
            now = time.monotonic()
            if now - self._last_flush >= _FLUSH_INTERVAL:
                self._fh.flush()  # type: ignore[union-attr]
                self._last_flush = now
        except Exception:
            self.close()
            self._disabled = True

    def flush(self) -> None:
        try:
            if self._fh is not None:
                self._fh.flush()
                self._last_flush = time.monotonic()
        except Exception:
            pass

    def close(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if self._dropped:
                fh.write(
                    json.dumps(
                        {"ev": "truncated", "t": time.time(), "dropped": self._dropped}
                    )
                    + "\n"
                )
            fh.close()
        except Exception:
            pass


# The writer for the store currently in use.  set_store_dir may point
# somewhere else mid-process, so the root is checked on every record -- but it
# is compared as given, never rebuilt: deriving the path here instead cost
# more than everything else in this function put together.
_writer: _Writer | None = None
_writer_root: Any = None


def record(store: Any, ev: str, **fields: Any) -> None:
    """Write one record to *store*'s event log.

    A no-op when the store has no directory.  Never raises: a diagnostic
    that can break a pipeline is worse than no diagnostic.
    """
    global _writer, _writer_root
    try:
        emit = getattr(store, "emit", None)
        if emit is not None:
            # A worker whose store is the main process's: the record goes there,
            # into the main process's own event file.
            record = {"ev": ev, "t": time.time()}
            record.update(fields)
            emit(json.loads(json.dumps(record, default=_unrepresentable)))
            return
        root = getattr(store, "root", None)
        if root is None:
            return
        if _writer is None or root != _writer_root:
            if _writer is not None:
                _writer.close()
            _writer = _Writer(Path(root) / "events")
            _writer_root = root
        _writer.write(ev, **fields)
    except Exception:
        pass


def _unrepresentable(obj: Any) -> str:
    """Last resort for a field json cannot encode: describe, never fail."""
    try:
        return repr(obj)[:200]
    except Exception:
        return f"<{type(obj).__name__}>"


def prune(directory: Path) -> None:
    """Drop event files that are old or surplus. Best effort."""
    try:
        files = sorted(
            (p for p in directory.glob("*.jsonl")),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    now = time.time()
    keep: list[Path] = []
    for p in files:
        try:
            if now - p.stat().st_mtime > _MAX_AGE:
                p.unlink(missing_ok=True)
            else:
                keep.append(p)
        except OSError:
            pass
    for p in keep[: max(0, len(keep) - _MAX_FILES + 1)]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _flush() -> None:
    """Push buffered records to disk without closing the file.

    Writes are batched on an interval so that a cache hit costs no syscall,
    which means a reader is normally a fraction of a second behind. A caller
    that must see everything now -- a test, above all -- asks here.
    """
    if _writer is not None:
        _writer.flush()


def _close() -> None:
    global _writer, _writer_root
    if _writer is not None:
        _writer.close()
        _writer = None
        _writer_root = None


atexit.register(_close)
