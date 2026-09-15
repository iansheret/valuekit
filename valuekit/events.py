"""The event log: what a run did, for another process to watch.

This module writes hits, misses, forced runs, batch starts and outcomes,
and host states as they happen, so that :mod:`valuekit.monitor` can show a
run in progress.  A step that should hit and does not is otherwise
indistinguishable from a slow one.

Records go under ``events/`` in the store directory, one file per main
process, ``events/<start>-<pid>.jsonl``; a worker's records reach the main
process's file.  With no store directory nothing is written.

The event log is a diagnostic: every failure here is swallowed, and an
unwritable directory, a full disk or a serialisation problem disables the
log for the process and changes nothing else.  The file is capped: past
the cap, records are counted rather than written.  The oldest files
beyond a count are removed when a new one is opened.

The events.  Every record is a JSON object with ``ev``, the event's name,
and ``t``, the time it was written; its other fields are these, and
every field listed is always present.  ``SCHEMA_VERSION`` is the version
of this table.

``process``
    First in every file.  ``v`` (the schema version), ``pid``, ``argv``,
    ``cwd``.  A worker writes no file: its events reach the main
    process's.
``hit``, ``miss``, ``forced``, ``error``
    One memoised call, written by :mod:`valuekit.pure`.  ``fn`` is the
    function's qualified name and ``function_hash`` its function hash.
    ``dur`` is the seconds the lookup took, on a hit and on a miss.  A
    miss also has ``exec``, the seconds the body ran, and ``stored``,
    whether a call record was written.  An error has ``exc``, the
    exception's type name.  A forced run has neither duration.
``batch``
    A :func:`valuekit.run_all` call begins.  ``id`` numbers the batch
    within the process, ``fn`` is the function's qualified name, ``n``
    the number of inputs, ``mode`` ``parallel`` or
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

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["record", "open_log"]

SCHEMA_VERSION = 1

_MAX_BYTES = 32 << 20  # per event file; then detail stops and drops are counted
_MAX_FILES = 50  # event files retained in a directory


class _Writer:
    """Appends records to one file, owned by one process.

    Never raises.  On any failure it disables itself and subsequent writes
    are no-ops.
    """

    __slots__ = ("_fh", "_bytes", "_dropped", "_disabled")

    def __init__(self, directory: Path):
        self._fh = None
        self._bytes = 0
        self._dropped = 0
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
        self.write({
            "ev": "process", "t": time.time(), "v": SCHEMA_VERSION,
            "pid": os.getpid(), "argv": sys.argv, "cwd": os.getcwd(),
        })

    def write(self, record: dict) -> None:
        if self._disabled:
            return
        if self._bytes >= _MAX_BYTES:
            self._dropped += 1
            return
        try:
            line = json.dumps(record, default=_unrepresentable) + "\n"
            self._fh.write(line)  # type: ignore[union-attr]
            self._fh.flush()  # type: ignore[union-attr]
            self._bytes += len(line)
        except Exception:
            self.close()
            self._disabled = True

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


def open_log(store) -> None:
    """Open *store*'s event log for this process: one file under
    ``events/``.  Called when the store becomes the current one."""
    store.events = _Writer(Path(store.root) / "events")


def record(store: Any, ev: str, **fields: Any) -> None:
    """Write one record to *store*'s event log.

    A no-op with no store.  Never raises: a diagnostic that can break a
    pipeline is worse than no diagnostic.
    """
    if store is None:
        return
    try:
        store.event({"ev": ev, "t": time.time(), **fields})
    except Exception:
        pass


def _unrepresentable(obj: Any) -> str:
    """Last resort for a field json cannot encode: describe, never fail."""
    try:
        return repr(obj)[:200]
    except Exception:
        return f"<{type(obj).__name__}>"


def prune(directory: Path) -> None:
    """Drop the oldest event files beyond the count kept. Best effort."""
    try:
        keep = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for p in keep[: max(0, len(keep) - _MAX_FILES + 1)]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
