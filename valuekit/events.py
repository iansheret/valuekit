"""Run events: a live record of what a pipeline is doing.

The cache's whole promise is that it can stay on, but nothing about it is
visible from outside: a step that should be hitting and silently is not
looks exactly like one that is.  This module writes what happened -- hits,
misses, forced runs, batch outcomes -- so that another process can watch a
run in progress.  See :mod:`valuekit.monitor` for the reader.

Events go in the configured cache directory, under ``runs/``.  That is the
whole configuration: the cache directory is where valuekit writes, and
nothing is written until one is named, so importing valuekit still has no
effect on its own.  A batch run with no cache directory is therefore
unobservable, which is the price of not inventing a second location.

One file per process, ``runs/<start>-<pid>.jsonl``, never appended to by
two processes: concurrent appends to a shared file are exactly what does
not work on Windows.  A spawned worker derives its own path from the cache
directory it is already given, so there is nothing extra to pass it.

Events are a diagnostic, never a dependency.  Every failure here is
swallowed: an unwritable directory, a full disk or a serialisation problem
disables emission for the process and changes nothing else.  Writes are
buffered and flushed on an interval rather than per event, so a cache hit
does not cost a syscall; a reader may therefore be a fraction of a second
behind, which is the right trade.

The file is capped.  A pipeline doing millions of hits stops appending
detail once it reaches the cap and records how many events it dropped,
rather than filling the disk.  Old run files are reaped when a new one is
opened.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["emit", "runs_dir"]

SCHEMA_VERSION = 1

_MAX_BYTES = 32 << 20  # per run file; then detail stops and drops are counted
_MAX_RUNS = 50  # run files kept in a directory
_MAX_AGE = 7 * 24 * 3600  # seconds
_FLUSH_INTERVAL = 0.25  # seconds between flushes


def _role() -> str:
    """"driver" or "worker".

    A batch spawns one process per input, each of which records its own
    file; without this a twelve-input batch reads as thirteen runs.  Imported
    here rather than at module scope to keep this module cheap for
    :mod:`valuekit.pure`, which imports it.
    """
    try:
        import multiprocessing

        return "driver" if multiprocessing.parent_process() is None else "worker"
    except Exception:
        return "driver"


def runs_dir(store: Any) -> Path | None:
    """The runs directory of *store*, or None if it has no directory.

    Duck-typed on ``.root`` rather than importing LocalStore: this module is
    imported by :mod:`valuekit.pure` and must not import it back.
    """
    root = getattr(store, "root", None)
    return None if root is None else Path(root) / "runs"


class _Emitter:
    """Appends events to one file, owned by one process.

    Never raises.  On any failure it sets itself dead and subsequent emits
    are no-ops.
    """

    __slots__ = ("_fh", "_bytes", "_dropped", "_last_flush", "_dead")

    def __init__(self, directory: Path):
        self._fh = None
        self._bytes = 0
        self._dropped = 0
        self._last_flush = 0.0
        self._dead = False
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _reap(directory)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            path = directory / f"{stamp}-{os.getpid()}.jsonl"
            self._fh = open(path, "a", encoding="utf-8")
        except Exception:
            self._dead = True
            return
        self.emit(
            "run",
            v=SCHEMA_VERSION,
            pid=os.getpid(),
            role=_role(),
            argv=sys.argv,
            cwd=os.getcwd(),
        )

    def emit(self, ev: str, **fields: Any) -> None:
        if self._dead:
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
            self._dead = True

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


# The emitter for the store currently in use.  set_cache_dir may point
# somewhere else mid-process, so the root is checked on every emit -- but it
# is compared as given, never rebuilt: deriving the path here instead cost
# more than everything else in this function put together.
_emitter: _Emitter | None = None
_emitter_root: Any = None


def emit(store: Any, ev: str, **fields: Any) -> None:
    """Record one event against *store*'s runs directory.

    A no-op when the store has no directory.  Never raises: a diagnostic
    that can break a pipeline is worse than no diagnostic.
    """
    global _emitter, _emitter_root
    try:
        root = getattr(store, "root", None)
        if root is None:
            return
        if _emitter is None or root != _emitter_root:
            if _emitter is not None:
                _emitter.close()
            _emitter = _Emitter(Path(root) / "runs")
            _emitter_root = root
        _emitter.emit(ev, **fields)
    except Exception:
        pass


def _unrepresentable(obj: Any) -> str:
    """Last resort for a field json cannot encode: describe, never fail."""
    try:
        return repr(obj)[:200]
    except Exception:
        return f"<{type(obj).__name__}>"


def _reap(directory: Path) -> None:
    """Drop run files that are old or surplus. Best effort."""
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
    for p in keep[: max(0, len(keep) - _MAX_RUNS + 1)]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _flush() -> None:
    """Push buffered events to disk without closing the file.

    Writes are batched on an interval so that a cache hit costs no syscall,
    which means a reader is normally a fraction of a second behind. A caller
    that must see everything now -- a test, above all -- asks here.
    """
    if _emitter is not None:
        _emitter.flush()


def _close() -> None:
    global _emitter, _emitter_root
    if _emitter is not None:
        _emitter.close()
        _emitter = None
        _emitter_root = None


atexit.register(_close)
