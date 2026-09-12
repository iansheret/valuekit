"""The run's log: the values a run logged, read back by their labels.

``log(labels, value)`` records *value* under a small mapping that says
what it is (``{"quantity": "residuals", "sid": 7}``).  valuekit gives no
key any meaning; labels are whatever identifies the observation to the
code that will read it.  Retrieval selects by containment::

    sel = valuekit.logs("physics").where(quantity="residuals")
    residuals = sel.where(sid=7).one().value
    for logged in sel:
        logged.labels, logged.value

A *run* is one driver process running a script, named by the script's
file stem.  Its log is the complete set of logged values that run
produced, as if the code had run from scratch: a step that executes
writes its logged values as it makes them, and a step served from cache
writes the ones its call record holds, nested calls included.  A new run
under the same name replaces the last, so the log never carries a value
from an earlier run of the script, and the main script and a debugging
script never touch each other's.  Every emission is its own logged
value: the same labels and value logged twice are two.

Layout, under the cache directory::

    logs/<name>/latest                # the id of the newest run
    logs/<name>/<run-id>/header.json
    logs/<name>/<run-id>/<pid>.jsonl  # one file per writing process

A line names the labels and the value by hash, both in the object store,
and carries the labels' entries in encoded form so a query needs no
decoding.  Workers on this machine write their own file into the driver's
run; a remote worker sends its lines to the driver, which writes them.
Nothing here imports or runs the pipeline.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from .map import ImmutableMap
from .store import CacheMiss, LocalStore, _atomic_write, dirname_for
from .values import content_hash, encode_key, freeze

__all__ = ["logs", "Logs", "Selection", "LoggedValue"]

LOG_VERSION = 1


def logs_dir(store: LocalStore) -> Path:
    return store.root / "logs"


def label_hashes(labels: ImmutableMap) -> dict[str, str]:
    """The labels' entries as ``{encoded key: value hash}``: what a log
    line carries, and what a query is compared against."""
    return {encode_key(k).hex(): content_hash(v) for k, v in labels.items()}


# ---------------------------------------------------------------------------
# this process's run
# ---------------------------------------------------------------------------


def _script_name() -> str:
    """The stem of the script this process is running, or ``python``."""
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0 or argv0 == "-c":
        return "python"
    return Path(argv0).stem or "python"


class _Run:
    """The run this process writes to: begun here (a driver) or
    joined (a worker the driver told which one it belongs to)."""

    __slots__ = ("root", "name", "id", "dir", "_fh", "_lock")

    def __init__(self, root: Path, name: str, run_id: str):
        self.root = root
        self.name = name
        self.id = run_id
        self.dir = root / "logs" / dirname_for(name) / run_id
        self._fh = None
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with self._lock:
            if self._fh is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.dir / f"{os.getpid()}.jsonl", "a", encoding="utf-8")
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None


_current: _Run | None = None
_adopted: tuple[str, str] | None = None  # (name, id) handed down by the driver
_begin_lock = threading.Lock()


def adopt(name: str, run_id: str) -> None:
    """Internal: a worker joins the driver's run instead of beginning
    its own.  Must be called before the worker's first memoised call."""
    global _adopted
    _adopted = (name, run_id)


def current() -> tuple[str, str] | None:
    """Internal: ``(name, id)`` of this process's run, if begun."""
    return None if _current is None else (_current.name, _current.id)


def current_run(store: Any) -> _Run | None:
    """The run this process writes to for *store*, begun if needed.

    None for a store without a directory (a worker whose store is the
    driver's sends its lines there instead).  Beginning a run
    writes its header, points ``latest`` at it and removes the older
    runs under the same name; a worker joins without any of that.
    """
    global _current
    root = getattr(store, "root", None)
    if root is None:
        return None
    run = _current
    if run is not None and run.root == root:
        return run
    with _begin_lock:
        run = _current
        if run is not None and run.root == root:
            return run
        if run is not None:
            run.close()
        if _adopted is not None:
            run = _Run(root, *_adopted)
        else:
            run = _begin(root)
        _current = run
        return run


def _begin(root: Path) -> _Run:
    name = _script_name()
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    run = _Run(root, name, run_id)
    header = {
        "v": LOG_VERSION,
        "name": name,
        "script": sys.argv[0] if sys.argv else "",
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "started": time.time(),
    }
    _atomic_write(run.dir / "header.json", json.dumps(header).encode())
    _atomic_write(run.dir.parent / "latest", run_id.encode())
    for old in list(run.dir.parent.iterdir()):
        if old.is_dir() and old.name != run_id:
            shutil.rmtree(old, ignore_errors=True)
    return run


def _close() -> None:
    if _current is not None:
        _current.close()


atexit.register(_close)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def emit(store: Any, labels_hash: str, value_hash: str, keys: dict[str, str]) -> None:
    """Write one logged value to this run's log.  In a worker whose store
    is the driver's, the line goes to the driver."""
    line = {"labels": labels_hash, "v": value_hash, "k": keys, "t": time.time()}
    send = getattr(store, "emit_line", None)
    if send is not None:
        send(json.dumps(line, separators=(",", ":")))
        return
    run = current_run(store)
    if run is not None:
        run.write(line)


def write_line(store: Any, line: str) -> None:
    """Internal: a line a remote worker sent; the driver writes it."""
    d = json.loads(line)
    run = current_run(store)
    if run is not None:
        run.write({"labels": d["labels"], "v": d["v"], "k": d["k"], "t": d.get("t", time.time())})


def collect(store: LocalStore, record: dict, strict: bool = True) -> list[list]:
    """Every ``[labels, value, keys]`` entry a call record and the calls
    nested in it hold, in call-record order.

    *strict* raises :class:`CacheMiss` when any nested call record, labels or
    value is gone, which is what a hit needs: it may only stand in for the
    call if it can emit everything the call would have.  Lenient, the
    missing parts are skipped.
    """
    out: list[list] = []

    def walk(t: dict) -> None:
        for _, key, h in t.get("calls", []):
            try:
                sub = store.get_record(key, h)
            except CacheMiss:
                if strict:
                    raise
                continue
            walk(sub)
        for entry in t.get("logs", []):
            labels, v = entry[0], entry[1]
            if store._find(labels) is None or store._find(v) is None:
                if strict:
                    raise CacheMiss(v)
                continue
            out.append(entry)

    walk(record)
    return out


def reemit(store: Any, function_hash: str, h: str, record: dict) -> None:
    """Emit again what a hit's call record holds, nested calls included.

    Raises :class:`CacheMiss` if the subtree cannot be read whole, and then
    emits nothing: the caller treats the hit as a miss.  In a remote
    worker the driver walks its own store and answers.
    """
    remote = getattr(store, "reemit", None)
    if remote is not None:
        remote(function_hash, h)
        return
    entries = collect(store, record)
    if not entries:
        return
    run = current_run(store)
    if run is None:
        return
    now = time.time()
    for labels, v, keys in entries:
        run.write({"labels": labels, "v": v, "k": keys, "t": now})


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


class LoggedValue:
    """One logged value with its labels.  Both load when asked for; arrays
    arrive as memory maps."""

    __slots__ = ("_logs", "_labels_hash", "_v", "_k")

    def __init__(self, logs: Logs, ctx: str, v: str, keys: dict[str, str]):
        self._logs = logs
        self._labels_hash = ctx
        self._v = v
        self._k = keys

    @property
    def labels(self) -> ImmutableMap:
        return self._logs._labels(self._labels_hash)

    @property
    def value(self) -> Any:
        return self._logs._store.get_value(self._v)

    def __repr__(self) -> str:
        try:
            return f"LoggedValue({dict(self.labels)!r})"
        except CacheMiss:
            return f"LoggedValue(<labels {self._labels_hash[:8]} unreadable>)"


def _query(mapping: Mapping | None, kw: dict) -> dict[str, str]:
    """A query in the log's encoded form.  Values are frozen first, so
    a dict in a query matches the map it became when it was logged."""
    q: dict = {}
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise TypeError(f"where() takes a mapping, got {type(mapping).__name__}")
        q.update(mapping)
    q.update(kw)
    return {encode_key(k).hex(): content_hash(freeze(v)) for k, v in q.items()}


class Selection:
    """Logged values whose labels contain every key/value pair asked for.
    Iterable; ``one()`` for a selection expected to hold exactly one;
    ``where()`` narrows further.  Order carries no meaning."""

    def __init__(self, logs: Logs, items: list[LoggedValue], asked: dict | None = None):
        self._logs = logs
        self._items = items
        self._asked = asked or {}

    def where(self, mapping: Mapping | None = None, /, **kw: Any) -> Selection:
        want = _query(mapping, kw)
        asked = {**self._asked, **(dict(mapping) if mapping else {}), **kw}
        kept = [
            it for it in self._items
            if all(it._k.get(k) == v for k, v in want.items())
        ]
        return Selection(self._logs, kept, asked)

    def one(self) -> LoggedValue:
        n = len(self._items)
        if n == 1:
            return self._items[0]
        asked = f" for {self._asked!r}" if self._asked else ""
        if n == 0:
            raise LookupError(f"no logged value{asked}")
        shown = ", ".join(repr(dict(it.labels)) for it in self._items[:5])
        more = f", and {n - 5} more" if n > 5 else ""
        raise LookupError(f"{n} logged values{asked}, expected one: {shown}{more}")

    def __iter__(self) -> Iterator[LoggedValue]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        asked = f" where {self._asked!r}" if self._asked else ""
        return f"Selection({len(self._items)} logged values{asked})"


class Logs(Selection):
    """What one run logged, or what one batch's call records hold.
    ``refresh()`` picks up lines written since it was opened."""

    def __init__(self, store: LocalStore, path: Path | None = None, items: list | None = None):
        super().__init__(self, [])
        self._store = store
        self._path = path
        self._offsets: dict[Path, int] = {}
        self._contexts: dict[str, ImmutableMap] = {}
        for entry in items or []:
            self._items.append(LoggedValue(self, entry[0], entry[1], entry[2]))
        if path is not None:
            self.refresh()

    def _labels(self, h: str) -> ImmutableMap:
        ctx = self._contexts.get(h)
        if ctx is None:
            ctx = self._contexts[h] = self._store.get_value(h)
        return ctx

    def refresh(self) -> None:
        if self._path is None:
            return
        try:
            files = sorted(self._path.glob("*.jsonl"))
        except OSError:
            return
        for p in files:
            start = self._offsets.get(p, 0)
            try:
                with open(p, "rb") as fh:
                    fh.seek(start)
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1  # a line still being written is left for next time
            for raw in data[:end].splitlines():
                try:
                    d = json.loads(raw)
                    self._items.append(LoggedValue(self, d["labels"], d["v"], d["k"]))
                except (ValueError, KeyError, TypeError):
                    continue
            self._offsets[p] = start + end

    def __repr__(self) -> str:
        return f"Logs({len(self._items)} logged values)"


def logs(name: str | None = None, cache_dir: str | os.PathLike | None = None) -> Logs:
    """Open the newest run's log.

    *name* is the script's file stem; with one script logged under the
    cache it may be omitted.  *cache_dir* defaults to the configured one.
    Raises ``LookupError`` when nothing has been logged under that name.
    """
    if cache_dir is None:
        from .pure import _current_store

        store = _current_store()
        if not isinstance(store, LocalStore):
            raise LookupError("no cache directory is configured; pass cache_dir=")
    else:
        store = LocalStore(cache_dir)
    base = logs_dir(store)
    try:
        names = sorted(p.name for p in base.iterdir() if p.is_dir())
    except OSError:
        names = []
    if name is None:
        if len(names) == 1:
            name = names[0]
        elif not names:
            raise LookupError(f"nothing has been logged under {store.root}")
        else:
            raise LookupError(
                f"several scripts have logged under {store.root}: "
                f"{', '.join(names)}; name one"
            )
    d = base / dirname_for(name)
    try:
        run_id = (d / "latest").read_text().strip()
        if not (d / run_id / "header.json").exists():
            raise OSError(run_id)
    except OSError as e:
        raise LookupError(f"nothing has been logged under {name!r} in {store.root}") from e
    return Logs(store, d / run_id)
