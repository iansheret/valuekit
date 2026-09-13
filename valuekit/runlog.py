"""The run's log: the values a run logged, read back by their labels.

``log(labels, value)`` records *value* under a small mapping that says
what it is (``{"quantity": "residuals", "sid": 7}``).  valuekit gives no
key any meaning; labels are whatever identifies the observation to the
code that will read it.  Retrieval selects by containment::

    sel = valuekit.logs("physics").where(quantity="residuals")
    residuals = sel.where(sid=7).one().value
    for logged in sel:
        logged.labels, logged.value

A *run* is one main process running a script, named by the script's
file stem.  Its log is the complete set of logged values that run
produced, as if the code had run from scratch: a step that executes
writes each logged value as it makes it, and a step served from cache
writes one line naming its call record, which holds what the step
logged, nested calls included.  A new run under the same name replaces
the last, so the log never carries a value from an earlier run of the
script, and the main script and a debugging script never touch each
other's.  Every emission is its own logged value: the same labels and
value logged twice are two.

The log is as current as the cache.  A logged value is the output of a
pure function, so whatever removes call records, ``clear_cache()`` or
the sweep, removes the logged values that came with them; a line naming
a call record that is gone reads as stale, and ``logs()`` says so.

Layout, under the store directory::

    logs/<name>/latest                # the id of the newest run
    logs/<name>/<run-id>/header.json
    logs/<name>/<run-id>/<pid>.jsonl  # one file per writing process

Two kinds of line.  An entry names the labels and the value by hash, both
in the object store, and carries the labels' entries in encoded form so a
query needs no decoding.  A reference names a call record by function
hash and record hash.  Workers on this machine write their own file into
the main process's run; a remote worker sends its lines to the main
process, which writes them.  Nothing here imports or runs the pipeline.
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
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from .map import ImmutableMap
from .store import CacheMiss, LocalStore, _atomic_write, dirname_for
from .values import content_hash, encode_key, freeze

__all__ = ["logs", "Logs", "Selection", "LoggedValue"]

LOG_VERSION = 2


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
    """The run this process writes to: begun here (a main process) or
    joined (a worker the main process told which one it belongs to)."""

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
_begin_lock = threading.Lock()

# The main process names its run in its own environment, so the workers it
# spawns on this machine, which inherit that environment, write into the
# same run rather than beginning their own.  The main process's pid is in
# the value so that the process which set it never reads it back as a
# child would.
_RUN_ENV = "VALUEKIT_RUN"


def _inherited_run() -> tuple[str, str] | None:
    """The ``(name, id)`` a parent process named, if this is its child."""
    value = os.environ.get(_RUN_ENV, "")
    pid, _, rest = value.partition(":")
    run_id, _, name = rest.partition(":")
    if not pid or pid == str(os.getpid()) or not run_id:
        return None
    return name, run_id


def current_run(store: Any) -> _Run | None:
    """The run this process writes to for *store*, begun if needed.

    None for a store without a directory (a worker whose store is the
    main process's sends its lines there instead).  A process spawned by
    a main process writes into that process's run, named in the
    environment; any other process begins its own: writes the header,
    points ``latest`` at it and removes the older runs under the same name.
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
        inherited = _inherited_run()
        run = _Run(root, *inherited) if inherited else _begin(root)
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
    os.environ[_RUN_ENV] = f"{os.getpid()}:{run_id}:{name}"
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
    is the main process's, the line goes to the main process."""
    line = {"labels": labels_hash, "v": value_hash, "k": keys, "t": time.time()}
    send = getattr(store, "emit_line", None)
    if send is not None:
        send(json.dumps(line, separators=(",", ":")))
        return
    run = current_run(store)
    if run is not None:
        run.write(line)


def refer(store: Any, function_hash: str, h: str) -> None:
    """Write a reference to call record *h* of *function_hash* to this
    run's log: a hit's line, standing for what the record holds."""
    line = {"record": [function_hash, h], "t": time.time()}
    send = getattr(store, "emit_line", None)
    if send is not None:
        send(json.dumps(line, separators=(",", ":")))
        return
    run = current_run(store)
    if run is not None:
        run.write(line)


def write_line(store: Any, line: str) -> None:
    """Internal: a line a remote worker sent; the main process writes it."""
    d = json.loads(line)
    run = current_run(store)
    if run is None:
        return
    if "record" in d:
        run.write({"record": list(d["record"]), "t": d.get("t", time.time())})
    else:
        run.write({"labels": d["labels"], "v": d["v"], "k": d["k"], "t": d.get("t", time.time())})


def expand(store: LocalStore, record: dict, stale: list[str]) -> list[list]:
    """Every ``[labels, value, keys]`` entry a call record and the calls
    nested in it hold, in call-record order.  A nested call record that
    cannot be read is skipped, and its hash added to *stale*."""
    out: list[list] = []

    def walk(t: dict) -> None:
        for _, key, h in t.get("calls", []):
            try:
                sub = store.get_record(key, h)
            except CacheMiss:
                stale.append(h)
                continue
            walk(sub)
        out.extend(t.get("logs", []))

    walk(record)
    return out


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


class LoggedValue:
    """One logged value with its labels.  Both load from the store when
    asked for; arrays arrive as memory maps."""

    __slots__ = ("_store", "_labels_hash", "_v", "_k")

    def __init__(self, store: LocalStore, labels_hash: str, v: str, keys: dict[str, str]):
        self._store = store
        self._labels_hash = labels_hash
        self._v = v
        self._k = keys

    @property
    def labels(self) -> ImmutableMap:
        return self._store.get_value(self._labels_hash)

    @property
    def value(self) -> Any:
        return self._store.get_value(self._v)

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

    def __init__(self, items: list[LoggedValue], asked: dict | None = None):
        self._items = items
        self._asked = asked or {}

    def where(self, mapping: Mapping | None = None, /, **kw: Any) -> Selection:
        want = _query(mapping, kw)
        asked = {**self._asked, **(dict(mapping) if mapping else {}), **kw}
        kept = [
            it for it in self._items
            if all(it._k.get(k) == v for k, v in want.items())
        ]
        return Selection(kept, asked)

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


def selection(store: LocalStore, record: dict) -> Selection:
    """The logged values a call record holds, nested calls included.  A
    nested call record that is gone is left out."""
    stale: list[str] = []
    return Selection([LoggedValue(store, *entry) for entry in expand(store, record, stale)])


class Logs(Selection):
    """One run's log, read from its directory.  ``refresh()`` picks up
    lines written since it was opened.  ``stale`` counts the references
    read so far whose call record, or a record nested in it, is gone."""

    def __init__(self, store: LocalStore, name: str, path: Path):
        super().__init__([])
        self._store = store
        self._name = name
        self._path = path
        self._offsets: dict[Path, int] = {}
        self.stale = 0
        self.refresh()

    def refresh(self) -> None:
        try:
            files = sorted(self._path.glob("*.jsonl"))
        except OSError:
            return
        stale: list[str] = []
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
                    if "record" in d:
                        self._expand(*d["record"], stale)
                    else:
                        self._items.append(LoggedValue(self._store, d["labels"], d["v"], d["k"]))
                except (ValueError, KeyError, TypeError):
                    continue
            self._offsets[p] = start + end
        if stale:
            self.stale += len(stale)
            warnings.warn(
                f"{len(stale)} logged calls of {self._name!r} are no longer in the "
                f"cache; run the script again",
                stacklevel=2,
            )

    def _expand(self, function_hash: str, h: str, stale: list[str]) -> None:
        try:
            record = self._store.get_record(function_hash, h)
        except CacheMiss:
            stale.append(h)
            return
        missing: list[str] = []
        for entry in expand(self._store, record, missing):
            self._items.append(LoggedValue(self._store, *entry))
        if missing:
            stale.append(h)

    def __repr__(self) -> str:
        return f"Logs({len(self._items)} logged values)"


def logs(name: str | None = None, store_dir: str | os.PathLike | None = None) -> Logs:
    """Open the newest run's log.

    *name* is the script's file stem; with one script logged in the
    store it may be omitted.  *store_dir* defaults to the configured one.
    Raises ``LookupError`` when nothing has been logged under that name.
    """
    if store_dir is None:
        from .pure import _current_store

        store = _current_store()
        if not isinstance(store, LocalStore):
            raise LookupError("no store directory is configured; pass store_dir=")
    else:
        store = LocalStore(store_dir)
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
    return Logs(store, name, d / run_id)
