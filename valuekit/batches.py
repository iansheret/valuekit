"""Batch records: what a ``run_all`` produced, by name, for analysis code.

A trace holds everything a computation bound -- its result, the memoised
calls it made, its ``log()`` values -- but a trace is found by matching
arguments, and analysis code should not have to reconstruct arguments.  So
``run_all`` writes a small record naming its root traces, and this module
reads it back::

    b = valuekit.batch("process")     # the newest batch of process()
    b[7]                              # the row for input 7
    b[7]["detrend"]                   # what detrend returned inside it
    b[7]["residuals"]                 # a value the function log()ged
    b.column("rms")                   # one value per input, in input order
    b.by("order")                     # rows grouped by a logged parameter

Nothing here imports or runs the pipeline.  Staleness within a batch is
impossible: every input ran under one fingerprint, which the record
carries.  Across code changes the question is only "has this batch been
re-run since the edit", and the record's ``fingerprint`` answers it.

Layout, under the cache directory::

    batches/<name>/latest               # the id of the newest batch
    batches/<name>/<batch_id>/header.json
    batches/<name>/<batch_id>/<index>.json   # one per finished input

One file per outcome and nothing rewritten, so a reader opened mid-batch
sees the inputs finished so far and never a torn record.  ``latest`` is
the one mutable name, replaced atomically; older batch directories under
the same name are removed when a new batch starts.  The batch id has no
meaning beyond uniqueness.  Names are directory names: characters outside
``[A-Za-z0-9._-]`` are written as ``_``; the header keeps the real one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from .store import CacheMiss, LocalStore, SerializationError, _atomic_write
from .values import content_hash

__all__ = ["batch", "Batch", "Row", "BatchWriter"]

RECORD_VERSION = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _dirname(name: str) -> str:
    return _UNSAFE.sub("_", name) or "_"


def batches_dir(store: LocalStore) -> Path:
    return store.root / "batches"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


class BatchWriter:
    """Records one ``run_all`` as it happens.  Never raises after
    construction: a record that cannot be written is a diagnostic lost, not
    a batch failed."""

    def __init__(
        self,
        store: LocalStore,
        name: str,
        fn: str,
        fn_key: str,
        inputs: list,
    ):
        self.dir = batches_dir(store) / _dirname(name)
        self.id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = self.dir / self.id
        hashes: list[str | None] = []
        for x in inputs:
            try:
                hashes.append(store.put_value(x))
            except (SerializationError, TypeError, ValueError):
                hashes.append(None)  # hashable but not storable: shown as None
        header = {
            "v": RECORD_VERSION,
            "name": name,
            "fn": fn,
            "fn_key": fn_key,
            "n": len(inputs),
            "inputs": hashes,
            "started": time.time(),
        }
        _atomic_write(self.path / "header.json", json.dumps(header).encode())
        _atomic_write(self.dir / "latest", self.id.encode())
        for old in list(self.dir.iterdir()):
            if old.is_dir() and old.name != self.id:
                shutil.rmtree(old, ignore_errors=True)

    def outcome(self, index: int, trace: str | None, exc: BaseException | None) -> None:
        record: dict[str, Any] = {"i": index}
        if exc is None:
            record["trace"] = trace
        else:
            record["error"] = type(exc).__name__
            record["message"] = str(exc)
        try:
            _atomic_write(self.path / f"{index}.json", json.dumps(record).encode())
        except Exception:
            pass


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _short(qualname: str) -> str:
    return qualname.rsplit(".", 1)[-1]


class Row:
    """One computation: a root input of a batch, or a call nested in one.

    ``row[name]`` is the value bound to *name*: what the call of that name
    returned, or what ``log(name, ...)`` recorded, anywhere in this
    computation or the calls nested in it -- analysis code need not know
    which step logged a value.  A name bound more than once gives a list,
    in order.  ``row.calls`` narrows to one nested step when that matters.
    Values load when asked for, and arrays arrive as memory maps.
    """

    __slots__ = ("_store", "fn", "trace_hash", "_trace", "input", "index")

    def __init__(self, store, fn: str, trace_hash: str, trace: dict, input=None, index=None):
        self._store = store
        self.fn = fn
        self.trace_hash = trace_hash
        self._trace = trace
        self.input = input
        self.index = index

    def __repr__(self) -> str:
        where = f"[{self.index}]" if self.index is not None else ""
        return f"Row({self.fn}{where})"

    @property
    def result(self) -> Any:
        return self._store.get_value(self._trace["result"])

    @property
    def calls(self) -> list[Row]:
        out = []
        for qn, fn_key, h in self._trace.get("calls", []):
            try:
                out.append(Row(self._store, qn, h, self._store.get_trace(fn_key, h)))
            except CacheMiss:
                continue  # swept, or cleared: the call is no longer readable
        return out

    def names(self) -> list[str]:
        """Every name bound in this computation or any call nested in it,
        in order of first binding."""
        seen: dict[str, None] = {}
        self._collect_names(seen)
        return list(seen)

    def _collect_names(self, seen: dict) -> None:
        for call in self.calls:
            call._collect_names(seen)
            seen.setdefault(_short(call.fn), None)
        for name, _ in self._trace.get("logs", []):
            seen.setdefault(name, None)

    def _bindings(self, name: str) -> list:
        """Every value bound to *name* here or in a nested call, in the
        order the bindings were made: what a call bound inside itself comes
        before the call's own result, and a level's logs follow its calls."""
        found = []
        for call in self.calls:
            found.extend(call._bindings(name))
            if call.fn == name or _short(call.fn) == name:
                found.append(call.result)
        for logged, h in self._trace.get("logs", []):
            if logged == name:
                found.append(self._store.get_value(h))
        return found

    def __getitem__(self, name: str) -> Any:
        found = self._bindings(name)
        if not found:
            raise KeyError(f"{name!r} is not bound in {self!r}; names: {self.names()}")
        return found[0] if len(found) == 1 else found

    def get(self, name: str, default: Any = None) -> Any:
        found = self._bindings(name)
        if not found:
            return default
        return found[0] if len(found) == 1 else found


class Batch:
    """The newest ``run_all`` recorded under a name.  See the module."""

    def __init__(self, store: LocalStore, path: Path, header: dict):
        self._store = store
        self._path = path
        self.name: str = header["name"]
        self.fn: str = header["fn"]
        self.fingerprint: str = header["fn_key"]
        self.n: int = header["n"]
        self._input_hashes: list[str | None] = header["inputs"]
        self.started: float = header.get("started", 0.0)
        self._outcomes: dict[int, dict] = {}
        self.refresh()

    def refresh(self) -> None:
        """Pick up inputs that have finished since this object was made."""
        for p in self._path.glob("*.json"):
            if p.name == "header.json":
                continue
            try:
                i = int(p.stem)
            except ValueError:
                continue
            if i in self._outcomes:
                continue
            try:
                self._outcomes[i] = json.loads(p.read_bytes())
            except (OSError, ValueError):
                continue

    def __repr__(self) -> str:
        done = len(self._outcomes)
        return f"Batch({self.name!r}, {done}/{self.n} finished)"

    @property
    def inputs(self) -> list:
        """The batch's inputs, in order; None where one was not storable."""
        out = []
        for h in self._input_hashes:
            try:
                out.append(None if h is None else self._store.get_value(h))
            except CacheMiss:
                out.append(None)
        return out

    @property
    def complete(self) -> bool:
        return len(self._outcomes) >= self.n

    @property
    def failures(self) -> list[tuple[Any, str, str]]:
        """``(input, exception type name, message)`` for each failed input."""
        inputs = self.inputs
        return [
            (inputs[i], o["error"], o.get("message", ""))
            for i, o in sorted(self._outcomes.items())
            if "error" in o
        ]

    @property
    def pending(self) -> list:
        """Inputs with no outcome yet."""
        inputs = self.inputs
        return [inputs[i] for i in range(self.n) if i not in self._outcomes]

    def _row(self, i: int) -> Row | None:
        o = self._outcomes.get(i)
        if o is None or "error" in o or not o.get("trace"):
            return None
        try:
            trace = self._store.get_trace(self.fingerprint, o["trace"])
        except CacheMiss:
            return None
        inputs = self._input_hashes
        x = None
        if inputs[i] is not None:
            try:
                x = self._store.get_value(inputs[i])
            except CacheMiss:
                pass
        return Row(self._store, self.fn, o["trace"], trace, input=x, index=i)

    @property
    def rows(self) -> list[Row]:
        """The rows of inputs that finished successfully, in input order."""
        return [r for r in (self._row(i) for i in range(self.n)) if r is not None]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __getitem__(self, x: Any) -> Row:
        """The row for input *x*, matched by content."""
        h = content_hash(x)
        for i, ih in enumerate(self._input_hashes):
            if ih == h:
                row = self._row(i)
                if row is None:
                    o = self._outcomes.get(i)
                    why = "has not finished" if o is None else f"failed: {o.get('error')}"
                    raise KeyError(f"input {x!r} {why}")
                return row
        raise KeyError(f"{x!r} is not an input of {self!r}")

    def column(self, name: str) -> list:
        """The value bound to *name* for each input, in input order; None
        where the input has no row or no such binding."""
        return [None if r is None else r.get(name) for r in (self._row(i) for i in range(self.n))]

    def by(self, name: str) -> dict:
        """Rows grouped by the value bound to *name*."""
        out: dict = {}
        for r in self.rows:
            v = r.get(name, _ABSENT)
            if v is _ABSENT:
                continue
            out.setdefault(v, []).append(r)
        return out


_ABSENT = object()


def batch(name: str, cache_dir: str | os.PathLike | None = None) -> Batch:
    """Open the newest batch recorded under *name*.

    *cache_dir* defaults to the configured one.  Raises ``LookupError`` if
    no batch of that name has been recorded.
    """
    if cache_dir is None:
        from .pure import _current_store

        store = _current_store()
        if not isinstance(store, LocalStore):
            raise LookupError("no cache directory is configured; pass cache_dir=")
    else:
        store = LocalStore(cache_dir)
    d = batches_dir(store) / _dirname(name)
    try:
        batch_id = (d / "latest").read_text().strip()
        header = json.loads((d / batch_id / "header.json").read_bytes())
    except (OSError, ValueError) as e:
        raise LookupError(f"no batch named {name!r} under {store.root}") from e
    return Batch(store, d / batch_id, header)
