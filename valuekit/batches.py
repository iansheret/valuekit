"""Batch records: what a ``run_all`` produced, by name, for analysis code.

A call record holds everything a call recorded -- its result, the memoised
calls it made, its logged values -- but a call record is found by matching
arguments, and analysis code should not have to reconstruct arguments.  So
``run_all`` writes a small record naming its root call records, and this module
reads it back::

    b = valuekit.batch("process")     # the newest batch of process()
    b[7]                              # the row for input 7
    b[7].result                       # what process(7) returned
    b.failures                        # (input, exception type, message)
    b.logs.where(quantity="rms")      # what the batch's inputs logged

Nothing here imports or runs the pipeline.  Staleness within a batch is
impossible: every input ran under one function hash, which the record
carries.  Across code changes the question is only "has this batch been
re-run since the edit", and the record's ``function_hash`` answers it.

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
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from .runlog import Logs, collect
from .store import CacheMiss, LocalStore, SerializationError, _atomic_write, dirname_for
from .values import content_hash

__all__ = ["batch", "Batch", "CallRecord", "BatchWriter"]

RECORD_VERSION = 1


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
        function_hash: str,
        inputs: list,
    ):
        self.dir = batches_dir(store) / dirname_for(name)
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
            "function_hash": function_hash,
            "n": len(inputs),
            "inputs": hashes,
            "started": time.time(),
        }
        _atomic_write(self.path / "header.json", json.dumps(header).encode())
        _atomic_write(self.dir / "latest", self.id.encode())
        for old in list(self.dir.iterdir()):
            if old.is_dir() and old.name != self.id:
                shutil.rmtree(old, ignore_errors=True)

    def outcome(self, index: int, record: str | None, exc: BaseException | None) -> None:
        outcome: dict[str, Any] = {"i": index}
        if exc is None:
            outcome["record"] = record
        else:
            outcome["error"] = type(exc).__name__
            outcome["message"] = str(exc)
        try:
            _atomic_write(self.path / f"{index}.json", json.dumps(outcome).encode())
        except Exception:
            pass


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


class CallRecord:
    """One call: a root input of a batch, or a call nested in one.

    ``row.result`` is what it returned, ``row.calls`` the memoised calls it
    made, ``row.logs`` what it and they logged.  Values load when asked
    for, and arrays arrive as memory maps.
    """

    __slots__ = ("_store", "fn", "record_hash", "_doc", "input", "index")

    def __init__(self, store, fn: str, record_hash: str, record: dict, input=None, index=None):
        self._store = store
        self.fn = fn
        self.record_hash = record_hash
        self._doc = record
        self.input = input
        self.index = index

    def __repr__(self) -> str:
        where = f"[{self.index}]" if self.index is not None else ""
        return f"CallRecord({self.fn}{where})"

    @property
    def result(self) -> Any:
        return self._store.get_value(self._doc["result"])

    @property
    def calls(self) -> list[CallRecord]:
        out = []
        for qn, function_hash, h in self._doc.get("calls", []):
            try:
                out.append(CallRecord(self._store, qn, h, self._store.get_record(function_hash, h)))
            except CacheMiss:
                continue  # swept, or cleared: the call is no longer readable
        return out

    @property
    def logs(self) -> Logs:
        """What this call logged, nested calls included.  A part
        that has been swept since is left out rather than failing."""
        return Logs(self._store, items=collect(self._store, self._doc, strict=False))


class Batch:
    """The newest ``run_all`` recorded under a name.  See the module."""

    def __init__(self, store: LocalStore, path: Path, header: dict):
        self._store = store
        self._path = path
        self.name: str = header["name"]
        self.fn: str = header["fn"]
        self.function_hash: str = header["function_hash"]
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

    def _row(self, i: int) -> CallRecord | None:
        o = self._outcomes.get(i)
        if o is None or "error" in o or not o.get("record"):
            return None
        try:
            record = self._store.get_record(self.function_hash, o["record"])
        except CacheMiss:
            return None
        inputs = self._input_hashes
        x = None
        if inputs[i] is not None:
            try:
                x = self._store.get_value(inputs[i])
            except CacheMiss:
                pass
        return CallRecord(self._store, self.fn, o["record"], record, input=x, index=i)

    @property
    def rows(self) -> list[CallRecord]:
        """The rows of inputs that finished successfully, in input order."""
        return [r for r in (self._row(i) for i in range(self.n)) if r is not None]

    def __iter__(self) -> Iterator[CallRecord]:
        return iter(self.rows)

    def __getitem__(self, x: Any) -> CallRecord:
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

    @property
    def logs(self) -> Logs:
        """What the batch's finished inputs logged, as :func:`valuekit.logs`
        would show it: every logged value their call records hold, nested calls
        included.  A part that has been swept since is left out."""
        items: list = []
        for r in self.rows:
            items.extend(collect(self._store, r._doc, strict=False))
        return Logs(self._store, items=items)


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
    d = batches_dir(store) / dirname_for(name)
    try:
        batch_id = (d / "latest").read_text().strip()
        header = json.loads((d / batch_id / "header.json").read_bytes())
    except (OSError, ValueError) as e:
        raise LookupError(f"no batch named {name!r} under {store.root}") from e
    return Batch(store, d / batch_id, header)
