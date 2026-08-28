"""The cache store.

Content-addressed, machine-local, and deletable at any moment with no
semantic effect: a missing or corrupt entry is treated as a miss.

Layout::

    <root>/format            # store format version; mismatch → refuse
    <root>/objects/ab/<hash>.npy   # a read-only ndarray (reloaded mmap)
    <root>/objects/ab/<hash>.npyw  # a writeable ndarray (reloaded in full)
    <root>/objects/ab/<hash>.bin   # any other value
    <root>/traces/<fnkey>.jsonl    # traces for one function, one per line

Values are stored structurally: composite values (tuples, lists, sets,
frozensets, maps) store the content hashes of their children, each of which
is its own object.  This deduplicates large arrays across traces and lets
read-only arrays reload as memory maps, which ``freeze`` then shares without
copying.

A stored value round-trips to an equal value of the same type, which is why
writeability splits the two array extensions and why lists, dicts and sets
keep their order: a hit must be indistinguishable from the miss that
recorded it.  Content-addressing makes this work only because the content
hash identifies a value exactly (see :mod:`valuekit.values`).

There is no pickle anywhere.  Storable types are a fixed set: None, bool,
int, float, complex, str, bytes, range, numpy scalars, numpy arrays, tuples,
lists, sets, frozensets, dicts, ImmutableMaps and plain-data dataclasses
(recursively of the same).  Anything else raises
:class:`SerializationError`.  A dataclass entry names its class, but nothing
is imported on the strength of a stored entry: the class is resolved only
among modules the process has already loaded, and a class that has changed
since reads as a miss (see :mod:`valuekit.plaindata`).

Writes are atomic: values go through a temp file and ``os.replace``, and
traces are appended with a single ``O_APPEND`` write, so any number of
processes may share a cache directory.  A concurrent duplicate trace line is
possible and harmless; a torn line from a crash is skipped on read.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .codec import SerializationError, decode, encode
from .values import content_hash

__all__ = ["CacheMiss", "SerializationError", "CacheStore", "LocalStore"]

FORMAT_VERSION = 5


class CacheMiss(Exception):
    """A value or trace could not be retrieved; recompute."""


class CacheStore(Protocol):
    """The methods a store must implement.

    Kept small so that a shared or remote store (e.g. S3 or Redis) can be
    added by implementing these methods.
    """

    def get_traces(self, fn_key: str) -> list[dict]: ...
    def put_trace(
        self, fn_key: str, trace: dict, units: Sequence[str] = ()
    ) -> None: ...
    def get_value(self, h: str) -> Any: ...
    def put_value(self, v: Any) -> str: ...


# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class LocalStore:
    """Content-addressed store in a local directory."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.traces = self.root / "traces"
        self.root.mkdir(parents=True, exist_ok=True)
        fmt = self.root / "format"
        if fmt.exists():
            found = fmt.read_text().strip()
            if found != str(FORMAT_VERSION):
                raise RuntimeError(
                    f"Cache at {self.root} has format {found}, this valuekit "
                    f"writes format {FORMAT_VERSION}. Delete the directory or "
                    "point set_cache_dir() elsewhere."
                )
        else:
            _atomic_write(fmt, f"{FORMAT_VERSION}\n".encode())
        self.objects.mkdir(exist_ok=True)
        self.traces.mkdir(exist_ok=True)

    # -- object paths -----------------------------------------------------

    def _obj(self, h: str, ext: str) -> Path:
        return self.objects / h[:2] / f"{h}{ext}"

    def _find(self, h: str) -> Path | None:
        for ext in (".npy", ".npyw", ".bin"):
            p = self._obj(h, ext)
            if p.exists():
                return p
        return None

    # -- values -------------------------------------------------------------

    def put_value(self, v: Any) -> str:
        h = content_hash(v)
        if self._find(h) is not None:
            return h
        if isinstance(v, np.ndarray):
            buf = BytesIO()
            np.save(buf, np.asarray(v), allow_pickle=False)
            # Writeability is in the content hash, so the two extensions are
            # distinct objects and never race for the same name.
            ext = ".npyw" if v.flags.writeable else ".npy"
            _atomic_write(self._obj(h, ext), buf.getvalue())
            return h
        _atomic_write(self._obj(h, ".bin"), self._encode(v))
        return h

    def _encode(self, v: Any) -> bytes:
        return encode(v, self.put_value)

    def get_value(self, h: str) -> Any:
        path = self._find(h)
        if path is None:
            raise CacheMiss(h)
        try:
            if path.suffix == ".npy":
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                return arr  # memmap 'r' → writeable=False → freeze shares it
            if path.suffix == ".npyw":
                return np.load(path, allow_pickle=False)  # writeable, as stored
            return self._decode(path.read_bytes())
        except CacheMiss:
            raise
        except Exception as e:  # corrupt entry → miss
            raise CacheMiss(f"{h}: {e}") from e

    def _decode(self, data: bytes) -> Any:
        return decode(data, self.get_value)

    # -- traces ---------------------------------------------------------------

    def _trace_path(self, fn_key: str) -> Path:
        return self.traces / f"{fn_key}.jsonl"

    def get_traces(self, fn_key: str) -> list[dict]:
        try:
            text = self._trace_path(fn_key).read_text()
        except OSError:
            return []
        out: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn or corrupt line: skip (a miss at worst)
            if isinstance(t, dict) and t not in out:
                out.append(t)
        return out

    def put_trace(self, fn_key: str, trace: dict, units: Sequence[str] = ()) -> None:
        deps = self.traces / f"{fn_key}.deps"
        if units and not deps.exists():
            _atomic_write(deps, "\n".join(units).encode())
        if trace in self.get_traces(fn_key):
            return
        # One O_APPEND write per trace: atomic under concurrency, so parallel
        # workers cannot drop each other's traces.  A duplicate line from a
        # write race is possible and harmless (get_traces de-duplicates).
        data = (json.dumps(trace) + "\n").encode()
        path = self._trace_path(fn_key)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def _drop(self, trace_path: Path) -> None:
        try:
            trace_path.unlink(missing_ok=True)
            trace_path.with_suffix(".deps").unlink(missing_ok=True)
        except OSError:
            pass

    def drop_dependents(self, unit_digests: set[str], value_hash: str | None) -> None:
        """Delete the traces of every function whose recorded closure
        contains any of *unit_digests*, plus any trace file that mentions
        *value_hash* (the function appearing as an argument). Stored values
        are left in place."""
        for deps in list(self.traces.glob("*.deps")):
            try:
                recorded = set(deps.read_text().split())
            except OSError:
                recorded = set()
            if recorded & unit_digests:
                self._drop(deps.with_suffix(".jsonl"))
        if value_hash:
            for tf in list(self.traces.glob("*.jsonl")):
                try:
                    if value_hash in tf.read_text():
                        self._drop(tf)
                except OSError:
                    pass

    # -- maintenance -----------------------------------------------------------

    def clear(self) -> None:
        """Delete all cached objects and traces (always safe)."""
        import shutil

        for sub in (self.objects, self.traces):
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(exist_ok=True)
