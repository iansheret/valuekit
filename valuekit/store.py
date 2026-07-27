"""The cache store.

Content-addressed, machine-local, and deletable at any moment with no
semantic effect: a missing or corrupt entry is treated as a miss.

Layout::

    <root>/format            # store format version; mismatch → refuse
    <root>/objects/ab/<hash>.npy   # a single ndarray (reloaded mmap, read-only)
    <root>/objects/ab/<hash>.bin   # any other value
    <root>/traces/<fnkey>.jsonl    # traces for one function, one per line

Values are stored structurally: composite values (tuples, frozensets, maps)
store the content hashes of their children, each of which is its own object.
This deduplicates large arrays across traces and lets arrays reload as
read-only memory maps, which ``freeze`` then shares without copying.

There is no pickle anywhere.  Storable types are a fixed set: None, bool,
int, float, complex, str, bytes, range, numpy scalars, numpy arrays, tuples,
frozensets, and ImmutableMaps (recursively of the same).  Anything else
raises :class:`SerializationError`, matching the closed-set behaviour of
``freeze``.

Writes are atomic: values go through a temp file and ``os.replace``, and
traces are appended with a single ``O_APPEND`` write, so any number of
processes may share a cache directory.  A concurrent duplicate trace line is
possible and harmless; a torn line from a crash is skipped on read.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .map import ImmutableMap
from .values import (
    content_hash,
    custom_reduce,
    custom_rebuild,
    encode_key,
    decode_key,
    _blob,
    _read_blob,
)

__all__ = ["CacheMiss", "SerializationError", "CacheStore", "LocalStore"]

FORMAT_VERSION = 4


class CacheMiss(Exception):
    """A value or trace could not be retrieved; recompute."""


class SerializationError(TypeError):
    """A value cannot be stored. Only the fixed set of storable types may
    appear in cached return values."""


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
        for ext in (".npy", ".bin"):
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
            _atomic_write(self._obj(h, ".npy"), buf.getvalue())
            return h
        _atomic_write(self._obj(h, ".bin"), self._encode(v))
        return h

    def _encode(self, v: Any) -> bytes:
        t = type(v)
        if t is tuple:
            hs = [bytes.fromhex(self.put_value(x)) for x in v]
            return _blob(b"T", b"".join(hs))
        if t is frozenset:
            hs = sorted(bytes.fromhex(self.put_value(x)) for x in v)
            return _blob(b"F", b"".join(hs))
        if isinstance(v, ImmutableMap):
            pairs = sorted(
                (bytes.fromhex(self.put_value(k)), bytes.fromhex(self.put_value(val)))
                for k, val in v.items()
            )
            return _blob(b"M", b"".join(k + val for k, val in pairs))
        reduced = custom_reduce(v)
        if reduced is not None:
            name, red = reduced
            h = bytes.fromhex(self.put_value(red))
            return _blob(b"C", _blob(b"s", name.encode("utf-8")) + h)
        try:
            return _blob(b"I", encode_key(v))  # atomics + np scalars
        except Exception:
            raise SerializationError(
                f"Cannot cache a value of type {t.__name__!r}. Cached return "
                "values are limited to: None, bool, int, float, complex, str, "
                "bytes, range, numpy scalars/arrays, tuples, frozensets, and "
                "(Immutable)Maps of the same -- or a type registered with a "
                "reduce_fn/rebuild_fn via valuekit.register_type()."
            ) from None

    def get_value(self, h: str) -> Any:
        path = self._find(h)
        if path is None:
            raise CacheMiss(h)
        try:
            if path.suffix == ".npy":
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                return arr  # memmap 'r' → writeable=False → freeze shares it
            return self._decode(path.read_bytes())
        except CacheMiss:
            raise
        except Exception as e:  # corrupt entry → miss
            raise CacheMiss(f"{h}: {e}") from e

    def _decode(self, data: bytes) -> Any:
        tag, body, pos = _read_blob(data, 0)
        if pos != len(data):
            raise ValueError("trailing bytes in stored value")
        if tag == b"I":
            return decode_key(body)
        if tag == b"C":
            _, name, p = _read_blob(body, 0)
            reduced = self.get_value(body[p:].hex())
            return custom_rebuild(name.decode("utf-8"), reduced)
        n = 20  # raw digest size
        hs = [body[i : i + n].hex() for i in range(0, len(body), n)]
        if tag == b"T":
            return tuple(self.get_value(x) for x in hs)
        if tag == b"F":
            return frozenset(self.get_value(x) for x in hs)
        if tag == b"M":
            d = {}
            for i in range(0, len(hs), 2):
                d[self.get_value(hs[i])] = self.get_value(hs[i + 1])
            return ImmutableMap(d)
        raise ValueError(f"unknown value tag {tag!r}")

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
