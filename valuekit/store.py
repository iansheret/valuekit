"""The cache store.

Content-addressed, machine-local, and deletable at any moment with no
semantic effect: a missing or corrupt entry is treated as a miss.

Layout::

    <root>/format                         # store format version; mismatch → refuse
    <root>/objects/ab/<hash>.npy          # a read-only ndarray (reloaded mmap)
    <root>/objects/ab/<hash>.npyw         # a writeable ndarray (reloaded in full)
    <root>/objects/ab/<hash>.bin          # any other value
    <root>/traces/<fnkey>/<tracehash>.json  # one trace of one function
    <root>/traces/<fnkey>.deps            # the function's code units, for clear_cache

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

Every write is a temp file and ``os.replace``, and every file is named by
the hash of its content, traces included.  Two processes writing the same
entry write the same bytes under the same name, so there is nothing to
coordinate: no appends, no locks, no read-before-write.  This is what lets
any number of processes share a cache directory, on Windows as well as
POSIX (Windows appends are not atomic, and Windows refuses to replace a file
another process has mapped -- both cases reduce to "already there").
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .codec import SerializationError, decode, encode
from .values import content_hash

__all__ = [
    "CacheMiss",
    "SerializationError",
    "CacheStore",
    "LocalStore",
    "trace_bytes",
    "trace_hash",
]

FORMAT_VERSION = 6


class CacheMiss(Exception):
    """A value or trace could not be retrieved; recompute."""


class CacheStore(Protocol):
    """The methods a store must implement.

    Kept small so that a store elsewhere (a peer over a connection, say) can
    be added by implementing these methods.
    """

    def get_traces(self, fn_key: str) -> list[tuple[str, dict]]:
        """``(trace_hash, trace)`` pairs, newest first."""

    def put_trace(
        self, fn_key: str, trace: dict, units: Sequence[str] = ()
    ) -> str:
        """Store *trace*; return its hash."""

    def get_value(self, h: str) -> Any: ...
    def put_value(self, v: Any) -> str: ...


# ---------------------------------------------------------------------------
# traces as content-addressed documents
# ---------------------------------------------------------------------------


def trace_bytes(trace: dict) -> bytes:
    """The canonical serialisation of a trace: what is written to disk, and
    what is hashed to name it."""
    return json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()


def trace_hash(trace: dict) -> str:
    return _hash_bytes(trace_bytes(trace))


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=20).hexdigest()


# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            os.replace(tmp, path)
        except (PermissionError, FileExistsError):
            # Windows refuses to replace a file another process has open or
            # mapped.  Every path here is content-addressed, so a target that
            # exists already holds these bytes: the write has happened.
            if not path.exists():
                raise
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _Listing:
    """What one function's trace directory held when last read.

    ``entries`` is newest-first ``(hash, trace)``; ``docs`` keeps every parsed
    document by hash so a re-listing parses only files it has not seen.
    """

    __slots__ = ("mtime_ns", "entries", "docs", "stale")

    def __init__(self) -> None:
        self.mtime_ns = -1
        self.entries: list[tuple[str, dict]] = []
        self.docs: dict[str, dict] = {}
        self.stale = True


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
        self._listings: dict[str, _Listing] = {}

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

    def put_object(self, h: str, ext: str, data: bytes) -> None:
        """Store one already-encoded object under its hash.

        *ext* is ``.npy``, ``.npyw`` or ``.bin`` and *data* is exactly what
        :meth:`put_value` would have written: a peer's object frames carry
        the same bytes, so they go in without being decoded.
        """
        if ext not in (".npy", ".npyw", ".bin"):
            raise ValueError(f"not an object kind: {ext!r}")
        if self._find(h) is None:
            _atomic_write(self._obj(h, ext), data)

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

    def _trace_dir(self, fn_key: str) -> Path:
        return self.traces / fn_key

    def _deps_path(self, fn_key: str) -> Path:
        return self.traces / f"{fn_key}.deps"

    def get_traces(self, fn_key: str) -> list[tuple[str, dict]]:
        """``(hash, trace)`` pairs for *fn_key*, newest first.

        The directory is re-read only when its modification time has moved
        or this store wrote to it, so a hit costs one ``stat``.  Another
        process's write or removal moves the directory's mtime, so it is
        seen on the next call.  A filesystem with coarse mtime can leave a
        listing stale for a moment; the consequence is a spurious miss and a
        rewrite of an identically named file, never a wrong hit.
        """
        d = self._trace_dir(fn_key)
        try:
            mtime_ns = d.stat().st_mtime_ns
        except OSError:
            self._listings.pop(fn_key, None)
            return []
        listing = self._listings.get(fn_key)
        if listing is not None and not listing.stale and listing.mtime_ns == mtime_ns:
            return listing.entries
        if listing is None:
            listing = _Listing()
            self._listings[fn_key] = listing
        found: list[tuple[int, str]] = []
        try:
            with os.scandir(d) as it:
                for entry in it:
                    name = entry.name
                    if not name.endswith(".json"):
                        continue  # a .tmp-* mid-write, or debris
                    try:
                        found.append((entry.stat().st_mtime_ns, name[:-5]))
                    except OSError:
                        continue
        except OSError:
            self._listings.pop(fn_key, None)
            return []
        found.sort(reverse=True)
        entries: list[tuple[str, dict]] = []
        for _, h in found:
            doc = listing.docs.get(h)
            if doc is None:
                doc = self._read_trace(d / f"{h}.json", h)
                if doc is None:
                    continue
                listing.docs[h] = doc
            entries.append((h, doc))
        listing.entries = entries
        listing.mtime_ns = mtime_ns
        listing.stale = False
        return entries

    @staticmethod
    def _read_trace(path: Path, h: str) -> dict | None:
        """The document at *path*, or None if it is not the trace its name
        claims (a torn or corrupt file: a miss at worst)."""
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if _hash_bytes(raw) != h:
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    def get_trace(self, fn_key: str, h: str) -> dict:
        """One trace by hash; CacheMiss if it is gone or corrupt."""
        listing = self._listings.get(fn_key)
        if listing is not None and h in listing.docs:
            return listing.docs[h]
        doc = self._read_trace(self._trace_dir(fn_key) / f"{h}.json", h)
        if doc is None:
            raise CacheMiss(f"trace {h} of {fn_key}")
        return doc

    def put_trace(self, fn_key: str, trace: dict, units: Sequence[str] = ()) -> str:
        deps = self._deps_path(fn_key)
        if units and not deps.exists():
            _atomic_write(deps, "\n".join(units).encode())
        data = trace_bytes(trace)
        h = _hash_bytes(data)
        path = self._trace_dir(fn_key) / f"{h}.json"
        if not path.exists():
            _atomic_write(path, data)
        listing = self._listings.get(fn_key)
        if listing is not None:
            listing.stale = True
        return h

    def _drop(self, fn_key: str) -> None:
        shutil.rmtree(self._trace_dir(fn_key), ignore_errors=True)
        try:
            self._deps_path(fn_key).unlink(missing_ok=True)
        except OSError:
            pass
        self._listings.pop(fn_key, None)

    def drop_dependents(self, unit_digests: set[str], value_hash: str | None) -> None:
        """Delete the traces of every function whose recorded closure
        contains any of *unit_digests*, plus those of any function with a
        trace that mentions *value_hash* (the function appearing as an
        argument). Stored values are left in place."""
        for deps in list(self.traces.glob("*.deps")):
            try:
                recorded = set(deps.read_text().split())
            except OSError:
                recorded = set()
            if recorded & unit_digests:
                self._drop(deps.stem)
        if value_hash:
            needle = value_hash.encode()
            for d in list(self.traces.iterdir()):
                if not d.is_dir():
                    continue
                for tf in list(d.glob("*.json")):
                    try:
                        if needle in tf.read_bytes():
                            self._drop(d.name)
                            break
                    except OSError:
                        pass

    # -- maintenance -----------------------------------------------------------

    def clear(self) -> None:
        """Delete all cached objects and traces (always safe)."""
        for sub in (self.objects, self.traces):
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(exist_ok=True)
        self._listings.clear()
