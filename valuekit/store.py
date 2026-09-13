"""The cache store.

Content-addressed, machine-local, and deletable at any moment with no
semantic effect: a missing or corrupt entry is treated as a miss.

Layout::

    <root>/format                         # store format version; mismatch → refuse
    <root>/objects/ab/<hash>.npy          # a read-only ndarray (reloaded mmap)
    <root>/objects/ab/<hash>.npyw         # a writeable ndarray (reloaded in full)
    <root>/objects/ab/<hash>.bin          # any other value
    <root>/records/<function hash>/<record hash>.json  # one call record of one function

Values are stored structurally: composite values (tuples, lists, sets,
frozensets, maps) store the content hashes of their children, each of which
is its own object.  This deduplicates large arrays across call records and lets
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
the hash of its content, call records included.  Two processes writing the same
entry write the same bytes under the same name, so there is nothing to
coordinate: no appends, no locks, no read-before-write.  This is what lets
any number of processes share a store directory, on Windows as well as
POSIX (Windows appends are not atomic, and Windows refuses to replace a file
another process has mapped -- both cases reduce to "already there").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
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
    "record_bytes",
    "record_hash",
]

FORMAT_VERSION = 8


class CacheMiss(Exception):
    """A value or call record could not be retrieved; recompute."""


class CacheStore(Protocol):
    """The methods a store must implement.

    Kept small so that a store elsewhere (a peer over a connection, say) can
    be added by implementing these methods.
    """

    def get_records(self, function_hash: str) -> list[tuple[str, dict]]:
        """``(record_hash, record)`` pairs, newest first."""

    def put_record(self, function_hash: str, record: dict) -> str:
        """Store *record*; return its hash."""

    def get_value(self, h: str) -> Any: ...
    def put_value(self, v: Any) -> str: ...


# ---------------------------------------------------------------------------
# call records as content-addressed documents
# ---------------------------------------------------------------------------


def record_bytes(record: dict) -> bytes:
    """The canonical serialisation of a call record: what is written to disk, and
    what is hashed to name it."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def record_hash(record: dict) -> str:
    return _hash_bytes(record_bytes(record))


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=20).hexdigest()


# ---------------------------------------------------------------------------


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def dirname_for(name: str) -> str:
    """A user-chosen name as a directory name: characters outside
    ``[A-Za-z0-9._-]`` become ``_``.  The record inside keeps the real one."""
    return _UNSAFE.sub("_", name) or "_"


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
    """What one function's call-record directory held when last read.

    ``entries`` is newest-first ``(hash, record)``; ``docs`` keeps every parsed
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
        self.records = self.root / "records"
        self.root.mkdir(parents=True, exist_ok=True)
        fmt = self.root / "format"
        if fmt.exists():
            found = fmt.read_text().strip()
            if found != str(FORMAT_VERSION):
                raise RuntimeError(
                    f"Cache at {self.root} has format {found}, this valuekit "
                    f"writes format {FORMAT_VERSION}. Delete the directory or "
                    "point set_store_dir() elsewhere."
                )
        else:
            _atomic_write(fmt, f"{FORMAT_VERSION}\n".encode())
        self.objects.mkdir(exist_ok=True)
        self.records.mkdir(exist_ok=True)
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
        :meth:`put_value` would have written: a peer's object messages carry
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

    # -- call records ---------------------------------------------------------------

    def _record_dir(self, function_hash: str) -> Path:
        return self.records / function_hash

    def get_records(self, function_hash: str) -> list[tuple[str, dict]]:
        """``(hash, record)`` pairs for *function_hash*, newest first.

        The directory is re-read only when its modification time has moved
        or this store wrote to it, so a hit costs one ``stat``.  Another
        process's write or removal moves the directory's mtime, so it is
        seen on the next call.  A filesystem with coarse mtime can leave a
        listing stale for a moment; the consequence is a spurious miss and a
        rewrite of an identically named file, never a wrong hit.
        """
        d = self._record_dir(function_hash)
        try:
            mtime_ns = d.stat().st_mtime_ns
        except OSError:
            self._listings.pop(function_hash, None)
            return []
        listing = self._listings.get(function_hash)
        if listing is not None and not listing.stale and listing.mtime_ns == mtime_ns:
            return listing.entries
        if listing is None:
            listing = _Listing()
            self._listings[function_hash] = listing
        found: list[tuple[int, str]] = []
        try:
            with os.scandir(d) as it:
                for entry in it:
                    name = entry.name
                    if not name.endswith(".json"):
                        continue  # a .tmp-* mid-write, or a leftover
                    try:
                        found.append((entry.stat().st_mtime_ns, name[:-5]))
                    except OSError:
                        continue
        except OSError:
            self._listings.pop(function_hash, None)
            return []
        found.sort(reverse=True)
        entries: list[tuple[str, dict]] = []
        for _, h in found:
            doc = listing.docs.get(h)
            if doc is None:
                doc = self._read_record(d / f"{h}.json", h)
                if doc is None:
                    continue
                listing.docs[h] = doc
            entries.append((h, doc))
        listing.entries = entries
        listing.docs = dict(entries)  # a record deleted since is not served
        listing.mtime_ns = mtime_ns
        listing.stale = False
        return entries

    @staticmethod
    def _read_record(path: Path, h: str) -> dict | None:
        """The document at *path*, or None if it is not the call record its name
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

    def get_record(self, function_hash: str, h: str) -> dict:
        """One record by hash; CacheMiss if it is gone or corrupt.

        Answered through the listing, so a record another process has
        deleted is missed here from the next call on, like any other."""
        self.get_records(function_hash)
        listing = self._listings.get(function_hash)
        doc = listing.docs.get(h) if listing is not None else None
        if doc is None:
            raise CacheMiss(f"record {h} of {function_hash}")
        return doc

    def put_record(self, function_hash: str, record: dict) -> str:
        data = record_bytes(record)
        h = _hash_bytes(data)
        path = self._record_dir(function_hash) / f"{h}.json"
        if not path.exists():
            _atomic_write(path, data)
        listing = self._listings.get(function_hash)
        if listing is not None:
            listing.stale = True
        return h

    def clear(self) -> None:
        """Delete everything computed or logged: the values, call records,
        batches and run logs.  The event log and host source trees stay."""
        for name in ("objects", "records", "batches", "logs"):
            shutil.rmtree(self.root / name, ignore_errors=True)
        self.objects.mkdir(exist_ok=True)
        self.records.mkdir(exist_ok=True)
        self._listings.clear()
