"""Framing and value transfer between a driver and a worker.

Messages are ``tag | 8-byte little-endian length | body`` -- the same shape
:mod:`valuekit.values` already uses for hashing and for the key codec, read
here from a stream rather than a buffer.  Two differences matter.  The
length is capped, because an unbounded one lets a corrupt or hostile header
ask for gigabytes.  And the tags live in a reserved low-byte range rather
than sharing the letter namespace those two codecs have already claimed, so
an envelope can never be confused with a value.

Values cross as a small object graph: :mod:`valuekit.codec` encodes a
composite as the content hashes of its children, so a value becomes one
root hash plus the objects reachable from it.  Since the hash identifies the
content exactly, an object shared by many values is sent once -- the same
property that deduplicates the store on disk.

Failures here are transport failures.  That distinction is why this module
does not reuse the store's read path: :meth:`LocalStore.get_value` turns a
corrupt entry into a CacheMiss, which correctly means "recompute" for a
cache and would wrongly mean "recompute" for a broken connection.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, BinaryIO

import numpy as np

from .codec import decode, encode
from .values import _blob, _read_blob, content_hash

__all__ = ["WireError", "read_frame", "write_frame", "pack", "unpack"]

# Bigger than any plausible message, small enough that a corrupt length is
# refused rather than acted on.
MAX_FRAME = 1 << 31

HELLO = b"\x01"  # driver -> worker: salt, module, qualname, fingerprint
READY = b"\x02"  # worker -> driver: empty if admitted, else the reason
OBJECT = b"\x03"  # either way: one content-addressed object
TASK = b"\x04"  # driver -> worker: the root hash of the input
RESULT = b"\x05"  # worker -> driver: ok or error
EVENT = b"\x06"  # reserved: worker -> driver events, once the
                 # worker is on another filesystem (step 3)


class WireError(Exception):
    """The connection said something impossible. Never a cache miss."""


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def write_frame(f: BinaryIO, tag: bytes, body: bytes = b"") -> None:
    f.write(_blob(tag, body))
    f.flush()


def _read_exact(f: BinaryIO, n: int) -> bytes:
    """Read exactly *n* bytes, or return b"" at a clean end of stream."""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = f.read(n - got)
        if not chunk:
            if got == 0:
                return b""  # clean EOF between frames
            raise WireError(f"stream ended {n - got} bytes into a frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_frame(f: BinaryIO) -> tuple[bytes, bytes] | None:
    """The next ``(tag, body)``, or None at a clean end of stream."""
    header = _read_exact(f, 9)
    if not header:
        return None
    tag = header[:1]
    n = int.from_bytes(header[1:9], "little")
    if n > MAX_FRAME:
        raise WireError(f"frame claims {n} bytes; refusing")
    body = _read_exact(f, n) if n else b""
    if n and not body:
        raise WireError("stream ended at a frame body")
    return tag, body


# ---------------------------------------------------------------------------
# values as object graphs
# ---------------------------------------------------------------------------
#
# An object's payload is one marker byte and its bytes.  Arrays get their
# own markers because they never reach the structural codec, and because
# writeability is part of the content hash and so has to survive the trip.

_STRUCT = b"b"
_ARRAY_RO = b"a"
_ARRAY_RW = b"w"


def pack(v: Any, seen: set[str] | None = None) -> tuple[str, dict[str, bytes]]:
    """Return ``(root_hash, objects)`` for *v*.

    *seen* names objects the peer already has; they are omitted from the
    result, so a value shared across many tasks crosses once.
    """
    objects: dict[str, bytes] = {}
    have = seen if seen is not None else set()

    def put(x: Any) -> str:
        h = content_hash(x)
        if h in have or h in objects:
            return h
        if isinstance(x, np.ndarray):
            buf = BytesIO()
            np.save(buf, np.asarray(x), allow_pickle=False)
            marker = _ARRAY_RW if x.flags.writeable else _ARRAY_RO
            objects[h] = marker + buf.getvalue()
            return h
        objects[h] = _STRUCT + encode(x, put)
        return h

    return put(v), objects


def unpack(root: str, objects: dict[str, bytes]) -> Any:
    """Rebuild the value *root* names from *objects*."""

    def get(h: str) -> Any:
        try:
            payload = objects[h]
        except KeyError:
            raise WireError(f"object {h} was never sent") from None
        marker, data = payload[:1], payload[1:]
        if marker == _ARRAY_RO:
            arr = np.load(BytesIO(data), allow_pickle=False)
            # Writeability is part of the content hash, so a read-only array
            # must come back read-only or it is a different value.
            arr.flags.writeable = False
            return arr
        if marker == _ARRAY_RW:
            return np.load(BytesIO(data), allow_pickle=False)
        if marker == _STRUCT:
            return decode(data, get)
        raise WireError(f"unknown object marker {marker!r}")

    return get(root)


def send_value(f: BinaryIO, v: Any, seen: set[str]) -> str:
    """Send whatever of *v* the peer lacks; return the root hash."""
    root, objects = pack(v, seen)
    for h, payload in objects.items():
        write_frame(f, OBJECT, bytes.fromhex(h) + payload)
        seen.add(h)
    return root


def recv_object(body: bytes, objects: dict[str, bytes]) -> None:
    """Record an OBJECT frame's contents."""
    if len(body) < 20:
        raise WireError("truncated object frame")
    objects[body[:20].hex()] = body[20:]


def strings(*parts: str) -> bytes:
    """Pack several strings into one body."""
    return b"".join(_blob(b"s", p.encode("utf-8")) for p in parts)


def unstrings(body: bytes) -> list[str]:
    out, pos = [], 0
    while pos < len(body):
        try:
            _, part, pos = _read_blob(body, pos)
        except ValueError as e:
            raise WireError(str(e)) from None
        out.append(part.decode("utf-8"))
    return out
