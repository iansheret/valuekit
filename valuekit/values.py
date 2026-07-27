"""The unified type registry: freeze, content-hash, and inline key codec.

Every type that participates in valuekit must be *frozen* (made immutable on
entry) and *content-hashable* (a stable digest of its value, identical across
processes and sessions).  The two capabilities are registered together:
``register_type`` rejects partial registration, since a type that could be
frozen but not hashed would enter maps and then fail later, inside a cache
lookup, far from the registration that caused it.

The Mapping handlers are registered in :mod:`valuekit.map` (they need the
ImmutableMap class); the function handlers are registered in
:mod:`valuekit.codehash` (they need the code hasher).
"""

from __future__ import annotations

import hashlib
import struct
from functools import singledispatch
from typing import Any, Callable

import numpy as np

_DIGEST_SIZE = 20  # bytes; 40 hex chars

__all__ = [
    "freeze",
    "content_hash",
    "digest",
    "hash_update",
    "register_type",
    "custom_reduce",
    "custom_rebuild",
    "encode_key",
    "decode_key",
    "UnsupportedKeyError",
]


# ---------------------------------------------------------------------------
# freeze — recursively make a value immutable
# ---------------------------------------------------------------------------


@singledispatch
def freeze(v: Any) -> Any:
    """Return an immutable form of *v*.

    * Already-immutable values (int, float, complex, bool, str, bytes, None,
      frozenset, range, numpy scalars) are returned unchanged.
    * Types with a known freezing strategy are returned as an immutable copy:
      numpy arrays are made read-only, sets become frozensets, dicts become
      ImmutableMaps, and tuples have their contents frozen.
    * Anything else raises TypeError; register new types with
      :func:`register_type`.
    """
    raise TypeError(
        f"Cannot store a {type(v).__name__!r}: not known to be immutable. "
        "Pass an already-immutable value, or register a handler with "
        "valuekit.register_type()."
    )


_identity: Any = lambda v: v  # noqa: E731

for _t in (int, float, complex, bool, str, bytes, type(None), frozenset, range):
    freeze.register(_t, _identity)


@freeze.register
def _freeze_numpy_scalar(v: np.generic) -> np.generic:
    """NumPy scalars are immutable; pass through unchanged."""
    return v


@freeze.register
def _freeze_ndarray(v: np.ndarray) -> np.ndarray:
    """Copy writeable arrays (preserving memory layout); share frozen ones.

    Pre-setting ``v.flags.writeable = False`` before insertion therefore
    avoids the copy — the array is shared as-is.
    """
    if v.flags.writeable:
        v = v.copy(order="K")
        v.flags.writeable = False
    return v


@freeze.register
def _freeze_tuple(v: tuple) -> tuple:
    """Freeze the contents of a tuple (the tuple shell is already immutable)."""
    return tuple(freeze(x) for x in v)


@freeze.register
def _freeze_set(v: set) -> frozenset:
    """Convert a mutable set to its immutable frozenset equivalent."""
    return frozenset(v)


# ---------------------------------------------------------------------------
# content hashing
# ---------------------------------------------------------------------------
#
# hash_update(v, h) feeds value *v* into hasher *h* using an unambiguous
# framed encoding (tag + length + payload) so that e.g. ("ab", "c") and
# ("a", "bc") hash differently.  Dispatch is on the value's type.


def _new_hasher() -> "hashlib._Hash":
    return hashlib.blake2b(digest_size=_DIGEST_SIZE)


def _frame(h: Any, tag: bytes, payload: bytes = b"") -> None:
    h.update(tag)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)


@singledispatch
def hash_update(v: Any, h: Any) -> None:
    """Feed the content of *v* into hasher *h*. Dispatch on type of *v*."""
    raise TypeError(
        f"No content-hash strategy for {type(v).__name__!r}. "
        "Register one with valuekit.register_type()."
    )


def digest(v: Any) -> bytes:
    """Return the raw content digest of *v*."""
    h = _new_hasher()
    hash_update(v, h)
    return h.digest()


def content_hash(v: Any) -> str:
    """Return the hex content hash of *v* (stable across sessions/machines)."""
    return digest(v).hex()


# --- atomics ---------------------------------------------------------------

# NOTE: bool must be registered separately from int (bool subclasses int);
# singledispatch picks the more specific handler.


@hash_update.register
def _h_none(v: None, h: Any) -> None:
    _frame(h, b"N")


@hash_update.register
def _h_bool(v: bool, h: Any) -> None:
    _frame(h, b"b", b"\x01" if v else b"\x00")


@hash_update.register
def _h_int(v: int, h: Any) -> None:
    _frame(h, b"i", str(v).encode("ascii"))


@hash_update.register
def _h_float(v: float, h: Any) -> None:
    _frame(h, b"f", struct.pack("<d", v))


@hash_update.register
def _h_complex(v: complex, h: Any) -> None:
    _frame(h, b"c", struct.pack("<dd", v.real, v.imag))


@hash_update.register
def _h_str(v: str, h: Any) -> None:
    _frame(h, b"s", v.encode("utf-8"))


@hash_update.register
def _h_bytes(v: bytes, h: Any) -> None:
    _frame(h, b"y", v)


@hash_update.register
def _h_range(v: range, h: Any) -> None:
    _frame(h, b"r", f"{v.start}:{v.stop}:{v.step}".encode("ascii"))


# --- numpy -----------------------------------------------------------------


@hash_update.register
def _h_np_scalar(v: np.generic, h: Any) -> None:
    _frame(h, b"g", v.dtype.str.encode("ascii") + b"|" + v.tobytes())


@hash_update.register
def _h_ndarray(v: np.ndarray, h: Any) -> None:
    meta = f"{v.dtype.str}|{v.shape}".encode("ascii")
    # .tobytes() always serialises in C order, independent of memory layout,
    # so two arrays equal under np.array_equal hash identically.
    _frame(h, b"a", meta)
    h.update(v.tobytes())


# --- containers ------------------------------------------------------------


@hash_update.register
def _h_tuple(v: tuple, h: Any) -> None:
    _frame(h, b"t", len(v).to_bytes(8, "little"))
    for x in v:
        hash_update(x, h)


@hash_update.register
def _h_frozenset(v: frozenset, h: Any) -> None:
    ds = sorted(digest(x) for x in v)
    _frame(h, b"F", len(v).to_bytes(8, "little") + b"".join(ds))


# ---------------------------------------------------------------------------
# registration API — freeze and hash travel together
# ---------------------------------------------------------------------------


_reduce_by_type: dict[type, tuple[str, Callable[[Any], Any]]] = {}
_rebuild_by_name: dict[str, Callable[[Any], Any]] = {}


def register_type(
    cls: type,
    *,
    freeze_fn: Callable[[Any], Any],
    hash_fn: Callable[[Any, Any], None],
    reduce_fn: Callable[[Any], Any] | None = None,
    rebuild_fn: Callable[[Any], Any] | None = None,
    name: str | None = None,
) -> None:
    """Register a new value type with a freeze and a hash strategy, and
    optionally a store codec.

    ``freeze_fn(v)`` must return an immutable form of *v*.
    ``hash_fn(v, hasher)`` must feed a stable, content-based encoding of the
    *frozen* form into *hasher* (use hashlib-style ``hasher.update``).

    Both are required: a type that could be frozen but not hashed would
    enter maps and then fail at the first cache lookup that reads it, so
    partial registration is rejected.

    With only those two, values of the type can be @pure inputs (hashed for
    the cache key) and map values, but not cached return values. To also let
    them appear in a cached return, pass a ``reduce_fn`` / ``rebuild_fn``
    pair: ``reduce_fn(v)`` returns a storable form (built from the fixed
    storable set) and ``rebuild_fn(reduced)`` reconstructs the value, so the
    store round-trips it. ``name`` identifies the type in stored entries
    (default ``"module:qualname"``); keep it stable, since entries reference
    it and a change reads as a miss.
    """
    if freeze_fn is None or hash_fn is None:
        raise TypeError("register_type requires BOTH freeze_fn and hash_fn")
    if (reduce_fn is None) != (rebuild_fn is None):
        raise TypeError(
            "register_type requires BOTH reduce_fn and rebuild_fn, or neither"
        )
    freeze.register(cls, freeze_fn)
    hash_update.register(cls, hash_fn)
    if reduce_fn is not None:
        codec_name = name or f"{cls.__module__}:{cls.__qualname__}"
        _reduce_by_type[cls] = (codec_name, reduce_fn)
        _rebuild_by_name[codec_name] = rebuild_fn


def custom_reduce(v: Any) -> tuple[str, Any] | None:
    """If *v*'s type has a store codec, return ``(name, reduced storable
    form)``; otherwise None."""
    entry = _reduce_by_type.get(type(v))
    if entry is None:
        return None
    name, reduce_fn = entry
    return name, reduce_fn(v)


def custom_rebuild(name: str, reduced: Any) -> Any:
    """Rebuild a custom value from its reduced form. Raises KeyError if the
    named type is not registered in this process."""
    return _rebuild_by_name[name](reduced)


# ---------------------------------------------------------------------------
# inline key codec
# ---------------------------------------------------------------------------
#
# Recorded read-paths must survive a JSON round trip, so map keys along a
# path are serialised with a small self-describing binary codec.  Keys that
# this codec cannot express cause the recorder to fall back to a whole-map
# dependency (correct, just coarser).


class UnsupportedKeyError(TypeError):
    """Raised when a map key cannot be encoded for fine-grained recording."""


def _blob(tag: bytes, body: bytes) -> bytes:
    return tag + len(body).to_bytes(8, "little") + body


def encode_key(v: Any) -> bytes:
    """Canonical self-describing encoding for map keys (atomics + tuples)."""
    if v is None:
        return _blob(b"N", b"")
    t = type(v)
    if t is bool:
        return _blob(b"b", b"\x01" if v else b"\x00")
    if t is int:
        return _blob(b"i", str(v).encode("ascii"))
    if t is float:
        return _blob(b"f", struct.pack("<d", v))
    if t is complex:
        return _blob(b"c", struct.pack("<dd", v.real, v.imag))
    if t is str:
        return _blob(b"s", v.encode("utf-8"))
    if t is bytes:
        return _blob(b"y", v)
    if t is range:
        return _blob(b"r", f"{v.start}:{v.stop}:{v.step}".encode("ascii"))
    if t is tuple:
        return _blob(b"t", b"".join(encode_key(x) for x in v))
    if t is frozenset:
        return _blob(b"F", b"".join(sorted(encode_key(x) for x in v)))
    if isinstance(v, np.generic):
        return _blob(b"g", _blob(b"s", v.dtype.str.encode("ascii")) + v.tobytes())
    raise UnsupportedKeyError(
        f"Map key of type {type(v).__name__!r} cannot be recorded "
        "fine-grained; falling back to whole-map dependency."
    )


def _read_blob(data: bytes, pos: int) -> tuple[bytes, bytes, int]:
    tag = data[pos : pos + 1]
    if not tag:
        raise ValueError("truncated key encoding")
    n = int.from_bytes(data[pos + 1 : pos + 9], "little")
    body = data[pos + 9 : pos + 9 + n]
    if len(body) != n:
        raise ValueError("truncated key encoding")
    return tag, body, pos + 9 + n


def _decode_one(data: bytes, pos: int) -> tuple[Any, int]:
    tag, body, pos = _read_blob(data, pos)
    if tag == b"N":
        return None, pos
    if tag == b"b":
        return body == b"\x01", pos
    if tag == b"i":
        return int(body.decode("ascii")), pos
    if tag == b"f":
        return struct.unpack("<d", body)[0], pos
    if tag == b"c":
        re_, im = struct.unpack("<dd", body)
        return complex(re_, im), pos
    if tag == b"s":
        return body.decode("utf-8"), pos
    if tag == b"y":
        return bytes(body), pos
    if tag == b"r":
        start, stop, step = (int(x) for x in body.decode("ascii").split(":"))
        return range(start, stop, step), pos
    if tag == b"t":
        items, p = [], 0
        while p < len(body):
            x, p = _decode_one(body, p)
            items.append(x)
        return tuple(items), pos
    if tag == b"F":
        items, p = [], 0
        while p < len(body):
            x, p = _decode_one(body, p)
            items.append(x)
        return frozenset(items), pos
    if tag == b"g":
        p = 0
        _, sbody, p = _read_blob(body, p)
        dt = np.dtype(sbody.decode("ascii"))
        return np.frombuffer(body[p:], dtype=dt)[0], pos
    raise ValueError(f"unknown key tag {tag!r}")


def decode_key(data: bytes) -> Any:
    """Inverse of :func:`encode_key`."""
    v, pos = _decode_one(data, 0)
    if pos != len(data):
        raise ValueError("trailing bytes in key encoding")
    return v
