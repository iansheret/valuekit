"""The unified type registry: freeze, content-hash, and inline key codec.

Content hashing is the primitive.  A content hash *identifies a value
exactly*: two values hash the same only if no Python program could tell
them apart.  So a list and a tuple of the same items hash differently, two
dicts hash differently if their iteration order differs, and a writeable
array differs from a read-only one.  This is what lets a content-addressed
store hand back, on a hit, exactly what the miss produced.

Freezing is separate, and narrower: it is what :class:`ImmutableMap` applies
to values on entry.  ``@pure`` does not freeze anything, so a type needs a
freeze strategy only if it is to be stored *in a map*.

Hence the three tiers of :func:`register_type`: a hash strategy alone makes
a type usable as a ``@pure`` argument; adding a freeze strategy lets it go
into an ImmutableMap; adding a reduce/rebuild pair lets it come back out of
a cached return value.

The ImmutableMap handlers are registered in :mod:`valuekit.map` (they need
the class itself); the function handlers are registered in
:mod:`valuekit.codehash` (they need the code hasher); plain-data dataclasses
are handled in :mod:`valuekit.plaindata`, which needs no registration at all
because the type it recognises is a family rather than a class.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
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
    """Return an immutable form of *v*, for storing it in an ImmutableMap.

    * Already-immutable values (int, float, complex, bool, str, bytes, None,
      frozenset, range, numpy scalars) are returned unchanged.
    * Types with a known freezing strategy are returned as an immutable copy:
      numpy arrays are made read-only, sets become frozensets, dicts become
      ImmutableMaps, and tuples have their contents frozen.
    * Anything else raises TypeError; register new types with
      :func:`register_type`.

    Note that this is applied by ImmutableMap on entry, and nowhere else:
    ``@pure`` hashes its arguments and returns without converting them.
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


# Handlers for type *families*, which singledispatch cannot key on: each is
# tried in turn on a value no registered type claims, and reports whether it
# hashed it.  Plain-data dataclasses register one in :mod:`valuekit.plaindata`.
_hash_fallbacks: list[Callable[[Any, Any], bool]] = []


@singledispatch
def hash_update(v: Any, h: Any) -> None:
    """Feed the content of *v* into hasher *h*. Dispatch on type of *v*."""
    for fallback in _hash_fallbacks:
        if fallback(v, h):
            return
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
    # Writeability is part of the identity: a caller can tell a read-only
    # array from a writeable one, so a cached return must come back as
    # whichever it was, and the store keys the two separately.
    flag = "w" if v.flags.writeable else "r"
    meta = f"{v.dtype.str}|{v.shape}|{flag}".encode("ascii")
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
def _h_list(v: list, h: Any) -> None:
    # Tagged apart from tuple: the two are distinguishable values, so
    # f([1, 2]) and f((1, 2)) are distinct calls.
    _frame(h, b"l", len(v).to_bytes(8, "little"))
    for x in v:
        hash_update(x, h)


@hash_update.register
def _h_frozenset(v: frozenset, h: Any) -> None:
    ds = sorted(digest(x) for x in v)
    _frame(h, b"F", len(v).to_bytes(8, "little") + b"".join(ds))


@hash_update.register
def _h_set(v: set, h: Any) -> None:
    ds = sorted(digest(x) for x in v)
    _frame(h, b"S", len(v).to_bytes(8, "little") + b"".join(ds))


@hash_update.register(Mapping)
def _h_mapping(v: Mapping, h: Any) -> None:
    """Hash a plain mapping in iteration order.

    A dict's order is observable, so two dicts differing only in it are
    different values.  ImmutableMap is order-independent by definition and
    registers its own handler in :mod:`valuekit.map`.
    """
    _frame(h, b"d", len(v).to_bytes(8, "little"))
    for k, val in v.items():
        hash_update(k, h)
        hash_update(val, h)


# ---------------------------------------------------------------------------
# registration API — hash always, freeze and codec as needed
# ---------------------------------------------------------------------------


_reduce_by_type: dict[type, tuple[str, Callable[[Any], Any]]] = {}
_rebuild_by_name: dict[str, Callable[[Any], Any]] = {}


def register_type(
    cls: type,
    *,
    hash_fn: Callable[[Any, Any], None],
    freeze_fn: Callable[[Any], Any] | None = None,
    reduce_fn: Callable[[Any], Any] | None = None,
    rebuild_fn: Callable[[Any], Any] | None = None,
    name: str | None = None,
) -> None:
    """Register a new value type, in as many of three tiers as it needs.

    ``hash_fn(v, hasher)`` is always required: it must feed a stable,
    content-based encoding of *v* into *hasher* (hashlib-style
    ``hasher.update``), identifying the value exactly, and identically
    across processes and machines. With it alone, values of the type can be
    ``@pure`` arguments.

    ``freeze_fn(v)`` returns an immutable form of *v*, and is what lets the
    type be stored *in an ImmutableMap*. Omit it for a type that is only
    ever passed as an argument; a map will then reject it.

    ``reduce_fn`` / ``rebuild_fn`` let the type appear in a cached *return*
    value: ``reduce_fn(v)`` returns a storable form (built from the fixed
    storable set) and ``rebuild_fn(reduced)`` reconstructs the value. Pass
    both or neither. ``name`` identifies the type in stored entries
    (default ``"module:qualname"``); keep it stable, since entries reference
    it and a change reads as a miss.
    """
    if hash_fn is None:
        raise TypeError("register_type requires a hash_fn")
    if (reduce_fn is None) != (rebuild_fn is None):
        raise TypeError(
            "register_type requires BOTH reduce_fn and rebuild_fn, or neither"
        )
    hash_update.register(cls, hash_fn)
    if freeze_fn is not None:
        freeze.register(cls, freeze_fn)
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
