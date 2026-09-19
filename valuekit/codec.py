"""The structural value format, independent of where values are stored.

A composite value is encoded as a tag and the content hashes of its
children; each child is its own object, reached through a callback.  That
one indirection is what lets the same format serve two purposes: a
:class:`~valuekit.store.LocalStore` passes its own ``put_value`` and
``get_value``, so children become files on disk, while a transport passes
"send this object unless the peer already has it" and "look it up among the
objects received", so children become messages.  Content-addressing then
deduplicates on the protocol for exactly the reason it deduplicates on disk --
an array shared by fifty values is transferred once.

There is no pickle here: the decode side reaches only a fixed set of
types, registered codecs, and dataclasses whose class the receiving
process has already imported, so bytes from another process describe
data and cannot name code.

ndarrays are the one thing this module does not handle.  Their
representation is ``np.save`` rather than a tag-and-hash structure, so each
caller frames them itself -- the store as ``.npy``/``.npyw`` files chosen by
writeability, a transport as its own message kind.  Writeability is part of
the content hash, so it must survive the round trip either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .map import ImmutableMap
from .plaindata import (
    is_dataclass_instance,
    plain_data_class,
    plain_data_state,
    rebuild_plain_data,
)
from .values import (
    _DIGEST_SIZE,
    _blob,
    _read_blob,
    custom_rebuild,
    custom_reduce,
    decode_key,
    encode_key,
)

__all__ = ["encode", "decode", "SerializationError"]


class SerializationError(TypeError):
    """A value cannot be stored. Only the fixed set of storable types may
    appear in cached return values."""


def encode(v: Any, put: Callable[[Any], str]) -> bytes:
    """Encode *v*, reaching its children through *put*.

    *put* takes a child value and returns its content hash, having made the
    child retrievable by that hash -- written to a store, or queued for a
    peer.  ndarrays are never passed here; the caller handles them.
    """
    t = type(v)
    if t is tuple:
        hs = [bytes.fromhex(put(x)) for x in v]
        return _blob(b"T", b"".join(hs))
    if t is list:
        hs = [bytes.fromhex(put(x)) for x in v]
        return _blob(b"L", b"".join(hs))
    if t is frozenset:
        hs = sorted(bytes.fromhex(put(x)) for x in v)
        return _blob(b"F", b"".join(hs))
    if t is set:
        hs = sorted(bytes.fromhex(put(x)) for x in v)
        return _blob(b"S", b"".join(hs))
    if isinstance(v, ImmutableMap):
        pairs = sorted(
            (bytes.fromhex(put(k)), bytes.fromhex(put(val))) for k, val in v.items()
        )
        return _blob(b"M", b"".join(k + val for k, val in pairs))
    if isinstance(v, Mapping):
        # Insertion order is preserved, matching how a plain mapping is
        # hashed: it is observable, so it is part of the value.
        pairs = [
            (bytes.fromhex(put(k)), bytes.fromhex(put(val))) for k, val in v.items()
        ]
        return _blob(b"D", b"".join(k + val for k, val in pairs))
    reduced = custom_reduce(v)
    if reduced is not None:
        name, red = reduced
        h = bytes.fromhex(put(red))
        return _blob(b"C", _blob(b"s", name.encode("utf-8")) + h)
    if is_dataclass_instance(v):
        # A plain-data dataclass stores its identity beside its field
        # values, so a class that has changed since is refused on the way
        # back out rather than rebuilt into something it no longer means.
        name, field_names, params_key, values = plain_data_state(v)
        try:
            plain_data_class(name)
        except ValueError as e:
            raise SerializationError(
                f"Cannot cache a {t.__name__!r}: {e}, so a stored entry "
                "could never be rebuilt. Define it at module level, or "
                "register it with valuekit.register_type()."
            ) from None
        h = bytes.fromhex(put(values))
        return _blob(
            b"P",
            _blob(b"s", name.encode("utf-8"))
            + _blob(b"s", ",".join(field_names).encode("utf-8"))
            + _blob(b"s", params_key.encode("utf-8"))
            + h,
        )
    try:
        return _blob(b"I", encode_key(v))  # atomics + np scalars
    except Exception:
        raise SerializationError(
            f"Cannot cache a value of type {t.__name__!r}. Cached return "
            "values are limited to: None, bool, int, float, complex, str, "
            "bytes, range, numpy scalars/arrays, tuples, lists, sets, "
            "frozensets, dicts, ImmutableMaps and plain-data dataclasses "
            "of the same -- or a type registered with a "
            "reduce_fn/rebuild_fn via valuekit.register_type()."
        ) from None


def decode(data: bytes, get: Callable[[str], Any]) -> Any:
    """Rebuild a value encoded by :func:`encode`, children through *get*."""
    tag, body, pos = _read_blob(data, 0)
    if pos != len(data):
        raise ValueError("trailing bytes in stored value")
    if tag == b"I":
        return decode_key(body)
    if tag == b"C":
        _, name, p = _read_blob(body, 0)
        reduced = get(body[p:].hex())
        return custom_rebuild(name.decode("utf-8"), reduced)
    if tag == b"P":
        _, name, p = _read_blob(body, 0)
        _, names, p = _read_blob(body, p)
        _, params_key, p = _read_blob(body, p)
        joined = names.decode("utf-8")
        return rebuild_plain_data(
            name.decode("utf-8"),
            tuple(joined.split(",")) if joined else (),
            params_key.decode("utf-8"),
            get(body[p:].hex()),
        )
    n = _DIGEST_SIZE
    hs = [body[i : i + n].hex() for i in range(0, len(body), n)]
    if tag == b"T":
        return tuple(get(x) for x in hs)
    if tag == b"L":
        return [get(x) for x in hs]
    if tag == b"F":
        return frozenset(get(x) for x in hs)
    if tag == b"S":
        return {get(x) for x in hs}
    if tag in (b"M", b"D"):
        d = {}
        for i in range(0, len(hs), 2):
            d[get(hs[i])] = get(hs[i + 1])
        return ImmutableMap(d) if tag == b"M" else d
    raise ValueError(f"unknown value tag {tag!r}")
