"""ImmutableMap — an immutable mapping for pipeline data.

Values that are already immutable (int, str, float, …) are stored as-is.
For types it knows how to handle (e.g. numpy arrays) it stores an immutable
copy. Unknown mutable types are rejected.

There is no in-place mutation: derive new versions with ``|``, and the
original is never modified::

    ctx  = ImmutableMap({"raw": signal, "fs": 1000.0})
    ctx2 = ctx  | {"scaled": ctx["raw"] * gain}   # add a key
    ctx3 = ctx2 | {"raw": detrended}              # override — ctx2 untouched
    ctx4 = ctx3.dissoc("tmp")                     # drop a key
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .values import _frame, _new_hasher, digest, freeze, hash_update

__all__ = ["ImmutableMap"]


class ImmutableMap(Mapping):
    """An immutable mapping. Values are frozen on entry via ``freeze``:
    already-immutable values are stored as-is, known types (e.g. numpy
    arrays) are stored as an immutable copy, and unknown mutable types
    are rejected.

    Parameters
    ----------
    data:
        A ``dict``, ``Mapping``, or ``ImmutableMap`` to initialise from.
        Omit or pass ``None`` for an empty map.
    """

    __slots__ = ("_d", "_digest")

    def __init__(self, data: Mapping | None = None) -> None:
        if isinstance(data, ImmutableMap):
            # Fast path: the source is already fully frozen; share its internals
            # (including any already-computed content digest).
            object.__setattr__(self, "_d", data._d)
            object.__setattr__(self, "_digest", data._digest)
        else:
            d: dict[Any, Any] = {}
            for k, v in (data or {}).items():
                d[k] = freeze(v)
            object.__setattr__(self, "_d", d)
            object.__setattr__(self, "_digest", None)

    @classmethod
    def _from_frozen_dict(cls, d: dict) -> "ImmutableMap":
        """Internal: build from a dict whose values are already frozen."""
        out = object.__new__(cls)
        object.__setattr__(out, "_d", d)
        object.__setattr__(out, "_digest", None)
        return out

    # ------------------------------------------------------------------
    # Mapping read interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    # __setitem__ and __delitem__ are intentionally absent; Python raises
    # TypeError naturally for `m["k"] = v` and `del m["k"]`.

    # ------------------------------------------------------------------
    # Block direct mutation of internal state
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ImmutableMap does not support attribute assignment")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ImmutableMap does not support attribute deletion")

    # ------------------------------------------------------------------
    # Persistent derivation
    # ------------------------------------------------------------------

    def __or__(self, other: Any) -> "ImmutableMap":
        """Return a new ImmutableMap merging self with other (other wins on conflict).

        Existing frozen values are shared by reference; no copies are made.
        """
        if not isinstance(other, Mapping):
            return NotImplemented
        d = dict(self._d)  # shallow copy: preserves frozen references
        for k, v in other.items():
            d[k] = freeze(v)
        return ImmutableMap._from_frozen_dict(d)

    def __ror__(self, other: Any) -> "ImmutableMap":
        """Return a new ImmutableMap merging other with self (self wins on conflict)."""
        if not isinstance(other, Mapping):
            return NotImplemented
        d: dict[Any, Any] = {}
        for k, v in other.items():
            d[k] = freeze(v)
        d.update(self._d)  # self wins; values already frozen
        return ImmutableMap._from_frozen_dict(d)

    def assoc(self, key: Any, value: Any) -> "ImmutableMap":
        """Return a new ImmutableMap with ``key`` set to ``value``."""
        return self | {key: value}

    def dissoc(self, *keys: Any) -> "ImmutableMap":
        """Return a new ImmutableMap without the specified keys."""
        drop = set(keys)
        return ImmutableMap._from_frozen_dict(
            {k: v for k, v in self._d.items() if k not in drop}
        )

    # ------------------------------------------------------------------
    # Equality, hashing, display
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableMap):
            return NotImplemented
        if self._d.keys() != other._d.keys():
            return False
        for k in self._d:
            a, b = self._d[k], other._d[k]
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]  # explicitly unhashable

    def __reduce__(self):
        """Support pickling (e.g. for process-based parallelism): rebuild
        from the underlying dict, re-freezing values on arrival (arrays
        unpickle writeable, so the receiving process copies them
        read-only)."""
        return (ImmutableMap, (dict(self._d),))

    def __repr__(self) -> str:
        return f"ImmutableMap({dict(self._d)!r})"


# ---------------------------------------------------------------------------
# Registry wiring (must follow ImmutableMap definition)
# ---------------------------------------------------------------------------


@freeze.register(Mapping)
def _freeze_mapping(v: Mapping) -> ImmutableMap:
    """Convert any Mapping to an ImmutableMap; pass ImmutableMaps through."""
    if isinstance(v, ImmutableMap):
        return v  # already fully frozen; share as-is
    return ImmutableMap(v)


def map_digest(m: ImmutableMap) -> bytes:
    """Content digest of an ImmutableMap, cached on the instance.

    Order-independent: computed from sorted (key digest, value digest)
    pairs, so two maps with equal contents digest identically regardless of
    insertion order. Derived maps share value references with their parents,
    so re-hashing a large derived map costs only the hashes of the new
    values.
    """
    cached = m._digest
    if cached is not None:
        return cached
    pairs = sorted(digest(k) + digest(v) for k, v in m._d.items())
    h = _new_hasher()
    _frame(h, b"m", len(pairs).to_bytes(8, "little"))
    for p in pairs:
        h.update(p)
    d = h.digest()
    object.__setattr__(m, "_digest", d)
    return d


@hash_update.register
def _h_immutable_map(v: ImmutableMap, h: Any) -> None:
    _frame(h, b"M", map_digest(v))


@hash_update.register(Mapping)
def _h_mapping(v: Mapping, h: Any) -> None:
    # A raw dict (or RecordingMap) hashes as its frozen equivalent.
    hash_update(_freeze_mapping(v), h)
