"""Read recording.

During a cache miss, each ImmutableMap argument of a @pure function is
wrapped in a :class:`RecordingMap`.  The proxy delegates to the underlying
map and records what the function read:

* ``m[k]`` / ``m.get(k)`` on a leaf  → value dependency (content hash);
* ``m[k]`` / ``m.get(k)`` missing    → absence dependency;
* ``k in m``                          → presence/absence dependency;
* ``m[k]`` yielding a sub-map        → a child proxy with an extended path
  (reads inside it are recorded path-qualified);
* iteration, ``len``, ``keys/items/values``, ``==``, ``repr``, deriving with
  ``|`` / ``assoc`` / ``dissoc``, and anything else that observes the whole
  map → whole-map dependency (conservative).

The proxy carries the whole of ImmutableMap's interface, so a @pure function
cannot tell that it received one: deriving a new map with ``|`` works inside a
recorded call exactly as it does outside, and yields a plain ImmutableMap.

The resulting trace is exactly the set of facts that must still hold for a
recorded result to be valid, which is what makes invalidation fine-grained:
an unrelated new key changes none of the recorded facts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .map import ImmutableMap, map_digest
from .values import UnsupportedKeyError, digest, encode_key

__all__ = ["Recorder", "RecordingMap"]

# Dependency kinds, in dominance order: a whole-map record at a path makes
# all records at or below that path redundant; a value record subsumes a
# presence record for the same path.
_RANK = {"present": 0, "absent": 0, "value": 1, "whole": 2}


class Recorder:
    """Accumulates (path → dependency) facts for one map argument."""

    def __init__(self) -> None:
        self.entries: dict[tuple, tuple] = {}  # path -> ("value", hex) | ("present",) | ("absent",) | ("whole", hex)

    def _put(self, path: tuple, dep: tuple) -> None:
        old = self.entries.get(path)
        if old is None or _RANK[dep[0]] >= _RANK[old[0]]:
            self.entries[path] = dep

    def value(self, path: tuple, d: bytes) -> None:
        self._put(path, ("value", d.hex()))

    def present(self, path: tuple) -> None:
        self._put(path, ("present",))

    def absent(self, path: tuple) -> None:
        self._put(path, ("absent",))

    def whole(self, path: tuple, d: bytes) -> None:
        self._put(path, ("whole", d.hex()))

    def finalize(self) -> list[dict]:
        """Compress and serialise: whole-map records shadow deeper records."""
        wholes = [p for p, dep in self.entries.items() if dep[0] == "whole"]
        out = []
        for path, dep in sorted(self.entries.items(), key=lambda kv: len(kv[0])):
            shadowed = any(len(w) < len(path) and path[: len(w)] == w for w in wholes)
            if shadowed:
                continue
            entry: dict[str, Any] = {
                "path": [encode_key(k).hex() for k in path],
                "dep": dep[0],
            }
            if len(dep) > 1:
                entry["hash"] = dep[1]
            out.append(entry)
        return out


class RecordingMap(Mapping):
    """A read-observing view over an ImmutableMap. Not itself persistent —
    it exists only for the duration of one recorded call."""

    __slots__ = ("_base", "_rec", "_path")

    def __init__(self, base: ImmutableMap, recorder: Recorder, path: tuple = ()):
        self._base = base
        self._rec = recorder
        self._path = path

    # -- fine-grained accesses ------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        try:
            keypath = self._path + (key,)
            encode_key(key)  # verify recordability before using the path
        except UnsupportedKeyError:
            self._rec.whole(self._path, map_digest(self._base))
            return self._base[key]
        try:
            v = self._base[key]
        except KeyError:
            self._rec.absent(keypath)
            raise
        if isinstance(v, ImmutableMap):
            # Defer: record only what is read *inside* the sub-map.
            return RecordingMap(v, self._rec, keypath)
        self._rec.value(keypath, digest(v))
        return v

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: Any) -> bool:
        try:
            encode_key(key)
        except UnsupportedKeyError:
            self._rec.whole(self._path, map_digest(self._base))
            return key in self._base
        hit = key in self._base
        if hit:
            self._rec.present(self._path + (key,))
        else:
            self._rec.absent(self._path + (key,))
        return hit

    # -- derivation -------------------------------------------------------------
    #
    # Every derived form reads all of self: ``|`` copies the underlying dict and
    # ``dissoc`` filters it.  So each records a whole-map dependency and returns
    # a plain ImmutableMap.  Reads of the result need no recording of their own,
    # because a result derived from the whole of self is already covered by the
    # dependency on the whole of self.

    def __or__(self, other: Any) -> ImmutableMap:
        if not isinstance(other, Mapping):
            return NotImplemented
        self._observe_all()
        return self._base | other

    def __ror__(self, other: Any) -> ImmutableMap:
        if not isinstance(other, Mapping):
            return NotImplemented
        self._observe_all()
        return other | self._base

    def assoc(self, key: Any, value: Any) -> ImmutableMap:
        self._observe_all()
        return self._base.assoc(key, value)

    def dissoc(self, *keys: Any) -> ImmutableMap:
        self._observe_all()
        return self._base.dissoc(*keys)

    def __reduce__(self):
        """Pickling reads everything, so it observes the whole map and reduces
        to the ImmutableMap being proxied: the proxy itself is meaningless
        outside the call that created it."""
        self._observe_all()
        return self._base.__reduce__()

    # -- whole-map observations (conservative fallback) -------------------------

    def _observe_all(self) -> None:
        self._rec.whole(self._path, map_digest(self._base))

    def __iter__(self) -> Iterator[Any]:
        self._observe_all()
        return iter(self._base)

    def __len__(self) -> int:
        self._observe_all()
        return len(self._base)

    def __eq__(self, other: object) -> Any:
        self._observe_all()
        base = self._base
        if isinstance(other, RecordingMap):
            other._observe_all()
            other = other._base
        return base == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        self._observe_all()
        return f"RecordingMap({dict(self._base._d)!r})"
