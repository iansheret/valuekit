"""Read recording.

Passing an :class:`ImmutableMap` to a @pure function opts that argument into
fine-grained invalidation: on a cache miss the map is wrapped in a
:class:`RecordingMap`, which records what the function read.

* ``m[k]`` / ``m.get(k)`` on a leaf  → value dependency (content hash);
* ``m[k]`` / ``m.get(k)`` missing    → absence dependency;
* ``k in m``                          → presence/absence dependency;
* ``m[k]`` yielding a sub-map        → a child proxy with an extended path
  (reads inside it are recorded path-qualified);
* iteration, ``len``, ``keys/items/values``, ``==``, ``repr``, hashing,
  deriving with ``|`` / ``assoc`` / ``dissoc``, and anything else that
  observes the whole map → whole-map dependency (conservative).

The resulting trace is exactly the set of facts that must still hold for a
recorded result to be valid, which is what makes invalidation fine-grained:
an unrelated new key changes none of the recorded facts.

RecordingMap *is* an ImmutableMap, sharing the underlying storage, so a
function cannot distinguish one from the other: ``isinstance`` succeeds,
equality and repr agree, and deriving with ``|`` yields a plain map exactly
as it does outside a recorded call.  A proxy that escapes its call is inert
rather than wrong — the recorder closes when the trace is finalised.

A map handed from an outer @pure call into an inner one carries both
recorders (each with its own path), so one read is recorded in both traces:
the inner gets its own fine-grained trace, and the outer stays valid only
for maps that would drive the inner the same way.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .map import ImmutableMap, _h_immutable_map, map_digest
from .values import UnsupportedKeyError, digest, encode_key, freeze, hash_update

__all__ = ["Recorder", "RecordingMap", "unwrap_proxies"]

# Dependency kinds, in dominance order: a whole-map record at a path makes
# all records at or below that path redundant; a value record subsumes a
# presence record for the same path.
_RANK = {"present": 0, "absent": 0, "value": 1, "whole": 2}


class Recorder:
    """Accumulates (path → dependency) facts for one map argument."""

    def __init__(self) -> None:
        self.entries: dict[tuple, tuple] = {}  # path -> ("value", hex) | ("present",) | ("absent",) | ("whole", hex)
        self.closed = False

    def _put(self, path: tuple, dep: tuple) -> None:
        if self.closed:
            return
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
        """Compress and serialise: whole-map records shadow deeper records.

        Closes the recorder, so that a proxy outliving its call records
        nothing further and behaves as the plain map it wraps.
        """
        self.closed = True
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


class RecordingMap(ImmutableMap):
    """A read-observing ImmutableMap, live for the duration of one recorded
    call.  Shares the wrapped map's storage; adds only the observation."""

    __slots__ = ("_obs",)  # ((recorder, path), ...) — one entry per watcher

    def __init__(self, base: ImmutableMap, recorder: Recorder, path: tuple = ()):
        ImmutableMap.__init__(self, base)  # shares _d and any cached digest
        inherited = getattr(base, "_obs", ())
        object.__setattr__(self, "_obs", inherited + ((recorder, path),))

    @classmethod
    def _at(cls, base: ImmutableMap, obs: tuple) -> "RecordingMap":
        """Internal: a proxy over *base* with pre-built observers, used when
        descending into a sub-map."""
        out = object.__new__(cls)
        ImmutableMap.__init__(out, base)
        object.__setattr__(out, "_obs", obs)
        return out

    # -- fine-grained accesses ------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        try:
            encode_key(key)  # verify recordability before using the path
        except UnsupportedKeyError:
            self._observe_all()
            return self._d[key]
        try:
            v = self._d[key]
        except KeyError:
            for rec, path in self._obs:
                rec.absent(path + (key,))
            raise
        if isinstance(v, ImmutableMap):
            # Defer: record only what is read *inside* the sub-map.
            return RecordingMap._at(v, tuple((r, p + (key,)) for r, p in self._obs))
        d = digest(v)
        for rec, path in self._obs:
            rec.value(path + (key,), d)
        return v

    def __contains__(self, key: Any) -> bool:
        try:
            encode_key(key)
        except UnsupportedKeyError:
            self._observe_all()
            return key in self._d
        hit = key in self._d
        for rec, path in self._obs:
            if hit:
                rec.present(path + (key,))
            else:
                rec.absent(path + (key,))
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
        return ImmutableMap.__or__(self, other)

    def __ror__(self, other: Any) -> ImmutableMap:
        if not isinstance(other, Mapping):
            return NotImplemented
        self._observe_all()
        return ImmutableMap.__ror__(self, other)

    def assoc(self, key: Any, value: Any) -> ImmutableMap:
        self._observe_all()
        return ImmutableMap.assoc(self, key, value)

    def dissoc(self, *keys: Any) -> ImmutableMap:
        self._observe_all()
        return ImmutableMap.dissoc(self, *keys)

    def __reduce__(self):
        """Pickling reads everything, so it observes the whole map and
        reduces to a plain ImmutableMap: the proxy is meaningless outside
        the call that created it."""
        self._observe_all()
        return ImmutableMap.__reduce__(self)

    # -- whole-map observations (conservative fallback) -------------------------

    def _observe_all(self) -> None:
        d = map_digest(self)
        for rec, path in self._obs:
            rec.whole(path, d)

    def __iter__(self) -> Iterator[Any]:
        self._observe_all()
        return iter(self._d)

    def __len__(self) -> int:
        self._observe_all()
        return len(self._d)

    def __eq__(self, other: object) -> Any:
        self._observe_all()
        if isinstance(other, RecordingMap):
            other._observe_all()
        return ImmutableMap.__eq__(self, other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        # Deliberately indistinguishable from the map it wraps: a repr taken
        # inside a recorded call can end up embedded in the stored result,
        # where naming the proxy would outlive it.
        self._observe_all()
        return f"ImmutableMap({dict(self._d)!r})"


def unwrap_proxies(v: Any) -> Any:
    """Return *v* with every recording proxy replaced by the plain map it
    wraps, rebuilding only the containers that held one.

    Returning a map depends on the whole of it, so each replacement records
    that.  Only list, tuple and dict are searched: a set cannot hold a map,
    and an ImmutableMap freezes proxies out on entry.
    """
    if isinstance(v, RecordingMap):
        v._observe_all()
        return ImmutableMap(v)
    t = type(v)
    if t is list or t is tuple:
        items = [unwrap_proxies(x) for x in v]
        if all(a is b for a, b in zip(items, v)):
            return v
        return t(items)
    if t is dict:
        items = {k: unwrap_proxies(x) for k, x in v.items()}
        if all(items[k] is x for k, x in v.items()):
            return v
        return items
    return v


# ---------------------------------------------------------------------------
# Registry wiring — both whole-map reads
# ---------------------------------------------------------------------------


@freeze.register(RecordingMap)
def _freeze_recording_map(v: RecordingMap) -> ImmutableMap:
    """Putting a proxy into a map reads all of it, and the plain map is what
    goes in, so the proxy cannot outlive the call that created it."""
    v._observe_all()
    return ImmutableMap(v)


@hash_update.register(RecordingMap)
def _h_recording_map(v: RecordingMap, h: Any) -> None:
    """Content-hashing a proxy reads all of it."""
    v._observe_all()
    _h_immutable_map(v, h)
