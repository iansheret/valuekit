"""The @pure decorator.

``@pure`` asserts that a function is pure: its output depends only on what
it reads from its inputs, and it has no observable effects.  Pure functions
can be memoised, so valuekit memoises them.

Arguments are hashed, not converted: a dict stays a dict, a list stays a
list, an array keeps its writeability.  A cache hit returns a value equal to
what the call would have produced, of the same type — including on the miss
that recorded it, where the function's own object is handed straight back.

Fine-grained invalidation is opt-in, and the opt-in is passing an
:class:`~valuekit.ImmutableMap`.  Such an argument is wrapped on a miss in a
recording proxy (itself an ImmutableMap, so the function cannot tell); the
function runs; the observed reads, plus whole-value content hashes of every
other argument, become a *trace*, stored alongside the content hash of the
return value.

On a lookup, the stored traces for the function's fingerprint are scanned.
If every recorded fact still holds against the current arguments (same
values at the read paths, same absences, same whole-map hashes where the
function observed everything), the stored result is returned without
executing.  Keys the function never read are irrelevant, so unrelated
additions to a data or config map do not invalidate — whereas a plain dict
argument is depended on whole, since nothing observed how it was used.

Note that the function body does not run on a hit: logging, plotting, and
any other side effect inside a @pure function is skipped.

With no cache directory configured, @pure is a plain call.  For debugging
see :mod:`valuekit.debughook`: a breakpoint anywhere in the function's
user-code closure forces execution, without writing.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import sys
from typing import Any, Callable

from .codehash import _module_unit, _unit_digest, function_fingerprint
from .debughook import breakpoints_force
from .map import ImmutableMap, map_digest
from .recording import Recorder, RecordingMap, unwrap_proxies
from .store import CacheMiss, CacheStore, LocalStore
from .values import content_hash, decode_key, digest

__all__ = ["pure", "set_cache_dir", "clear_cache"]

_MISSING = object()

# ---------------------------------------------------------------------------
# store configuration
# ---------------------------------------------------------------------------

_store: CacheStore | None = None

# Incremented whenever a @pure call executes because a debugger forced it.
# A recording snapshots it before running and is not stored if it changed:
# a forced run inside a recorded call invalidates the enclosing recording
# too (e.g. a breakpoint added mid-run, after the outer entry check passed).
_force_epoch = 0


def _current_store() -> "CacheStore | None":
    """Internal: the configured store, late-bound (used by valuekit.parallel;
    the name ``pure`` in the package namespace shadows this module)."""
    return _store


def set_cache_dir(path: str | os.PathLike | None) -> None:
    """Configure the cache directory (or None to disable caching).

    Nothing is configured until this is called, so importing valuekit never
    enables disk caching by itself.  To drive it from the environment, read
    the variable explicitly::

        set_cache_dir(os.environ.get("VALUEKIT_CACHE"))
    """
    global _store
    _store = None if path is None else LocalStore(path)


def clear_cache(fn: Callable | None = None) -> None:
    """Delete cached results. Always safe: the worst case is recomputation.

    ``clear_cache()`` deletes everything in the configured cache.
    ``clear_cache(fn)`` states that *fn* has changed and behaves as if it
    had: it deletes the recorded traces of *fn* and of every @pure function
    whose closure contains it (callers, transitively), plus any traces
    where *fn* appeared as an argument.  Matching is conservative (a caller
    that reached *fn* through its module clears everything that referenced
    that module, and textually identical function bodies clear together):
    clearing too much means recomputing, while clearing too little would
    mean wrong results.  Stored values are content-addressed and shared, so
    they are not deleted; orphaned objects only occupy disk space and are
    removed by a full clear.
    """
    if not isinstance(_store, LocalStore):
        return
    if fn is None:
        _store.clear()
        return
    raw = getattr(fn, "__wrapped__", None)
    if not getattr(fn, "_valuekit_pure", False) or raw is None:
        raise TypeError(
            f"clear_cache() takes a @pure-decorated function; got "
            f"{getattr(fn, '__qualname__', fn)!r}"
        )
    units = {_unit_digest(raw.__code__)}
    mod = sys.modules.get(getattr(raw, "__module__", "") or "")
    if mod is not None:
        u = _module_unit(mod)
        if u:
            units.add(u)
    try:
        value_hash = content_hash(raw)
    except Exception:
        value_hash = None
    _store.drop_dependents(units, value_hash)


# Every stored entry is salted with this, so bumping it invalidates every
# cache everywhere.  Bump it when a release changes what a fingerprint or a
# trace means; releases that do not are then free to leave caches intact.
CACHE_EPOCH = 2


def _salt() -> str:
    return (
        f"valuekit-epoch{CACHE_EPOCH}"
        f"|py{sys.version_info.major}.{sys.version_info.minor}"
    )


# ---------------------------------------------------------------------------
# trace matching
# ---------------------------------------------------------------------------


def _navigate(m: Any, keys: tuple) -> Any:
    """Walk a decoded path through nested ImmutableMaps; _MISSING on absence."""
    cur = m
    for k in keys:
        if not isinstance(cur, ImmutableMap) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


def _match_map(entries: list[dict], m: ImmutableMap) -> bool:
    for e in entries:
        path = tuple(decode_key(bytes.fromhex(p)) for p in e["path"])
        dep = e["dep"]
        if dep == "whole":
            sub = _navigate(m, path)
            if not isinstance(sub, ImmutableMap):
                return False
            if map_digest(sub).hex() != e["hash"]:
                return False
        elif dep == "value":
            v = _navigate(m, path)
            if v is _MISSING or isinstance(v, ImmutableMap):
                return False
            if digest(v).hex() != e["hash"]:
                return False
        elif dep == "present":
            if _navigate(m, path) is _MISSING:
                return False
        elif dep == "absent":
            parent = _navigate(m, path[:-1])
            if not isinstance(parent, ImmutableMap) or path[-1] in parent:
                return False
        else:  # unknown dep kind from a future format: never match
            return False
    return True


def _match_trace(
    trace: dict, arguments: dict[str, Any], arg_hashes: dict[str, str]
) -> bool:
    deps = trace.get("deps", {})
    if set(deps) != set(arguments):
        return False
    for name, dep in deps.items():
        arg = arguments[name]
        if dep["kind"] == "value":
            if isinstance(arg, ImmutableMap):
                return False
            if arg_hashes[name] != dep["hash"]:
                return False
        elif dep["kind"] == "map":
            if not isinstance(arg, ImmutableMap):
                return False
            if not _match_map(dep["entries"], arg):
                return False
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# the decorator
# ---------------------------------------------------------------------------


def pure(fn: Callable):
    """Assert that *fn* is pure; memoise it on that basis.

    Purity contract (the caller's promise):
      1. Determinism over inputs: the same read values give the same
         result. No ambient RNG or clock reads reaching the result, no
         file or network reads, no dependence on mutable module state.
      2. No observable effects that matter: on a cache hit the body does
         not run, so prints, plots, and file writes inside it will not
         happen.
      3. No mutation of the arguments. They are passed through as they
         were given, so a function that writes to one is not pure and the
         trace recorded against it will be wrong.
      4. The result is reachable from the arguments plus the function's own
         definition (hashed: everything reachable by name through user
         code).

    Arguments and results must be content-hashable, and results must also
    be storable; both sets are extended with ``register_type``. Pass an
    ImmutableMap to get key-level invalidation for that argument; any other
    argument is depended on whole.

    Takes no options. If a dependency is not reachable by name, pass it as
    an argument; if something invisible changed anyway, call
    ``clear_cache(fn)``.
    """

    qn = getattr(fn, "__qualname__", "")
    parts = qn.split(".")
    if len(parts) >= 2 and parts[-2] != "<locals>":
        raise TypeError(
            f"@pure does not support methods ({qn!r}): what would hashing "
            "'self' mean? Use a module-level function taking explicit values."
        )
    sig = inspect.signature(fn)
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(
                f"@pure requires an explicit signature; {qn!r} uses "
                f"*{p.name} / **{p.name}, which defeats stable cache keys."
            )

    # The function's own code object is captured now, before any debugger
    # patches its bytecode, but names are resolved and the fingerprint
    # computed at the first call, when the module is fully loaded: definition
    # order does not matter, forward references are tracked, and mutual
    # recursion between @pure functions works.
    orig_code = fn.__code__
    _ident: list = []  # [(fn_key, spans, units)] once computed; benign races

    def _identity() -> tuple[str, list, list]:
        if not _ident:
            fingerprint, spans, units = function_fingerprint(fn, code=orig_code)
            fn_key = hashlib.blake2b(
                (_salt() + "|" + fingerprint).encode(), digest_size=20
            ).hexdigest()
            _ident.append((fn_key, spans, units))
        return _ident[0]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        store = _store
        if store is None:
            return fn(*args, **kwargs)

        fn_key, spans, units = _identity()

        # A live breakpoint in this function's closure: execute without
        # reading or writing the cache, so nothing from a debug session can
        # enter the store.
        if breakpoints_force(spans):
            global _force_epoch
            _force_epoch += 1
            return fn(*args, **kwargs)

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)  # the caller's own objects

        # Hash the non-map arguments once, before the call: the same hashes
        # match every candidate trace and then go into the trace written for
        # a miss, so what is recorded is the arguments as they were passed.
        arg_hashes = {
            name: content_hash(v)
            for name, v in arguments.items()
            if not isinstance(v, ImmutableMap)
        }

        # -- lookup ---------------------------------------------------------
        for trace in store.get_traces(fn_key):
            if _match_trace(trace, arguments, arg_hashes):
                try:
                    return store.get_value(trace["result"])
                except CacheMiss:
                    continue  # value evicted/corrupt: try others, else rerun

        # Wrap ImmutableMap arguments so their reads are observed; everything
        # else is passed through untouched and depended on whole.
        recorders: dict[str, Recorder] = {}
        for name, v in arguments.items():
            if isinstance(v, ImmutableMap):
                rec = Recorder()
                recorders[name] = rec
                bound.arguments[name] = RecordingMap(v, rec)

        epoch_before = _force_epoch
        result = fn(*bound.args, **bound.kwargs)  # exceptions: cache untouched

        # A proxy must not outlive the call it belongs to, wherever in the
        # result it sits.
        if recorders:
            result = unwrap_proxies(result)

        if _force_epoch != epoch_before:
            # Something in this call's dynamic extent was debugger-forced
            # (a breakpoint appeared after our own entry check): this result
            # may reflect a debug session, so it must not be persisted.
            return result

        # Store before finalising the recorders, so that a proxy reached only
        # by the encoder (inside a custom reduced form, say) still records the
        # whole-map dependency that returning it implies.
        result_hash = store.put_value(result)
        deps: dict[str, dict] = {}
        for name in arguments:
            if name in recorders:
                deps[name] = {"kind": "map", "entries": recorders[name].finalize()}
            else:
                deps[name] = {"kind": "value", "hash": arg_hashes[name]}
        store.put_trace(
            fn_key,
            {"fn": qn, "deps": deps, "result": result_hash},
            units=units,
        )
        return result

    wrapper.uncached = fn  # override: call the raw function directly
    wrapper.__wrapped__ = fn
    wrapper._valuekit_pure = True
    wrapper._valuekit_identity = _identity
    return wrapper
