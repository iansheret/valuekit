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

A trace also records what the computation *bound*: every memoised call made
inside it (function name, key and trace hash, in completion order) and
every :func:`log` call (name and value hash).  These are facts about the
computation, stored with its result, so a later hit needs nothing replayed:
the bindings are in the trace it found.  :mod:`valuekit.batches` reads them.

``@pure_local`` is memoised identically but promises less about the code
and more about the world: the result may depend on this machine's
environment (credentials, local files, a network the function can reach),
which the user promises does not change.  It runs only on the machine the
user configured, never on a remote worker.

Note that the function body does not run on a hit: prints, plots, and any
other side effect inside a @pure function are skipped.

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
import time
from contextvars import ContextVar
from typing import Any, Callable

from . import runlog
from .codehash import _module_unit, _unit_digest, function_fingerprint
from .debughook import breakpoints_force
from .map import ImmutableMap, map_digest
from .recording import Recorder, RecordingMap, unwrap_proxies
from .store import CacheMiss, CacheStore, LocalStore
from .values import content_hash, decode_key, digest

__all__ = ["pure", "pure_local", "log", "set_cache_dir", "clear_cache"]

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


def set_store(store: CacheStore | None) -> None:
    """Internal: use *store* directly (a worker whose store is a peer)."""
    global _store
    _store = store


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
CACHE_EPOCH = 3


def _salt() -> str:
    return (
        f"valuekit-epoch{CACHE_EPOCH}"
        f"|py{sys.version_info.major}.{sys.version_info.minor}"
    )


# ---------------------------------------------------------------------------
# the executing call
# ---------------------------------------------------------------------------


class _Frame:
    """What the memoised call currently executing has bound so far.

    ``calls`` and ``logs`` go into its trace.  A ``discard`` frame belongs
    to a debugger-forced run: it accepts nothing and stores nothing.
    """

    __slots__ = ("store", "calls", "logs", "discard")

    def __init__(self, store: CacheStore, discard: bool = False):
        self.store = store
        self.calls: list[list] = []
        self.logs: list[list] = []
        self.discard = discard


# The innermost executing call.  A context variable rather than a global:
# each thread sees its own, and the token reset in ``finally`` restores the
# enclosing frame on any exit.  A spawned worker starts at None, which is
# right: its root call has no enclosing computation in that process.
_ctx: ContextVar[_Frame | None] = ContextVar("valuekit_call", default=None)


def _record_call(qn: str, fn_key: str, h: str) -> None:
    """Note a completed memoised call in the enclosing computation, if any."""
    frame = _ctx.get()
    if frame is not None and not frame.discard:
        frame.calls.append([qn, fn_key, h])


def log(name: str, value: Any) -> None:
    """Bind *name* to *value* in the computation currently executing.

    The value is stored like a result (it must be storable) and the binding
    is recorded in the enclosing call's trace, so it is found again on every
    later hit without the body running.  Read it back through
    :func:`valuekit.batch`.

    Outside a memoised call there is no computation to bind to: with a cache
    configured this raises, since a dropped log is worse than a refused one;
    with none it does nothing, like everything else here.
    """
    if not isinstance(name, str):
        raise TypeError(f"log() name must be a str, got {type(name).__name__}")
    frame = _ctx.get()
    if frame is None:
        if _store is not None:
            raise RuntimeError(
                f"valuekit.log({name!r}, ...) was called outside a @pure "
                "function; there is no computation to attach it to."
            )
        return
    if frame.discard:
        return
    frame.logs.append([name, frame.store.put_value(value)])


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


def _matches(store: CacheStore, fn_key: str, arguments: dict, arg_hashes: dict):
    """The stored traces that hold for these arguments, newest first."""
    for h, trace in store.get_traces(fn_key):
        if _match_trace(trace, arguments, arg_hashes):
            yield h, trace


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

    The wrapper has ``cached(*args)``, which returns the stored result or
    raises :class:`CacheMiss` without ever executing, and ``uncached``, the
    raw function.
    """
    return _pure(fn, local=False)


def pure_local(fn: Callable):
    """Memoise *fn* as ``@pure`` does, on a different promise.

    A ``@pure`` function's result depends on its arguments and its
    definition, and nothing else.  A ``@pure_local`` function's result may
    also depend on something outside the program -- a file on this machine,
    a database, a download that needs this machine's credentials -- and the
    promise is that this thing, as seen through these arguments, never
    changes: the same arguments give the same value, now and later.  If that
    cannot be promised, put the version in the arguments (a date, a commit,
    an etag), where it is tracked like everything else.

    Because the function reads an environment, it runs only on the machine
    that has that environment: this one, where the pipeline is driven.  A
    batch running elsewhere sends such calls back here.  Effects that do not
    reach the result (a scratch file, a download cache) are permitted, since
    on a hit none of them happen.  The result must be a value, never a path:
    a path from this machine means nothing on another.
    """
    return _pure(fn, local=True)


def _pure(fn: Callable, *, local: bool):
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

    def _bind(args, kwargs):
        """Bind the call and hash its non-map arguments, once.

        The same hashes match every candidate trace and then go into the
        trace written for a miss, so what is recorded is the arguments as
        they were passed.
        """
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)  # the caller's own objects
        arg_hashes = {
            name: content_hash(v)
            for name, v in arguments.items()
            if not isinstance(v, ImmutableMap)
        }
        return bound, arguments, arg_hashes

    def _hit(store, fn_key, arguments, arg_hashes, t_lookup):
        """The first matching trace whose value loads, or None.

        Reports the hit and records it in the enclosing computation only
        once the value is in hand: a CacheMiss on the value falls through to
        the next candidate, and reporting a match before that would
        overcount.
        """
        for h, trace in _matches(store, fn_key, arguments, arg_hashes):
            try:
                value = store.get_value(trace["result"])
            except CacheMiss:
                continue  # value evicted/corrupt: try others, else rerun
            runlog.record(
                store, "hit", fn=qn, key=fn_key, dur=time.perf_counter() - t_lookup
            )
            _record_call(qn, fn_key, h)
            return (value,)
        return None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        store = _store
        if store is None:
            return fn(*args, **kwargs)

        fn_key, spans, units = _identity()

        # In a worker whose store is the driver's, a @pure_local call is
        # the driver's to make: it has the environment, and it does its own
        # lookup, so nothing is asked of the cache from here.
        if local:
            local_call = getattr(store, "local_call", None)
            if local_call is not None:
                value, h = local_call(fn.__module__, qn, args, kwargs)
                if h:
                    _record_call(qn, fn_key, h)
                return value

        # A live breakpoint in this function's closure: execute without
        # reading or writing the cache, so nothing from a debug session can
        # enter the store.  The discard frame keeps log() calls in the body
        # from raising, and from writing.
        if breakpoints_force(spans):
            global _force_epoch
            _force_epoch += 1
            runlog.record(store, "forced", fn=qn, key=fn_key)
            token = _ctx.set(_Frame(store, discard=True))
            try:
                return fn(*args, **kwargs)
            finally:
                _ctx.reset(token)

        bound, arguments, arg_hashes = _bind(args, kwargs)

        # -- lookup ---------------------------------------------------------
        t_lookup = time.perf_counter()
        found = _hit(store, fn_key, arguments, arg_hashes, t_lookup)
        if found is not None:
            return found[0]

        # Wrap ImmutableMap arguments so their reads are observed; everything
        # else is passed through untouched and depended on whole.
        recorders: dict[str, Recorder] = {}
        for name, v in arguments.items():
            if isinstance(v, ImmutableMap):
                rec = Recorder()
                recorders[name] = rec
                bound.arguments[name] = RecordingMap(v, rec)

        frame = _Frame(store)
        epoch_before = _force_epoch
        t_exec = time.perf_counter()
        token = _ctx.set(frame)
        try:
            result = fn(*bound.args, **bound.kwargs)  # exceptions: cache untouched
        except BaseException as e:
            # Report and re-raise unchanged: the cache is still untouched,
            # and a body that raises is otherwise invisible from outside.
            runlog.record(store, "error", fn=qn, key=fn_key, exc=type(e).__name__)
            raise
        finally:
            _ctx.reset(token)
        exec_dur = time.perf_counter() - t_exec

        # A proxy must not outlive the call it belongs to, wherever in the
        # result it sits.
        if recorders:
            result = unwrap_proxies(result)

        if _force_epoch != epoch_before:
            # Something in this call's dynamic extent was debugger-forced
            # (a breakpoint appeared after our own entry check): this result
            # may reflect a debug session, so it must not be persisted.
            runlog.record(
                store,
                "miss",
                fn=qn,
                key=fn_key,
                dur=time.perf_counter() - t_lookup,
                exec=exec_dur,
                stored=False,
            )
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
        h = store.put_trace(
            fn_key,
            {
                "fn": qn,
                "deps": deps,
                "result": result_hash,
                "calls": frame.calls,
                "logs": frame.logs,
            },
            units=units,
        )
        runlog.record(
            store,
            "miss",
            fn=qn,
            key=fn_key,
            dur=time.perf_counter() - t_lookup,
            exec=exec_dur,
            stored=True,
        )
        _record_call(qn, fn_key, h)
        return result

    def cached(*args, **kwargs):
        """The stored result for these arguments, without executing.

        Raises :class:`CacheMiss` when nothing is stored, or when no cache
        is configured.  A hit counts as one in the run log and in the
        enclosing computation, exactly as a hit inside a call would.
        """
        store = _store
        if store is None:
            raise CacheMiss(f"{qn}: no cache directory is configured")
        fn_key, _, _ = _identity()
        _, arguments, arg_hashes = _bind(args, kwargs)
        found = _hit(store, fn_key, arguments, arg_hashes, time.perf_counter())
        if found is None:
            raise CacheMiss(f"{qn}: no stored result for these arguments")
        return found[0]

    def _lookup(*args, **kwargs):
        """Internal: the ``(trace_hash, trace)`` that holds for these
        arguments, or None.  Loads no value and records nothing."""
        store = _store
        if store is None:
            return None
        fn_key, _, _ = _identity()
        _, arguments, arg_hashes = _bind(args, kwargs)
        return next(_matches(store, fn_key, arguments, arg_hashes), None)

    wrapper.uncached = fn  # override: call the raw function directly
    wrapper.cached = cached
    wrapper.__wrapped__ = fn
    wrapper._valuekit_pure = True
    wrapper._valuekit_local = local
    wrapper._valuekit_identity = _identity
    wrapper._valuekit_lookup = _lookup
    return wrapper
