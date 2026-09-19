"""The @pure decorator: memoisation by function hash and traced reads.

A call of a memoised function is looked up by its function hash (see
:mod:`valuekit.functionhash`) and its arguments.  An :class:`~valuekit.ImmutableMap`
argument is wrapped on a miss in a recording proxy, so the call record
holds the keys the function read and their values; every other argument is
hashed whole.  On a lookup the stored call records for the function hash
are matched against the arguments, and the one that holds gives the result
without executing.

A call record holds the reads, the result's hash, the memoised calls made
inside the call (name, function hash and record hash, in completion
order), and the values logged inside it (labels and value, by hash).  A
hit writes one line to the run's log naming the record;
:mod:`valuekit.runlog` reads the logged values out of the record, nested
calls included.

The user's contract is on :func:`pure` and :func:`pure_local`.  With no
store directory configured a memoised function is a plain call.  A
breakpoint anywhere in the function's reachable set forces execution
without a call record (see :mod:`valuekit.debughook`).
"""

from __future__ import annotations

import atexit
import functools
import inspect
import os
import time
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Callable

from . import runlog, events
from .functionhash import reachable_set
from .debughook import breakpoints_force
from .map import ImmutableMap, map_digest
from .recording import Recorder, RecordingMap, unwrap_proxies
from .store import CacheMiss, CacheStore, LocalStore
from .values import content_hash, decode_key, digest, freeze

__all__ = ["pure", "pure_local", "log", "set_store_dir", "clear_cache"]

_MISSING = object()

# ---------------------------------------------------------------------------
# store configuration
# ---------------------------------------------------------------------------

_store: CacheStore | None = None


def _current_store() -> "CacheStore | None":
    """The configured store, or None."""
    return _store


def set_store_dir(path: str | os.PathLike | None) -> None:
    """Configure the store directory (or None to disable caching).

    Nothing is configured until this is called, so importing valuekit never
    enables disk caching by itself.  To drive it from the environment, read
    the variable explicitly::

        set_store_dir(os.environ.get("VALUEKIT_STORE"))
    """
    global _store
    if isinstance(_store, LocalStore):
        _store.close()
    _store = None
    if path is not None:
        store = LocalStore(path)
        runlog.begin_run(store)
        events.open_log(store)
        _store = store


def _close_store() -> None:
    if isinstance(_store, LocalStore):
        _store.close()


atexit.register(_close_store)


def set_store(store: CacheStore | None) -> None:
    """Internal: use *store* directly (a worker whose store is a peer)."""
    global _store
    _store = store


def clear_cache() -> None:
    """Delete everything computed or logged: every stored value, call
    record and run log.  Always safe: the worst case is
    recomputation.  To invalidate one function, edit it, or put a version
    in its arguments; either gives it a new function hash.
    """
    if isinstance(_store, LocalStore):
        _store.clear()


def _take(store: CacheStore, function_hash: str, h: str, record: dict) -> Any:
    """The result of call record *h* of *function_hash*, taken as a hit:
    the value is loaded and the record is named in the run's log.  Raises
    :class:`CacheMiss` if the value is gone, and then writes nothing."""
    value = store.get_value(record["result"])
    runlog.refer(store, function_hash, h)
    return value


class Memoised:
    """What :mod:`valuekit.parallel` and :mod:`valuekit.hosts` use of a
    memoised function, at ``fn._valuekit``: whether it is local, its
    reachable set, and the two store operations on its arguments.
    """

    __slots__ = ("fn", "local", "_sig", "_code", "_reach")

    def __init__(self, fn: Callable, local: bool, sig: inspect.Signature, code):
        self.fn = fn
        self.local = local
        self._sig = sig
        self._code = code
        self._reach: list = []  # [ReachableSet] once computed; benign races

    @property
    def reachable(self):
        """The function's reachable set, computed at first use: names are
        resolved when the module is fully loaded, so definition order and
        forward references do not matter."""
        if not self._reach:
            self._reach.append(reachable_set(self.fn, code=self._code))
        return self._reach[0]

    def bind(self, args, kwargs):
        """Bind the call and hash its non-map arguments, once.

        The same hashes match every candidate call record and then go into
        the call record written for a miss, so what is recorded is the
        arguments as they were passed.
        """
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)  # the caller's own objects
        arg_hashes = {
            name: content_hash(v)
            for name, v in arguments.items()
            if not isinstance(v, ImmutableMap)
        }
        return bound, arguments, arg_hashes

    def record_hash(self, *args, **kwargs) -> str | None:
        """The hash of the call record that holds for these arguments, or
        None.  Loads no value and writes nothing."""
        store = _store
        if store is None:
            return None
        _, arguments, arg_hashes = self.bind(args, kwargs)
        found = _match(store, self.reachable.hash, arguments, arg_hashes)
        return None if found is None else found[0]

    def hit(self, *args, **kwargs) -> Any:
        """The stored result for these arguments, taken as a hit: the value
        is loaded and the record named in the run's log.  Raises
        :class:`CacheMiss` when no record holds or its value is gone."""
        store = _store
        if store is None:
            raise CacheMiss("no store directory is configured")
        function_hash = self.reachable.hash
        _, arguments, arg_hashes = self.bind(args, kwargs)
        found = _match(store, function_hash, arguments, arg_hashes)
        if found is None:
            raise CacheMiss("no call record holds for these arguments")
        return _take(store, function_hash, *found)


# ---------------------------------------------------------------------------
# the executing call
# ---------------------------------------------------------------------------


class _Frame:
    """What the memoised call currently executing has recorded so far.

    ``calls`` and ``logs`` go into its call record.  ``parent`` is the
    frame of the enclosing memoised call, or None.  A frame with
    ``discard`` set writes no call record and notes nothing, while its
    logged values still go to the run's log: it is a debugger-forced run,
    or a call inside which one happened (a breakpoint added after the
    call's own entry check), whose result may reflect the debug session.
    """

    __slots__ = ("calls", "logs", "parent", "discard")

    def __init__(self, parent: "_Frame | None", discard: bool = False):
        self.calls: list[list] = []
        self.logs: list[list] = []
        self.parent = parent
        self.discard = discard

    def discard_all(self) -> None:
        """Mark this frame and every enclosing one as not to be stored."""
        frame: _Frame | None = self
        while frame is not None:
            frame.discard = True
            frame = frame.parent


# The innermost executing call.  A contextvar rather than a global:
# each thread sees its own, and the token reset in ``finally`` restores the
# enclosing frame on any exit.  A worker starts at None, which is
# right: its root call has no enclosing call in that process.
_ctx: ContextVar[_Frame | None] = ContextVar("valuekit_call", default=None)


def _note_call(qn: str, function_hash: str, h: str) -> None:
    """Note a completed memoised call in the enclosing call, if any."""
    frame = _ctx.get()
    if frame is not None and not frame.discard:
        frame.calls.append([qn, function_hash, h])


def log(labels: Mapping, value: Any) -> None:
    """Record *value* under *labels*, a small mapping saying what it is.

    The labels are frozen to an :class:`ImmutableMap` and both they and
    the value are stored like results (they must be storable).  The logged
    value goes into this run's log at once and, inside a memoised call,
    into the call's record as well, which is where a later hit's line in
    the run's log points.  Read it back through :func:`valuekit.logs`.

    valuekit gives no label key a meaning.  Outside a memoised call, and
    in a call a debugger forced to run, the logged value goes to the run's
    log only.  With no store configured this does nothing, like everything
    else here.
    """
    if not isinstance(labels, Mapping):
        raise TypeError(
            f"log() takes a mapping as its labels, got {type(labels).__name__}"
        )
    frame = _ctx.get()
    store = _store
    if store is None:
        return
    labels = freeze(labels)
    labels_hash = store.put_value(labels)
    value_hash = store.put_value(value)
    keys = runlog.label_hashes(labels)
    if frame is not None and not frame.discard:
        frame.logs.append([labels_hash, value_hash, keys])
    runlog.emit(store, labels_hash, value_hash, keys)


# ---------------------------------------------------------------------------
# call-record matching
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


def _match_record(
    record: dict, arguments: dict[str, Any], arg_hashes: dict[str, str]
) -> bool:
    deps = record.get("deps", {})
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


def _match(store: CacheStore, function_hash: str, arguments: dict, arg_hashes: dict):
    """The stored call record that holds for these arguments, as ``(hash,
    record)``, or None.  A pure function reads the same keys given the
    same values, so at most one record can hold."""
    for h, record in store.get_records(function_hash):
        if _match_record(record, arguments, arg_hashes):
            return h, record
    return None


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
         call record recorded against it will be wrong.
      4. The result is reachable from the arguments plus the function's own
         definition (hashed: everything reachable by name through user
         code).

    Arguments and results must be content-hashable, and results must also
    be storable; both sets are extended with ``register_type``. Pass an
    ImmutableMap to get key-level invalidation for that argument; any other
    argument is depended on whole.

    Takes no options. If a dependency is not reachable by name, pass it as
    an argument; if something invisible changed anyway, edit the function,
    which gives it a new function hash.

    The wrapper has ``uncached``, the raw function.
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

    Because the function reads an environment, it runs only on this
    machine, never on a remote host.  A batch running elsewhere sends such
    calls back here and receives the value.  Effects that do not
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
    # patches its bytecode; the reachable set is computed at the first
    # call (see Memoised.reachable).
    memo = Memoised(fn, local, sig, fn.__code__)

    def _hit(store, function_hash, arguments, arg_hashes, t_lookup):
        """The matching record's value as a one-tuple, or None: no record
        holds, or its value is gone, and the call runs.

        Reports the hit and records it in the enclosing call only once the
        value has loaded; reporting a match before that would overcount.
        """
        found = _match(store, function_hash, arguments, arg_hashes)
        if found is None:
            return None
        h, record = found
        try:
            value = _take(store, function_hash, h, record)
        except CacheMiss:
            return None
        events.record(
            store, "hit", fn=qn, function_hash=function_hash, dur=time.perf_counter() - t_lookup
        )
        _note_call(qn, function_hash, h)
        return (value,)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        store = _store
        if store is None:
            return fn(*args, **kwargs)

        reach = memo.reachable
        function_hash, spans = reach.hash, reach.spans

        # In a worker whose store is the main process's, a @pure_local call is
        # the main process's to make: it has the environment, and it does its own
        # lookup, so nothing is asked of the cache from here.
        if local:
            local_call = getattr(store, "local_call", None)
            if local_call is not None:
                value, h = local_call(fn.__module__, qn, args, kwargs)
                if h:
                    _note_call(qn, function_hash, h)
                return value

        # A live breakpoint in this function's reachable set: execute without
        # reading or writing the cache, so nothing from a debug session can
        # enter a call record.  Logged values still reach the run's log: a
        # run being debugged is one whose values are wanted.
        if breakpoints_force(spans):
            events.record(store, "forced", fn=qn, function_hash=function_hash)
            forced = _Frame(_ctx.get())
            forced.discard_all()
            token = _ctx.set(forced)
            try:
                return fn(*args, **kwargs)
            finally:
                _ctx.reset(token)

        bound, arguments, arg_hashes = memo.bind(args, kwargs)

        # -- lookup ---------------------------------------------------------
        t_lookup = time.perf_counter()
        found = _hit(store, function_hash, arguments, arg_hashes, t_lookup)
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

        frame = _Frame(_ctx.get())
        t_exec = time.perf_counter()
        token = _ctx.set(frame)
        try:
            result = fn(*bound.args, **bound.kwargs)  # exceptions: cache untouched
        except BaseException as e:
            # Report and re-raise unchanged: the cache is still untouched,
            # and a body that raises is otherwise invisible from outside.
            events.record(store, "error", fn=qn, function_hash=function_hash, exc=type(e).__name__)
            raise
        finally:
            _ctx.reset(token)
        exec_dur = time.perf_counter() - t_exec

        # A proxy must not outlive the call it belongs to, wherever in the
        # result it sits.
        if recorders:
            result = unwrap_proxies(result)

        if frame.discard:
            # A debugger forced a call inside this one: the result may
            # reflect the debug session and is not stored.
            events.record(
                store,
                "miss",
                fn=qn,
                function_hash=function_hash,
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
        h = store.put_record(
            function_hash,
            {
                "fn": qn,
                "deps": deps,
                "result": result_hash,
                "calls": frame.calls,
                "logs": frame.logs,
            },
        )
        events.record(
            store,
            "miss",
            fn=qn,
            function_hash=function_hash,
            dur=time.perf_counter() - t_lookup,
            exec=exec_dur,
            stored=True,
        )
        _note_call(qn, function_hash, h)
        return result

    wrapper.uncached = fn  # override: call the raw function directly
    wrapper.__wrapped__ = fn
    wrapper._valuekit = memo
    return wrapper
