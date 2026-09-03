"""Parallel batch execution.

``run_all(fn, inputs)`` runs ``fn`` over ``inputs`` in parallel and returns
a :class:`BatchResult` of per-input outcomes, in input order.

``fn`` must be memoised (``@pure`` or ``@pure_local``).  Three things rest
on that.  An input whose result the cache already holds needs no worker,
and is answered here.  Each input's root trace can be named in a *batch
record* (see :mod:`valuekit.batches`), which is how analysis code reaches
what the batch produced without reconstructing arguments.  And running
somewhere other than this machine is safe only for a function whose
effects do not matter, which is what memoisation already assumes.

Each input runs in its own process (spawned per task, up to ``max_workers``
at once) rather than in a shared worker pool.  Isolation is the point: a
timeout kills exactly one process, a segfault loses exactly one input, and
neither affects the inputs running beside it or the number of workers
available to the rest of the batch.  The cost is one process start per
input, roughly 0.4 s including a numpy import; this overlaps across
workers, and is noise for inputs that take seconds or more.  For very
small inputs, batch them inside ``fn``.

Every input is processed, and every failure is recorded against the input
that caused it.  There are three kinds, treated alike:

* an exception raised by ``fn``, carrying the string-form traceback
  captured in the worker;
* a timeout.  ``timeout=`` limits the seconds each input may spend
  running; a breach kills that input's process promptly and records a
  TimeoutError;
* a process that dies without raising (a segfault or an out-of-memory
  kill), recorded as a RuntimeError naming the input and the exit code.

``.values`` returns the plain list of results, raising an ExceptionGroup if
any input failed, so failures cannot be dropped by accident; ``.failures``
lists ``(input, exception)`` pairs for callers that handle them explicitly.

Nothing is replayed automatically.  To debug one input, call ``fn(x)`` on
it: the cached prefix replays without executing and the failing step runs
inline, in this process, with a live stack — and you choose which input you
land in, rather than whichever one happened to fail first.

One debugger accommodation remains, because breakpoints do not reach
spawned workers: if a live breakpoint intersects anything reachable by name
from ``fn``, the whole batch runs sequentially in this process, where
breakpoints fire and the usual debugger rules apply.  The sequential
fallback does not enforce the timeout.

This module owns the scheduling -- admission, deadlines, input-order
reassembly, and attributing each failure to the input that caused it --
while :mod:`valuekit.backend` owns starting and killing a unit of work.
The split is what lets work run somewhere other than this machine without
the scheduling being written twice.

Workers are configured automatically (each process applies the parent's
cache directory before running) and share the cache: every write is a
content-named file, so concurrent writers cannot drop each other's
results.  ``fn`` must be a module-level function (it is sent to workers by
reference).  Worker processes are daemonic: they are cleaned up if the
parent exits, and ``fn`` cannot itself start processes (parallelise in this
driver, not inside it).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Iterable, Iterator

from . import runlog
from .backend import LocalBackend
from .batches import BatchWriter
from .codehash import function_fingerprint
from .debughook import breakpoints_force
from .store import CacheMiss, LocalStore

__all__ = ["run_all", "BatchResult", "Outcome"]

_POLL = 0.2  # seconds between timeout checks while tasks are running

# Distinguishes concurrent batches within one process in the run log;
# the run file already carries the pid, so a counter is identifier enough.
_batch_seq = 0

# Which backend a batch runs on.  Private and local-only for now: choosing
# where work happens is a deployment question, so when it becomes settable
# it will be settable from configuration, never from the call site -- a
# host list in code could reach a fingerprint, and where a computation ran
# must not be able to affect its result.
_backend_factory = LocalBackend


class Outcome:
    """The outcome of one input of a batch.

    ``input`` is the element of the submitted inputs that produced this
    outcome, carried for attribution: exceptions do not record which input
    started the call chain, and a filtered subset of outcomes would
    otherwise lose its alignment with the inputs.  ``result()`` returns the
    value, or re-raises the input's exception; ``exception()`` returns the
    exception, or None.
    """

    __slots__ = ("input", "_value", "_exc")

    def __init__(
        self, input: Any, value: Any = None, exc: BaseException | None = None
    ):
        self.input = input
        self._value = value
        self._exc = exc

    def result(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._value

    def exception(self) -> BaseException | None:
        return self._exc

    def __repr__(self) -> str:
        if self._exc is None:
            return f"Outcome({self.input!r}, ok)"
        return f"Outcome({self.input!r}, {type(self._exc).__name__})"


class BatchResult:
    """Per-input outcomes of :func:`run_all`, in input order.

    Iterating yields :class:`Outcome` objects.  Two accessors cover the two
    ways of handling failure:

    * ``.values``: the plain list of results.  If any input failed, this
      raises an ExceptionGroup instead, so failures cannot be dropped by
      accident; each grouped exception carries an ``input: ...`` note.
    * ``.failures``: the ``(input, exception)`` pairs of the failed inputs,
      for callers that handle failures explicitly and continue.
    """

    __slots__ = ("_outcomes",)

    def __init__(self, outcomes: Iterable[Outcome]):
        self._outcomes = list(outcomes)

    def __iter__(self) -> Iterator[Outcome]:
        return iter(self._outcomes)

    def __len__(self) -> int:
        return len(self._outcomes)

    def __getitem__(self, i):
        return self._outcomes[i]

    @property
    def failures(self) -> list[tuple[Any, BaseException]]:
        return [(o.input, o._exc) for o in self._outcomes if o._exc is not None]

    @property
    def values(self) -> list:
        failed = [o for o in self._outcomes if o._exc is not None]
        if failed:
            for o in failed:
                note = f"input: {o.input!r}"
                if note not in getattr(o._exc, "__notes__", []):
                    o._exc.add_note(note)
            raise BaseExceptionGroup(
                f"{len(failed)} of {len(self._outcomes)} inputs failed",
                [o._exc for o in failed],
            )
        return [o._value for o in self._outcomes]

    def __repr__(self) -> str:
        n_failed = len(self.failures)
        n_ok = len(self._outcomes) - n_failed
        return f"BatchResult({n_ok} ok, {n_failed} failed)"


class _RemoteTraceback(Exception):
    """Carries the string-form traceback captured in the worker, attached
    as the ``__cause__`` of a collected exception so that it prints."""

    def __init__(self, tb: str):
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


class _Task:
    __slots__ = ("x", "idx", "handle", "deadline", "timed_out")

    def __init__(self, x, idx, handle, deadline):
        self.x = x
        self.idx = idx
        self.handle = handle
        self.deadline = deadline
        self.timed_out = False


def _harvest(t: _Task, msg, name: str, timeout) -> Outcome:
    """Turn a finished task into an Outcome."""
    if msg is None:
        if t.timed_out:
            exc: BaseException = TimeoutError(
                f"{name}({t.x!r}) exceeded the {timeout} s limit and was "
                f"killed. Completed steps are cached; to debug a "
                f"deterministic hang, call {name}({t.x!r}) and pause the "
                f"debugger."
            )
        else:
            exc = RuntimeError(
                f"a worker died without raising while processing "
                f"{name}({t.x!r}) ({t.handle.death()}). Completed steps are "
                f"cached; call {name}({t.x!r}) yourself to debug it."
            )
        return Outcome(t.x, exc=exc)
    kind = msg[0]
    if kind == "ok":
        return Outcome(t.x, value=msg[1])
    if kind == "err":
        exc = msg[1]
        exc.__cause__ = _RemoteTraceback(f"\n{msg[2]}")
        return Outcome(t.x, exc=exc)
    # "err_str": the worker's exception was not picklable
    exc = RuntimeError(f"{msg[1]}: {msg[2]}")
    exc.__cause__ = _RemoteTraceback(f"\n{msg[3]}")
    return Outcome(t.x, exc=exc)


class _Record:
    """Everything one batch reports about an input as it finishes: the run
    log, the batch record, and the enclosing computation's call list."""

    def __init__(self, store, fn, name: str, batch_id: int, inputs: list):
        from .pure import _record_call

        self._store = store
        self._fn = fn
        self._name = name
        self._id = batch_id
        self._record_call = _record_call
        self._writer = None
        if isinstance(store, LocalStore):
            try:
                self._writer = BatchWriter(
                    store, name, fn.__qualname__, fn._valuekit_identity()[0], inputs
                )
            except Exception:
                self._writer = None  # the record is a diagnostic, never a failure

    def outcome(self, i: int, o: Outcome, host: str = "local") -> None:
        exc = o.exception()
        trace = None
        if exc is None:
            found = self._fn._valuekit_lookup(o.input)
            if found is not None:
                trace = found[0]
                self._record_call(
                    self._fn.__qualname__, self._fn._valuekit_identity()[0], trace
                )
        runlog.record(
            self._store,
            "outcome",
            id=self._id,
            i=i,
            ok=exc is None,
            host=host,
            exc=None if exc is None else type(exc).__name__,
        )
        if self._writer is not None:
            self._writer.outcome(i, trace, exc)


def _cached(fn, x) -> Outcome | None:
    """The stored outcome for *x*, if this store already holds one."""
    found = fn._valuekit_lookup(x)
    if found is None:
        return None
    from .pure import _current_store

    try:
        return Outcome(x, value=_current_store().get_value(found[1]["result"]))
    except CacheMiss:
        return None


def run_all(
    fn: Callable[[Any], Any],
    inputs: Iterable[Any],
    max_workers: int | None = None,
    *,
    timeout: float | None = None,
    name: str | None = None,
) -> BatchResult:
    """Run ``fn`` over ``inputs`` in parallel; return a BatchResult in
    input order.

    ``fn`` must be ``@pure`` or ``@pure_local``.  Each input runs in its
    own process; ``max_workers`` caps how many run at once (default: the
    CPU count).  An input whose result is already cached is answered
    without a worker.  ``timeout`` limits the seconds each input may spend
    running; a breach kills that input's process and records a
    TimeoutError on its outcome, leaving the rest of the batch unaffected.
    Worker deaths are likewise recorded per input.

    Every input is processed and failures are collected on the
    BatchResult: ``.values`` raises an ExceptionGroup if any input failed,
    and ``.failures`` gives the ``(input, exception)`` pairs.  Nothing is
    replayed automatically; to debug one input, call ``fn(x)`` on it.

    The batch is recorded under ``name`` (default: the function's qualified
    name) for :func:`valuekit.batch` to read.
    """
    if not getattr(fn, "_valuekit_pure", False):
        raise TypeError(
            f"run_all() takes a @pure or @pure_local function; got "
            f"{getattr(fn, '__qualname__', fn)!r}. Decorate it: a batch's "
            "results are recorded by the function that produced them."
        )
    inputs = list(inputs)

    # Resolved before the breakpoint check below, so that the sequential
    # fallback reports itself too: a batch being debugged is one you most
    # want to watch.
    from .pure import _current_store

    store = _current_store()
    cache_dir = str(store.root) if isinstance(store, LocalStore) else None
    qualname = getattr(fn, "__qualname__", repr(fn))
    name = name or qualname

    global _batch_seq
    _batch_seq += 1
    batch = _batch_seq

    # A live breakpoint anywhere reachable from fn: run sequentially, in
    # this process, so the breakpoint fires and the debugger contract
    # applies.  (Also covers VALUEKIT_ALWAYS_RUN.)
    try:
        _, spans, _ = function_fingerprint(fn)
    except Exception:
        spans = []
    if breakpoints_force(spans):
        runlog.record(
            store, "batch", id=batch, fn=qualname, name=name, n=len(inputs),
            mode="sequential",
        )
        record = _Record(store, fn, name, batch, inputs)
        seq: list[Outcome] = []
        try:
            for i, x in enumerate(inputs):
                o = Outcome(x, value=fn(x))  # exceptions propagate
                seq.append(o)
                record.outcome(i, o)
        finally:
            runlog.record(store, "end", id=batch)
        return BatchResult(seq)

    runlog.record(
        store, "batch", id=batch, fn=qualname, name=name, n=len(inputs), mode="parallel"
    )
    record = _Record(store, fn, name, batch, inputs)

    outcomes: list[Outcome | None] = [None] * len(inputs)
    queue = deque()
    for i, x in enumerate(inputs):
        o = _cached(fn, x) if store is not None else None
        if o is None:
            queue.append((i, x))
        else:
            outcomes[i] = o
            record.outcome(i, o)
    if not queue:
        runlog.record(store, "end", id=batch)
        return BatchResult(o for o in outcomes if o is not None)

    backend = _backend_factory(fn, cache_dir)

    # Whatever a backend must do once before it can take work -- syncing the
    # project, checking the environment -- happens here, before any input is
    # claimed. A failure is a fact about the host, so it is raised as one
    # rather than recorded identically against every input.
    ensure_ready = getattr(backend, "ensure_ready", None)
    if ensure_ready is not None:
        ensure_ready()

    workers = max_workers or backend.default_workers()

    # A deadline needs checking even while nothing is ready to read; a
    # backend may also want waking periodically for its own reasons.
    intervals = [
        p
        for p in (_POLL if timeout is not None else None, backend.poll_interval)
        if p is not None
    ]
    poll = min(intervals) if intervals else None

    running: list[_Task] = []

    def _start(idx: int, x: Any) -> None:
        handle = backend.start(x)
        deadline = time.monotonic() + timeout if timeout is not None else None
        running.append(_Task(x, idx, handle, deadline))

    try:
        while queue or running:
            while queue and len(running) < workers:
                _start(*queue.popleft())
            if not running:
                break
            ready = backend.wait([t.handle for t in running], timeout=poll)
            now = time.monotonic()
            for t in list(running):
                msg = None
                if t.handle in ready:
                    msg = t.handle.recv()
                elif t.deadline is not None and now >= t.deadline:
                    t.handle.kill()
                    if t.handle.poll():  # finished just before the kill landed
                        msg = t.handle.recv()
                    t.timed_out = msg is None
                else:
                    continue
                running.remove(t)
                t.handle.reap()
                o = _harvest(t, msg, qualname, timeout)
                outcomes[t.idx] = o
                record.outcome(t.idx, o)
    finally:
        # Covers KeyboardInterrupt: no orphans.
        for t in running:
            try:
                t.handle.kill()
            except Exception:
                pass
        backend.close()
        # In the finally, not after the return: an interrupted batch is
        # exactly the one whose final state is worth having.
        runlog.record(store, "end", id=batch)

    return BatchResult(o for o in outcomes if o is not None)
