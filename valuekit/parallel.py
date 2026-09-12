"""Parallel batch execution.

``run_all(fn, inputs)`` runs ``fn`` over ``inputs`` in parallel and returns
a :class:`BatchResult` of per-input outcomes, in input order.

``fn`` must be memoised (``@pure`` or ``@pure_local``).  Three things rest
on that.  An input whose result the cache already holds needs no worker,
and is answered here.  Each input's root call record can be named in a *batch
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

Nothing is re-run automatically.  To debug one input, call ``fn(x)`` on
it: the cached prefix is served without executing and the failing step runs
inline, in this process, with a live stack — and you choose which input you
land in, rather than whichever one happened to fail first.

One debugger accommodation remains, because breakpoints do not reach
spawned workers: if a live breakpoint intersects anything reachable by name
from ``fn``, the whole batch runs sequentially in this process, where
breakpoints fire and the usual debugger rules apply.  The sequential
fallback does not enforce the timeout.

This module owns the scheduling -- admission, deadlines, input-order
reassembly, and attributing each failure to the input that caused it --
while :mod:`valuekit.hosts` owns starting and killing a task.
The split is what lets work run somewhere other than this machine without
the scheduling being written twice.

Workers are configured automatically (each process applies the parent's
cache directory before running) and share the cache: every write is a
content-named file, so concurrent writers cannot drop each other's
results.  ``fn`` must be a module-level function (it is sent to workers by
reference).  Worker processes are daemonic: they are cleaned up if the
parent exits, and ``fn`` cannot itself start processes (parallelise in this
main process, not inside it).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable, Iterator

from . import bootstrap, events, localfile, modes, runlog, sync
from .hosts import RemoteHost, LocalHost, ProcessConnection
from .batches import BatchWriter
from .functionhash import reachable_set
from .debughook import breakpoints_force
from .store import CacheMiss, LocalStore

__all__ = ["run_all", "BatchResult", "Outcome"]

_POLL = 0.2  # seconds between checks of deadlines and of the mode file

# Distinguishes concurrent batches within one process in the run log;
# the run file already carries the pid, so a counter is identifier enough.
_batch_seq = 0

# Where work happens is configuration (the local file), never a call-site
# argument: a host list in code could reach a function hash, and where a call
# ran must not be able to affect its result.  This private hook stands in
# for the local file's hosts in tests: a name to the command
# that runs a Python 3 to bootstrap with, or to ``(command, workers)``; a host
# with no capacity given reports its own.
_host_commands: dict[str, Any] | None = None


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
    __slots__ = ("x", "idx", "handle", "deadline", "timed_out", "where")

    def __init__(self, x, idx, handle, deadline, where: str):
        self.x = x
        self.idx = idx
        self.handle = handle
        self.deadline = deadline
        self.timed_out = False
        self.where = where


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


class _BatchRecorder:
    """Everything one batch reports about an input as it finishes: the run
    log, the batch record, and the enclosing call's call list."""

    def __init__(self, store, fn, name: str, batch_id: int, inputs: list):
        from .pure import _note_call

        self._store = store
        self._fn = fn
        self._name = name
        self._id = batch_id
        self._note_call = _note_call
        self._writer = None
        if isinstance(store, LocalStore):
            try:
                self._writer = BatchWriter(
                    store, name, fn.__qualname__, fn._valuekit_reachable().hash, inputs
                )
            except Exception:
                self._writer = None  # the record is a diagnostic, never a failure

    def outcome(self, i: int, o: Outcome, host: str = "local") -> None:
        exc = o.exception()
        record = None
        if exc is None:
            found = self._fn._valuekit_lookup(o.input)
            if found is not None:
                record = found[0]
                self._note_call(
                    self._fn.__qualname__, self._fn._valuekit_reachable().hash, record
                )
        events.record(
            self._store,
            "outcome",
            id=self._id,
            i=i,
            ok=exc is None,
            host=host,
            exc=None if exc is None else type(exc).__name__,
        )
        if self._writer is not None:
            self._writer.outcome(i, record, exc)


def _cached(fn, x) -> Outcome | None:
    """The stored outcome for *x*, if this store already holds one."""
    found = fn._valuekit_lookup(x)
    if found is None:
        return None
    from .pure import _current_store

    store = _current_store()
    try:
        value = store.get_value(found[1]["result"])
        # The stored result stands in for the call only with everything the
        # call would have logged; otherwise a worker runs it afresh.
        runlog.reemit(store, fn._valuekit_reachable().hash, found[0], found[1])
    except CacheMiss:
        return None
    return Outcome(x, value=value)


def _launcher(command: list[str]):
    return lambda: ProcessConnection(command)


class _Hosts:
    """The hosts a batch may run on, and how many tasks each may hold.

    Remote hosts come from the local file (or the test hook); each syncs
    on its own thread and counts only once it is ready.  The
    mode is re-read from the file every time capacities are asked for, so a
    switch made while the batch runs applies to the next task started.
    """

    def __init__(self, fn, cache_dir: str | None, completions: queue.Queue, store, batch: int):
        self.local = LocalHost(fn, cache_dir, completions)
        self.hosts: list[RemoteHost] = []
        self._states: dict[str, str] = {}  # name -> syncing | ready | failed
        self._store = store
        self._batch = batch
        self._cache_dir = cache_dir
        self._closed = False
        # The local file lives in the function's project; a function with
        # no project (defined in __main__, or exec'd) has no file and so no
        # hosts.
        try:
            self._root: str | None = sync.sync_root(fn)
        except sync.SyncError:
            self._root = None
        config = localfile.load_local(self._root)
        self.local_workers = config.local_workers
        if _host_commands is not None:
            project = sync.Project(fn, config.project) if _host_commands else None
            source_root = os.path.join(cache_dir, "source") if cache_dir else ""
            for name, spec in _host_commands.items():
                command, workers = spec if isinstance(spec, tuple) else (spec, None)
                self.hosts.append(
                    RemoteHost(
                        project, cache_dir, completions, name,
                        _launcher([*command, "-c", bootstrap.STAGE0]),
                        source_root, workers,
                    )
                )
        else:
            # One project for every host: the tree is walked once per batch.
            project = sync.Project(fn, config.project) if config.hosts else None
            for h in config.hosts:
                command = [
                    "ssh", "-T", "-o", "BatchMode=yes", h.ssh,
                    bootstrap.remote_command(h.python),
                ]
                self.hosts.append(
                    RemoteHost(
                        project, cache_dir, completions, h.name, _launcher(command),
                        h.source_root, h.workers,
                    )
                )

    def all(self) -> list:
        return [*self.hosts, self.local]

    def sync(self, wait: bool) -> None:
        """Sync every host not yet tried; block if *wait*."""
        threads = []
        for b in self.hosts:
            if b.name in self._states:
                continue
            self._states[b.name] = "syncing"
            t = threading.Thread(target=self._sync_one, args=(b,), daemon=True)
            t.start()
            threads.append(t)
        if wait:
            for t in threads:
                t.join()

    def _sync_one(self, b: RemoteHost) -> None:
        reason = b.sync()
        self._states[b.name] = "failed" if reason else "ready"
        if self._closed:
            return  # the batch ended first; nothing to report it to
        events.record(
            self._store, "host", id=self._batch, name=b.name, ok=not reason,
            reason=reason or None, capacity=b.capacity,
        )

    def syncing(self) -> bool:
        """Whether any host is still syncing, and so may yet take work."""
        return any(s == "syncing" for s in self._states.values())

    def capacities(self) -> tuple[str, dict[str, int]]:
        """The mode in force and each host's capacity under it.

        Syncing is started here, never waited for: this machine's workers
        start at once and a host joins when it is ready, whether the mode
        named it from the start or a switch mid-batch brought it in.
        """
        mode = localfile.read_mode(self._root, self._cache_dir)
        if mode != "local" and self.hosts:
            self.sync(wait=False)
        remote = {}
        for b in self.hosts:
            if b.name not in self._states:
                continue  # never asked, under a local mode
            usable = self._states[b.name] == "ready" and not b.dead
            remote[b.name] = (b.capacity or 0) if usable else 0
            if b.dead and self._states[b.name] == "ready":
                self._states[b.name] = "failed"
                events.record(
                    self._store, "host", id=self._batch, name=b.name, ok=False,
                    reason=b.failure or "the connection closed", capacity=b.capacity,
                )
        return mode, modes.capacities(mode, self.local_workers, remote, self.syncing())

    def close(self) -> None:
        self._closed = True
        for b in self.all():
            try:
                b.close()
            except Exception:
                pass


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
    re-run automatically; to debug one input, call ``fn(x)`` on it.

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
    runlog.current_run(store)  # begun here, so that every worker joins this run

    global _batch_seq
    _batch_seq += 1
    batch = _batch_seq

    # A live breakpoint anywhere reachable from fn: run sequentially, in
    # this process, so the breakpoint fires and the debugger contract
    # applies.  (Also covers VALUEKIT_ALWAYS_RUN.)
    try:
        spans = reachable_set(fn).spans
    except Exception:
        spans = []
    if breakpoints_force(spans):
        events.record(
            store, "batch", id=batch, fn=qualname, name=name, n=len(inputs),
            mode="sequential",
        )
        recorder = _BatchRecorder(store, fn, name, batch, inputs)
        seq: list[Outcome] = []
        try:
            for i, x in enumerate(inputs):
                o = Outcome(x, value=fn(x))  # exceptions propagate
                seq.append(o)
                recorder.outcome(i, o)
        finally:
            events.record(store, "end", id=batch)
        return BatchResult(seq)

    events.record(
        store, "batch", id=batch, fn=qualname, name=name, n=len(inputs), mode="parallel"
    )
    recorder = _BatchRecorder(store, fn, name, batch, inputs)

    outcomes: list[Outcome | None] = [None] * len(inputs)
    pending = deque()
    for i, x in enumerate(inputs):
        o = _cached(fn, x) if store is not None else None
        if o is None:
            pending.append((i, x))
        else:
            outcomes[i] = o
            recorder.outcome(i, o)
    if not pending:
        events.record(store, "end", id=batch)
        return BatchResult(o for o in outcomes if o is not None)

    completions: queue.Queue = queue.Queue()
    hosts = _Hosts(fn, cache_dir, completions, store, batch)

    running: list[_Task] = []
    busy: dict[str, int] = {}  # host name -> tasks running there
    moved: set[int] = set()  # inputs already run again after losing their host

    def _start(host, idx: int, x: Any) -> None:
        handle = host.start(x)
        deadline = time.monotonic() + timeout if timeout is not None else None
        running.append(_Task(x, idx, handle, deadline, host.name))
        busy[host.name] = busy.get(host.name, 0) + 1
        events.record(store, "start", id=batch, i=idx, host=host.name)

    def _accept() -> None:
        _, caps = hosts.capacities()
        total = max_workers or sum(caps.values())
        for host in hosts.all():
            while (
                pending
                and len(running) < total
                and busy.get(host.name, 0) < caps.get(host.name, 0)
            ):
                _start(host, *pending.popleft())

    def _drain(block: bool) -> None:
        """Feed handles whatever the hosts have delivered."""
        try:
            handle, payload = completions.get(timeout=_POLL if block else 0)
        except queue.Empty:
            return
        handle.feed(payload)
        while True:
            try:
                handle, payload = completions.get_nowait()
            except queue.Empty:
                return
            handle.feed(payload)

    try:
        while pending or running:
            _accept()
            if not running:
                if not pending:
                    break
                if not hosts.syncing():
                    raise RuntimeError(
                        "no host can run this batch: every capacity is zero"
                    )
                _drain(block=True)  # a host is on its way; wait for it
                continue
            _drain(block=True)
            now = time.monotonic()
            for t in list(running):
                msg = None
                if t.handle.settled():
                    msg = t.handle.recv()
                elif t.deadline is not None and now >= t.deadline:
                    t.handle.kill()
                    _drain(block=False)
                    if t.handle.settled():  # finished just before the kill landed
                        msg = t.handle.recv()
                    t.timed_out = msg is None
                else:
                    continue
                running.remove(t)
                busy[t.where] -= 1
                t.handle.reap()
                if msg is None and not t.timed_out and t.handle.lost():
                    # The host went away; the input is not done, not
                    # failed.  Run it again elsewhere, once: an input that
                    # takes a host down each time is a failure after all.
                    if t.idx not in moved:
                        moved.add(t.idx)
                        pending.appendleft((t.idx, t.x))
                        events.record(store, "requeue", id=batch, i=t.idx, host=t.where)
                        continue
                o = _harvest(t, msg, qualname, timeout)
                outcomes[t.idx] = o
                recorder.outcome(t.idx, o, host=t.where)
    finally:
        # Covers KeyboardInterrupt: no orphans.
        for t in running:
            try:
                t.handle.kill()
            except Exception:
                pass
        hosts.close()
        # In the finally, not after the return: an interrupted batch is
        # exactly the one whose final state is worth having.
        events.record(store, "end", id=batch)

    return BatchResult(o for o in outcomes if o is not None)
