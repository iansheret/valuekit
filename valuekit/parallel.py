"""Parallel batch execution.

``run_all(fn, inputs)`` runs ``fn`` over ``inputs`` in parallel and returns
a :class:`BatchResult` of per-input outcomes, in input order.

``fn`` must be memoised (``@pure`` or ``@pure_local``).  Three things rest
on that.  An input whose result the cache already holds needs no worker,
and is served here.  Each input's root call record can be named in a *batch
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
* a process that exits without raising (a segfault or an out-of-memory
  kill), recorded as a RuntimeError naming the input and the exit code.

``.values`` returns the plain list of results, raising an ExceptionGroup if
any input failed, so failures cannot be dropped by accident; ``.failures``
lists ``(input, exception)`` pairs for callers that handle them explicitly.

Nothing is re-run automatically.  To debug one input, call ``fn(x)`` on
it: the cached prefix is served without executing and the failing step runs
inline, in this process, with a live stack — and you choose which input to
debug, rather than whichever one happened to fail first.

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
store directory before running) and share the cache: every write is a
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

from . import bootstrap, events, localfile, modes, runlog
from .project import Project, ProjectError, project_root
from .hosts import Finished, LocalHost, ProcessConnection, RemoteHost
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
# ran must not be able to affect its result.  This private hook replaces
# the local file's hosts in tests: a name to the command
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
    __slots__ = ("x", "idx", "handle", "deadline", "where")

    def __init__(self, x, idx, handle, deadline, where: str):
        self.x = x
        self.idx = idx
        self.handle = handle
        self.deadline = deadline
        self.where = where


def _outcome_of(t: _Task, done: Finished, name: str, timeout) -> Outcome:
    """A finished task's result as an Outcome against its input."""
    if done.kind == "ok":
        return Outcome(t.x, value=done.value)
    if done.kind == "error":
        exc: BaseException = done.exc
        exc.__cause__ = _RemoteTraceback(f"\n{done.tb}")
        return Outcome(t.x, exc=exc)
    if done.kind == "error_text":  # the worker's exception could not be sent
        exc = RuntimeError(done.text)
        exc.__cause__ = _RemoteTraceback(f"\n{done.tb}")
        return Outcome(t.x, exc=exc)
    if done.kind == "timed_out":
        return Outcome(t.x, exc=TimeoutError(
            f"{name}({t.x!r}) exceeded the {timeout} s limit and was "
            f"killed. Completed steps are cached; to debug a "
            f"deterministic hang, call {name}({t.x!r}) and pause the "
            f"debugger."
        ))
    # "exited", or "connection_closed" for the second time
    return Outcome(t.x, exc=RuntimeError(
        f"a worker exited without raising while processing "
        f"{name}({t.x!r}) ({done.text}). Completed steps are "
        f"cached; call {name}({t.x!r}) yourself to debug it."
    ))


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
    from .pure import _current_store, take_hit

    try:
        value = take_hit(_current_store(), fn._valuekit_reachable().hash, found[0], found[1])
    except CacheMiss:
        return None  # the result value is gone: a worker runs the input
    return Outcome(x, value=value)


def _launcher(command: list[str]):
    return lambda: ProcessConnection(command)


class _Hosts:
    """The hosts a batch may run on, and how many tasks each may hold.

    Remote hosts come from the local file (or the test hook); each syncs
    on its own thread and counts only once it is ready.  The mode is
    re-read from the file every time capacities are asked for, so a switch
    made while the batch runs applies to the next task started; a file
    that cannot be read leaves the mode last read in force.
    """

    def __init__(
        self, fn, store_dir: str | None, store, batch: int, max_workers: int | None = None
    ):
        # Every host puts (handle, payload) here as a worker returns, sends
        # a request, or exits; deliver() passes each to its handle.
        completions: queue.Queue = queue.Queue()
        self._completions = completions
        self.local = LocalHost(fn, store_dir, completions)
        self.hosts: list[RemoteHost] = []
        self._states: dict[str, str] = {}  # name -> syncing | ready | failed
        self._store = store
        self._batch = batch
        self._store_dir = store_dir
        self._closed = False
        # The local file is in the function's project; a function with
        # no project (defined in __main__, or exec'd) has no file and so no
        # hosts.
        try:
            self._root: str | None = project_root(fn)
        except ProjectError:
            self._root = None
        config = localfile.load_local(self._root)
        self.local_workers = config.local_workers if max_workers is None else max_workers
        self._mode = config.mode
        if store_dir is None:
            # A remote host sends every result to this process's store;
            # with none configured there is nowhere to put them.
            pass
        elif _host_commands is not None:
            project = Project(fn, config.project) if _host_commands else None
            source_root = os.path.join(store_dir, "source") if store_dir else ""
            for name, spec in _host_commands.items():
                command, workers = spec if isinstance(spec, tuple) else (spec, None)
                self.hosts.append(
                    RemoteHost(
                        project, store_dir, completions, name,
                        _launcher([*command, "-c", bootstrap.STAGE0]),
                        source_root, workers,
                    )
                )
        else:
            # One project for every host: the tree is walked once per batch.
            project = Project(fn, config.project) if config.hosts else None
            for h in config.hosts:
                command = [
                    "ssh", "-T", "-o", "BatchMode=yes", h.ssh,
                    bootstrap.remote_command(h.python),
                ]
                self.hosts.append(
                    RemoteHost(
                        project, store_dir, completions, h.name, _launcher(command),
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
        start at once and a host starts taking tasks when it is ready,
        whether the mode named it from the start or a switch mid-batch added it.
        """
        try:
            self._mode = localfile.load_local(self._root).mode
        except RuntimeError:
            pass  # the file cannot be read since the last time: that mode stays in force
        mode = self._mode
        if mode != "local" and self.hosts:
            self.sync(wait=False)
        remote = {}
        for b in self.hosts:
            if b.name not in self._states:
                continue  # never asked, under a local mode
            usable = self._states[b.name] == "ready" and not b.dropped
            remote[b.name] = (b.capacity or 0) if usable else 0
            if b.dropped and self._states[b.name] == "ready":
                self._states[b.name] = "failed"
                events.record(
                    self._store, "host", id=self._batch, name=b.name, ok=False,
                    reason=b.failure or "the connection closed", capacity=b.capacity,
                )
        return mode, modes.capacities(mode, self.local_workers, remote, self.syncing())

    def deliver(self, block: bool) -> None:
        """Pass what the hosts have delivered to the handles it is for;
        with *block*, wait up to ``_POLL`` seconds for the first."""
        try:
            handle, payload = self._completions.get(timeout=_POLL if block else 0)
        except queue.Empty:
            return
        handle.feed(payload)
        while True:
            try:
                handle, payload = self._completions.get_nowait()
            except queue.Empty:
                return
            handle.feed(payload)

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
    own process; ``max_workers`` is how many run at once on this machine
    (default: the ``[local] workers`` line of the local file, else the CPU
    count).  Remote hosts, if the local file names any, add their own
    capacity.  An input whose result is already cached is served without
    a worker.  ``timeout`` limits the seconds each input may spend
    running; a breach kills that input's process and records a
    TimeoutError on its outcome, leaving the rest of the batch unaffected.
    Worker deaths are likewise recorded per input.  When no host may run
    anything (``max_workers=0`` and no remote host is usable) the inputs
    run one at a time in this process, without the timeout.

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
    store_dir = str(store.root) if isinstance(store, LocalStore) else None
    qualname = getattr(fn, "__qualname__", repr(fn))
    name = name or qualname
    runlog.current_run(store)  # begun here, so that every worker writes into this run

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

    hosts = _Hosts(fn, store_dir, store, batch, max_workers)

    running: list[_Task] = []
    busy: dict[str, int] = {}  # host name -> tasks running there
    requeued: set[int] = set()  # inputs already run again once after losing their host

    def _start_pending() -> None:
        _, caps = hosts.capacities()
        for host in hosts.all():
            while pending and busy.get(host.name, 0) < caps.get(host.name, 0):
                idx, x = pending.popleft()
                handle = host.start(x)
                deadline = time.monotonic() + timeout if timeout is not None else None
                running.append(_Task(x, idx, handle, deadline, host.name))
                busy[host.name] = busy.get(host.name, 0) + 1
                events.record(store, "start", id=batch, i=idx, host=host.name)

    def _run_here() -> None:
        """Run the next pending input in this process: no host may run
        anything, and a batch must always be able to run."""
        idx, x = pending.popleft()
        events.record(store, "start", id=batch, i=idx, host="main")
        try:
            o = Outcome(x, value=fn(x))
        except Exception as e:  # noqa: BLE001 - collected on the outcome
            o = Outcome(x, exc=e)
        outcomes[idx] = o
        recorder.outcome(idx, o, host="main")

    try:
        while pending or running:
            _start_pending()
            if not running:
                if not pending:
                    break
                if hosts.syncing():
                    hosts.deliver(block=True)  # a host is still syncing; wait for it
                else:
                    _run_here()
                continue
            hosts.deliver(block=True)
            now = time.monotonic()
            for t in list(running):
                done = t.handle.finished()
                if done is None and t.deadline is not None and now >= t.deadline:
                    t.handle.kill()
                    hosts.deliver(block=False)
                    done = t.handle.finished()
                    if done is None or done.kind in ("exited", "connection_closed"):
                        done = Finished("timed_out")  # only a result means it finished first
                if done is None:
                    continue
                running.remove(t)
                busy[t.where] -= 1
                t.handle.release()
                if done.kind == "connection_closed" and t.idx not in requeued:
                    # The input is not done, not failed.  Run it again
                    # elsewhere, once: an input that takes a host down each
                    # time is a failure after all.
                    requeued.add(t.idx)
                    pending.appendleft((t.idx, t.x))
                    events.record(store, "requeue", id=batch, i=t.idx, host=t.where)
                    continue
                o = _outcome_of(t, done, qualname, timeout)
                outcomes[t.idx] = o
                recorder.outcome(t.idx, o, host=t.where)
    finally:
        # Covers KeyboardInterrupt: no worker process outlives the batch.
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
