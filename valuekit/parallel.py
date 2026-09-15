"""Parallel batch execution.

``run_all(fn, inputs)`` runs ``fn`` over ``inputs`` in parallel and returns
a :class:`BatchResult` of per-input outcomes, in input order.

A memoised function (``@pure`` or ``@pure_local``) gets two things an
undecorated one does not.  An input whose result the cache already holds
needs no worker, and is served here.  And running somewhere other than
this machine is safe only for a function whose effects do not matter,
which is what memoisation already assumes, so only a memoised function
runs on a host.

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
import socket
import sys
import time
import traceback
from collections import deque
from typing import Any, Callable, Iterable

from . import bootstrap, events, localfile, modes
from .project import Project, ProjectError, project_root
from .hosts import Failure, Finished, Host, ProcessConnection, hello_message
from .functionhash import reachable_set
from .debughook import breakpoints_force
from .store import CacheMiss, LocalStore

__all__ = ["run_all", "BatchError"]

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


class BatchError(Exception):
    """An input of a batch produced no result, so the batch stopped.

    ``input`` is the input, ``host`` where it ran, ``failure`` why there
    is no result (see :class:`valuekit.hosts.Failure`).  Inputs that had
    finished are in the cache; the failed input is the one to call by
    hand.
    """

    def __init__(self, fn: str, input: Any, host: str, failure: Failure):
        self.fn = fn
        self.input = input
        self.host = host
        self.failure = failure
        super().__init__(f"{fn}({input!r}) on {host}: {failure}")


class _Task:
    __slots__ = ("x", "idx", "handle", "deadline", "where")

    def __init__(self, x, idx, handle, deadline, where: str):
        self.x = x
        self.idx = idx
        self.handle = handle
        self.deadline = deadline
        self.where = where


class _BatchRecorder:
    """What one batch reports about an input as it finishes: an event, and
    for a memoised function that produced a result, the call record noted
    in the enclosing call's call list, as a call made directly would be."""

    def __init__(self, store, fn, batch_id: int):
        from .pure import _note_call

        self._store = store
        self._fn = fn
        self._memo = getattr(fn, "_valuekit", None)
        self._id = batch_id
        self._note_call = _note_call

    def outcome(self, i: int, x: Any, failure: Failure | None, host: str = "local") -> None:
        if failure is None and self._memo is not None:
            h = self._memo.record_hash(x)
            if h is not None:
                self._note_call(self._fn.__qualname__, self._memo.reachable.hash, h)
        events.record(
            self._store,
            "outcome",
            id=self._id,
            i=i,
            ok=failure is None,
            host=host,
            exc=None if failure is None else failure.type,
        )


_NOT_CACHED = object()


def _cached(fn, x) -> Any:
    """The stored result for *x*, or ``_NOT_CACHED``."""
    try:
        return fn._valuekit.hit(x)
    except CacheMiss:
        return _NOT_CACHED  # no record holds, or its value is gone: a worker runs the input


def _refused(reason: str):
    """A ``connect`` for a host the project cannot go to: fails with the reason."""

    def connect():
        raise RuntimeError(reason)

    return connect


def _remote_connect(name: str, command: list[str], project: Project, source_root: str):
    """A ``connect`` for a remote host: run *command* (an ssh invocation
    that starts the bootstrap there), send the project, and return the
    connection once the host process is about to send its first message.
    Raises with the reason the host cannot be used."""
    from .functionhash import PYTHON

    def connect() -> ProcessConnection:
        conn = ProcessConnection(command)
        try:
            reason = bootstrap.offer(
                conn.rx, conn.tx, source_root, project.name, project.project_hash, PYTHON,
                project.entries, project.pack,
                f"a run of {os.path.basename(sys.argv[0]) or 'python'} on "
                f"{socket.gethostname()} (pid {os.getpid()}, started {time.strftime('%H:%M:%S')})",
                project.dist, project.build_inputs,
            )
        except (OSError, ValueError) as e:
            reason = f"the connection to host {name!r} broke: {e}"
        if reason:
            detail = conn.failure()
            conn.close()
            raise RuntimeError(f"{reason}:\n{detail}" if detail else reason)
        return conn

    return connect


class _Hosts:
    """The hosts a batch may run on, and how many tasks each may hold.

    This machine is a host through a host process started here.  Remote
    hosts come from the local file (or the test hook); each syncs on its
    own thread and counts only once it is ready.  The mode is re-read from
    the file every time capacities are asked for, so a switch made while
    the batch runs applies to the next task started; a file that cannot
    be read leaves the mode last read in force.
    """

    def __init__(
        self, fn, store_dir: str | None, store, batch: int, max_workers: int | None = None,
        remote: bool = True,
    ):
        # Every host puts (handle, payload) here as a worker returns, sends
        # a request, or exits; deliver() passes each to its handle.
        completions: queue.Queue = queue.Queue()
        self._completions = completions
        self.hosts: list[Host] = []
        self._reported: dict[str, str] = {}  # host name -> the state last recorded as an event
        self._store = store
        self._batch = batch
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

        hello = hello_message(fn, "", [], store_dir, sys.path)
        self.local = Host(
            "local", lambda: ProcessConnection([sys.executable, "-m", "valuekit.hostprocess", "--local"]),
            hello, store, completions, self.local_workers, check=False,
        )
        self.local.sync()  # a local host that fails has no capacity; the scheduler runs inputs here

        if store_dir is None or not remote:
            # A remote host sends every result to this process's store;
            # with none configured there is nowhere to put them.  An
            # undecorated function has no function hash for a host to check
            # and makes no promise about its effects, so it runs here only.
            return
        entries = []
        if _host_commands is not None:
            entries = [
                (name, [*(spec[0] if isinstance(spec, tuple) else spec), "-c", bootstrap.STAGE0],
                 spec[1] if isinstance(spec, tuple) else None, os.path.join(store_dir, "source"))
                for name, spec in _host_commands.items()
            ]
        else:
            entries = [
                (h.name, ["ssh", "-T", "-o", "BatchMode=yes", h.ssh, bootstrap.remote_command(h.python)],
                 h.workers, h.source_root)
                for h in config.hosts
            ]
        if not entries:
            return
        # One project for every host: the tree is walked once per batch.
        project = Project(fn, config.project)
        refusal = project.refusal()
        hello = hello_message(fn, project.project_hash, project.roots, "")
        for name, command, workers, source_root in entries:
            if refusal:
                connect = _refused(refusal)
            else:
                connect = _remote_connect(name, command, project, source_root)
            self.hosts.append(Host(name, connect, hello, store, completions, workers))

    def all(self) -> list:
        return [*self.hosts, self.local]

    def mode(self) -> str:
        """The mode in force: the local file's, re-read now; the mode last
        read when the file cannot be read."""
        try:
            self._mode = localfile.load_local(self._root).mode
        except RuntimeError:
            pass
        return self._mode

    def start_syncs(self) -> None:
        """Start syncing every host not yet tried, each on its own thread.
        Never waited for: this machine's workers start at once and a host
        takes tasks once it is ready."""
        for b in self.hosts:
            if b.state == "new":
                threading.Thread(target=b.sync, daemon=True).start()

    def syncing(self) -> bool:
        """Whether any host is still syncing, and so may yet take work."""
        return any(b.state == "syncing" for b in self.hosts)

    def report(self) -> None:
        """Record a ``host`` event for each remote host whose state settled
        or changed since the last report, and for this machine's host only
        when it failed: a local host process that started is not news."""
        for b in (*self.hosts, self.local):
            if b is self.local and b.state != "failed":
                continue
            if b.state in ("ready", "failed") and self._reported.get(b.name) != b.state:
                self._reported[b.name] = b.state
                events.record(
                    self._store, "host", id=self._batch, name=b.name, ok=b.state == "ready",
                    reason=b.failure if b.state == "failed" else None, capacity=b.capacity,
                )

    def capacities(self, mode: str) -> dict[str, int]:
        """Each host's capacity under *mode*, from the hosts' states."""
        remote = {b.name: (b.capacity or 0) if b.state == "ready" else 0 for b in self.hosts}
        local = self.local_workers if self.local.state == "ready" else 0
        return modes.capacities(mode, local, remote, self.syncing())

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
) -> list:
    """Run ``fn`` over ``inputs`` in parallel; return their results in
    input order.

    Each input runs in its own process; ``max_workers`` is how many run at once on this machine
    (default: the ``[local] workers`` line of the local file, else the CPU
    count).  Remote hosts, if the local file names any, add their own
    capacity.  An input whose result is already cached is served without
    a worker.  ``timeout`` limits the seconds each input may spend
    running; a breach kills that input's process and records a
    TimeoutError on its outcome, leaving the rest of the batch unaffected.
    When no host may run anything (``max_workers=0`` and no remote host is
    usable) the inputs run one at a time in this process, without the
    timeout.

    The first input that produces no result ends the batch: running tasks
    are killed and :class:`BatchError` is raised, naming the input, the
    host it ran on and why there is no result (an exception, by type name
    and message with the worker's traceback as text; a timeout; a worker
    that exited without a result).  Inputs that had finished are in the
    cache.  A function that expects bad inputs returns a value that says
    so.  To debug the failed input, call ``fn(x)`` on it.

    A ``@pure`` or ``@pure_local`` function's inputs are served from the
    cache where it holds them, and may run on the local file's hosts.  An
    undecorated function runs on this machine only, with nothing cached:
    it has no function hash for a host to check.  Either way the function
    must be importable by name in a worker.
    """
    decorated = hasattr(fn, "_valuekit")
    inputs = list(inputs)

    # Resolved before the breakpoint check below, so that the sequential
    # fallback reports itself too: a batch being debugged is one you most
    # want to watch.
    from .pure import _current_store

    store = _current_store()
    store_dir = str(store.root) if isinstance(store, LocalStore) else None
    qualname = getattr(fn, "__qualname__", repr(fn))

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
        events.record(store, "batch", id=batch, fn=qualname, n=len(inputs), mode="sequential")
        recorder = _BatchRecorder(store, fn, batch)
        results: list = []
        try:
            for i, x in enumerate(inputs):
                results.append(fn(x))  # an exception propagates: a debugger is attached
                recorder.outcome(i, x, None, host="main")
        finally:
            events.record(store, "end", id=batch)
        return results

    events.record(store, "batch", id=batch, fn=qualname, n=len(inputs), mode="parallel")
    recorder = _BatchRecorder(store, fn, batch)

    results: list = [None] * len(inputs)
    pending = deque()
    for i, x in enumerate(inputs):
        value = _cached(fn, x) if store is not None and decorated else _NOT_CACHED
        if value is _NOT_CACHED:
            pending.append((i, x))
        else:
            results[i] = value
            recorder.outcome(i, x, None)
    if not pending:
        events.record(store, "end", id=batch)
        return results

    hosts = _Hosts(fn, store_dir, store, batch, max_workers, remote=decorated)

    running: list[_Task] = []
    busy: dict[str, int] = {}  # host name -> tasks running there
    requeued: set[int] = set()  # inputs already run again once after losing their host

    def _start_pending() -> None:
        mode = hosts.mode()
        if mode != "local":
            hosts.start_syncs()
        hosts.report()
        caps = hosts.capacities(mode)
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
            results[idx] = fn(x)
        except Exception as e:
            failure = Failure(type(e).__name__, str(e), traceback.format_exc())
            recorder.outcome(idx, x, failure, host="main")
            raise BatchError(qualname, x, "main", failure) from None
        recorder.outcome(idx, x, None, host="main")

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
                    done = Finished("failed", failure=Failure(
                        "timeout", f"exceeded the {timeout} s limit and was killed"
                    ))
                if done is None:
                    continue
                running.remove(t)
                busy[t.where] -= 1
                t.handle.release()
                if done.kind == "closed":
                    if t.idx not in requeued:
                        # The input is not done, not failed.  Run it again
                        # elsewhere, once: an input that takes a host down
                        # each time is a failure after all.
                        requeued.add(t.idx)
                        pending.appendleft((t.idx, t.x))
                        events.record(store, "requeue", id=batch, i=t.idx, host=t.where)
                        continue
                    done = Finished("failed", failure=Failure(
                        "connection", f"the connection to host {t.where!r} closed under this input twice"
                    ))
                recorder.outcome(t.idx, t.x, done.failure, host=t.where)
                if done.kind == "failed":
                    raise BatchError(qualname, t.x, t.where, done.failure)
                results[t.idx] = done.value
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

    return results
