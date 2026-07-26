"""Parallel batch execution.

``run_all(fn, inputs)`` runs ``fn`` over ``inputs`` in parallel and returns
a :class:`BatchResult` of per-input outcomes, in input order.

Each input runs in its own process (spawned per task, up to ``max_workers``
at once) rather than in a shared worker pool.  Isolation is the point: a
timeout kills exactly one process, a segfault loses exactly one input, and
neither affects the inputs running beside it or the number of workers
available to the rest of the batch.  The cost is one process start per
input, roughly 0.4 s including a numpy import; this overlaps across
workers, and is noise for inputs that take seconds or more.  For very
small inputs, batch them inside ``fn``.

Timeouts and worker deaths are per-input failures in every mode, recorded
on the BatchResult with the input that caused them:

* ``timeout=`` limits the seconds each input may spend running.  A breach
  kills that input's process promptly and records a TimeoutError; the rest
  of the batch is unaffected.  There is no replay of a timeout, since
  replaying a deterministic hang would hang this process (to debug one,
  call ``fn(x)`` and pause the debugger).
* A process that dies without raising (a segfault or an out-of-memory
  kill) records a RuntimeError naming the input and the exit code, and is
  deliberately not replayed: an automatic replay would reproduce the crash
  in this process.

Genuine exceptions raised by ``fn`` depend on whether a debugger is
attached:

* No debugger (a batch or production run): failures are collected.  Every
  input is processed; ``.values`` returns the plain list of results,
  raising an ExceptionGroup if any input failed, and ``.failures`` lists
  ``(input, exception)`` pairs for explicit handling.  Collected
  exceptions carry the string-form traceback captured in the worker.
* Debugger attached: fail fast and replay.  On the first failure, the
  remaining processes are killed (cheap: every @pure step persists when it
  completes, so at most one in-flight step per process is discarded) and
  the failing input is rerun sequentially in this process.  The cached
  prefix replays without executing; the failing step executes and raises
  here, with a live stack and a working REPL.
* Debugger attached, with a live breakpoint intersecting anything
  reachable by name from ``fn``: the batch runs sequentially in this
  process, where breakpoints fire and the usual debugger rules apply
  (breakpoints do not reach worker processes; the sequential fallback does
  not enforce the timeout).

A fully successful batch returns the same BatchResult in every mode, and
the mode never changes any result: purity means each input produces the
same value or the same exception either way.

Workers are configured automatically (each process applies the parent's
cache directory before running) and share the cache: value writes are
idempotent and trace writes are atomic appends, so concurrent writers
cannot drop each other's results.  ``fn`` must be a module-level function
(it is sent to workers by reference) and need not itself be @pure;
typically it is a plain driver calling @pure steps.  Worker processes are
daemonic: they are cleaned up if the parent exits, and ``fn`` cannot
itself start processes (parallelise in this driver, not inside it).
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from collections import deque
from multiprocessing import connection as _mp_connection
from typing import Any, Callable, Iterable, Iterator

from .codehash import function_fingerprint
from .debughook import breakpoints_force, debugger_attached
from .store import LocalStore

__all__ = ["run_all", "BatchResult", "Outcome"]

_POLL = 0.2  # seconds between timeout checks while tasks are running


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


def _child_main(conn, cache_dir: str | None, fn, x) -> None:
    """Runs in the worker process: configure the cache, run one input, send
    one message back: ("ok", value) or ("err", exc, tb) or, when the
    exception or value cannot be pickled, ("err_str", type_name, text, tb).
    """
    try:
        if cache_dir is not None:
            from .pure import set_cache_dir

            set_cache_dir(cache_dir)
        try:
            value = fn(x)
        except BaseException as e:
            tb = traceback.format_exc()
            try:
                conn.send(("err", e, tb))
            except Exception:
                conn.send(("err_str", type(e).__name__, str(e), tb))
            return
        try:
            conn.send(("ok", value))
        except Exception as e:
            conn.send(
                (
                    "err_str",
                    type(e).__name__,
                    f"the result could not be sent back: {e}",
                    traceback.format_exc(),
                )
            )
    finally:
        conn.close()


class _Task:
    __slots__ = ("x", "idx", "proc", "conn", "deadline", "timed_out")

    def __init__(self, x, idx, proc, conn, deadline):
        self.x = x
        self.idx = idx
        self.proc = proc
        self.conn = conn
        self.deadline = deadline
        self.timed_out = False


def _harvest(t: _Task, msg, name: str, timeout) -> tuple[Outcome, bool]:
    """Turn a finished task into an Outcome. Returns (outcome, genuine):
    *genuine* is True only for an exception raised by fn itself — the only
    kind that debug mode fails fast on and replays."""
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
                f"{name}({t.x!r}) (exit code {t.proc.exitcode}; a segfault "
                f"or an out-of-memory kill?). Not replaying automatically: "
                f"a replay would reproduce the crash in this process. Call "
                f"{name}({t.x!r}) yourself to debug it."
            )
        return Outcome(t.x, exc=exc), False
    kind = msg[0]
    if kind == "ok":
        return Outcome(t.x, value=msg[1]), False
    if kind == "err":
        exc = msg[1]
        exc.__cause__ = _RemoteTraceback(f"\n{msg[2]}")
        return Outcome(t.x, exc=exc), True
    # "err_str": the worker's exception was not picklable
    exc = RuntimeError(f"{msg[1]}: {msg[2]}")
    exc.__cause__ = _RemoteTraceback(f"\n{msg[3]}")
    return Outcome(t.x, exc=exc), True


def run_all(
    fn: Callable[[Any], Any],
    inputs: Iterable[Any],
    max_workers: int | None = None,
    *,
    timeout: float | None = None,
) -> BatchResult:
    """Run ``fn`` over ``inputs`` in parallel; return a BatchResult in
    input order.

    Each input runs in its own process; ``max_workers`` caps how many run
    at once (default: the CPU count).  ``timeout`` limits the seconds each
    input may spend running; a breach kills that input's process and
    records a TimeoutError on its outcome, leaving the rest of the batch
    unaffected.  Worker deaths are likewise recorded per input.

    Without a debugger, every input is processed and failures are
    collected on the BatchResult.  With a debugger attached, the first
    exception raised by ``fn`` kills the remaining processes and reruns
    the failing input sequentially in this process, where it raises with a
    live stack; if the rerun does not raise, the original failure was
    nondeterministic, which violates the @pure contract, and a
    RuntimeError chaining the worker's exception is raised so that this is
    not silently absorbed.
    """
    inputs = list(inputs)

    # A live breakpoint anywhere reachable from fn: run sequentially, in
    # this process, so the breakpoint fires and the debugger contract
    # applies.  (Also covers VALUEKIT_ALWAYS_RUN.)
    try:
        _, spans, _ = function_fingerprint(fn)
    except Exception:
        spans = []
    if breakpoints_force(spans):
        return BatchResult(Outcome(x, value=fn(x)) for x in inputs)

    debug = debugger_attached()

    from .pure import _current_store

    store = _current_store()
    cache_dir = str(store.root) if isinstance(store, LocalStore) else None

    ctx = multiprocessing.get_context("spawn")
    workers = max_workers or os.cpu_count() or 1
    name = getattr(fn, "__qualname__", repr(fn))

    outcomes: list[Outcome | None] = [None] * len(inputs)
    queue = deque(enumerate(inputs))
    running: list[_Task] = []
    genuine: tuple[Any, BaseException] | None = None

    def _start(idx: int, x: Any) -> None:
        recv_end, send_end = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(send_end, cache_dir, fn, x), daemon=True
        )
        proc.start()
        send_end.close()  # keep only the child's handle: EOF then means death
        deadline = time.monotonic() + timeout if timeout is not None else None
        running.append(_Task(x, idx, proc, recv_end, deadline))

    try:
        while queue or running:
            while queue and len(running) < workers and genuine is None:
                _start(*queue.popleft())
            if not running:
                break
            ready = _mp_connection.wait(
                [t.conn for t in running],
                timeout=_POLL if timeout is not None else None,
            )
            now = time.monotonic()
            for t in list(running):
                msg = None
                if t.conn in ready:
                    try:
                        msg = t.conn.recv()
                    except EOFError:
                        msg = None  # died without sending
                elif t.deadline is not None and now >= t.deadline:
                    t.proc.kill()
                    t.proc.join()
                    if t.conn.poll():  # finished just before the kill landed
                        try:
                            msg = t.conn.recv()
                        except EOFError:
                            msg = None
                    t.timed_out = msg is None
                else:
                    continue
                running.remove(t)
                t.proc.join()
                t.conn.close()
                outcome, is_genuine = _harvest(t, msg, name, timeout)
                outcomes[t.idx] = outcome
                if debug and is_genuine and genuine is None:
                    genuine = (t.x, outcome._exc)
            if genuine is not None:
                break

        if genuine is not None:
            for t in running:
                t.proc.kill()
            for t in running:
                t.proc.join()
                t.conn.close()
            running.clear()
            x, exc = genuine
            fn(x)  # replays the cached prefix, then raises inline
            raise RuntimeError(
                f"{name}({x!r}) failed in a worker but succeeded on "
                f"sequential replay: the failure is nondeterministic, which "
                f"violates the @pure contract."
            ) from exc
    finally:
        # Covers KeyboardInterrupt and the replay raise: no orphans.
        for t in running:
            try:
                t.proc.kill()
            except Exception:
                pass

    return BatchResult(o for o in outcomes if o is not None)
