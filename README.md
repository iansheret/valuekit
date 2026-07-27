# valuekit

An immutable map for pipeline data, plus disk memoisation for pure
functions. Both rest on the same idea: pipeline data as immutable *values*,
identified by content. The two parts are independent; use either without the
other.

## The immutable map

```python
from valuekit import ImmutableMap

ctx = ImmutableMap({"raw": signal, "fs": 1000.0})

ctx2 = ctx | {"scaled": ctx["raw"] * gain}   # derive; ctx is unchanged
ctx3 = ctx2.assoc("window", "hann")          # single-key derivation
ctx4 = ctx3.dissoc("tmp")                    # drop keys
```

Values are frozen on entry: numpy arrays become read-only (copied only if
writeable; set `arr.flags.writeable = False` beforehand to share without a
copy), sets become frozensets, nested dicts become ImmutableMaps, and
unknown mutable types are rejected with a `TypeError`. The rejection is
deliberate: a type must be registered (`register_type`) before it can be
stored, so nothing mutable gets in by accident. Deriving with `|` shares
unchanged values by reference, so adding one key to a 2 GB context copies
one dict, not 2 GB of data.

Using the map activates nothing else: no cache, no decorator, no
configuration.

## `@pure`

```python
from valuekit import ImmutableMap, pure, set_cache_dir

set_cache_dir("~/.cache/mypipeline")     # nothing is cached until this is called

@pure
def calculate_geometry(obs, config):
    order = config["geometry"]["order"]
    ...
    return {"az": az, "el": el}          # the returned dict is the diff

obs = obs | calculate_geometry(obs, config)
```

`@pure` states a *contract*: the function's output depends only on what it
reads from its arguments, and it has no effects that matter. valuekit does
not verify this; it memoises to disk on the assumption that it holds. The
decorator takes no options, so there is nothing to configure per function.

Note that on a cache hit the function body does not run. Prints, plots,
progress bars, and file writes inside a `@pure` function will not happen on
replays. If a side effect matters, it does not belong in a pure function.

### The contract

The guarantee: a cache hit returns exactly what executing the current
definition on the current arguments would return. The user's promise: the
result depends only on what the function reads from its arguments, plus its
definition. "The definition" means everything reachable by name from the
function's code; if go-to-definition in an IDE can reach it from the
function, it is part of the function's *fingerprint*. Names are resolved at
the function's first call, once the module is fully loaded, so definition
order does not matter and mutual recursion works.

A result is recomputed when any of these change:

| What changed | Why it is tracked |
|---|---|
| a key the call read (or probed and found absent) in a map argument | each call records a trace of exactly what it read |
| the content of any non-map argument | arguments are hashed whole |
| the function's code, or any user function it calls, recursively (helpers, lambdas, methods of user classes, other `@pure` functions) | the recursive code hash |
| an immutable module constant it uses (numbers, strings, tuples, frozensets, read-only arrays) | constants are part of the definition; `x / SPEED_OF_LIGHT` and `x / 299792458.0` invalidate identically |
| default and closure values | part of the definition |
| the version of an installed package it uses, or the Python version | package and standard-library boundaries contribute version markers |

Whitespace, comments, and the function's name are not changes.

A stale result is served when the change was invisible to the fingerprint.
This is the user's responsibility, by design:

| Invisible to the fingerprint | Remedy |
|---|---|
| mutable globals (lists, dicts, sets, writeable arrays), whether rebound, mutated, or edited in source | make them constant (tuple, frozenset, `arr.flags.writeable = False`) or pass them as arguments |
| dispatch through data: `getattr(mod, name)()`, registries, callables stored in structures | pass the function as an argument; functions are hashed by fingerprint, so lambdas work |
| file contents read inside the function | pass the data, or its path and a version, as arguments |
| runtime purity violations: unseeded RNG or clock reads that reach the result, mutation of arguments or globals | none; these break the promise |

The remedy column repeats one idea: arguments are always tracked, so moving
a dependency into the arguments makes it visible. If something invisible
changed anyway, clear it. `clear_cache(fn)` means "`fn` has changed" and
behaves as if it had: it deletes the recorded results of `fn` and of every
`@pure` function that computed through it (callers, transitively, and uses
of `fn` as an argument), across processes, using a small on-disk dependency
index. Matching is conservative: clearing too much means recomputing, while
clearing too little would mean wrong results, so ties resolve towards
clearing more. `clear_cache()` deletes everything.

Tunables belong in config maps rather than in module globals. A traced
config read is exact per call (change an unread key and hits are kept),
while a module constant is definition-wide (edit it and every function
naming it recomputes).

Decoration emits no warnings. Side effects in a `@pure` function (logging,
progress bars, metrics) are permitted by the contract precisely because
they will not happen on a hit; whether that is acceptable is the user's
decision.

### Granularity

- `config["filter"]["order"]` records a dependency on that one leaf. Taking
  `f = config["filter"]` and then iterating, printing, or comparing `f`
  observes the whole subtree and records a whole-map dependency. Anything
  that looks at all keys (`len`, iteration, `==`, `keys()`) is a whole-map
  read: correct, but coarser.
- Absence is a dependency. `config.get("detrend", 0)` on a map without
  `"detrend"` records the absence; adding that key later invalidates, and
  adding other keys does not.
- Conditional reads produce separate traces. A function that reads different
  keys on different branches accumulates one trace per observed read-set,
  each matched independently.
- Plain dicts work. Arguments are frozen at the boundary, so a raw `dict`
  gets the same per-key treatment as an `ImmutableMap`.
- Non-map arguments (arrays, scalars, tuples, lambdas) key the cache by
  content hash, whole.
- A recorded map passed into a nested `@pure` call records a whole-map
  dependency at that boundary.

### Debugging

Caching stays on while a debugger is attached. A hit is bypassed, and the
function runs, only when a live breakpoint intersects the function or
anything in its user-code dependency closure. Set a breakpoint in a step or
in one of its helpers and that step executes; clear the breakpoint and hits
resume. Through nested `@pure` calls this applies to the path from the
breakpoint to the root: a breakpoint in an inner function also forces its
`@pure` callers to execute, since a cached caller would otherwise skip the
breakpoint, while sibling stages inside a forced caller are unaffected and
continue to hit and to record. Forced runs never write to the cache, and a
recording whose execution contained a forced run (e.g. a breakpoint added
while paused mid-pipeline) is discarded rather than stored, so nothing done
in a debug session, such as evaluating expressions or modifying locals, can
enter the cache.

Supported debuggers: pydevd (PyCharm, and VS Code's debugpy) and anything
built on `bdb` (pdb, ipdb). Their breakpoint tables are internal APIs, so
access is defensive: if a debugger is detected but its table cannot be
read, valuekit behaves as if there were breakpoints everywhere, which costs
cache hits but never skips a breakpoint. Coverage tools and profilers are
recognised as non-debuggers and do not disable caching.

Manual overrides, from narrowest to broadest:

```python
step.uncached(obs, cfg)   # call the raw function; the cache is untouched
VALUEKIT_ALWAYS_RUN=1     # env var: execute everything, write nothing
clear_cache(step)         # "step changed": deletes step's results and its callers'
clear_cache()             # or delete the cache directory; always safe
```

### The store

The cache directory holds content-addressed files: arrays as `.npy`,
reloaded as read-only memory maps that `freeze` shares without copying (a
hit on a function returning a 2 GB array copies nothing), and everything
else in a small structural format in which composite values reference their
children by hash, so an array shared by many results is stored once. There
is no pickle anywhere. Cacheable return values are a fixed set: `None`,
`bool`, `int`, `float`, `complex`, `str`, `bytes`, `range`, numpy scalars
and arrays, tuples, frozensets, and (Immutable)Maps of the same.

Writes are atomic; directories can be shared between processes; a missing
or corrupt entry is treated as a miss. Deleting the cache is always safe. A
call that raises caches nothing. There is no eviction in this version: the
cache is a directory, so check its size with `du -sh` and delete it when it
grows too large.

### Parallelism

``run_all(fn, inputs)`` runs a module-level function over a batch of
inputs in parallel and returns a ``BatchResult`` of per-input outcomes, in
input order. Each input runs in its own process, spawned per task with at
most ``max_workers`` at once. Isolation is the point: a timeout kills
exactly one process, a segfault loses exactly one input, and neither
affects the inputs running beside it or the capacity available to the
rest of the batch. The cost is one process start per input (roughly 0.4 s
including a numpy import), which overlaps across workers and is noise for
inputs that take seconds or more; for very small inputs, batch them
inside ``fn``. Workers configure themselves from the parent's cache
directory and share the cache: value writes are idempotent and trace
writes are atomic appends, so concurrent writers cannot drop each other's
results.

Every input is processed, and every failure is recorded against the input
that caused it. An exception raised by ``fn`` carries the string-form
traceback captured in the worker. ``timeout=`` limits the seconds each
input may spend running; a breach kills that input's process promptly and
records a ``TimeoutError``. A process that dies without raising (a
segfault or an out-of-memory kill) records a ``RuntimeError`` naming the
input and the exit code.

```python
result = run_all(process_scenario, session_ids)

result.values                 # plain list of results; raises an
                              # ExceptionGroup if any input failed
for sid, exc in result.failures:
    ...                       # explicit handling; the batch completed
result[i].input               # the input that produced outcome i
result[i].result()            # the value, or re-raises the exception
```

``.values`` is the accessor to reach for by default: it is the plain list
of results when everything succeeded, and it raises when something
failed, so failures cannot be dropped by accident. ``.failures`` is for
callers that handle failures explicitly and continue.

Nothing is replayed automatically. To debug a failure, call the function
on that one input yourself:

```python
process_scenario(sid)         # the cached prefix replays in milliseconds;
                              # the failing step executes and raises here
```

with a live stack and a working REPL. Picking the input is the point:
which scenario fails first in a parallel batch is a race, so an automatic
replay would drop you into whichever one happened to lose it.

One debugger accommodation remains, because breakpoints do not reach
worker processes. If a live breakpoint intersects anything reachable by
name from ``fn``, the whole batch runs sequentially in this process, where
breakpoints fire and the usual debugger rules apply. The sequential
fallback does not enforce the timeout. Merely having a debugger attached
changes nothing on its own.

Two rules for using other pools (joblib, dask, a bare executor) around
``@pure`` code: parallelise in the driver, between ``@pure`` calls, never
inside a ``@pure`` function's body (reads performed in worker processes are
not recorded, which produces traces with missing dependencies and therefore
stale results); and call ``set_cache_dir`` at module top level, since a call
inside an ``if __name__ == "__main__":`` block, or in a notebook, does not
reach spawn-based workers. (``run_all`` is exempt: it passes the cache
directory to each worker explicitly.) To drive the location from the
environment, read the variable yourself, at top level:

```python
import os
from valuekit import set_cache_dir

set_cache_dir(os.environ.get("VALUEKIT_CACHE"))   # None disables caching
```

Nothing is cached until `set_cache_dir` is called: importing valuekit has no
effect on its own.

## Install

```
pip install valuekit        # Python >= 3.11; depends only on numpy
```

## Development

```
pip install -e .[dev]
pytest
```
