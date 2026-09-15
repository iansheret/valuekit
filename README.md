# valuekit

Disk memoisation for pure functions, plus an immutable map for the
pipeline data they run over. Both apply the same idea: pipeline data as
immutable *values*, identified by content. The two parts are independent;
use either without the other. On top of them: a batch runner, a log of the
values a run produced that analysis code reads back by their labels, and a
monitor for watching a run from another process.

The usual ways of caching a pipeline fail in one of two directions. If the
key is too coarse (a file path, a manual version tag), results go stale silently and hits stop
being trusted; if it is too broad (whole-argument hashes), one config edit
recomputes everything and hits stop happening. Either way the cache ends
up cleared before every run that matters, at which point it saves nothing.

valuekit is built so the cache can stay on. Invalidation follows what each
call actually read (change one config key and only the steps that read it
recompute) and what the code actually is (edit a helper and everything
that depends on it recomputes). What the tracking cannot see is a short
documented list, each entry with a remedy, and an uncertain match
recomputes rather than risk a stale result. Caching stays on under a
debugger too: the cached prefix is served in milliseconds, the step under
the breakpoint executes and stops there, and nothing done while paused
enters the cache.

## `@pure`

```python
from valuekit import pure, set_store_dir

set_store_dir("~/.cache/mypipeline")     # nothing is cached until this is called
                                         # (and deleting the directory is always safe)

@pure
def calculate_geometry(obs, config):
    order = config["geometry"]["order"]
    ...
    return {"az": az, "el": el}          # the returned dict is the diff

obs = obs | calculate_geometry(obs, config)
```

`obs` and `config` are `ImmutableMap`s here, and that is what gives the
per-key tracing: reads of a map argument are recorded individually, so a
change to a key the function never read does not invalidate it. Any other
argument — including a plain `dict` — is hashed whole. `ImmutableMap` has
its own section below.

Nothing else about the call changes. Arguments are passed as the objects you
passed, and the result is the object the function built, so a cache hit
differs from a miss only in that the body did not run.

`@pure` states a *contract*: the function's output depends only on what it
reads from its arguments, and it has no effects that matter. valuekit does
not verify this; it memoises to disk on the assumption that it holds. The
decorator takes no options, so there is nothing to configure per function.

Note that on a cache hit the function body does not run. Prints, plots,
progress bars, and file writes inside a `@pure` function will not happen on
hits. If a side effect matters, it does not belong in a pure function.

### The contract

The guarantee: a cache hit returns exactly what executing the current
definition on the current arguments would return. The user's promise: the
result depends only on what the function reads from its arguments, plus its
definition. "The definition" means everything reachable by name from the
function's code; if go-to-definition in an IDE can reach it from the
function, it is part of the function's *function hash*. Names are resolved at
the function's first call, once the module is fully loaded, so definition
order does not matter and mutual recursion works.

A result is recomputed when any of these change:

| What changed | Why it is tracked |
|---|---|
| a key the call read (or probed and found absent) in a map argument | each call records exactly what it read |
| the content of any non-map argument | arguments are hashed whole |
| the function's code, or any user function it calls, recursively (helpers, lambdas, methods of user classes, other `@pure` functions) | the recursive function hash |
| an immutable module constant it uses (numbers, strings, tuples, frozensets, read-only arrays) | constants are part of the definition; `x / SPEED_OF_LIGHT` and `x / 299792458.0` invalidate identically |
| default and closure values | part of the definition |
| the version of an installed package it uses, or the Python version | package and standard-library boundaries contribute version markers |
| the compiled contents of a native extension it calls, where that extension is your own | an extension installed from a local directory is rebuilt in place under an unchanged version, so its binary is hashed instead |

Whitespace, comments, and the function's name are not changes.

A stale result is served when the change was invisible to the function hash.
This is the user's responsibility, by design:

| Invisible to the function hash | Remedy |
|---|---|
| mutable globals (lists, dicts, sets, writeable arrays), whether rebound, mutated, or edited in source | make them constant (tuple, frozenset, `arr.flags.writeable = False`) or pass them as arguments |
| dispatch through data: `getattr(mod, name)()`, registries, callables stored in structures | pass the function as an argument; functions are hashed by function hash, so lambdas work |
| file contents read inside the function | pass the data, or its path and a version, as arguments |
| runtime purity violations: unseeded RNG or clock reads that reach the result, mutation of arguments or globals | none; these break the promise |

The remedy column repeats one idea: arguments are always tracked, so moving
a dependency into the arguments makes it visible. If something invisible
changed anyway, edit the function: any change to its source gives it a
new function hash, and every function that reaches it a new one too, so
all of them recompute at their next call. `clear_cache()` deletes
everything computed or logged.

Tunables belong in config maps rather than in module globals. A traced
config read is exact per call (change an unread key and hits continue),
while a module constant is definition-wide (edit it and every function
naming it recomputes).

### Native extensions

A compiled extension (nanobind, pybind11, Cython, plain C) has no source to
walk, so its binary is its marker. That matters where the version marker
would be wrong: a package installed from a released wheel changes
only through a reinstall, which moves its version, while a package installed
from a local directory — `pip install -e .` or `pip install .` — is rebuilt
in place under the same version. Extensions in the second group are hashed
by content, so rebuilding your own C++ recomputes what depends on it, and
released wheels keep their version markers, which cost nothing to compute.

The binary is a stricter dependency than the sources it was built from: it
carries the compiler, its flags, and any library linked statically into the
result, none of which the sources mention. It is also the code that actually
runs, so editing a source file without rebuilding correctly changes nothing.
The cost is one read of the file per build, on the order of a millisecond
for a few megabytes.

The build is the project's own: its lock tool installs the project and
rebuilds it (for uv, `uv sync --reinstall-package <name>`), through the
build backend `pyproject.toml` names. Nothing rebuilds on import, and a
backend's rebuild-on-import option is not supported with parallel workers
on any backend: many processes importing at once would run the build tool
at once in one build directory. To rebuild without a step of your own, call
`valuekit.build()` at the top of the script, before the project is
imported:

```python
import valuekit
valuekit.build()          # rebuilds the project if a build input changed
import mypipeline         # imports the current binary
```

It runs the rebuild command when the project's *build inputs* changed since
the last build, and does nothing otherwise. The build inputs are what the
project says they are, as globs in `pyproject.toml`:

```toml
[tool.valuekit]
build-inputs = ["CMakeLists.txt", "src/**/*.c", "src/**/*.h"]
```

Without the list, every file that is not a Python source counts.
`pyproject.toml` and the lock file always count. A host applies the same
rule when a sync sends changed files (below). Set the backend's build
directory (`build-dir` for scikit-build-core and meson-python) so a
rebuild is incremental.

Decoration emits no warnings. Side effects in a `@pure` function (printing,
progress bars, metrics) are permitted by the contract precisely because
they will not happen on a hit; whether that is acceptable is the user's
decision.

## `@pure_local`

A `@pure` function's result depends on its arguments and its definition and
nothing else. Most pipelines also have a few functions whose result depends
on something outside the program: a file on this machine, a database, a
download that needs this machine's credentials. `@pure_local` memoises such
a function exactly as `@pure` does, on a different promise: that the thing
it reads, as seen through these arguments, never changes. The same
arguments give the same value, now and later.

```python
@pure_local
def fetch_session(session_id):
    return download(session_id, token=os.environ["TOKEN"])   # bytes, or arrays

@pure
def process(session_id):
    return analyse(fetch_session(session_id))
```

If the promise cannot be made for a source that does change, put the
version in the arguments (a date, a commit, an etag), where it is tracked
like everything else; editing the function says "what this function sees
has changed". Effects that do not reach the result, such as a scratch
file or a download cache, are permitted, since on a hit none of them
happen. The result must be a value, never a path: a path from this machine
means nothing on another.

Because the function reads an environment, it runs only on this machine,
never on a remote host. A batch running elsewhere sends such calls back
here and receives the value. Expect a
pipeline to have a handful of these at the top and `@pure` everywhere else;
`@pure_local` is not the way out when `@pure` feels strict.

## The immutable map

`@pure` does not require the map — any hashable argument works — but the
map is how a function opts an argument into per-key invalidation. There
are three places it is worth using.

The first is granularity, and it is the reason the other two matter under
`@pure`. A plain `dict` argument is a single opaque value: nothing observed
which keys the function used, so any edit anywhere in it invalidates the
result. The same dict as an `ImmutableMap` is traced key by key.

The second is config. Tunables belong in a config passed as an argument,
where every read is traced; the same tunables in a module-level dict are a
mutable global, the first row of the stale-results table above. An
`ImmutableMap` config removes that hazard, since nothing can edit it in
place, and it makes derivation the way to vary settings: an override is
`cfg | {"gain": 2.0}`, a sweep is `[cfg.assoc("order", n) for n in orders]`,
and each variant is a distinct value that recomputes only the steps that
read the changed key.

The third is the data flowing between steps. Frozen state means no step
can mutate another's input, by design or by accident, and each step
returns a derived map instead of editing a shared one:

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

Freezing is the map's behaviour, and only the map's: `@pure` never converts
an argument or a result. Putting a value into a map is where you ask for it.

Using the map activates nothing else: no cache, no decorator, no
configuration.

## Read granularity

- `config["filter"]["order"]` records a dependency on that one leaf. Taking
  `f = config["filter"]` and then iterating, printing, or comparing `f`
  observes the whole subtree and records a whole-map dependency. Anything
  that looks at all keys (`len`, iteration, `==`, `keys()`) is a whole-map
  read: correct, but coarser.
- Deriving inside a `@pure` function works exactly as it does outside: `|`,
  `assoc` and `dissoc` are all available on the map you were passed, and
  return a plain `ImmutableMap`. Each copies every key, so each is a
  whole-map read; derive from the narrowest map you can.
- Absence is a dependency. `config.get("detrend", 0)` on a map without
  `"detrend"` records the absence; adding that key later invalidates, and
  adding other keys does not.
- Conditional reads produce separate call records. A function that reads different
  keys on different branches accumulates one call record per observed read-set,
  each matched independently.
- Every other argument (plain dicts, lists, arrays, scalars, tuples,
  lambdas, plain-data dataclasses) keys the cache by content hash, whole.
  Nothing observed how the function used it, so any change to it
  invalidates.
- A map passed from one `@pure` call into a nested one is traced in both:
  the inner call gets its own per-key call record, and the outer stays valid only
  for maps that would drive the inner the same way.

## Debugging

Caching stays on while a debugger is attached. A hit is bypassed, and the
function runs, only when a live breakpoint intersects the function or
anything in its reachable set. Set a breakpoint in a step or
in one of its helpers and that step executes; clear the breakpoint and hits
resume. Through nested `@pure` calls this applies to the path from the
breakpoint to the root: a breakpoint in an inner function also forces its
`@pure` callers to execute, since a cached caller would otherwise skip the
breakpoint, while sibling stages inside a forced caller are unaffected and
continue to hit and to record. Forced runs write no call record, and a
recording whose execution contained a forced run (e.g. a breakpoint added
while paused mid-pipeline) is discarded rather than stored, so nothing done
in a debug session, such as evaluating expressions or modifying locals, can
enter the cache. Values a forced run logs still go to the run's log: a run
being debugged is one whose values you want to see.

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
clear_cache()             # everything computed or logged; the same as deleting the directory
```

## The store

The store directory holds the values, call records and run logs a
project's code produced. `clear_cache()` deletes all of them, and
so does deleting the directory; both are always safe. Values are
content-addressed files: read-only arrays as
`.npy`, reloaded as memory maps that `freeze` shares without copying (a hit
on a function returning a 2 GB read-only array copies nothing), writeable
arrays as `.npyw`, and everything else in a small structural format in which
composite values reference their children by hash, so an array shared by
many results is stored once. There is no pickle anywhere. Cacheable return
values are a fixed set: `None`, `bool`, `int`, `float`, `complex`, `str`,
`bytes`, `range`, numpy scalars and arrays, tuples, lists, sets, frozensets,
dicts, `ImmutableMap`s and plain-data dataclasses of the same.

A stored value reloads as an equal value of the same type, which is what
lets a hit be returned in place of running the call. That is also why the content hash
distinguishes a list from a tuple, two dicts that differ only in order, and
a writeable array from a read-only one: a hash has to identify a value
exactly for a content-addressed store to be able to hand it back.

### Plain-data dataclasses

A dataclass whose whole meaning is its fields needs no registration: pass
one as an argument, return one from a `@pure` function, nest them, and they
are hashed and stored like any other value. Its identity is its qualified
name, its dataclass parameters and its ordered fields, so renaming a field,
adding one, or flipping `order=` is a different value and recomputes.
`frozen=` and `slots=` are both fine.

Plain data is checked, not assumed. The class must be built by assignment
alone — no `__post_init__`, no `InitVar`, no `init=False` fields — the
instance must hold exactly its declared fields, and neither the class nor
any base may define a method, property, `staticmethod` or `classmethod`.
Anything else raises, and points at `register_type`.

Methods are the line because a method reached through an argument —
`obs.magnitude()` — is an attribute name, so it resolves to nothing at
module scope and never enters the calling function's function hash: edit it
and the cache would serve a stale result. So adding a method to a dataclass
you already cache turns it into a `register_type` job, where you take on
hashing it yourself. That cliff is deliberate.

A stored entry names its class, but nothing is imported because an entry
names it: the class is resolved only among modules the process has already
loaded, and a class that has changed since the entry was written reads as a
miss rather than being rebuilt into something it no longer means.

Caching a type is not the same as freezing it, so a dataclass still cannot
go into an `ImmutableMap` without a `freeze_fn`.

Every file is named by the hash of its content, call records included, and is
written whole: two processes writing the same entry write the same bytes
under the same name, so directories can be shared between any number of
processes, on Windows as well as POSIX, with nothing to coordinate. A
missing or corrupt entry is treated as a miss. A call that raises caches
nothing.

### Retention

Nothing is removed for being old: a result whose function has not
changed is current whatever its age. The store grows until
`clear_cache()` empties it, or you delete the directory.

## Parallelism

``run_all(fn, inputs)`` runs a module-level ``@pure`` (or ``@pure_local``)
function over a batch of inputs in parallel and returns their results as
a list, in input order. An input whose result is already cached is served
without a worker. Each other input runs in its own process, started per
task by a host process on this machine, with ``max_workers`` running at
once (default: the ``[local] workers`` line of ``valuekit.local.toml``,
else the CPU count). Isolation is the point: a timeout kills exactly one
process and a segfault loses exactly one input. The cost is one process
start per input (roughly 0.4 s including a numpy import). Starts overlap
across workers, and for inputs that take seconds or more the cost does
not matter; for very small inputs, batch them inside ``fn``. A worker on
this machine reads and writes values and call records in the store
directory itself; every write is a content-named file, so concurrent
writers cannot drop each other's results.

A function that is not decorated runs on this machine only, with nothing
cached: it has no function hash for a host to check. Everything else
about the batch is the same.

The first input that produces no result ends the batch: running tasks
are killed and ``BatchError`` is raised, naming the input, the host it
ran on, and why there is no result. An exception raised by ``fn`` is
reported by its type name and message, with the worker's traceback as
text; ``timeout=`` limits the seconds each input may spend running, and a
breach kills that input's process and reports a timeout; a process that
exits without raising (a segfault or an out-of-memory kill) reports its
exit code. Inputs that had finished are in the cache, so the rerun after
the fix serves them. A function that expects bad inputs returns a value
that says so; that is where the author knows what is recoverable.

```python
try:
    results = run_all(process_scenario, session_ids)
except BatchError as e:
    e.input, e.host, e.failure.type, e.failure.message, e.failure.traceback
```

Nothing is re-run automatically. To debug a failure, call the function
on that one input yourself:

```python
process_scenario(sid)         # the cached prefix is served in milliseconds;
                              # the failing step executes and raises here
```

with a live stack and a working REPL. Choosing the input yourself is
deliberate: which input fails first in a parallel batch differs from run
to run, so an automatic re-run would pick one arbitrarily.

One debugger accommodation remains, because breakpoints do not reach
worker processes. If a live breakpoint intersects anything reachable by
name from ``fn``, the whole batch runs sequentially in this process, where
breakpoints fire and the usual debugger rules apply. The sequential
fallback does not enforce the timeout. Merely having a debugger attached
changes nothing on its own. A batch that no host may run, because
``max_workers=0`` and no remote host is usable, likewise runs its inputs
one at a time in this process.

Two rules for using other pools (joblib, dask, a bare executor) around
``@pure`` code: parallelise in the main process, between ``@pure`` calls, never
inside a ``@pure`` function's body (reads performed in worker processes are
not recorded, which produces call records with missing dependencies and therefore
stale results); and call ``set_store_dir`` at module top level, since a call
inside an ``if __name__ == "__main__":`` block, or in a notebook, does not
reach the processes such a pool starts. (``run_all`` is exempt: it tells
each worker the store directory.) To drive the location from the
environment, read the variable yourself, at top level:

```python
import os
from valuekit import set_store_dir

set_store_dir(os.environ.get("VALUEKIT_STORE"))   # None disables caching
```

Nothing is cached until `set_store_dir` is called: importing valuekit has no
effect on its own.

## Logging values

Processing code and the code that looks at what it produced belong in
different places. A pipeline step should not know what will be plotted,
and a plotting script should neither import the pipeline nor run any of
it. What connects them is `log`:

```python
from valuekit import log

@pure
def analyse(obs, cfg):
    residuals = fit(obs, cfg)
    log({"quantity": "residuals", "sid": obs["sid"], "filter": cfg["filter"]}, residuals)
    return summarise(residuals)
```

`log(labels, value)` records a value under *labels*: a small mapping
that says what the value is, in whatever terms the code that reads it
will use. valuekit gives no key any meaning, and prescribes nothing about
how labels are built or passed around; a project wraps `log` in its own
conventions. The value is stored like a result (it must be storable), and
so are the labels, so label values are valuekit values: strings,
numbers, tuples, maps. A label the step reads from an argument is a read
like any other, so renaming an experiment in a config map recomputes the
steps that log it.

The plotting script reads by *containment*: a logged value matches when
its labels hold every key/value pair asked for, and may hold more.

```python
L = valuekit.logs("physics")            # what the last run of physics.py produced
sel = L.where(quantity="residuals")      # kwargs, or a mapping: where({"sid": 7})
for logged in sel:
    plot(logged.value, label=logged.labels["sid"])
r = sel.where(sid=7).one().value         # exactly one logged value, or LookupError
```

Matching is exact by value, so `1` and `1.0` are different labels. A
selection is iterable and can be narrowed again; `one()` refuses zero
logged values or several. Every emission is its own logged value: the
same labels logged twice give two, and a step that logs in a loop puts
the step number in the labels if it matters, since order carries no
meaning.

What `logs("physics")` holds is the complete set of logged values the
last run of that script produced, as if the code had run from scratch. A
step that executes writes each logged value as it makes it; a step served
from cache writes one line naming its call record, which holds what the
step logged, nested calls included, and `logs()` reads them out of the
record. So a re-run after an edit shows exactly the current code's logged
values, the unchanged steps' from cache and the edited steps' fresh, and
nothing from before. Each script keeps its own log, named by its file
stem, begins when `set_store_dir` is called, and replaces the previous
run of the same script; a debugging
script never touches the main script's log. With one script logged under
a store directory, `logs()` needs no name. `log` outside a memoised call,
in the script itself, goes to the log with no call record; with no store
directory configured it does nothing.

The log is as current as the cache: `clear_cache()` deletes the logs
with everything else. A reference whose call record was deleted by other
means reads as stale; `logs()` warns, counts it in `L.stale`, and running
the script again repairs it.

A script whose work is one `@pure` function, `main()`, has one line in its
log after an unchanged re-run, and every logged value inside it is
reachable from that one record. Nothing requires this shape; it costs
nothing and keeps the log small.

A log is readable while its run is going: `L.refresh()` picks up new
logged values. Arrays come back as memory maps. Logs go in `logs/` in the
store directory with everything else, so there is nothing to configure and
nothing recorded without a store directory.

## Running on other machines

A batch can run on any machine you can reach with ssh, as if this machine
had that many more cores. The rule for what a machine needs is the rule
for a person: if they could check your project out there and run it, so
can valuekit. Concretely the host needs a Python 3 to start with, the tool
your project locks its dependencies with, a compiler if you build an
extension, and the network. Nothing of yours, valuekit included, is
installed there beforehand: your project's own lock file says what the
environment is, and the host builds it.

The cache stays here: every value, call record, logged value and event a
remote worker produces is sent back over the connection, every lookup asks this machine,
and `@pure_local` calls run here. Nothing you compute elsewhere has to be
fetched, and the remote machine keeps no results. From the analysis code's
point of view a batch that ran on three machines is indistinguishable from
one that ran on this one.

Your project has to be a *locked* project: its tree must carry a lock file
from a tool valuekit recognises. Today that is `uv.lock`; a tree
without one is refused before anything is sent, and the refusal names the
lock files valuekit understands. The lock pins the Python version and every
dependency, so what the host builds is what you have.

Which machines a checkout may use is written in `valuekit.local.toml`,
beside `pyproject.toml`. The file is about this checkout on this machine,
not about the project, so add it to `.gitignore`; valuekit warns if git
tracks it. A checkout without the file runs on this machine only.

```toml
project = "residuals-experiment"              # optional; the project's directory name on each host
mode = "all"                                  # optional; all, local or remote (below)

[local]
workers = 8                                   # optional; default: CPU count

[hosts.mac]
ssh = "ian@mac.local"                         # anything ssh accepts
python = "python3"                            # optional; any Python 3 there ("python" on Windows)
workers = 8                                   # optional; default: the host's CPU count
source_root = "~/.cache/valuekit/source"      # optional; this is the default
```

Login must work without a prompt (`ssh mac.local true`), which means a key
and, on macOS, Remote Login switched on; on Windows the OpenSSH Server
feature. A key with a passphrase needs an agent holding it wherever the
main process runs, so a main process that is itself reached over ssh needs `ssh -A`. The lock tool must be on the PATH a *non-interactive* ssh session
sees, which is shorter than your login shell's; valuekit also looks in
`~/.local/bin` and `~/.cargo/bin`. A Windows host is reached through sshd's
default shell: leave that as `cmd.exe`, because PowerShell in that role
strips the quotes the bootstrap command needs.

What happens on the host: your project is sent as a source tree (tracked
files plus untracked files that are not ignored, never build artefacts,
never `valuekit.local.toml`) into `source_root/<project>`, the lock tool syncs it there (for uv, `uv sync
--frozen`, for the same Python minor version you are running; uv fetches
that interpreter if the host lacks it), and workers run in the environment
that produced. A native extension is built on the host from the same
sources, by the project's own build backend. Its binary differs from yours,
and that is expected: an extension's marker in the function hash is the
hash of *your* build of it, and a worker is given that marker rather than
hashing its own binary, so results computed anywhere are keyed by what you
ran. The host's binary must then be a build of the same sources, which the
sync guarantees: the first sync installs the project through its lock
tool, and a later sync whose changed files include a build input (see
"Native extensions") runs the tool's rebuild command, `uv sync --frozen
--reinstall-package <name>`. A worker imports, and only imports. Put the
build tools in the project (`cmake` and `ninja` are on PyPI) and build
without isolation (for uv, `no-build-isolation-package` under
`[tool.uv]`, with `scikit-build-core` among the dependencies), so the
build directory's tools exist between builds. Workers run in the
environment activated, so whatever the lock installed beside the
interpreter is on their PATH.

Each host keeps one directory per project, named by the project (the
`project` key, else the name in `pyproject.toml`), and updates it in
place: a run sends only the files whose content changed and the names of
those removed, and touches nothing else there, so the build directory and
the environment persist and a native extension rebuilds incrementally. An
unchanged project costs one comparison. Two checkouts of one project that
should not share a host directory give one of them a different `project`
name. A host holds one version at a time: a run that needs a different
version while an earlier run is still using that host is refused, with the
earlier run named, and continues on its other hosts and this machine.
Stop the earlier run, or wait for it. A run that needs the same version
proceeds.

Where work goes is a *mode*, the `mode` line of `valuekit.local.toml`:

| mode | this machine | hosts |
|---|---|---|
| `all` (default) | full capacity | full capacity |
| `local` | everything | nothing |
| `remote` | nothing, unless no host is reachable | everything |

The default is `all`: a host named in the file is used unless the mode
excludes it. Edit the line, switch it from the monitor (below), or run
`python -m valuekit.monitor --mode remote <cache-dir>` from inside the
project. The main process re-reads the file each time it starts a task, so a switch during a batch applies to the next
task; tasks already running finish where they are. Syncing a host never
delays the batch: this machine starts at once and a host starts taking
tasks when it is ready (under `remote`, this machine waits for it instead). A host that
cannot be reached or synced is dropped with the reason recorded once,
and the batch continues elsewhere. When a host's connection drops mid-batch
no input is lost: the inputs that were running there run again elsewhere,
once, and the host takes no more.

`run_all` takes no argument about any of this. Where a call ran must
not be able to affect its result, so it cannot be named in code where a
function hash could reach it.

## Watching a run

A cache that works is silent, which makes it hard to tell from one that
doesn't: a step that ought to be hitting and isn't looks exactly
like a step that is slow. `python -m valuekit.monitor` shows what is
actually happening, from a separate process:

```
$ python -m valuekit.monitor ~/.cache/mypipeline

runs: 1 live, 9 workers, 0 finished
mode: all          l local  r remote  a all  q quit
  new tasks go to mac, here

  pid 97702    process_scenarios.py         up 2.7s

hosts
  host              capacity  running   done  failed  state
  mac                      8        6      3       0  ready
  local                    4        3      2       1

batches
  nightly                  [################........] 8/12  2.7s  1 failed

this run
  function                        hits  misses    rate  forced  errors      time
  calculate_geometry                11       0    100%       0       0        1ms
  detrend                            0       9      0%       0       0       3.7s

failures
      1.9s ago  process[5]                               RuntimeError
```

The hit rate is the number to look at. Everything else is context for it.

Start it whenever you like, including twenty minutes into a long run — it
reads what has been recorded so far rather than needing to have been
watching from the beginning. Watching has no effect on the run. The one
thing the monitor writes is the `mode` line of the project's
`valuekit.local.toml`, on a keystroke: `l`, `r` and `a` set `local`,
`remote` and `all`. The project is the one enclosing the directory the
monitor is run from. The header shows the mode and, while a batch runs,
what it means for the next task given which hosts are ready: under
`remote` with the host still syncing, "waiting for mac to sync; none
start here". Tasks already running finish where they are. The `hosts`
block shows each host's state (syncing, ready, or dropped with the
reason), its capacity under the mode, and what is running and finished
there.

The event log goes in `events/` inside the store directory, one file per process, and
nothing is recorded until `set_store_dir` has been called — the same rule as
everything else here. That does mean a `run_all` batch with no cache
directory is not observable. The monitor takes the store directory as an
argument, falling back to `$VALUEKIT_STORE`.

There is nothing to switch on and no way to get it wrong: writing the log never
fails a run, an unwritable directory just disables it, old run files are
pruned, and a run that produces a huge number of records stops recording
detail rather than filling a disk. It costs about 3 µs per `@pure` call,
a small fraction of a cache hit.

## Install

```
pip install valuekit        # Python >= 3.11; depends only on numpy
```

## Development

```
uv sync                    # the environment, from uv.lock, into .venv
uv run pytest
```

The repository is a locked project, like the ones it runs: `uv.lock` pins
what the suite runs against, and the host tests build a test project's
environment with `uv`, so it must be on the PATH. Without uv,
`pip install -e ".[dev]"` (quoted: zsh globs the brackets) and `pytest`
also work, but the host tests then skip.
