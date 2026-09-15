# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Caches written by earlier versions are refused; delete the directory. The
call-record layout changed (each call record is its own file, under
`records/`) and a call record now holds more than it did.

Three API changes. `set_cache_dir` is `set_store_dir`. `clear_cache(fn)`
is gone: edit the function, or put a version in its arguments. `run_all`
returns the list of results and raises `BatchError` at the first input
that produces no result, instead of returning a `BatchResult` of
per-input outcomes; a function that expects bad inputs returns a value
that says so.

### Added

- `log(labels, value)` records a value under a small mapping saying what
  it is, and `valuekit.logs(script)` reads it back by containment:
  `logs("physics").where(quantity="residuals", sid=7).one().value`. No key
  means anything to valuekit; labels are the project's own terms. A
  script's log is the complete set of logged values its last run produced,
  as if the code had run from scratch: a step that executes writes each
  logged value as it makes it, and a step served from cache writes one
  line naming its call record, which holds what the step logged, nested
  calls included, so a re-run after an edit shows the current code's
  logged values and nothing stale. The log is as current as the cache:
  whatever removes call records removes the logged values that came with
  them, and `logs()` reports a line it can no longer resolve. Each
  script's log replaces its previous run's and never touches another
  script's. Every emission is its own logged value; order carries no
  meaning. This is the boundary between processing code and the plotting
  code that reads it, which imports nothing.

- `@pure_local`: memoised exactly like `@pure`, on a different promise. The
  result may depend on something outside the program -- a file on this
  machine, a database, a download that needs this machine's credentials --
  which the user promises does not change for the same arguments. It runs
  only on the main process; a worker elsewhere sends the
  call back. This is what lets a batch that fetches data run remotely
  without credentials leaving the main process.

- `run_all` serves an already-cached input without starting a worker.

- A running pipeline can be watched from a separate process.
  `python -m valuekit.monitor <cache-dir>` shows, live, the hit rate per
  function, batch progress, and failures. It attaches whenever you start it,
  including part-way through a long run, and only ever reads. The event log is
  written to `events/` inside the cache directory; a batch run with no cache
  directory is therefore not observable. Writing it never fails a run.

### Changed

- Nothing depends on a file's modification time any more. The store lists
  a function's call-record directory on every lookup instead of caching
  the listing on the directory's mtime, so a record another process wrote
  within the same clock tick is seen at once. A native extension's binary
  is hashed once per process, as the binary the process loaded, instead
  of being re-read when its size or mtime changed: an extension module
  cannot be reloaded, so a rebuild is seen by the next process.

- `set_cache_dir` is `set_store_dir`, and `cache_dir=` on `logs()` is
  `store_dir=`. The directory holds everything the project's code
  produced: values, call records and run logs.

- The project's build is the project's lock tool's. A host installs the
  project with `uv sync --frozen` and, when a sync sends changed files that
  include a build input, rebuilds it with `uv sync --frozen
  --reinstall-package <name>`. `valuekit.build()`, called at the top of a
  script before the project is imported, does the same on this machine.
  Build inputs are the project's `[tool.valuekit] build-inputs` globs,
  else every file that is not a Python source; `pyproject.toml` and the
  lock file always count. Rebuild-on-import is no longer the mechanism
  and is not supported with parallel workers: many processes importing at
  once run the build tool at once in one build directory, which failed
  under MSBuild. A local worker receives the function by name and imports
  it itself, as a host's worker does.

- A run a debugger forces (a live breakpoint, or `VALUEKIT_ALWAYS_RUN`)
  still writes its logged values to the run's log; it writes no call
  record, as before.

- `run_all`'s `max_workers` is how many tasks run at once on this
  machine, whatever the CPU count, and overrides the local file's
  `[local] workers`. When no host may run anything (`max_workers=0` and
  no usable remote host) the inputs run one at a time in the main
  process rather than the batch raising.

- A native extension's marker in the function hash is the hash of the
  main process's build of it, per extension, sent to workers rather than
  computed by them. Editing a file the extension's build does not read no
  longer re-keys the functions that reach it; the project hash keeps its
  other job of saying whether a host's copy of the project is current.

- Remote configuration is one file per checkout, `valuekit.local.toml`
  beside `pyproject.toml`: the hosts, this machine's worker cap, the mode
  (`all`, `local` or `remote`) and the project's directory name on each
  host. It replaces the `VALUEKIT_HOSTS` environment variable and the
  `placement` file in the cache directory. It is per checkout, so it
  belongs in `.gitignore` (valuekit warns if git tracks it); it is never
  sent to a host and never hashed. A checkout that runs on this machine
  only needs no file. The monitor's keys and `--mode` edit the file's
  `mode` line and find the project from the current directory. A file
  that cannot be read while a batch runs leaves the mode last read in
  force, and the monitor shows the error in place of the mode.

- A host keeps one directory per project, updated in place. A run sends
  only the files whose content changed and the names of those removed,
  and touches nothing else in the directory, so a native extension's
  build directory persists and it rebuilds incrementally; before, each
  edit made a new directory named by the project hash and a from-scratch
  build. A manifest beside the directory names the project hash it holds, and
  a lock file serialises updates. A host holds one version at a time: a
  run that wants a different version while an earlier run still uses the
  host is refused, naming that run; stop it or wait.

- Names say what things are. The *function hash* (was fingerprint, with a
  salted key on top) names a function's *call records* (were traces), kept
  under `records/`. `logs()` reads the *run's log* (was ledger); the
  monitor reads the *event log* (was run log), under `events/`. A *logged
  value* carries *labels* (were item and context). A *run* (was execution)
  is one main process process. Work runs on a *host*, this machine included (was place or
  backend); a *connection* carries *messages* (were link, wire and frame); a
  function's *reachable set* (was closure) is what the function hash covers; a
  *project hash* (was tree id) identifies a version of the project's files. Modules follow:
  `runlog.py` holds the run's log, `events.py` the event log,
  `protocol.py` the messages, `hosts.py` the hosts,
  `functionhash.py` the function hash, `project.py` the project's files
  (was `sync.py`). A host *syncs* when it is brought to where it can run
  this version of the project; the word means nothing else.

- One function hash and one version number. The Python version is a marker
  inside the function hash rather than a salt applied on top, and the store's
  format version is the only version: the cache epoch is gone.

- `clear_cache(fn)` is gone; `clear_cache()` deletes everything computed
  or logged. To invalidate one function, edit it, or put a version in its
  arguments: either gives it and every function that reaches it a new
  function hash. The targeted form was the only way a live call record
  could name a nested record that no longer existed, and so the only
  reason a hit read its whole subtree; a hit now loads its result and
  nothing else. The per-function dependency index is gone with it.

- `run_all` with an undecorated function runs on this machine only, with
  nothing cached, as before. A decorated function's already-cached inputs
  need no worker, and only a decorated function may run on another
  machine.

- This machine is a host like any other: a host process started here
  runs one worker per input, and a worker reports a failure as the
  exception's type name, message and traceback text, the same from any
  machine. A worker on this machine reads and writes values and call
  records in the store directory itself and sends everything else to the
  main process, so there is one event file and one run per main process.
  An event's `t` is the time the main process wrote it, so one file holds
  one clock's times whichever machine the event came from. The
  `multiprocessing` worker path, the pickled exceptions and the
  worker-role detection in the event log are gone.

- Each call record is its own file, named by the hash of its content, under
  `records/<function hash>/`. Two processes writing the same record write the same
  bytes under the same name, so there are no appends and nothing to
  coordinate. This fixes lost call records on Windows, where an append is not
  atomic, and makes a hit cost one `stat` instead of a re-read of the
  function's whole call-record file.

- A call record holds the memoised calls made inside it and its logged
  values, in order. Matching is unchanged.

- The Windows CI job is gating. Besides the appends, three defects were
  fixed there: replacing an object file another process has memory-mapped
  is treated as the completed write it is; polling a killed worker's pipe
  reports death instead of raising; and the host connection waits on reader
  threads rather than `select`, which Windows refuses for pipes and which
  made a batch spin forever.

- The repository is a locked project: `uv.lock` pins the development
  environment and CI syncs from it. The suite's test project pins numpy to
  the main process's version and Python to the main process's minor, since a package
  version is part of the function hash of every function using it.

- `valuekit` itself is never classified as user code, wherever it is
  installed from, so a function naming `log` or `ImmutableMap` does not hash
  the library's module state.

### Running on other machines

- A batch can run on hosts declared in the project's `valuekit.local.toml`,
  over ssh, as if this machine had more cores. A host needs what a person
  would need to check the project out and run it: a Python 3 to start with,
  the project's lock tool, a compiler for an extension, and the network.
  Nothing of the project's, valuekit included, is installed there first.
- The project must be locked: its tree carries a lock file from a tool
  valuekit can invoke (`uv.lock` today; the table has one row per tool). A
  small stdlib-only bootstrap, sent over the connection, receives the tree,
  runs the tool's sync in it (`uv sync --frozen`, for the main process's Python
  minor) and starts the host process from the environment that produced,
  activated: the tools the lock installed beside the interpreter (cmake and
  ninja for an extension that rebuilds on import, say) are on the workers'
  PATH. A tree with no known lock is refused before anything is sent.
- One connection per host carries every task: a host process
  (`python -m valuekit.hostprocess`) starts a worker per task and multiplexes their
  streams, so a thousand inputs cost one ssh handshake. The host process
  exits, killing its workers, when the connection closes.
- A native extension's marker in the function hash is the hash of the main
  process's build of it, sent to workers (see the entry under Changed).

### Remote execution groundwork

- Scheduling is separated from where work runs. `run_all` keeps the admission
  limit, deadlines, input ordering and failure attribution; a host starts
  and kills a task and reports on it.
- The value codec is free functions parameterised by how a child value is
  reached, so the same pickle-free format serves a directory on disk and a
  connection to a peer. A peer's object messages carry the same bytes the
  store writes, so they are stored without being decoded.
- A worker that speaks a message protocol over a pipe, with a handshake that
  recomputes the function's function hash and refuses if the code it would run
  is not the code the main process meant. It runs on this machine, which is the
  point: everything is exercised in CI with no network involved.
- Code sync. The main process describes its project as a manifest -- tracked files
  plus untracked ones that are not ignored -- and the host unpacks an
  immutable source tree named by the manifest hash, which workers import
  from.  Build artefacts are never shipped, whatever platform names them.
- A sync, once per host rather than once per input, and an checks
  after importing that every user module actually came from the source tree.
- A worker holds no cache. Its store is the main process's store, reached over the
  connection: every value, call record and event it produces goes to the
  main process, every lookup asks the main process, and a `@pure_local` call runs on the
  main process. A batch's results exist in one host however many machines ran it.
- A worker's environment is an allowlist of what a process needs to start,
  plus `VALUEKIT_*`. Nothing else of the main process's crosses.

## 0.3.1 — 2026-08-27

### Fixed

- Editing a submodule reached through attribute access now recomputes what
  depends on it. `mypkg.sub.f()` spells `sub` and `f` as attribute names, which
  resolve to nothing at module scope, and a package's source file is only its
  `__init__.py` — so the walk stopped there and nothing sub.py said reached the
  function hash. Editing sub.py left the function hash unchanged and `@pure` served
  the old result without executing. The same shape defeated `clear_cache(fn)`,
  since callers never recorded the submodule. Submodules named by the
  referencing function are now followed, at any depth, each classified in its
  own right so that a compiled extension inside a package is still identified
  by its binary rather than read as source. Reaching a submodule as a name
  (`from mypkg.sub import f`) was always tracked and is unchanged, as is a flat
  module referenced as a module.

  `@pure` functions using that import style will recompute once. Results they
  cached before this release may have been computed from code that has since
  changed — if you have relied on this style, clearing the cache directory is
  the cautious move, though results are not silently reused: the function hash
  now differs, so the affected entries are simply never consulted again.

## 0.3.0 — 2026-08-06

### Added

- Plain-data dataclasses are `@pure` arguments and cached return values
  without registration. A dataclass's identity is its qualified name, its
  dataclass parameters and its ordered fields, so a change to any of them
  recomputes. "Plain data" is checked rather than assumed: the class must be
  built by assignment alone, hold exactly its declared fields, and carry no
  methods, properties, or static and class methods, since a method reached
  through an argument is invisible to the calling function's fingerprint.
  Anything else raises and points at `register_type`. Stored entries name
  their class but never import it, and a class that has changed since an
  entry was written reads as a miss.

### Fixed

- Rebuilding a native extension you maintain yourself now recomputes what
  depends on it. Its binary is hashed whenever the distribution providing it
  was installed from a local directory, editable or not, since such a package
  is rebuilt in place under an unchanged version; released wheels keep their
  version markers and are never read. Objects a compiled module defines are
  also recognised now: a nanobind function is neither a Python function nor a
  builtin, so it previously reached the fingerprint as an untracked opaque
  value and nothing about the extension was tracked at all.

## 0.2.0 — 2026-07-28

### Changed

- `@pure` no longer converts anything. Arguments are hashed and passed through
  as the objects the caller gave, and the result is the object the function
  built, so a cache hit is indistinguishable from a miss apart from the skipped
  execution. Previously arguments were frozen on entry (dicts became
  `ImmutableMap`s, sets became frozensets, writeable arrays were copied
  read-only, lists were rejected outright) and the return value was frozen too
  — and none of it happened at all when no cache directory was configured, so
  the decorator's contract changed shape with the configuration.
- Per-key invalidation is now opt-in, and passing an `ImmutableMap` is the
  opt-in. Every other argument, plain `dict` included, is depended on whole.
- The recording proxy is an `ImmutableMap` subclass and cannot escape the call
  that made it, so a function cannot tell a recorded map from a plain one.
- A map passed from one `@pure` call into a nested one is now traced in both,
  replacing the whole-map dependency previously recorded at that boundary.
- A content hash now identifies a value exactly: lists and tuples of the same
  items differ, dicts differ if their order differs, and writeable arrays
  differ from read-only ones. This is what lets the content-addressed store
  return the type it was given.
- `register_type` requires only `hash_fn`; `freeze_fn` is needed just for types
  that go into an `ImmutableMap`.

### Added

- Lists, sets and dicts may be `@pure` arguments and cached return values,
  and round-trip as themselves.

### Fixed

- Argument hashes are computed once per call rather than once per candidate
  trace, and before the function runs, so a trace records the arguments as they
  were passed.

Caches from 0.1.0 are refused rather than misread; delete the cache directory.

## 0.1.0 — 2026-07-28

First public release.

- `ImmutableMap`: an immutable mapping for pipeline data. Values are frozen on
  entry, derivation with `|` / `assoc` / `dissoc` shares unchanged values by
  reference, and unknown mutable types are rejected.
- `@pure`: disk memoisation for pure functions, keyed on a recursive content
  hash of everything reachable by name from the function, and invalidated per
  read rather than per argument.
- `register_type`: extend the frozen/hashable type set, optionally with a store
  codec so custom types can appear in cached return values.
- `run_all`: parallel batch execution with per-input isolation, timeouts that
  kill, and every failure recorded against the input that caused it.
- Debugger integration: a live breakpoint anywhere in a `@pure` function's
  dependency closure forces execution, and forced runs never write to the store.
