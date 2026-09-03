# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Caches written by earlier versions are refused; delete the directory. The
trace layout changed (each trace is now its own file) and a trace now
records more than it did.

### Added

- `log(name, value)` binds a name to a value inside a `@pure` function. The
  value is stored like a result and the binding is recorded in the call's
  trace, so it is found again on every later hit without the body running.
  A memoised call made inside another is recorded the same way, under its
  function name, with no call needed.

- `run_all` records each batch under a name (default: the function's
  qualified name; `name=` to choose) and `valuekit.batch(name)` reads it
  back: rows by input, values by name (`b[7]["detrend"]`, whether `detrend`
  was a nested step or a `log` name, at any depth), a column across inputs,
  rows grouped by a logged parameter, failures with their messages. Nothing
  is imported or run to read a batch, and a batch can be read while it is
  still running. Every input of a batch ran under one fingerprint, which the
  record carries, so a batch cannot mix results from two versions of the
  code.

- `@pure_local`: memoised exactly like `@pure`, on a different promise. The
  result may depend on something outside the program -- a file on this
  machine, a database, a download that needs this machine's credentials --
  which the user promises does not change for the same arguments. It runs
  only on the machine driving the pipeline; a worker elsewhere sends the
  call back. This is what lets a batch that fetches data run remotely
  without credentials leaving the driver.

- `fn.cached(...)` returns a `@pure` function's stored result without
  executing, or raises `CacheMiss`. `run_all` uses the same lookup to answer
  an already-cached input without starting a worker.

- `python -m valuekit.sweep <module>...` deletes what the current code can no
  longer reach: traces and batches of functions whose fingerprint no
  importable function produces, then objects no remaining trace or batch
  names. Retention is by code version, not by age; nothing is removed for
  being old.

- A running pipeline can be watched from a separate process.
  `python -m valuekit.monitor <cache-dir>` shows, live, the hit rate per
  function, batch progress, and failures. It attaches whenever you start it,
  including part-way through a long run, and only ever reads. The run log is
  written to `runs/` inside the cache directory; a batch run with no cache
  directory is therefore not observable. Writing it never fails a run.

### Changed

- `run_all` requires a `@pure` or `@pure_local` function and raises
  `TypeError` otherwise. A batch's results are recorded by the function that
  produced them, an already-cached input needs no worker, and a function
  whose effects do not matter is the only kind that can safely run
  elsewhere.

- Each trace is its own file, named by the hash of its content, under
  `traces/<fnkey>/`. Two processes writing the same trace write the same
  bytes under the same name, so there are no appends and nothing to
  coordinate. This fixes lost traces on Windows, where an append is not
  atomic, and makes a hit cost one `stat` instead of a re-read of the
  function's whole trace file.

- A trace records the memoised calls made inside it and its `log` bindings,
  in order. Matching is unchanged.

- The Windows CI job is gating. Besides the appends, three defects were
  fixed there: replacing an object file another process has memory-mapped
  is treated as the completed write it is; polling a killed worker's pipe
  reports death instead of raising; and the pipe backend waits on reader
  threads rather than `select`, which Windows refuses for pipes and which
  made a batch spin forever.

- `valuekit` itself is never classified as user code, wherever it is
  installed from, so a function naming `log` or `ImmutableMap` does not hash
  the library's module state.

### Remote execution (not yet reachable by users)

Groundwork for running a batch somewhere other than this machine. `run_all`
keeps its signature apart from `name=`, and every batch still runs in local
processes unless a private hook selects the pipe backend.

- Scheduling is separated from where work runs. `run_all` keeps the admission
  limit, deadlines, input ordering and failure attribution; a backend starts,
  waits for and kills a unit of work.
- The value codec is free functions parameterised by how a child value is
  reached, so the same pickle-free format serves a directory on disk and a
  connection to a peer. A peer's object frames carry the same bytes the
  store writes, so they are stored without being decoded.
- A worker that speaks a framed protocol over a pipe, with a handshake that
  recomputes the function's fingerprint and refuses if the code it would run
  is not the code the driver meant. It runs on this machine, which is the
  point: everything is exercised in CI with no network involved.
- Code sync. The driver describes its project as a manifest -- tracked files
  plus untracked ones that are not ignored -- and the worker unpacks an
  immutable source tree named by the manifest hash and imports from that.
  Build artefacts are never shipped, whatever platform names them.
- A readiness phase, once per host rather than once per input, and an audit
  after importing that every user module actually came from the source tree.
- A worker holds no cache. Its store is the driver's store, reached over the
  connection: every value, trace and run-log record it produces goes to the
  driver, every lookup asks the driver, and a `@pure_local` call runs on the
  driver. A batch's results exist in one place however many machines ran it.
- A worker's environment is an allowlist of what a process needs to start,
  plus `VALUEKIT_*`. Nothing else of the driver's crosses.

## 0.3.1 — 2026-08-27

### Fixed

- Editing a submodule reached through attribute access now recomputes what
  depends on it. `mypkg.sub.f()` spells `sub` and `f` as attribute names, which
  resolve to nothing at module scope, and a package's source file is only its
  `__init__.py` — so the walk stopped there and nothing sub.py said reached the
  fingerprint. Editing sub.py left the fingerprint unchanged and `@pure` served
  the old result without executing. The same shape defeated `clear_cache(fn)`,
  since callers never recorded the submodule's unit. Submodules named by the
  referencing function are now followed, at any depth, each classified in its
  own right so that a compiled extension inside a package is still identified
  by its binary rather than read as source. Reaching a submodule as a name
  (`from mypkg.sub import f`) was always tracked and is unchanged, as is a flat
  module referenced as a module.

  `@pure` functions using that import style will recompute once. Results they
  cached before this release may have been computed from code that has since
  changed — if you have relied on this style, clearing the cache directory is
  the cautious move, though results are not silently reused: the fingerprint
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
