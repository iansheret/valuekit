# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Groundwork for running a batch somewhere other than this machine. Nothing here
is user-visible: `run_all` keeps its signature and its behaviour, and every
batch still runs in local processes.

- Scheduling is separated from where work runs. `run_all` keeps the admission
  limit, deadlines, input ordering and failure attribution; a backend starts,
  waits for and kills a unit of work. Waiting is a backend method because
  readiness is not portable — on Windows `multiprocessing.connection.wait`
  needs a real Win32 handle, so a subprocess pipe is not interchangeable there.
- The value codec is now free functions parameterised by how a child value is
  reached, so the same pickle-free format serves a directory on disk and a
  connection to a peer. Content-addressing then deduplicates on the wire for
  the reason it deduplicates on disk.
- A worker that speaks a framed protocol over a pipe, with a handshake that
  recomputes the function's fingerprint and refuses if the code it would run is
  not the code the driver meant. It runs on this machine, which is the point:
  the framing, codec, handshake and failure mapping are all exercised in CI
  with no network involved.
- Code sync. The driver describes its project as a manifest — tracked files
  plus untracked ones that are not ignored, since the helper you just wrote and
  have not `git add`ed is the commonest thing to be editing — and the worker
  unpacks an immutable source tree named by the manifest hash and imports
  from that. Build artefacts are never shipped, and neither is anything outside
  the project: a dependency is the environment's job on both machines.
- A readiness phase, once per host rather than once per input. Syncing,
  importing and verifying happen before any input is claimed, so a sync
  failure, a missing dependency or a compile error is reported as one fact
  about the host instead of as an identical failure against every input.
- After importing, the worker checks that every user module actually came from
  the source tree. A path entry is not proof: an editable install's meta-path
  finder runs first and can silently win. That check is what separates running
  the driver's code from running whatever the worker happened to have.

## 0.4.0 — 2026-08-28

### Added

- A running pipeline can now be watched from a separate process.
  `python -m valuekit.monitor <cache-dir>` shows, live, the hit rate per
  function, batch progress, and failures. It attaches whenever you start it,
  including part-way through a long run, and only ever reads: nothing it does
  can affect the run it is watching.

  The hit rate is the point. It is what the library promises and the one thing
  that was previously invisible — a step that ought to be hitting and silently
  is not looks exactly like a slow step.

  The run log is written to `runs/` inside the configured cache directory, one file
  per process, and nothing is written until `set_cache_dir` is called: the rule
  is unchanged, the cache directory is where valuekit writes. A batch run with
  no cache directory is therefore not observable. Old run files are pruned, and
  a file that reaches its size cap stops recording detail and counts what it
  dropped rather than filling a disk.

  Writing the log never fails a run: an unwritable directory or a full disk stops it
  for that process and changes nothing else. It costs roughly 3 µs per `@pure`
  call, under a tenth of the cost of a cache hit, which is dominated by reading
  and parsing the function's trace file.

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

- Lists, sets and dicts may be `@pure` arguments and cached return values, and
  round-trip as themselves.

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
