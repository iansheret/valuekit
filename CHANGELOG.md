# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0

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
