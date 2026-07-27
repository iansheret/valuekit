# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

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
