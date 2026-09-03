# Remote execution: state and outstanding work

This records the state of the remote-execution work so it can be resumed after a break.
It covers what is built, what was decided and why, and what is left. Decisions are
recorded with their reasons so they need not be reargued.

## Repository state

| Branch | Contents |
|---|---|
| `main` | Released as `v0.3.1`, tagged and published to PyPI. |
| `remote-seam` | Everything below, unreleased. Branched from `main` via the run-log work. |

**Nothing after 0.3.1 is released, and nothing will be until SSH execution works.** The
CHANGELOG's "Unreleased" entry accumulates the run log and monitor, the per-file trace
layout, `log`, batch records, `@pure_local`, the sweep, and the remote groundwork.
`_version.py` still reads `0.4.0` and is not to be bumped before the release that
includes SSH.

The suite passes on Windows (the machine this was developed on) and on Linux and macOS in
CI. The Windows job is gating.

## What works today

`run_all` can execute a batch in worker subprocesses that hold no cache and import the
driver's code from a synced copy. The sequence is:

1. The driver builds a manifest of its project, hashes it, and starts one readiness
   process for the host.
2. That process reports whether it already holds the source tree for that manifest hash;
   if not, the driver sends the tree and the process unpacks it under the source root the
   driver named.
3. The process puts the source tree on `sys.path`, imports the target function, checks
   that every user module actually came from the source tree, recomputes the function's
   fingerprint, and compares it with the driver's. It then exits.
4. Each input runs in its own worker process against the finished source tree. The
   worker's store is a `WireStore`: every value, trace and run-log record it produces is
   sent to the driver, every lookup asks the driver, and a `@pure_local` call is sent to
   the driver to run there. The driver writes received objects into its store as they
   arrive (a wire object's bytes are the store's bytes for that value).
5. The driver records each input's root trace in the batch record (`batches/<name>/`),
   and answers an input whose result it already holds without starting a worker.

None of it is reachable by a user. The backend is selected by a private module-level hook
(`parallel._backend_factory`) that defaults to local processes.

## Module map

| File | Responsibility |
|---|---|
| `valuekit/parallel.py` | Scheduling: admission, deadlines, input ordering, failure attribution, cached-input short-circuit, batch recording. |
| `valuekit/backend.py` | Where work runs. `Backend`/`Handle` protocols, `LocalBackend`, `PipeBackend`. The pipe handle answers a worker's store requests and runs its `@pure_local` calls on a thread pool. |
| `valuekit/remotestore.py` | `WireStore`: the worker's side of the store, over the connection. |
| `valuekit/wire.py` | Message framing and value transfer as content-addressed object graphs. |
| `valuekit/codec.py` | The structural value format, parameterised by how a child value is reached. Shared by the store and the wire. |
| `valuekit/worker.py` | The worker process: a readiness mode and a single-task mode. |
| `valuekit/sync.py` | Manifest, source-tree packing, import roots, and the path predicate that separates project from environment. |
| `valuekit/batches.py` | Batch records: written by `run_all`, read by `valuekit.batch()`. |
| `valuekit/sweep.py` | Retention: delete what the current code cannot reach. |
| `valuekit/runlog.py` | Records hits, misses, forced runs, errors and batch progress; forwards a worker's records through its store. |
| `valuekit/monitor.py` | Reads the run log from another process. |

## Decisions taken

**`run_all` gains no placement parameter.** Where work runs is configuration, not a
call-site argument. A host list in code could be reached by a fingerprint, and where a
computation ran must not be able to affect its result. (`name=` is a parameter; a name
cannot reach a fingerprint or change a result.)

**`run_all` requires `@pure` or `@pure_local`.** A batch's results are recorded by the
function that produced them, an already-cached input needs no worker, and a function
whose effects do not matter is the only kind that can safely run elsewhere.

**Workers hold no cache.** With several hosts and no scheduling affinity, which host holds
a given result is arbitrary, so per-host caches fragment. The worker's store is the
driver's store over the connection. Consequence: the source tree lives at a root the
driver names in the greeting (the pipe backend passes `<cache_dir>/source`); the SSH
transport must choose a default on the remote machine.

**`@pure_local` runs on the driver.** A function that reads outside its arguments reads an
environment, and environments differ between machines. The driver has the credentials
and the files; a worker sends the call back as a request and receives a value. This is
the only mechanism for acquisition; there is no credential forwarding.

**Verify code identity; do not ship binaries.** Two machines cannot produce identical
compiled output, so a native extension must be built on the machine that runs it.

**Cross-architecture execution is permitted, with no architecture tag in the cache key.**
Whether results from two architectures are equivalent is the user's judgement, and `@pure`
already requires that judgement.

**No prepare hook.** The contract is that importing the package must be sufficient to make
the worker correct. The documented remedy is a build backend that rebuilds on import:
scikit-build-core with `editable.rebuild` set, or meson-python.

**`fn` must be in an importable module, never `__main__`.** Refused during the handshake.

**Sync ships the user-code partition only.** `git ls-files --cached --others
--exclude-standard`. Compiled artefacts are excluded by a fixed cross-platform suffix
list. Submodules are out of scope. Dependencies are the environment's responsibility on
both machines.

**Whole-tree transfer over the existing channel, not rsync.** Testable in CI with no
network.

**Source trees are immutable and named by their manifest hash.**

**Three independent checks, in this order.** Source tree on `sys.path`; post-import audit;
fingerprint handshake.

**Readiness is a separate process, once per host.**

**File digests are not memoised on `(mtime, size)`.** Hashing the whole tree costs 5 ms
for this repository.

**The worker environment is an allowlist.** What a process needs to start (`PATH`, the
Windows system variables, temp and home directories, locale) plus `VALUEKIT_*`. No
`PYTHONPATH`, no credentials.

**The run log lives in the cache directory**, under `runs/`. A worker's records arrive as
`EVENT` frames and are written into the driver's own run file; a worker opens no file.

**Terminology.** "Source tree" is the project's files on a worker. "Run log" is the record
of what happened during a run. "Trace" is a memoised call's recorded reads, result,
nested calls and log bindings. "Batch record" is what `run_all` writes under a name.

## Outstanding work

### Blocks remote execution

1. **Split the fingerprint for locally-built native extensions.** `_classify` hashes the
   built binary for any extension whose distribution came from a local directory. Two
   machines cannot produce identical binaries, and the source tree cannot carry the
   driver's. The binary hash must stay in the local cache key and be omitted from the
   cross-machine handshake. Until this is done, any project with an editable-installed
   native extension refuses every remote host.

### SSH

2. **Host configuration.** TOML, located by an environment variable, with no default
   location. Includes the remote source root.

3. **Multiple hosts with per-host capacity.** Admission is a single scalar and there is no
   placement. What `max_workers` means across hosts; replacing the hardcoded
   `host="local"` in `parallel.py`.

4. **Per-host readiness with partial-failure tolerance.** Drop a failing host and continue
   when local capacity remains; raise when it does not. Promote `ensure_ready` into the
   `Backend` protocol.

5. **The SSH transport.** `-T`, `BatchMode=yes`, `ControlMaster`/`ControlPersist` with a
   short `%C` socket path, exit code 255 disambiguation, `exec` in the remote command. The
   pipe handle already reads stdout on a thread and can drain stderr the same way.

6. **Per-host store traffic.** A `@pure_local` call from a remote worker runs on the driver
   thread pool; a download-heavy batch may want a cap on concurrent calls. Not needed for
   the pipe backend.

### Correctness

7. **`_salt()` covers only `major.minor`.** Adding `micro` converts a silent risk into an
   explicit refusal, at the cost of a `CACHE_EPOCH` bump.

8. **`_classify` identifies installed packages by the substring `"site-packages"`.**
   `sync.is_environment` already derives this from `sysconfig` and `site`.

9. **Verify a reported instability in `_unit_digest`** (a frozenset constant's `repr`
   varying with `PYTHONHASHSEED`). The evidence offered did not support the claim.

### Documentation

10. The remote contract: importing the package must be sufficient; a data file beside the
    project locally may not exist on a worker, and `@pure_local` is the remedy.

### Release

11. Bump `_version.py`, move the "Unreleased" changelog entry under the version, release.
    Only after SSH works end to end.

## Open questions

**A content-addressed value that can produce a file path on demand.** Some libraries accept
only filenames. A `@pure_local` function returning `bytes` covers acquisition; a raw
object kind whose store path could be handed out on any machine would cover the library
case. Undecided.

## Checking that it still works

```
.venv/Scripts/python.exe -m pytest -ra          # Windows; python -m pytest elsewhere
```

To exercise the remote path manually, select the pipe backend through the private hook and
run a batch:

```python
import valuekit as vk
from valuekit import parallel
from valuekit.backend import PipeBackend

vk.set_cache_dir(cache)
parallel._backend_factory = PipeBackend
vk.run_all(mymodule.work, [1, 2, 3])
vk.batch("work")[1]
```

The cache directory then contains `format`, `objects`, `traces`, `batches`, `runs` and
`source`. `python -m valuekit.monitor <cache-dir>` reads the run log; the workers'
hits and misses appear in the driver's own run file.
