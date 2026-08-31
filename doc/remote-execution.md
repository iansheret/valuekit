# Remote execution: state and outstanding work

This records the state of the remote-execution work so it can be resumed after a break.
It covers what is built, what was decided and why, and what is left. Decisions are
recorded with their reasons so they need not be reargued.

## Repository state

| Branch | Contents |
|---|---|
| `main` | Released as `v0.3.1`, tagged and published to PyPI. |
| `run-events` | 0.4.0: the run log and the monitor. One commit, branched from the submodule fix. |
| `remote-seam` | The backend seam, the pipe worker, code sync, the file-digest fix, the terminology rename, and this document. Branched from `run-events`. |

`v0.3.1` is on PyPI, verified by installing it into a clean environment and confirming
the submodule fingerprint fix behaves correctly there. Tags are `v0.1.0`, `v0.2.0`,
`v0.3.0`, `v0.3.1`.

**0.4.0 is not released.** `valuekit/_version.py` on this branch reads `0.4.0` and the
CHANGELOG dates it, but the run log and the monitor are still in the unmerged stack.
Whether 0.4.0 ships on its own before the remote work resumes is undecided. Earlier
releases used a release candidate on TestPyPI first; 0.3.1 went straight to PyPI after a
local `build` and `twine check --strict`.

`main` also carries one commit the stack does not: a correction to the 0.3.1 changelog
date, made as a separate commit rather than an amend so that the stack stays based on an
ancestor of `main` and will merge without conflict.

The suite is 213 tests and passes on `remote-seam`. Note that `main` has 162, because the
run log and everything above it are not there.

## What works today

`run_all` can execute a batch in worker subprocesses that import the driver's code from a
synced copy rather than from the driver's own filesystem. The sequence is:

1. The driver builds a manifest of its project, hashes it, and starts one readiness
   process for the host.
2. That process reports whether it already holds the source tree for that manifest hash;
   if not, the driver sends the tree and the process unpacks it.
3. The process puts the source tree on `sys.path`, imports the target function, checks
   that every user module actually came from the source tree, recomputes the function's
   fingerprint, and compares it with the driver's. It then exits.
4. Each input runs in its own worker process against the finished source tree, exchanging
   values in a pickle-free content-addressed format.

This is verified by deleting the driver's source file after readiness completes and
confirming that tasks still return correct results.

None of it is reachable by a user. The backend is selected by a private module-level hook
(`parallel._backend_factory`) that defaults to local processes, and `run_all`'s signature
and behaviour are unchanged.

## Module map

| File | Responsibility |
|---|---|
| `valuekit/parallel.py` | Scheduling: admission, deadlines, input ordering, failure attribution. |
| `valuekit/backend.py` | Where work runs. `Backend`/`Handle` protocols, `LocalBackend`, `PipeBackend`. |
| `valuekit/wire.py` | Message framing and value transfer as content-addressed object graphs. |
| `valuekit/codec.py` | The structural value format, parameterised by how a child value is reached. Shared by the store and the wire. |
| `valuekit/worker.py` | The worker process: a readiness mode and a single-task mode. |
| `valuekit/sync.py` | Manifest, source-tree packing, import roots, and the path predicate that separates project from environment. |
| `valuekit/runlog.py` | Records hits, misses, forced runs, errors and batch progress. |
| `valuekit/monitor.py` | Reads the run log from another process. |

## Decisions taken

**`run_all` gains no parameter.** Where work runs is configuration, not a call-site
argument. A host list in code could be reached by a fingerprint, and where a computation
ran must not be able to affect its result.

**Verify code identity; do not ship binaries.** Two machines cannot produce identical
compiled output, so a native extension must be built on the machine that runs it.

**Cross-architecture execution is permitted, with no architecture tag in the cache key.**
Whether results from two architectures are equivalent is the user's judgement, and `@pure`
already requires that judgement.

**No prepare hook.** The contract is that importing the package must be sufficient to make
the worker correct. A declared command would prove nothing, since any command satisfies
"a command was declared". The documented remedy is a build backend that rebuilds on
import: scikit-build-core with `editable.rebuild` set (it is not the default), or
meson-python.

**`fn` must be in an importable module, never `__main__`.** A worker cannot import a
driver script. Refused during the handshake with a message naming the fix.

**Sync ships the user-code partition only.** `git ls-files --cached --others
--exclude-standard`, so tracked files plus untracked files that are not ignored; a file
just created and not yet added is the common case while editing. Compiled artefacts are
excluded unconditionally rather than by trusting ignore rules. Submodules are out of
scope, and `git ls-files` cannot combine `--recurse-submodules` with `--others` in any
case. Dependencies are the environment's responsibility on both machines.

**Whole-tree transfer over the existing channel, not rsync.** This keeps the protocol
exercisable by the pipe backend with no network, which is what makes it testable in CI.
The boundary rule keeps trees small enough for this to be cheap.

**Source trees are immutable and named by their manifest hash.** Several versions coexist,
a running batch cannot have its source changed underneath it, and an unchanged tree
transfers nothing.

**Three independent checks, in this order.** The source tree determines `sys.path`; the
post-import audit confirms that every non-environment module in `sys.modules` actually
came from the source tree; the fingerprint handshake confirms the code matches the
driver's. The audit is required because a path entry does not guarantee that imports
resolved through it — a PEP 660 editable install places a finder on `sys.meta_path`, which
runs first.

**Readiness is a separate process, once per host.** A sync failure, a missing dependency
or a compile error is a property of the host. Running readiness inside the first task
would attribute such a failure to whichever input happened to run first, which conflicts
with the rule that every failure is recorded against the input that caused it.

**File digests are not memoised on `(mtime, size)`.** That key fails in the direction that
matters here: content changing without moving mtime or size (a same-size edit within one
tick on a coarse-mtime filesystem) would leave the manifest hash unchanged and let a
worker reuse an out-of-date source tree. Hashing the whole tree costs 5 ms for this
repository and 74 ms for 2000 files across 48 MiB.

**The run log lives in the cache directory**, under `runs/`, as does the source tree,
under `source/`. One rule applies to both: the cache directory is where valuekit writes,
and nothing is written until one is named.

**Terminology.** "Source tree" is the project's files on a worker. "Run log" is the record
of what happened during a run. "Trace" continues to mean a `@pure` call's recorded reads
and result, as it has since 0.1.0.

## Decided but not implemented

**Require `fn` to be `@pure`.** Raise `TypeError` at the top of `run_all`, with a message
in the style of the one `clear_cache` already raises for the same mistake. Reasons: an
impure function run on another machine writes its side effects to that machine, silently;
implicit purity would assert on the user's behalf a promise whose content is "the body may
not run"; and it would break the equivalence between `fn(x)` and `run_all(fn, [x])` that
the documented debugging advice depends on. It also removes the current difference where
the local path accepts anything picklable while the wire requires storable types.

The cost is that a batch whose purpose is per-input side effects can no longer use
`run_all`. Assessed against one real pipeline (a GNSS batch solver calling `run_all` over
session identifiers): the change there is about five lines, and two other scripts in that
repository already use the required shape.

**Check the driver's cache before dispatching.** With `fn` pure, an input whose result is
already held needs no worker at all.

**Record returned results and their traces in the driver's cache.** This makes re-running
after a partial failure inexpensive and makes the driver the single authority for
function-level results across hosts.

**Do not give workers a cache.** With several hosts and no scheduling affinity, which host
holds a given result is arbitrary, so per-host caches fragment. Consequence to resolve:
the source tree currently lives under the cache directory and would need a fixed
conventional path on the worker instead.

## Outstanding work

### Blocks remote execution

1. **Split the fingerprint for locally-built native extensions.** `_classify` hashes the
   built binary for any extension whose distribution came from a local directory. Two
   machines cannot produce identical binaries, and the source tree cannot carry the
   driver's (it is ignored by git, and would be the wrong architecture). The binary hash
   must stay in the local cache key, where a rebuild has to invalidate results, and be
   omitted from the cross-machine handshake. Until this is done, any project with an
   editable-installed native extension refuses every remote host.

2. **Replace the bulk environment copy with an explicit allowlist.** `PipeBackend` copies
   `os.environ` minus `PYTHONPATH`. That cannot cross an SSH boundary, and should not: it
   would carry the driver's `PATH`, `VIRTUAL_ENV` and any credentials to another machine.

### Step 4: SSH

3. **Host configuration.** TOML, located by an environment variable, with no default
   location.

4. **Multiple hosts with per-host capacity.** Admission is currently a single scalar
   (`len(running) < workers`) and there is no placement. Two sub-decisions: what
   `max_workers` means across hosts, and replacing the hardcoded `host="local"` at two
   sites in `parallel.py`.

5. **Per-host readiness with partial-failure tolerance.** Currently one blocking call that
   raises. It should drop a failing host and continue when local capacity remains, and
   raise when it does not. Consider promoting `ensure_ready` into the `Backend` protocol
   rather than discovering it with `getattr`.

6. **The SSH transport.** Required: `-T` (a pseudo-tty would corrupt binary frames),
   `BatchMode=yes`, `ControlMaster`/`ControlPersist` with a short `%C` socket path,
   stderr drained concurrently (the present `_why` reads stderr once at the end, which
   deadlocks when a 64 KiB pipe fills), disambiguation of exit code 255 (ssh's own
   failures) from a worker's, and `exec` in the remote command so a kill reaches the
   remote process rather than the local ssh client.

### Correctness

7. **`_salt()` covers only `major.minor`.** A 3.12.4 driver and a 3.12.7 worker pass the
   salt check and then depend on identical bytecode. `MAGIC_NUMBER` stability across patch
   releases is a guarantee about the eval loop, not about the compiler emitting the same
   bytes. Adding `micro` converts a silent risk into an explicit refusal, at the cost of a
   `CACHE_EPOCH` bump.

8. **`_classify` identifies installed packages by the substring `"site-packages"`.** This
   makes environment layout part of the fingerprint and misclassifies `pip --target`
   installs, vendored trees and Nix or conda prefixes. `sync.is_environment` already
   derives this from `sysconfig` and `site` and could be reused.

9. **Verify a reported instability in `_unit_digest`.** It was claimed that `repr` of a
   frozenset constant varies with `PYTHONHASHSEED`, making `clear_cache`'s dependency
   index inconsistent across processes. The evidence offered did not support the claim —
   the digests shown were identical. Worth a short check; if real, it is a defect
   independent of remote execution.

### Performance

10. **`get_traces` re-reads and re-parses a function's entire trace file on every `@pure`
    call**, and candidates are scanned oldest-first with no bound on how many accumulate.
    This dominates the roughly 40 µs cost of a cache hit. Two inexpensive improvements:
    scan newest-first, and avoid re-reading within a process.

### Known limitation to document

11. **The pipe and SSH backends are POSIX-only.** `PipeBackend.wait` uses `select.select`
    over subprocess pipes; on Windows `multiprocessing.connection.wait` requires
    overlapped-capable Win32 handles. The backend seam confines this to one
    implementation, but it should be stated rather than discovered.

### Documentation

12. The remote contract: importing the package must be sufficient, with scikit-build-core
    (`editable.rebuild`) and meson-python named as the remedy. Include a note that the
    existing list of dependencies invisible to the fingerprint applies with more
    consequence remotely — a data file that exists beside the project locally may not
    exist on a worker, and nothing detects that in advance.

## Open questions

**A separate entry point for effectful batches.** `run_all` provides per-input isolation,
timeouts that kill, and failure attribution. A batch that only wants those and no caching
(for example, downloading many files) would be excluded once `@pure` is required. Whether
to serve it with a second function is undecided; it should not be built speculatively.

**A content-addressed value that can produce a file path on demand.** Some libraries accept
only filenames (RTKLIB's `readrnxt` is one concrete case). Expressing acquisition as a
value currently requires the caller to write bytes to a temporary file. Whether to support
this in the library is undecided.

## Checking that it still works

```
python -m pytest -ra                 # 213 tests
```

To exercise the remote path manually, select the pipe backend through the private hook,
run a batch, then delete the project's source file and confirm that tasks still succeed
from the source tree:

```python
import valuekit as vk
from valuekit import parallel
from valuekit.backend import PipeBackend

vk.set_cache_dir(cache)
parallel._backend_factory = PipeBackend
vk.run_all(mymodule.work, [1, 2, 3])
```

The cache directory should then contain `format`, `objects`, `traces`, `runs` and
`source`. `python -m valuekit.monitor <cache-dir>` reads the run log and shows per-function
hit rates, batch progress and failures.
