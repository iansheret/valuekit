# Remote execution: state and outstanding work

This records the state of the remote-execution work so it can be resumed after a break.
It covers what is built, what was decided and why, and what is left. Decisions are
recorded with their reasons so they need not be reargued.

## Repository state

| Branch | Contents |
|---|---|
| `main` | Released as `v0.3.1`, tagged and published to PyPI. |
| `remote-seam` | Everything below, unreleased. |

**Nothing after 0.3.1 is released, and nothing will be until ssh execution is validated
between two real machines.** The CHANGELOG's "Unreleased" entry holds it all. `_version.py`
reads `0.4.0` and becomes `0.5.0` at release.

The suite passes on Windows (the machine this was developed on) and on Linux and macOS in
CI; the Windows job is gating. Everything below except the ssh command line itself is
exercised in the suite through a host process launched on the same machine.

## What works today

`run_all` runs a batch across this machine and any hosts named in the hosts file, under
a mode chosen in the monitor. The sequence for a host:

1. The driver opens one connection to the host: `ssh -T -o BatchMode=yes <target>
   "<python> -m valuekit.host"`. The host process greets with its salt and CPU count.
2. On a readiness channel, the driver sends its manifest hash and import roots; the host
   process starts `valuekit.worker --ready`, which reports whether it holds the source tree,
   receives it if not, installs it on `sys.path`, imports the function, audits that every
   user module came from the tree, and compares fingerprints. Readiness runs for every
   host concurrently before any task starts, and a host that fails is dropped with its
   reason recorded once.
3. Each task is a channel: the host process starts one `valuekit.worker` per task and
   carries its stdin and stdout as `DATA` frames. The worker's store is a `WireStore`, so
   its values, traces and run-log records go to the driver, its lookups ask the driver,
   and a `@pure_local` call is sent to the driver to run there.
4. The driver records each outcome under the host it ran on, and each input's root trace
   in the batch record.

The mode file (`<cache>/placement`) is read each time a task is started. A switch to a
mode that needs hosts not yet prepared starts their readiness on threads; they join as
they become ready. A host whose connection closes fails the inputs running there and is
dropped.

## Module map

| File | Responsibility |
|---|---|
| `valuekit/parallel.py` | Scheduling: capacities per place from the mode file, remote places first, deadlines, input ordering, failure attribution, cached-input short-circuit, batch recording. |
| `valuekit/placement.py` | The hosts file, the mode file, capacities per mode, the worker environment allowlist. |
| `valuekit/backend.py` | `LocalBackend` (a process per input), `HostBackend` (one connection, a channel per task), and the handle that answers a worker's store requests and runs its `@pure_local` calls. |
| `valuekit/host.py` | The host process: starts a worker per channel, multiplexes their streams, exits on EOF. |
| `valuekit/worker.py` | The worker process: a readiness mode and a single-task mode. |
| `valuekit/remotestore.py` | `WireStore`: the worker's side of the store, over its channel. |
| `valuekit/wire.py` | Message framing, channel framing, and value transfer as content-addressed object graphs. |
| `valuekit/codec.py` | The structural value format, and the child-hash walk the sweep uses. |
| `valuekit/sync.py` | Manifest, source-tree packing, import roots, and the path predicate that separates project from environment. |
| `valuekit/codehash.py` | Fingerprints, including the extension-digest override a worker takes from the driver. |
| `valuekit/batches.py` | Batch records: written by `run_all`, read by `valuekit.batch()`. |
| `valuekit/sweep.py` | Retention: delete what the current code cannot reach. |
| `valuekit/runlog.py` | Records hits, misses, forced runs, errors, batch progress, placement and host events. |
| `valuekit/monitor.py` | Reads the run log; shows and sets the placement mode. |

## Decisions taken

**`run_all` gains no placement parameter.** Where work runs is configuration: the hosts
file for what exists, the mode file for what is used. A host list in code could be reached
by a fingerprint, and where a computation ran must not be able to affect its result.

**`run_all` requires `@pure` or `@pure_local`.** A batch's results are recorded by the
function that produced them, an already-cached input needs no worker, and a function whose
effects do not matter is the only kind that can safely run elsewhere.

**Workers hold no cache.** The worker's store is the driver's store over the connection.
The only thing a host keeps is the source tree, at `source_root` (default
`~/.cache/valuekit/source`, expanded on the host).

**`@pure_local` runs on the driver.** A function that reads outside its arguments reads an
environment, and environments differ between machines. The driver has the credentials and
the files; a worker sends the call back as a request and receives a value. There is no
credential forwarding.

**One connection per host, a host process at the other end.** The Windows ssh client has
no connection sharing, so a connection per task would cost a handshake per input. The host
process multiplexes worker streams by channel and exits when its stdin closes, so a dropped
connection cannot orphan work. The same host process, launched as a plain subprocess, is
how the suite exercises everything without a network.

**Three modes over per-place capacities.** `local`, `remote`, `all` are presets over a
capacity per place; `remote` with no reachable host is local at full capacity, recorded as
such. Remote places are filled before local, in configuration order.

**The mode lives in the cache directory; the hosts file does not.** The mode changes from
run to run and is safely reset by deleting the cache. The hosts file is authored once and
must survive that.

**The monitor's only write is the mode file.** Watching has no effect on a run; a
keystroke has exactly the intended effect and no other. The run log carries the mode the
driver applied, so the monitor shows both and a stale driver cannot be mistaken for a
switched one.

**A worker fingerprints as the driver would.** Two machines cannot hold the same compiled
binary, so the driver sends the digests its fingerprint walk hashed and the worker uses
them in place of its own. Every key computed on a worker then equals the driver's, and the
handshake and the traces need no second form. Cross-architecture execution is permitted;
whether results from two architectures are equivalent is the user's judgement, as `@pure`
already requires.

**Verify code identity; do not ship binaries.** A native extension must be built on the
machine that runs it, by a build backend that rebuilds on import (scikit-build-core with
`editable.rebuild`, or meson-python). There is no prepare hook.

**Sync ships the user-code partition only**, with a fixed cross-platform list of build
artefact suffixes excluded. Dependencies are the environment's responsibility on both
machines.

**The worker environment is an allowlist.** What a process needs to start, plus
`VALUEKIT_*`. No `PYTHONPATH`, no credentials.

**Terminology.** "Source tree": the project's files on a host. "Run log": the record of
what happened during a run. "Trace": a memoised call's recorded reads, result, nested calls
and log bindings. "Batch record": what `run_all` writes under a name. "Place": somewhere
work can run, this machine included. "Mode": which places are used.

## Outstanding work

### Before release

1. **Validate between the PC and the Mac, both directions.** Steps in the README's
   "Running on other machines" and in the plan. Not exercisable in CI.
2. **A real native extension across machines.** The suite covers the digest substitution
   with a fake extension in-process; a minimal scikit-build-core project on both machines
   confirms it end to end.
3. **Release.** `_version.py` to `0.5.0`, CHANGELOG dated, CI green on Linux, macOS and
   Windows, `python -m build`, `twine check --strict`, tag `v0.5.0`, publish.

### Correctness

4. **`_salt()` covers only `major.minor`.** Adding `micro` converts a silent risk into an
   explicit refusal, at the cost of a `CACHE_EPOCH` bump.

5. **`_classify` identifies installed packages by the substring `"site-packages"`.**
   `sync.is_environment` already derives this from `sysconfig` and `site`.

6. **Verify a reported instability in `_unit_digest`** (a frozenset constant's `repr`
   varying with `PYTHONHASHSEED`). The evidence offered did not support the claim.

### Later

7. **Moving a running task.** A mode switch applies to the next task started; a task
   already running finishes where it is. Correctness never needs more.

8. **A cap on concurrent `@pure_local` calls on the driver.** They run on a thread pool
   sized by CPU count; a download-heavy batch may want a smaller number.

9. **Re-running an input whose host died elsewhere.** Today it is recorded as a failure
   against the input, as a dead local worker is.

## Open questions

**A content-addressed value that can produce a file path on demand.** Some libraries accept
only filenames. A `@pure_local` function returning `bytes` covers acquisition; a raw
object kind whose store path could be handed out on any machine would cover the library
case. Undecided.

## Checking that it still works

```
.venv/Scripts/python.exe -m pytest -ra          # Windows; python -m pytest elsewhere
```

To exercise a host by hand without ssh, point the private hook at a host process on this
machine and set a mode:

```python
import sys
import valuekit as vk
from valuekit import parallel, placement

vk.set_cache_dir(cache)
parallel._host_commands = {"here": [sys.executable, "-m", "valuekit.host"]}
placement.write_mode(cache, "remote")
vk.run_all(mymodule.work, [1, 2, 3])
vk.batch("work")[1]
```

With a hosts file at `$VALUEKIT_HOSTS`, drop the hook and `python -m valuekit.monitor
<cache-dir>` shows the hosts block and switches modes with `l`, `r` and `a`.
