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
exercised in the suite through the real bootstrap: a host process on this machine,
started the way an ssh session would start it, building the test project's environment
with `uv sync` from a lock file. The suite therefore needs `uv` on the PATH.

## The three ideas this design answers

1. **A project runs remotely iff it could be checked out and run there.** The host
   provides the toolchain (a Python 3 to start with, the lock tool, a compiler, the
   network); the project provides everything else through its lock file, valuekit
   included. The host builds the environment; the project builds its own extension.
2. **The user's model is "my machine has extra cores".** Hosts in the hosts file are used
   by default. Preparing one never holds work back. A host that dies loses nothing.
3. **The design extends to serverless.** The connection is a seam; the bootstrap is a
   container entrypoint; requeue on connection loss is what a recycled instance needs.

## What works today

`run_all` runs a batch across this machine and any hosts named in the hosts file. The
sequence for a host:

1. The driver refuses before connecting if the project cannot go: no lock file valuekit
   knows, or user code outside the project tree (`sync.Project.refusal`).
2. The driver opens one connection: `ssh -T -o BatchMode=yes <target> "<python> -c
   <stage 0>"`, where stage 0 is a one-line Python program that reads a script off stdin
   up to a NUL byte and runs it. The driver sends `valuekit/bootstrap.py` as that script.
3. The bootstrap speaks a short JSON-line protocol on the same streams: it says whether it
   already holds the tree, receives the tarball if not, extracts it under `source_root`,
   runs the lock tool's sync in it (`uv sync --frozen --python <driver's minor>`), writes a
   marker beside the tree naming the interpreter, and starts `python -m valuekit.host` from
   that interpreter with `VALUEKIT_TREE` set, on the same streams.
4. The host process greets with its salt, CPU count and pid. On a readiness channel the
   driver sends its greeting (salt, function, fingerprint, tree id, import roots); the host
   starts `valuekit.worker --ready`, which puts the tree's roots on `sys.path`, imports the
   function, audits that every user module came from the tree, and compares fingerprints.
   Readiness runs on a thread per host; the batch never waits for it.
5. Each task is a channel: one `valuekit.worker` per task. The worker's store is a
   `WireStore`, so its values, traces and run-log records go to the driver, its lookups ask
   the driver, and a `@pure_local` call is sent to the driver to run there.
6. The driver records each outcome under the host it ran on. An input whose host
   connection closed is requeued elsewhere, once; a worker that exits with a code fails
   its input as a local one would.

The mode file (`<cache>/placement`) is read each time a task is started and defaults to
`all`. A switch to a mode that needs hosts not yet prepared starts their readiness; they
join as they become ready.

## Layers

```
Link          bytes to a process on the host       backend.ProcessLink (later: a socket)
bootstrap     tree -> environment -> host process   bootstrap.py, both halves, stdlib-only
host          workers as numbered channels          host.py
worker        verify identity, run one task         worker.py
store         the driver's store over the channel   remotestore.py
```

## Module map

| File | Responsibility |
|---|---|
| `valuekit/parallel.py` | Scheduling: capacities per place from the mode file, deadlines, input ordering, failure attribution, requeue on host loss, cached-input short-circuit, batch recording. |
| `valuekit/placement.py` | The hosts file, the mode file, capacities per mode, the worker environment allowlist. |
| `valuekit/backend.py` | `Link`/`ProcessLink` (the connection), `LocalBackend` (a process per input), `HostBackend` (one link, a channel per task), and the handle that answers a worker's store requests and runs its `@pure_local` calls. |
| `valuekit/bootstrap.py` | How a tree becomes an environment on a host: the lock-tool table, the layout under `source_root`, extraction, the sync, starting the host process. Both halves of its protocol. Stdlib only. |
| `valuekit/host.py` | The host process: starts a worker per channel, multiplexes their streams, exits on EOF. |
| `valuekit/worker.py` | The worker process: a readiness mode and a single-task mode; install, admit, audit. |
| `valuekit/remotestore.py` | `WireStore`: the worker's side of the store, over its channel. |
| `valuekit/wire.py` | Message framing, channel framing, and value transfer as content-addressed object graphs. |
| `valuekit/codec.py` | The structural value format, and the child-hash walk the sweep uses. |
| `valuekit/sync.py` | `Project` (the project as shipped, once per batch), manifest, tree identity, packing, import roots, and the path predicate that separates project from environment. |
| `valuekit/codehash.py` | Fingerprints, including a native extension's identity as the tree it was built from. |
| `valuekit/batches.py` | Batch records: written by `run_all`, read by `valuekit.batch()`. |
| `valuekit/sweep.py` | Retention: delete what the current code cannot reach. |
| `valuekit/runlog.py` | Records hits, misses, forced runs, errors, batch progress, placement, host and requeue events. |
| `valuekit/monitor.py` | Reads the run log; shows and sets the placement mode. |

## Decisions taken

**A project runs remotely iff its tree carries a lock file from a tool valuekit can
invoke, and that tool is on the host.** The capability needed is: given the tree and
nothing else, produce an interpreter on the host that imports the project with the
versions the driver has. Lock tools provide it; the requirement is the capability, not
uv. Detection is by lock filename through a table (`bootstrap._TOOLS`) with one row per
tool; uv is the first row because it is what can be validated between the two machines
to hand. Nothing above the bootstrap knows which row was used. The README says
"requires a locked project", never "requires uv". Rejected on the way: pip with
`--no-deps --no-build-isolation` (kept the old boundary rather than the clone-and-run
reading), pip with defaults on any pyproject (no lock, so versions drift), a
project-configured prepare command (a hook by another name).

**Dependencies are the project's job; the toolchain is the environment's.** This replaces
"dependencies are the environment's job on both machines". The environment partition in
the fingerprint still exists, populated by the sync rather than by a human.

**The bootstrap is stdlib-only and sent over the connection.** Nothing of valuekit's is
on the host before the project's environment exists, and valuekit is one of the project's
dependencies, so it arrives with the rest. Stage 0 reads the script one byte at a time up
to a NUL so nothing meant for the protocol is consumed ahead. The bootstrap stays as the
host process's parent (spawn-and-wait on every platform, since `exec` is unreliable on
Windows) and never touches the streams after the spawn. It is also a container entrypoint.

**The tree is renamed into place before the sync, and the marker is written after.** An
editable install records the directory it was made in, so syncing in a temporary
directory would bake that path in. A directory without a marker is either being built by
another driver (waited for) or debris (removed once older than an hour). A sync that fails
removes the tree, so a retry starts clean.

**A native extension's identity is the tree it was built from.** Each host builds its own
binary from the same sources, so the tree is what they share; a key computed anywhere
equals a key computed anywhere else with nothing sent between them. The cost is coarseness
(any edit in the project re-keys functions that reach an extension) and reliance on the
build being current: the key describes the sources, so a build backend that rebuilds on
import is what keeps the driver honest. On a worker the identity is the tree's name,
known before anything is imported. An extension with no project marker above it at all is
identified by its binary, which is all there is.

**Modes stay; the default is `all`; readiness never blocks.** On review, modes are a
preset over per-place capacities, which is the shape the extra-cores model wants
underneath; the objection to them was aesthetic. What the model concretely requires was
changed instead: configured hosts are used without a keystroke, local work starts at once
and hosts join when ready, and (under `remote` only) this machine stays idle while a host
is still preparing, so that mode keeps its meaning for a short batch.

**An input whose host connection closed is requeued, once.** From the scheduler's view the
input is neither done nor failed; `@pure` makes the retry safe. A worker that exits with a
code still fails its input, as locally. A second loss fails the input, so one that takes a
host down each time does not cycle. `Handle.lost()` is the distinction.

**The connection is a seam.** `Link` is two byte streams and how they ended;
`ProcessLink` is the one implementation (a child process's pipes, whether the child is
`python` here or `ssh`). A websocket later is another `Link`. Nothing above it changes.

**`run_all` gains no placement parameter.** Where work runs is configuration: the hosts
file for what exists, the mode file for what is used. A host list in code could be reached
by a fingerprint, and where a computation ran must not be able to affect its result.

**`run_all` requires `@pure` or `@pure_local`.** A batch's results are recorded by the
function that produced them, an already-cached input needs no worker, and a function whose
effects do not matter is the only kind that can safely run elsewhere.

**Workers hold no cache.** The worker's store is the driver's store over the connection.
The only things a host keeps are source trees and their environments, under `source_root`.

**`@pure_local` runs on the driver.** A function that reads outside its arguments reads an
environment, and environments differ between machines. The driver has the credentials and
the files; a worker sends the call back as a request and receives a value. There is no
credential forwarding.

**One connection per host, a host process at the other end.** The Windows ssh client has
no connection sharing, so a connection per task would cost a handshake per input. The host
process multiplexes worker streams by channel and exits when its stdin closes.

**The mode lives in the cache directory; the hosts file does not.** The mode changes from
run to run and is safely reset by deleting the cache. The hosts file is authored once and
must survive that.

**The monitor's only write is the mode file.** Watching has no effect on a run.

**The worker environment is an allowlist.** What a process needs to start, plus
`VALUEKIT_*` (which carries `VALUEKIT_TREE`). No `PYTHONPATH`, no credentials.

**The suite runs the real bootstrap.** The bootstrap is the riskiest code here (raw
file-descriptor I/O, Windows, shell quoting); a test-only bypass would leave exactly it
untested. A session fixture builds a valuekit wheel from the checkout, writes a template
project depending on it by a relative path inside the tree, and locks it with uv once;
every test tree gets those files. The cost is `uv` in CI and a few seconds per new tree.

**Terminology.** "Source tree": the project's files on a host. "Tree id": its manifest
hash, the tree's name and a native extension's identity. "Run log": the record of what
happened during a run. "Trace": a memoised call's recorded reads, result, nested calls and
log bindings. "Batch record": what `run_all` writes under a name. "Place": somewhere work
can run, this machine included. "Mode": which places are used. "Link": the connection.

## Outstanding work

### Before release

**Validation progress (2026-09-08).** The Mac (`pidge.local`, user `ians`, 18 cores) and
the PC (`hunk.local`, user `iansh`, 28 cores, Windows, sshd default shell `cmd.exe`) log
into each other by key. Mac-as-driver, PC-as-host is verified: tree shipped and synced
under `C:\Users\iansh\.cache\valuekit\source`, outcomes recorded under the host, a
second run skipped the transfer (host ready in 1s instead of 4s), mode `all` shared a
60-input batch between both machines, and `valuekit.batch()` read the record back. Two
defects found and fixed on the way, both in `bootstrap.py`:

- *uv could not inspect its managed Pythons on the PC* ("untrusted mount point", os
  error 448): sshd gives an administrator an elevated token, and Windows refuses an
  elevated process the junctions uv makes for its minor-version links. The bootstrap now
  hands the lock tool its own interpreter's path when that interpreter has the driver's
  minor, and the bare minor otherwise; uv then does no discovery.
- *Stage 0 never ended at EOF*: a driver that died before sending the script left it
  joining empty reads forever, at full CPU and growing without bound (seen on both
  machines). It now exits.

PC-as-driver is not yet verified. It cannot be exercised through a non-pty ssh session
into the PC: the Windows ssh client forwards nothing on a piped stdin, not even EOF,
unless the process has a console (verified: a pty session works, `CREATE_NO_WINDOW` does
not). Run the driver from a terminal on the PC. Trial material there:
`C:\Users\iansh\trial` holds a locked project (`proj/`, valuekit as a wheel), `drive.py`,
`hosts-mac.toml` naming `ians@pidge.local`, and a plain venv for driving (`venv/`; the
checkout's own `.venv` is a uv trampoline behind the same junction, unusable over ssh).
The Mac's ssh Python is the Xcode 3.9, so this direction also exercises the
bare-minor branch, where uv finds or fetches a 3.14 on the Mac.

1. **Validate between the PC and the Mac, both directions.** Steps in the README's
   "Running on other machines". Not exercisable in CI. On each side: uv installed, ssh
   login without a prompt; a project with a `uv.lock`. Confirm outcomes recorded under the
   host, the tree and `.venv` under `~/.cache/valuekit/source`, a second run skipping the
   transfer, and a Windows host (`python = "python"`) reached from the Mac. On the
   Windows host sshd's default shell must be `cmd.exe`: checked here by running the
   stage-0 command line verbatim through `cmd /c` (works) and `powershell -c` (strips
   the quotes, a documented Windows OpenSSH limitation); `sh` works too.
2. **A real native extension across machines.** A minimal scikit-build-core project with
   `editable.rebuild` on both machines: the host builds it, the keys match. Also whether a
   persistent build directory outside the tree gives incremental rebuilds, which is what
   keeps a host joining within seconds of an edit.
3. **Release.** `_version.py` to `0.5.0`, CHANGELOG dated, CI green on Linux, macOS and
   Windows, `python -m build`, `twine check --strict`, tag `v0.5.0`, publish.

### Correctness

4. **`_salt()` covers only `major.minor`.** Adding `micro` converts a silent risk into an
   explicit refusal, at the cost of a `CACHE_EPOCH` bump. Less pressing now that the
   bootstrap asks the lock tool for the driver's minor.

5. **Verify a reported instability in `_unit_digest`** (a frozenset constant's `repr`
   varying with `PYTHONHASHSEED`). The evidence offered did not support the claim.

### Later

6. **A websocket `Link` and a container image**, for Cloud Run services. The image is the
   toolchain; the bootstrap is the entrypoint; requeue covers a recycled instance. The
   real limit is bandwidth to the driver's store; a bucket holding objects by hash would
   be a second tier, after the streaming version works.

7. **Collapse `LocalBackend` into a host process launched as a subprocess.** One code path;
   inputs narrowed to storable types locally as they are remotely.

8. **Moving a running task.** A mode switch applies to the next task started; a task
   already running finishes where it is. Correctness never needs more.

9. **A cap on concurrent `@pure_local` calls on the driver.** They run on a thread pool
   sized by CPU count; a download-heavy batch may want a smaller number.

## Open questions

**A content-addressed value that can produce a file path on demand.** Some libraries accept
only filenames. A `@pure_local` function returning `bytes` covers acquisition; a raw
object kind whose store path could be handed out on any machine would cover the library
case. Undecided.

## Checking that it still works

```
.venv/Scripts/python.exe -m pytest -ra          # Windows; python -m pytest elsewhere
```

`uv` must be on the PATH: the host tests build the test project's environment with it,
and skip with a message when it is absent.

To exercise a host by hand without ssh, point the private hook at a Python on this
machine and set a mode:

```python
import sys
import valuekit as vk
from valuekit import parallel, placement

vk.set_cache_dir(cache)
parallel._host_commands = {"here": [sys.executable]}
vk.run_all(mymodule.work, [1, 2, 3])          # mode defaults to "all"
vk.batch("work")[1]
```

With a hosts file at `$VALUEKIT_HOSTS`, drop the hook and `python -m valuekit.monitor
<cache-dir>` shows the hosts block and switches modes with `l`, `r` and `a`.
