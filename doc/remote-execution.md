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
2. **The user's model is "my machine has extra cores".** Hosts in the local file are used
   by default. Preparing one never holds work back. A host that dies loses nothing.
3. **The design extends to serverless.** The connection is one interface; the bootstrap is a
   container entrypoint; requeue on connection loss is what a recycled instance needs.

## What works today

`run_all` runs a batch across this machine and any hosts named in the project's
`valuekit.local.toml`. The sequence for a host:

1. The main process refuses before connecting if the project cannot go: no lock file valuekit
   knows, or user code outside the project tree (`project.Project.refusal`).
2. The main process opens one connection: `ssh -T -o BatchMode=yes <target> "<python> -c
   <stage 0>"`, where stage 0 is a one-line Python program that reads a script off stdin
   up to a NUL byte and runs it. The main process sends `valuekit/bootstrap.py` as that script.
3. The bootstrap speaks a short JSON-line protocol on the same streams: it says whether
   `source_root/<project>` already holds this project hash, else reports the files it has;
   the main process sends the names to delete and a tar of the files whose hash differs; the
   bootstrap applies both in place, runs the lock tool's install command (`uv sync --frozen --python
   <main process's minor>`), writes the manifest beside the directory (project hash, interpreter,
   every file's hash), and starts `python -m valuekit.hostprocess` from that interpreter with
   `VALUEKIT_TREE` and `VALUEKIT_PROJECT_HASH` set, on the same streams, in the environment
   activated (its interpreter's directory first on `PATH`, `VIRTUAL_ENV` set).
4. The host process's first message carries its Python version, CPU count and pid. On a
   check channel the main process sends HELLO (Python version, function, function hash, project hash,
   import roots); the host
   starts `valuekit.worker --check`, which puts the tree's roots on `sys.path`, imports the
   function, checks that every user module came from the tree, and compares function hashes.
   A host syncs on its own thread; the batch never waits for it.
5. Each task is a channel: one `valuekit.worker` per task. The worker's store is a
   `RemoteStore`, so its values, call records, logged values and events go to the main process, its lookups ask
   the main process, and a `@pure_local` call is sent to the main process to run there.
6. The main process records each outcome under the host it ran on. An input whose host
   connection closed is requeued elsewhere, once; a worker that exits with a code fails
   its input as a local one would.

The `mode` line of `valuekit.local.toml` is read each time a task is started and defaults to
`all`. A switch to a mode that needs hosts not yet synced starts their sync; they
join as they become ready.

## Layers

```
Connection    bytes to a process on the host       hosts.ProcessConnection (later: a socket)
bootstrap     tree -> environment -> host process   bootstrap.py, both halves, stdlib-only
host          workers as numbered channels          hostprocess.py
worker        check the function hash, run one task      worker.py
store         the main process's store over the channel   remotestore.py
```

## Module map

| File | Responsibility |
|---|---|
| `valuekit/parallel.py` | Scheduling: capacities per host from the local file's mode, deadlines, input ordering, failure attribution, requeue on host loss, cached-input short-circuit, batch recording. |
| `valuekit/localfile.py` | The local file: hosts, worker cap, mode, project name; reading and setting the mode line. |
| `valuekit/modes.py` | What each mode means: the capacity each host has under it. |
| `valuekit/hosts.py` | `Connection`/`ProcessConnection`, `LocalHost` (a process per input), `RemoteHost` (one connection, a channel per task), and the handle that answers a worker's store requests and runs its `@pure_local` calls. |
| `valuekit/bootstrap.py` | How a tree becomes an environment on a host: the lock-tool table, the layout under `source_root`, extraction, the environment build, starting the host process. Both halves of its protocol. Stdlib only. |
| `valuekit/hostprocess.py` | The host process: starts a worker per channel, multiplexes their streams, exits on EOF. |
| `valuekit/worker.py` | The worker process: a check mode and a single-task mode; install, admit, audit. |
| `valuekit/remotestore.py` | `RemoteStore`: the worker's side of the store, over its channel. |
| `valuekit/protocol.py` | Message framing, channel framing, and value transfer as content-addressed object graphs. |
| `valuekit/codec.py` | The structural value format, and the child-hash walk the sweep uses. |
| `valuekit/project.py` | `Project` (the project as sent, once per batch), manifest, project hash, packing, import roots, and the path predicate that separates project from environment. |
| `valuekit/functionhash.py` | Function hashes (the hash of a function's reachable set), including a native extension's marker as the project hash of the tree it was built from. |
| `valuekit/batches.py` | Batch records: written by `run_all`, read by `valuekit.batch()`. |
| `valuekit/runlog.py` | The run's log: the values a run logged, under `logs/<script>/`, written as steps log or hit, read by `valuekit.logs()`. |
| `valuekit/sweep.py` | Retention: delete what the current code cannot reach. |
| `valuekit/events.py` | Records hits, misses, forced runs, errors, batch progress, host and requeue events. |
| `valuekit/monitor.py` | Reads the event log; shows the mode and what it means for the next task; sets the mode. |

## Decisions taken

**A project runs remotely iff its tree carries a lock file from a tool valuekit can
invoke, and that tool is on the host.** The capability needed is: given the tree and
nothing else, produce an interpreter on the host that imports the project with the
versions the main process has. Lock tools provide it; the requirement is the capability, not
uv. Detection is by lock filename through a table (`bootstrap._TOOLS`) with one row per
tool; uv is the first row because it is what can be validated between the two machines
to hand. Nothing above the bootstrap knows which row was used. The README says
"requires a locked project", never "requires uv". Rejected on the way: pip with
`--no-deps --no-build-isolation` (kept the old boundary rather than the clone-and-run
reading), pip with defaults on any pyproject (no lock, so versions drift), a
project-configured prepare command (a hook by another name).

**Dependencies are the project's job; the toolchain is the environment's.** This replaces
"dependencies are the environment's job on both machines". The environment partition in
the function hash still exists, populated by the sync rather than by a human.

**The bootstrap is stdlib-only and sent over the connection.** Nothing of valuekit's is
on the host before the project's environment exists, and valuekit is one of the project's
dependencies, so it arrives with the rest. Stage 0 reads the script one byte at a time up
to a NUL so nothing meant for the protocol is consumed ahead. The bootstrap stays as the
host process's parent (spawn-and-wait on every platform, since `exec` is unreliable on
Windows) and never touches the streams after the spawn. It is also a container entrypoint.

**One directory per project on a host, updated in place.** Each version of the tree used
to get its own directory named by the project hash, which made every edit a from-scratch
build of a native extension: CMake keys its cache on the source path. Now
`source_root/<project>` is the one directory, its manifest beside it names the project hash
it holds and every file's hash, and an update sends only the difference and touches
nothing the manifest never listed, so `build/` and the environment persist. The manifest
is removed before an update and written after; a lock file beside the directory says an
update is in progress, a second main process waits for it, and a lock older than an hour is
broken. A sync that fails keeps the tree and drops the manifest, so the next update starts
from what is there. A host holds one version at a time: while a host process runs, its
bootstrap keeps a busy marker (refreshed every few seconds; ignored once stale) naming
the run, and a run wanting a different version is refused with that name rather than
updating the directory under a running batch. Starting a second run before stopping the
first is the user's error, and the refusal says what to stop. The project hash still says whether a host is current and still
stands for a native extension in the function hash.

**A native extension's marker is the hash of the main process's binary, sent to workers.**
The marker was the project hash for a while, on the reasoning that each host builds its own
binary and the tree is what they share. That was too coarse (any edit anywhere re-keyed
every function reaching an extension) and the reasoning missed that a worker never computes
the marker: it takes the main process's from HELLO. So the main process hashes its own
build, per extension, and sends the markers; the walk on a worker substitutes them. Results
computed anywhere are keyed by the main process's build; the sync guarantees a host's
binary was built from the same sources. An extension the main process never reached has
no marker on the worker, and the worker is refused. A released wheel keeps its version
marker, as before.

**Modes stay; the default is `all`; syncing never blocks.** On review, modes are a
preset over per-host capacities, which is the shape the extra-cores model wants
underneath; the objection to them was aesthetic. What the model concretely requires was
changed instead: configured hosts are used without a keystroke, local work starts at once
and hosts join when ready, and (under `remote` only) this machine stays idle while a host
is still syncing, so that mode keeps its meaning for a short batch.

**An input whose host connection closed is requeued, once.** From the scheduler's view the
input is neither done nor failed; `@pure` makes the retry safe. A worker that exits with a
code still fails its input, as locally. A second loss fails the input, so one that takes a
host down each time does not cycle. `Handle.lost()` is the distinction.

**The connection is one interface.** `Connection` is two byte streams and how they ended;
`ProcessConnection` is the one implementation (a child process's pipes, whether the child is
`python` here or `ssh`). A websocket later is another `Connection`. Nothing above it changes.

**`run_all` gains no placement parameter.** Where work runs is configuration: the hosts
file for what exists, the mode file for what is used. A host list in code could be reached
by a function hash, and where a call ran must not be able to affect its result.

**`run_all` requires `@pure` or `@pure_local`.** A batch's results are recorded by the
function that produced them, an already-cached input needs no worker, and a function whose
effects do not matter is the only kind that can safely run elsewhere.

**Workers hold no cache.** The worker's store is the main process's store over the connection.
The only things a host keeps are source trees and their environments, under `source_root`.

**`@pure_local` runs on the main process.** A function that reads outside its arguments reads an
environment, and environments differ between machines. The main process has the credentials and
the files; a worker sends the call back as a request and receives a value. There is no
credential forwarding.

**One connection per host, a host process at the other end.** The Windows ssh client has
no connection sharing, so a connection per task would cost a handshake per input. The host
process multiplexes worker streams by channel and exits when its stdin closes.

**Remote configuration is one file per checkout, `valuekit.local.toml`.** Hosts, the
local worker cap, the mode and the project's host directory name. It replaced
`VALUEKIT_HOSTS` and `<cache>/placement`: it is about this checkout on this machine, so it
is ignored by git (valuekit warns if it is tracked), never shipped, and never hashed. It
does not configure the cache; `set_store_dir` stays in code, so a checkout that never uses
other machines needs no file. A checkout that wants its own host directory, a git
worktree say, sets `project`.

**The monitor's only write is the `mode` line of that file.** Watching has no effect on a
run. The monitor finds the project from the directory it is run in.

**The worker environment is an allowlist.** What a process needs to start, plus
`VALUEKIT_*` (which carries `VALUEKIT_TREE`). No `PYTHONPATH`, no credentials.

**The bootstrap activates the environment it built.** A shell that runs a project activates
its environment; on a host there is no shell, so the bootstrap puts the interpreter's
directory first on the `PATH` the host process (and so every worker) sees, and sets
`VIRTUAL_ENV`. Found with scikit-build-core's `editable.rebuild`, which runs `cmake` by
name at import time: with `cmake` and `ninja` from PyPI in the lock, they are in the
environment's `bin`, and nothing else would put that on the PATH. The bootstrap is the
one place that knows where the environment keeps its tools.

**A project that rebuilds on import must build without isolation.** Not valuekit's
decision but a fact for the README: uv builds a wheel in a temporary environment, and
scikit-build-core's persistent build directory records the absolute path of the ninja it
used, which is gone by the first import (`CMakeCache.txt`'s `CMAKE_MAKE_PROGRAM`). With
`no-build-isolation-package = ["<project>"]` under `[tool.uv]` and `scikit-build-core`,
`cmake`, `ninja` among the dependencies, `uv sync` installs those first, the build uses the
environment's own tools, and the recorded paths survive. This is what scikit-build-core's
own documentation prescribes for `editable.rebuild`.

**The suite runs the real bootstrap.** The bootstrap is the riskiest code here (raw
file-descriptor I/O, Windows, shell quoting); a test-only bypass would leave exactly it
untested. A session fixture builds a valuekit wheel from the checkout, writes a template
project depending on it by a relative path inside the tree, and locks it with uv once;
every test tree gets those files. The cost is `uv` in CI and a few seconds per new tree.

**Terminology.** "Source tree": the project's files on a host, one directory per project.
"Project hash": its manifest hash, what the host's manifest names and a native extension's marker. "Event log": the diagnostic record of
what happened during a run, for the monitor. "Call record": a memoised call's recorded reads, result, nested calls and
logged values. "Batch record": what `run_all` writes under a name. "Run": one main process
process running a script. "Run log": the values a run logged, under `logs/`. "Host": a machine that can run
workers, this one included; a remote host is one reached over ssh. "Host process": the process on a
remote host that starts its workers. "Main process": the process the user started, in which the script runs; it owns the cache,
and during a batch it schedules the inputs and answers the workers. "Worker": a process that runs
one input and exits, started by the main process here or by a host process on a remote host.
"Check": the worker-module process that imports the function on a host and checks it,
running no input. "Mode": which hosts are used. "Connection": the
two byte streams to a process on a remote host.

## Outstanding work

### Before release

**Validation progress (2026-09-08).** The Mac (`pidge.local`, user `ians`, 18 cores) and
the PC (`hunk.local`, user `iansh`, 28 cores, Windows, sshd default shell `cmd.exe`) log
into each other by key. Mac-as-main process, PC-as-host is verified: tree shipped and synced
under `C:\Users\iansh\.cache\valuekit\source`, outcomes recorded under the host, a
second run skipped the transfer (host ready in 1s instead of 4s), mode `all` shared a
60-input batch between both machines, and `valuekit.batch()` read the record back. Two
defects found and fixed on the way, both in `bootstrap.py`:

- *uv could not inspect its managed Pythons on the PC* ("untrusted mount point", os
  error 448): sshd gives an administrator an elevated token, and Windows refuses an
  elevated process the junctions uv makes for its minor-version links. The bootstrap now
  hands the lock tool its own interpreter's path when that interpreter has the main process's
  minor, and the bare minor otherwise; uv then does no discovery.
- *Stage 0 never ended at EOF*: a main process that died before sending the script left it
  joining empty reads forever, at full CPU and growing without bound (seen on both
  machines). It now exits.

**PC-as-main process, Mac-as-host is verified (2026-09-08).** From a PowerShell on the PC
(`C:\Users\iansh\trial`, `venv\Scripts\python.exe drive.py`, `VALUEKIT_HOSTS` naming
`ians@pidge.local`): the tree was shipped and synced under
`/Users/ians/.cache/valuekit/source`, uv on the Mac took the bare-minor branch (its ssh
Python is the Xcode 3.9) and used its managed 3.14.6, six inputs ran there in 3.6s
including the build; a second run skipped the transfer (host ready in 2s, batch 2.0s);
mode `all` shared 60 inputs 32 on the Mac and 28 here with no failures or requeues, local
work starting a second before the host joined. Two things to know when repeating it:

- The main process must run under Windows' own ssh client (`C:\Windows\System32\OpenSSH`,
  which PowerShell's PATH gives) with a console. The key is passphrase-protected and held
  by the Windows ssh-agent service; Git Bash's MSYS ssh offers the key file, cannot ask
  for the passphrase under `BatchMode`, and is refused. A piped stdin is forwarded only
  when the process has a console (verified: a pty session works, `CREATE_NO_WINDOW` does
  not), which is why yesterday's `run1.cmd` through a non-pty session stalled at
  "started" and left the "never ran the bootstrap" host event in the trial's event log.
- The bootstrap finds uv in `~/.local/bin` on the Mac; the non-interactive PATH there is
  `~/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin`.
- The trial's `venv` drives with the checkout as an editable install, so the main process side
  is the working tree; the host side is the wheel named in the lock (`proj/wheels/`),
  rebuilt from the checkout with `uv build --wheel` and relocked with `uv lock --refresh`
  (a plain `uv lock` keeps the old hash when the filename is unchanged).

**A native extension, on the Mac, through a same-machine host process (2026-09-08).**
`C:\Users\iansh\trial\ext` is a minimal scikit-build-core project (`fastproj._core`, one
C function, `editable.rebuild`, `build-dir = "build/{wheel_tag}"`, `cmake` and `ninja`
from PyPI, non-isolated build; `drive.py` uses `parallel._host_commands` when
`VALUEKIT_HOSTS` is unset, `hosts-mac.toml` and `hosts-pc.toml` otherwise). Copied to
`/Users/ians/exttrial` and driven there in mode `remote`: the host built its own binary in
its tree under `cache/source/<project hash>/build/`, the check passed (the worker compares
function hashes and refuses a difference, so they matched), three inputs ran through the
host in 3.1s from a cold tree. Editing `_core.c` (`s + 1`) produced a new project hash, a fresh
host build, and the changed sums, again in 3.0s; nothing was sent but sources. On the way
the two findings above (activation; non-isolated build) were made and fixed or
documented. At the time, incremental rebuilds across trees were not available: each tree
was a new directory and CMake refuses a build directory whose recorded source directory
differs, so a host built each version from scratch. Closed on 2026-09-12 by the
one-directory-per-project decision above: repeated with the same project through a host
process on this machine, an edit to `_core.c` reached the host as one file, the ninja log
showed one object recompiled and relinked in the same build directory, and the results
carried the edit. One defect found on the way: the tar carries no times, so a file written
over an older tree read as older than the build, and a rebuild-on-import backend ran the
old binary on the new source; extracted files now take the host's current time.

**The extension across the two machines, both directions (2026-09-08).** The PC had no
C compiler at all; Visual Studio Build Tools 2022 with the C++ workload (MSVC 14.44) was
installed for this through winget. Then:

- *PC main process, Mac host.* `uv sync --frozen` on the PC built the extension with MSVC
  (scikit-build-core finds the compiler itself; no developer prompt) and, with
  `hosts-mac.toml`, three inputs ran on the Mac in 4.0s from a cold tree: a `.pyd` here,
  a `.so` there, the check passed, so one key for both binaries.
- *Mac main process, PC host.* The Mac's copy carries the `s + 1` edit, so its project hash is the
  one the Mac's own same-machine trial had produced; the PC built that tree under sshd
  (elevated token, no console) with MSVC, using the Visual Studio generator (the binary
  sits under `build/<tag>/Release/`), and the check passed: 22.7s cold including the
  build, 7.1s warm with the transfer skipped. Outcomes were recorded under `hunk`, and
  the sums carry the edit.

How the Mac was driven from the PC: the same key is on both machines, passphrase-
protected, and a non-interactive session on the Mac has no agent, so `ssh -A` from the PC
forwarded the Windows agent into the session (`ssh -A -T ians@pidge.local "cd ~/exttrial
&& PATH=$PWD/.venv/bin:$HOME/.local/bin:$PATH VALUEKIT_HOSTS=$PWD/hosts-pc.toml
.venv/bin/python drive.py 7 8 9"`). The Mac's `~/.ssh/agent/` socket from June is stale
and hangs `ssh-add`; do not use it. The main process's own venv must be activated (or its
`Scripts`/`bin` put on PATH) for the import-time rebuild on the main process, which is the
user's shell's job, as the README says.

1. ~~**Validate between the PC and the Mac, both directions.**~~ Done both ways (above).
   For the record, on the Windows host sshd's default shell must be `cmd.exe`: checked by
   running the stage-0 command line verbatim through `cmd /c` (works) and `powershell -c`
   (strips the quotes, a documented Windows OpenSSH limitation); `sh` works too.
2. ~~**A real native extension across machines.**~~ Done both ways (above). The
   incremental rebuild question is answered: not across trees, with CMake.
3. **Release.** `_version.py` to `0.5.0`, CHANGELOG dated, CI green on Linux, macOS and
   Windows, `python -m build`, `twine check --strict`, tag `v0.5.0`, publish.

### Correctness

4. **The Python version marker covers only `major.minor`.** Adding `micro` converts a silent
   risk into an explicit refusal, at the cost of a format-version bump. Less pressing now that
   the bootstrap asks the lock tool for the main process's minor.

5. **Verify a reported instability in the function hash** (a frozenset constant's `repr`
   varying with `PYTHONHASHSEED`). The evidence offered did not support the claim.

### Later

6. **A websocket `Connection` and a container image**, for Cloud Run services. The image is the
   toolchain; the bootstrap is the entrypoint; requeue covers a recycled instance. The
   real limit is bandwidth to the main process's store; a bucket holding objects by hash would
   be a second tier, after the streaming version works.

7. **Collapse `LocalHost` into a host process launched as a subprocess.** One code path;
   inputs narrowed to storable types locally as they are remotely.

8. **Moving a running task.** A mode switch applies to the next task started; a task
   already running finishes where it is. Correctness never needs more.

9. **A cap on concurrent `@pure_local` calls on the main process.** They run on a thread pool
   sized by CPU count; a download-heavy batch may want a smaller number.

## Open questions

**A content-addressed value that can produce a file path on demand.** Some libraries accept
only filenames. A `@pure_local` function returning `bytes` covers acquisition; a raw
object kind whose store path could be handed out on any machine would cover the library
case. Undecided.

## Checking that it still works

```
uv sync && uv run pytest -ra
```

The repository is itself a locked project: `uv.lock` pins the suite's environment, and
CI syncs from it. The host tests build a test project's environment with `uv` (which
must therefore be on the PATH; they skip with a message when it is absent), pinning
numpy to the main process's version and Python to the main process's minor, because a package's
version is part of the function hash of every function that uses it and a host whose
numpy differed would refuse the main process's functions as out of sync.

To exercise a host by hand without ssh, point the private hook at a Python on this
machine and set a mode:

```python
import sys
import valuekit as vk
from valuekit import parallel, localfile

vk.set_store_dir(cache)
parallel._host_commands = {"here": [sys.executable]}
vk.run_all(mymodule.work, [1, 2, 3])          # mode defaults to "all"
vk.batch("work")[1]
```

With hosts in the project's `valuekit.local.toml`, drop the hook and `python -m valuekit.monitor
<cache-dir>` shows the hosts block and switches modes with `l`, `r` and `a`.
