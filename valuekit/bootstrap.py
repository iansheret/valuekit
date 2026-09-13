"""How a project tree becomes an environment on a host, and how the host
process starts inside it.

A host needs only what a person would need to check the project out and
run it: a Python to start with, the tool the project locks its
dependencies with, a compiler if it builds an extension, and the network.
Nothing of valuekit's is installed there beforehand.  valuekit itself is
one of the project's dependencies, so it arrives with the rest.

That leaves a gap: something has to run on the host before the project's
environment exists, to receive the tree and build that environment.  This
module is that something.  The main process sends its source over the
connection, a one-line Python program (:data:`STAGE0`) reads it and runs
it, and it then speaks a short protocol on the same two streams::

    main   -> the source of this module, then a NUL byte
    main   -> {"root": ..., "project": ..., "project_hash": ..., "python": "3.13",
               "run": "<who is asking>"}
    host   -> {"have": true}                    the tree here is this one already
            | {"have": false, "files": {...}}   what is here: relpath -> file hash
    main   -> {"delete": [...], "files": {...}}  only if not have: what to remove,
                                                 and the full new manifest
    main   -> 8-byte length, tar                 the files whose hash differs
    host   -> {"ok": true, "python": <interpreter>} | {"ok": false, "reason": ...}
    host   -> python -m valuekit.hostprocess, from that interpreter, on these streams

Only the standard library is used, and only what Python 3.8 has, because
the interpreter this runs under is whatever the host happens to have.
Bytes are read from file descriptor 0 one request at a time and never
ahead of it, so that when the host process takes over the streams nothing
meant for it has been consumed.

*The tree becomes an environment the way the project says.*  Which tool to
run is read off the lock file the tree carries -- the one fact that makes
"check it out and run it" true for a person -- through a table with one
row per tool.  Nothing above this module knows which row was used.  A tree
with no lock valuekit knows is refused, and the refusal names the ones it
does.

*One directory per project, updated in place.*  Under the source root each
project has one directory, named by the project, and beside it a manifest
file naming the project hash the directory holds, the interpreter the sync
made, and every file with its hash.  A main process whose tree differs sends the
files that changed and the names of those removed; nothing else in the
directory is touched, so a build directory and the environment persist and
a native extension rebuilds incrementally.  The manifest is removed before
an update and written after, so a directory with no manifest is either
being updated or was left by a failed sync; either way the next main process
updates it in place.  A lock file beside the directory says an update is
in progress; a second main process waits for it, then proceeds with its own if
the tree is still not the one it wants.

*A host holds one version at a time.*  While a host process runs, its
bootstrap keeps a busy marker beside the directory, refreshed every few
seconds, naming the run it serves.  A run that wants a different version
while a live marker exists is refused with that name: stop the earlier run
or wait.  A run that wants the same version joins.  A marker that has
stopped being refreshed belongs to a run that died and is ignored.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

__all__ = [
    "KNOWN_LOCKS",
    "STAGE0",
    "lock_tool",
    "remote_command",
    "local_command",
    "offer",
    "main",
]

# The table.  A row is what one lock tool needs said about it: its
# executable, how to sync a tree into an environment for a given Python
# (``{python}``: this interpreter's path when it has the minor version the
# main process runs, else that minor for the tool to find or fetch), where the
# interpreter then lives relative to the tree, and where the executable
# hides when a non-interactive shell's PATH is short.
_TOOLS = {
    "uv.lock": {
        "tool": "uv",
        "sync": ("sync", "--frozen", "--python", "{python}"),
        "interpreters": (".venv/bin/python", ".venv/Scripts/python.exe"),
        "search": ("~/.local/bin", "~/.cargo/bin"),
    },
}

KNOWN_LOCKS = tuple(_TOOLS)

# Reads this module's source off stdin up to a NUL and runs it; exits if the
# stream ends first (a main process that died before sending it), rather than
# reading empty strings forever.  Safe to pass through sh, cmd.exe and
# PowerShell inside double quotes: no dollar, backslash, percent, caret,
# ampersand, pipe or angle bracket.
STAGE0 = "import os;exec(b''.join(iter(lambda:os.read(0,1) or os._exit(1),bytes(1))))"

_STALE = 3600  # seconds after which a lock counts as abandoned
_REFRESH = 5  # seconds between refreshes of a busy marker
_BUSY_STALE = 60  # seconds without a refresh after which a busy marker is ignored
_WAIT = 600  # seconds to wait for another main process's update to finish
_TAIL = 64 << 10


def lock_tool(names) -> str | None:
    """The lock file valuekit knows among *names*, or None."""
    present = set(names)
    for lock in KNOWN_LOCKS:
        if lock in present:
            return lock
    return None


def remote_command(python: str) -> str:
    """The command an ssh session runs: *python* is any Python 3 there."""
    return f'{python} -c "{STAGE0}"'


def local_command(python: str) -> list[str]:
    """The same as a plain argv, for a host on this machine."""
    return [python, "-c", STAGE0]


# ---------------------------------------------------------------------------
# the main process's half
# ---------------------------------------------------------------------------

_source: bytes | None = None


def _script() -> bytes:
    global _source
    if _source is None:
        with open(__file__, "rb") as f:
            _source = f.read()
    return _source


def offer(rx, tx, source_root: str, project: str, project_hash: str, py_minor: str, entries, pack, run: str = "") -> str:
    """Bring the host at the far end of *rx*/*tx* to a running host process.

    *entries* is the manifest, ``(relpath, hash)`` pairs; *pack* is called
    with the subset of them the host lacks, only if it lacks any.  *run*
    names the main process, for the refusal a busy host gives another run.
    Returns "" once the host process is about to send its first message,
    else why not.
    """
    tx.write(_script() + b"\0")
    tx.write(
        json.dumps(
            {
                "root": source_root, "project": project, "project_hash": project_hash,
                "python": py_minor, "run": run,
            }
        ).encode()
        + b"\n"
    )
    tx.flush()
    reply = _reply(rx)
    if reply is None:
        return "the host's Python never ran the bootstrap"
    if not reply.get("have"):
        theirs = reply.get("files") or {}
        ours = dict(entries)
        delete = sorted(set(theirs) - set(ours))
        changed = [(rel, h) for rel, h in entries if theirs.get(rel) != h]
        tx.write(json.dumps({"delete": delete, "files": ours}).encode() + b"\n")
        data = pack(changed)
        tx.write(len(data).to_bytes(8, "little") + data)
        tx.flush()
    reply = _reply(rx)
    if reply is None:
        return "the host never reported on the project"
    if not reply.get("ok"):
        return str(reply.get("reason") or "the host refused the project")
    return ""


def _reply(rx) -> dict | None:
    line = rx.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# the host's half
# ---------------------------------------------------------------------------


def _binary_stdio() -> None:
    if sys.platform == "win32":
        import msvcrt

        for fd in (0, 1):
            msvcrt.setmode(fd, os.O_BINARY)


def _read_line() -> bytes | None:
    buf = bytearray()
    while True:
        c = os.read(0, 1)
        if not c:
            return None
        if c == b"\n":
            return bytes(buf)
        buf += c


def _read_exact(n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(0, min(1 << 16, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _send(obj: dict) -> None:
    data = json.dumps(obj).encode() + b"\n"
    while data:
        n = os.write(1, data)
        data = data[n:]


def _busy(tree: str) -> str:
    """The run a live busy marker beside *tree* names, or ""."""
    root, name = os.path.split(tree)
    try:
        names = os.listdir(root)
    except OSError:
        return ""
    for n in names:
        if not n.startswith(name + ".busy-"):
            continue
        path = os.path.join(root, n)
        try:
            if time.time() - os.stat(path).st_mtime > _BUSY_STALE:
                os.remove(path)  # its run died without cleaning up
                continue
            with open(path, encoding="utf-8") as f:
                return f.read().strip() or "another run"
        except OSError:
            continue
    return ""


def _hold_busy(tree: str, run: str):
    """Create this run's busy marker and keep refreshing it; returns the
    function that removes it."""
    path = f"{tree}.busy-{os.getpid()}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(run + "\n")
    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(_REFRESH):
            try:
                os.utime(path, None)
            except OSError:
                return

    threading.Thread(target=refresh, daemon=True).start()

    def release() -> None:
        stop.set()
        try:
            os.remove(path)
        except OSError:
            pass

    return release


def _paths(root: str, project: str) -> tuple[str, str, str]:
    """The tree directory, its manifest file and its lock file."""
    tree = os.path.join(root, project)
    return tree, tree + ".manifest", tree + ".lock"


def _read_manifest(path: str) -> dict | None:
    """The manifest at *path* if it is whole and its interpreter exists."""
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or not isinstance(m.get("files"), dict):
        return None
    python = m.get("python")
    if not python or not os.path.exists(python):
        return None
    return m


def _write_manifest(path: str, project_hash: str, python: str, files: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"project_hash": project_hash, "python": python, "files": files}, f)
    os.replace(path + ".tmp", path)


def _stale(path: str) -> bool:
    try:
        return time.time() - os.stat(path).st_mtime > _STALE
    except OSError:
        return False


def _take_lock(lock: str) -> str:
    """Hold *lock* for this update; wait for another main process's first.

    Returns "" once held, else why not.  A lock older than ``_STALE`` was
    left by a main process that died and is removed.
    """
    deadline = time.time() + _WAIT
    while True:
        if _stale(lock):
            try:
                os.remove(lock)
            except OSError:
                pass
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() > deadline:
                return (
                    f"another main process was updating this project ({lock} is held) and "
                    "did not finish. If nothing is running there, delete that file."
                )
            time.sleep(0.5)
            continue
        os.close(fd)
        return ""


def _release_lock(lock: str) -> None:
    try:
        os.remove(lock)
    except OSError:
        pass


def _extract(data: bytes, dest: str) -> str:
    """Unpack *data* over *dest*, replacing files it names and no others."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r")
    except tarfile.TarError as e:
        return f"the project tree could not be read: {e}"
    with tar:
        for m in tar.getmembers():
            parts = m.name.replace("\\", "/").split("/")
            if (
                not m.isfile()
                or m.name.startswith("/")
                or ".." in parts
                or (len(m.name) > 1 and m.name[1] == ":")
            ):
                return f"refusing to unpack {m.name!r}: not a plain file inside the tree"
        os.makedirs(dest, exist_ok=True)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)  # every member was just checked
        # The tar carries no times.  A file written over an older tree must
        # read as newer than any build made from the old one, or a build
        # backend that rebuilds on import sees nothing to do and the worker
        # runs the old binary on the new source.
        now = time.time()
        for m in tar.getmembers():
            try:
                os.utime(os.path.join(dest, *m.name.split("/")), (now, now))
            except OSError:
                pass
    return ""


def _delete(tree: str, rels) -> None:
    """Remove the named files from *tree*; a path outside it is ignored."""
    real = os.path.realpath(tree)
    for rel in rels:
        path = os.path.realpath(os.path.join(tree, *rel.replace("\\", "/").split("/")))
        if not path.startswith(real + os.sep):
            continue
        try:
            os.remove(path)
        except OSError:
            continue
        # Directories the manifest no longer reaches are left in place only
        # if something else is in them.
        parent = os.path.dirname(path)
        while parent != real:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)


def _find(tool: str, search) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    for d in search:
        for name in (tool, tool + ".exe"):
            candidate = os.path.join(os.path.expanduser(d), name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _python_request(py_minor: str) -> str:
    """What to ask the lock tool for: this interpreter, when it already has
    the main process's minor version, else the version.  Naming an interpreter
    spares a download when the host has one, and spares the tool's search
    through its own managed installations, which an ssh session on Windows
    cannot always traverse (their junctions are refused to an elevated
    process, and sshd gives an administrator an elevated token)."""
    if "%d.%d" % sys.version_info[:2] == py_minor:
        return sys.executable
    return py_minor


def _sync(tree: str, py_minor: str) -> tuple[str, str]:
    """Run the tree's lock tool in it; the interpreter it made, or why not."""
    lock = lock_tool(os.listdir(tree))
    if lock is None:
        return "", (
            "the project has no lock file valuekit knows how to use "
            f"({', '.join(KNOWN_LOCKS)}), so it cannot be run here"
        )
    row = _TOOLS[lock]
    exe = _find(row["tool"], row["search"])
    if exe is None:
        looked = ", ".join(row["search"])
        return "", (
            f"{row['tool']} is not on this host's PATH (a non-interactive ssh "
            f"session has a short one) and was not found in {looked}. Install "
            "it there, or put it on the PATH that non-interactive shells see."
        )
    cmd = [exe] + [a.format(python=_python_request(py_minor)) for a in row["sync"]]
    try:
        p = subprocess.run(
            cmd, cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except OSError as e:
        return "", f"could not run {' '.join(cmd)}: {e}"
    if p.returncode:
        tail = p.stdout[-_TAIL:].decode("utf-8", "replace").strip()
        return "", f"{' '.join(cmd)} failed in {tree} (exit {p.returncode}):\n{tail}"
    for rel in row["interpreters"]:
        python = os.path.join(tree, *rel.split("/"))
        if os.path.exists(python):
            return python, ""
    return "", (
        f"{row['tool']} finished but left no interpreter at any of "
        f"{', '.join(row['interpreters'])} under {tree}"
    )


def _prepare(
    root: str, project: str, project_hash: str, py_minor: str, delete, files: dict, data: bytes
) -> tuple[str, str]:
    """Update the project's directory to *project_hash* and build its
    environment; the interpreter, or why not.

    *delete* names the files to remove, *files* is the full new manifest,
    *data* a tar of the files whose content differs.  Holds the project's
    lock throughout.  A main process that arrives while another holds the lock
    waits; if the other main process's update produced this project hash there is
    nothing left to do.
    """
    tree, manifest, lock = _paths(root, project)
    os.makedirs(root, exist_ok=True)
    reason = _take_lock(lock)
    if reason:
        return "", reason
    try:
        current = _read_manifest(manifest)
        if current is not None and current.get("project_hash") == project_hash:
            return current["python"], ""  # another main process just did this
        try:
            os.remove(manifest)
        except OSError:
            pass
        _delete(tree, delete)
        reason = _extract(data, tree)
        if reason:
            return "", reason
        python, reason = _sync(tree, py_minor)
        if reason:
            return "", reason  # the tree stays; the next update starts from it
        _write_manifest(manifest, project_hash, python, files)
        return python, ""
    finally:
        _release_lock(lock)


def main() -> int:
    _binary_stdio()
    line = _read_line()
    if line is None:
        return 1
    req = json.loads(line)
    root = os.path.expanduser(req["root"])
    project = req["project"]
    project_hash = req["project_hash"]
    tree, manifest, _ = _paths(root, project)
    current = _read_manifest(manifest)
    have = current is not None and current.get("project_hash") == project_hash
    busy = "" if have else _busy(tree)
    if busy:
        _send({"have": True})  # nothing to send: the update is refused
        _send({"ok": False, "reason": (
            f"the project directory on this host is in use by {busy}, which runs a "
            "different version; stop that run or wait for it"
        )})
        return 1
    if have:
        _send({"have": True})
        python, reason = current["python"], ""
    else:
        _send({"have": False, "files": current["files"] if current else {}})
        line = _read_line()
        head = _read_exact(8)
        data = _read_exact(int.from_bytes(head, "little")) if head else None
        if line is None or data is None:
            return 1
        change = json.loads(line)
        try:
            python, reason = _prepare(
                root, project, project_hash, req["python"], change["delete"], change["files"], data
            )
        except Exception as e:  # anything else is still a reason, not a crash
            python, reason = "", f"{type(e).__name__}: {e}"
    if reason:
        _send({"ok": False, "reason": reason})
        return 1
    _send({"ok": True, "python": python})
    release = _hold_busy(tree, req.get("run") or "another run")
    env = dict(os.environ)
    env["VALUEKIT_TREE"] = tree
    env["VALUEKIT_PROJECT_HASH"] = project_hash
    # The environment is activated, as a shell would: the tools the lock
    # installed beside the interpreter (cmake and ninja for an extension that
    # rebuilds on import, say) are on the PATH the workers see.  Nothing
    # above the bootstrap knows where the environment keeps them.
    bindir = os.path.dirname(python)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.dirname(bindir)
    try:
        return subprocess.call([python, "-m", "valuekit.hostprocess"], env=env)
    finally:
        release()


if __name__ == "__main__":
    sys.exit(main())
