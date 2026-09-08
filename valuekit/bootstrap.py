"""How a project tree becomes an environment on a host, and how the host
process starts inside it.

A host needs only what a person would need to check the project out and
run it: a Python to start with, the tool the project locks its
dependencies with, a compiler if it builds an extension, and the network.
Nothing of valuekit's is installed there beforehand.  valuekit itself is
one of the project's dependencies, so it arrives with the rest.

That leaves a gap: something has to run on the host before the project's
environment exists, to receive the tree and build that environment.  This
module is that something.  The driver sends its source over the
connection, a one-line Python program (:data:`STAGE0`) reads it and runs
it, and it then speaks a short protocol on the same two streams::

    driver -> the source of this module, then a NUL byte
    driver -> {"root": ..., "tree": <tree id>, "python": "3.13"}
    host   -> {"have": bool}          whether the tree is already here
    driver -> 8-byte length, tar      only if not
    host   -> {"ok": true, "python": <interpreter>} | {"ok": false, "reason": ...}
    host   -> python -m valuekit.host, from that interpreter, on these streams

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

The layout under the source root is one directory per tree, named by the
tree's identity, and a marker file beside it holding the interpreter path.
The marker is written last, so a directory without one is either being
built by another driver or is debris; the first is waited for, the second
is removed once it is old enough to be sure.  The tree is renamed into
place *before* the sync runs, since an editable install records the
directory it was made in.  A sync that fails removes the tree, so a retry
starts clean rather than waiting on a marker that will never come.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

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
# driver runs, else that minor for the tool to find or fetch), where the
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
# stream ends first (a driver that died before sending it), rather than
# reading empty strings forever.  Safe to pass through sh, cmd.exe and
# PowerShell inside double quotes: no dollar, backslash, percent, caret,
# ampersand, pipe or angle bracket.
STAGE0 = "import os;exec(b''.join(iter(lambda:os.read(0,1) or os._exit(1),bytes(1))))"

_MAX_TREES = 10  # complete trees kept per source root
_STALE = 3600  # seconds after which an unfinished tree is debris
_WAIT = 600  # seconds to wait for another driver's tree to finish
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
# the driver's half
# ---------------------------------------------------------------------------

_source: bytes | None = None


def _script() -> bytes:
    global _source
    if _source is None:
        with open(__file__, "rb") as f:
            _source = f.read()
    return _source


def offer(rx, tx, source_root: str, tree_id: str, py_minor: str, pack) -> str:
    """Bring the host at the far end of *rx*/*tx* to a running host process.

    *pack* is called for the tree's tarball only if the host asks for it.
    Returns "" once the host process is about to greet, else why not.
    """
    tx.write(_script() + b"\0")
    tx.write(
        json.dumps({"root": source_root, "tree": tree_id, "python": py_minor}).encode()
        + b"\n"
    )
    tx.flush()
    reply = _reply(rx)
    if reply is None:
        return "the host's Python never ran the bootstrap"
    if not reply.get("have"):
        data = pack()
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


def _stale(path: str) -> bool:
    try:
        return time.time() - os.stat(path).st_mtime > _STALE
    except OSError:
        return False


def _ready(marker: str) -> str | None:
    """The interpreter a finished tree's marker names, if it still exists."""
    try:
        with open(marker, encoding="utf-8") as f:
            python = f.read().strip()
    except OSError:
        return None
    return python if python and os.path.exists(python) else None


def _await(marker: str, tree: str) -> tuple[str, str]:
    """Wait for another driver to finish the tree it is building."""
    deadline = time.time() + _WAIT
    while time.time() < deadline:
        python = _ready(marker)
        if python:
            return python, ""
        if not os.path.isdir(tree):
            break
        time.sleep(0.5)
    return "", (
        f"another driver was preparing {tree} and it did not finish. If nothing "
        f"is running there, delete that directory and try again."
    )


def _extract(data: bytes, dest: str) -> str:
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
        os.makedirs(dest)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)  # every member was just checked
    return ""


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
    the driver's minor version, else the version.  Naming an interpreter
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


def _prune(root: str) -> None:
    """Keep the source root to a few finished trees; drop old debris."""
    try:
        names = os.listdir(root)
    except OSError:
        return
    markers = []
    for name in names:
        path = os.path.join(root, name)
        if name.endswith(".complete"):
            try:
                markers.append((os.stat(path).st_mtime, name[: -len(".complete")]))
            except OSError:
                pass
        elif os.path.isdir(path) and name + ".complete" not in names and _stale(path):
            shutil.rmtree(path, ignore_errors=True)  # unfinished, and old
    markers.sort()
    for _, tree_id in markers[: max(0, len(markers) - _MAX_TREES + 1)]:
        shutil.rmtree(os.path.join(root, tree_id), ignore_errors=True)
        try:
            os.remove(os.path.join(root, tree_id + ".complete"))
        except OSError:
            pass


def _prepare(root: str, tree_id: str, py_minor: str, data: bytes | None) -> tuple[str, str]:
    """The tree's interpreter, building the environment if it is not there."""
    tree = os.path.join(root, tree_id)
    marker = tree + ".complete"
    python = _ready(marker)
    if python:
        return python, ""
    if data is None:
        return _await(marker, tree)  # the driver was told it is here

    os.makedirs(root, exist_ok=True)
    _prune(root)
    if os.path.isdir(tree) and _stale(tree):
        shutil.rmtree(tree, ignore_errors=True)
    tmp = os.path.join(root, ".tmp-" + uuid.uuid4().hex)
    try:
        reason = _extract(data, tmp)
        if reason:
            return "", reason
        try:
            os.rename(tmp, tree)
        except OSError:
            if os.path.isdir(tree):
                return _await(marker, tree)  # someone else got there first
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    python, reason = _sync(tree, py_minor)
    if reason:
        shutil.rmtree(tree, ignore_errors=True)
        return "", reason
    with open(marker + ".tmp", "w", encoding="utf-8") as f:
        f.write(python + "\n")
    os.replace(marker + ".tmp", marker)
    return python, ""


def main() -> int:
    _binary_stdio()
    line = _read_line()
    if line is None:
        return 1
    req = json.loads(line)
    root = os.path.expanduser(req["root"])
    tree_id = req["tree"]
    tree = os.path.join(root, tree_id)
    have = _ready(tree + ".complete") is not None or (
        os.path.isdir(tree) and not _stale(tree)
    )
    _send({"have": have})
    data = None
    if not have:
        head = _read_exact(8)
        data = _read_exact(int.from_bytes(head, "little")) if head else None
        if data is None:
            return 1
    try:
        python, reason = _prepare(root, tree_id, req["python"], data)
    except Exception as e:  # anything else is still a reason, not a crash
        python, reason = "", f"{type(e).__name__}: {e}"
    if reason:
        _send({"ok": False, "reason": reason})
        return 1
    _send({"ok": True, "python": python})
    env = dict(os.environ)
    env["VALUEKIT_TREE"] = tree
    # The environment is activated, as a shell would: the tools the lock
    # installed beside the interpreter (cmake and ninja for an extension that
    # rebuilds on import, say) are on the PATH the workers see.  Nothing
    # above the bootstrap knows where the environment keeps them.
    bindir = os.path.dirname(python)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.dirname(bindir)
    return subprocess.call([python, "-m", "valuekit.host"], env=env)


if __name__ == "__main__":
    sys.exit(main())
