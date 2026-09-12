"""Where a batch runs: the local file.

Everything about running a checkout on other machines is one file beside
``pyproject.toml``, ``valuekit.local.toml``, which git should ignore.  It
is per checkout: the hosts this checkout may use, how many workers each
may run, the mode in force, and the name of this project's directory on
each host.  A checkout that never uses other machines has no such file.

::

    project = "residuals-experiment"   # optional: the host directory name
    mode = "all"                       # optional: all | local | remote

    [local]
    workers = 8                        # optional: this machine's cap

    [hosts.mac]
    ssh = "ian@mac.local"                        # anything ssh accepts
    python = "python3"                           # optional; any Python 3 there
    workers = 8                                  # optional; default: the host's CPU count
    source_root = "~/.cache/valuekit/source"     # optional; this is the default

``python`` is only what starts the bootstrap (:mod:`valuekit.bootstrap`);
the interpreter that runs the project comes from the project's own lock
file, built on the host.  Nothing of the project's, valuekit included, has
to be installed there.  A Windows host has ``python`` rather than
``python3``.

The *mode*: ``all`` uses every reachable remote host and this machine at
full capacity; ``local`` runs everything on this machine; ``remote`` runs as
little here as possible, which means nothing here while any host is
reachable or still syncing, and everything here when none is.  Absent,
it is ``all``: a host in the file is there to be used, the way a core is.
The scheduler reads the file each time it is about to start a task, so an
edit takes effect for the next task started; tasks already running finish
where they are.  The monitor's keys and its ``--mode`` flag edit the
``mode`` line in place and touch nothing else in the file.

The file is never part of the source tree sent to a host and never part of
any hash: it says where a computation runs, which must not be able to
affect a result.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .store import _atomic_write

__all__ = [
    "LOCAL_FILE",
    "MODES",
    "HostEntry",
    "LocalConfig",
    "load_local",
    "read_mode",
    "write_mode",
    "capacities",
    "worker_env",
]

LOCAL_FILE = "valuekit.local.toml"
MODES = ("local", "remote", "all")
DEFAULT_MODE = "all"
DEFAULT_SOURCE_ROOT = "~/.cache/valuekit/source"
DEFAULT_PYTHON = "python3"


@dataclass(frozen=True)
class HostEntry:
    name: str
    ssh: str
    python: str  # any Python 3 on the host, to bootstrap with
    workers: int | None  # None: whatever the host reports
    source_root: str


@dataclass(frozen=True)
class LocalConfig:
    project: str | None  # the host directory name, if chosen
    mode: str
    local_workers: int
    hosts: tuple[HostEntry, ...]


def local_path(root: str | os.PathLike | None) -> Path | None:
    return None if root is None else Path(root) / LOCAL_FILE


def load_local(root: str | os.PathLike | None) -> LocalConfig:
    """The local file in *root*, or the defaults when there is none."""
    default_local = os.cpu_count() or 1
    p = local_path(root)
    if p is None or not p.exists():
        return LocalConfig(None, DEFAULT_MODE, default_local, ())
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise RuntimeError(f"cannot read {p}: {e}") from e

    project = data.get("project")
    if project is not None and (not isinstance(project, str) or not project):
        raise RuntimeError(f"{p}: project must be a non-empty string")
    mode = data.get("mode", DEFAULT_MODE)
    if mode not in MODES:
        raise RuntimeError(f"{p}: mode must be one of {', '.join(MODES)}, not {mode!r}")

    local = data.get("local", {})
    if not isinstance(local, dict):
        raise RuntimeError(f"{p}: [local] must be a table")
    local_workers = _workers(p, "local", local.get("workers", default_local))

    hosts: list[HostEntry] = []
    table = data.get("hosts", {})
    if not isinstance(table, dict):
        raise RuntimeError(f"{p}: [hosts] must be a table of tables")
    for name, h in table.items():
        if not isinstance(h, dict):
            raise RuntimeError(f"{p}: [hosts.{name}] must be a table")
        if not isinstance(h.get("ssh"), str) or not h["ssh"]:
            raise RuntimeError(f"{p}: [hosts.{name}] needs a string 'ssh'")
        python = h.get("python", DEFAULT_PYTHON)
        if not isinstance(python, str) or not python:
            raise RuntimeError(f"{p}: [hosts.{name}] python must be a non-empty string")
        workers = h.get("workers")
        if workers is not None:
            workers = _workers(p, f"hosts.{name}", workers)
        hosts.append(
            HostEntry(
                name=str(name),
                ssh=h["ssh"],
                python=python,
                workers=workers,
                source_root=str(h.get("source_root") or DEFAULT_SOURCE_ROOT),
            )
        )
    return LocalConfig(project, mode, local_workers, tuple(hosts))


def _workers(path: Path, section: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{path}: [{section}] workers must be a non-negative integer")
    return value


# ---------------------------------------------------------------------------
# the mode
# ---------------------------------------------------------------------------

_MODE_LINE = re.compile(r"^\s*mode\s*=")


def read_mode(root: str | os.PathLike | None, cache_dir: str | os.PathLike | None) -> str:
    """The mode in force for the project at *root*; ``all`` when unset or
    the file is unreadable.

    With no cache directory a host's results would have nowhere to land:
    that case is ``local``.
    """
    if cache_dir is None:
        return "local"
    try:
        return load_local(root).mode
    except RuntimeError:
        return DEFAULT_MODE


def write_mode(root: str | os.PathLike, mode: str) -> None:
    """Set the ``mode`` line of the local file in *root*, creating the file
    if needed and leaving every other line as it was."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    p = Path(root) / LOCAL_FILE
    try:
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        lines = []
    new = f'mode = "{mode}"\n'
    for i, line in enumerate(lines):
        if _MODE_LINE.match(line):
            lines[i] = new
            break
    else:
        lines.insert(0, new)
    _atomic_write(p, "".join(lines).encode("utf-8"))


def capacities(
    mode: str, local: int, remote: dict[str, int], syncing: bool = False
) -> dict[str, int]:
    """How many tasks each host may run at once under *mode*.

    *remote* maps each host to its capacity, 0 until it is ready; *syncing*
    says whether any host is still syncing.  Every name is present in the
    result, at 0 where the mode excludes it, so a display can show what is
    switched off as well as what is on.

    Under ``remote`` this machine stays idle while a remote host is still on
    its way: the person who chose that mode wants their machine free, and a
    short batch would otherwise be over before the host arrived.
    """
    if mode == "local":
        return {**{name: 0 for name in remote}, "local": local}
    if mode == "all":
        return {**remote, "local": local}
    if mode == "remote":
        if any(remote.values()) or syncing:
            return {**remote, "local": 0}
        return {**remote, "local": local}
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# what a worker process is given
# ---------------------------------------------------------------------------

# What a process needs from the environment to start and to find its
# interpreter's own files; everything else stays with the main process.  No
# PYTHONPATH (imports must resolve through the source tree, or the check
# that they did proves nothing) and no credentials, which a @pure_local
# call keeps on the main process.
_WORKER_ENV = frozenset(
    {
        "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
        "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "WINDIR",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "USERNAME", "USER",
        "PYTHONHOME", "PYTHONUTF8", "PYTHONIOENCODING", "VIRTUAL_ENV",
    }
)


def worker_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() in _WORKER_ENV or k.upper().startswith("VALUEKIT_")
    }
