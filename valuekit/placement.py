"""Where a batch runs: the hosts file and the mode file.

Two pieces of state, kept apart because they have different lifetimes.

The *hosts file* is configuration the user writes once: which machines can
take work, how to reach them, and how many workers each may run.  It is
TOML, located by ``$VALUEKIT_HOSTS``; with no file there are no remote
hosts.  ``[local]`` may cap this machine's workers; each ``[hosts.<name>]``
names an ssh target::

    [local]
    workers = 8

    [hosts.mac]
    ssh = "ian@mac.local"
    python = "python3"                           # omitted: this; any Python 3 there
    workers = 8                                  # omitted: the host's CPU count
    source_root = "~/.cache/valuekit/source"     # omitted: this default

``python`` is only what starts the bootstrap (:mod:`valuekit.bootstrap`);
the interpreter that runs the project comes from the project's own lock
file, built on the host.  Nothing of the project's, valuekit included, has
to be installed there.  A Windows host has ``python`` rather than
``python3``.

The *mode* is a choice that changes from run to run and during one: one
word in ``<cache>/placement``.  ``all`` uses every reachable host and this
machine at full capacity; ``local`` runs everything on this machine;
``remote`` runs as little here as possible, which means nothing here while
any host is reachable or still preparing, and everything here when none is.
The file is absent by default, which reads as ``all``: a host in the hosts
file is there to be used, the way a core is, and needs no switching on.  It
is written by the monitor on a keystroke or by ``python -m valuekit.monitor
--mode``, and deleting the cache directory resets it.

The scheduler reads the mode each time it is about to start a task, so a
change takes effect for the next task started; tasks already running finish
where they are, and their results land in this machine's cache either way.
Preparing a host never holds a task back: this machine starts at once and
a host joins when it is ready.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .store import _atomic_write

__all__ = [
    "MODES",
    "Host",
    "Hosts",
    "load_hosts",
    "read_mode",
    "write_mode",
    "capacities",
    "worker_env",
]

MODES = ("local", "remote", "all")
DEFAULT_SOURCE_ROOT = "~/.cache/valuekit/source"
DEFAULT_PYTHON = "python3"


@dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    python: str  # any Python 3 on the host, to bootstrap with
    workers: int | None  # None: whatever the host reports
    source_root: str


@dataclass(frozen=True)
class Hosts:
    local: int
    hosts: tuple[Host, ...]


def load_hosts(path: str | os.PathLike | None = None) -> Hosts:
    """The hosts file at *path*, or at ``$VALUEKIT_HOSTS``, or none."""
    path = path or os.environ.get("VALUEKIT_HOSTS")
    default_local = os.cpu_count() or 1
    if not path:
        return Hosts(default_local, ())
    p = Path(os.path.expanduser(path))
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise RuntimeError(f"cannot read the hosts file {p}: {e}") from e

    local = data.get("local", {})
    if not isinstance(local, dict):
        raise RuntimeError(f"{p}: [local] must be a table")
    local_workers = _workers(p, "local", local.get("workers", default_local))

    hosts: list[Host] = []
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
            Host(
                name=str(name),
                ssh=h["ssh"],
                python=python,
                workers=workers,
                source_root=str(h.get("source_root") or DEFAULT_SOURCE_ROOT),
            )
        )
    return Hosts(local_workers, tuple(hosts))


def _workers(path: Path, section: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{path}: [{section}] workers must be a non-negative integer")
    return value


# ---------------------------------------------------------------------------
# the mode
# ---------------------------------------------------------------------------


def _mode_path(cache_dir: str | os.PathLike) -> Path:
    return Path(cache_dir) / "placement"


DEFAULT_MODE = "all"


def read_mode(cache_dir: str | os.PathLike | None) -> str:
    """The mode in force for *cache_dir*; ``all`` when unset or unreadable.

    With no cache directory there is no mode file and no run log, and a
    host's results would have nowhere to land: that case is ``local``.
    """
    if cache_dir is None:
        return "local"
    try:
        mode = _mode_path(cache_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_MODE
    return mode if mode in MODES else DEFAULT_MODE


def write_mode(cache_dir: str | os.PathLike, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    _atomic_write(_mode_path(cache_dir), f"{mode}\n".encode())


def capacities(
    mode: str, local: int, remote: dict[str, int], pending: bool = False
) -> dict[str, int]:
    """How many tasks each place may run at once under *mode*.

    *remote* maps each host to its capacity, 0 until it is ready; *pending*
    says whether any host is still preparing.  Every name is present in the
    result, at 0 where the mode excludes it, so a display can show what is
    switched off as well as what is on.

    Under ``remote`` this machine stays idle while a host is still on its
    way: the person who chose that mode wants their machine free, and a
    short batch would otherwise be over before the host arrived.
    """
    if mode == "local":
        return {**{name: 0 for name in remote}, "local": local}
    if mode == "all":
        return {**remote, "local": local}
    if mode == "remote":
        if any(remote.values()) or pending:
            return {**remote, "local": 0}
        return {**remote, "local": local}
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# what a worker process is given
# ---------------------------------------------------------------------------

# What a process needs from the environment to start and to find its
# interpreter's own files; everything else stays with the driver.  No
# PYTHONPATH (imports must resolve through the source tree, or the check
# that they did proves nothing) and no credentials, which a @pure_local
# call keeps on the driver.
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
