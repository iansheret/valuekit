"""Where a batch runs: the hosts file and the mode file.

Two pieces of state, kept apart because they have different lifetimes.

The *hosts file* is configuration the user writes once: which machines can
take work, how to reach them, and how many workers each may run.  It is
TOML, located by ``$VALUEKIT_HOSTS``; with no file there are no remote
hosts.  ``[local]`` may cap this machine's workers; each ``[hosts.<name>]``
names an ssh target and the interpreter there that has valuekit installed::

    [local]
    workers = 8

    [hosts.mac]
    ssh = "ian@mac.local"
    python = "/Users/ian/.venvs/vk/bin/python"
    workers = 8                                  # omitted: the host's CPU count
    source_root = "~/.cache/valuekit/source"     # omitted: this default

The *mode* is a choice that changes from run to run and during one: one
word in ``<cache>/placement``.  ``local`` runs everything on this machine;
``remote`` runs as little here as possible, which means nothing here while
any host is reachable and everything here when none is; ``all`` uses every
reachable host and this machine at full capacity.  The file is absent by
default, which reads as ``local``, and is written by the monitor on a
keystroke or by ``python -m valuekit.monitor --mode``.  Deleting the cache
directory resets it, which is the safe direction.

The scheduler reads the mode each time it is about to start a task, so a
change takes effect for the next task started; tasks already running finish
where they are, and their results land in this machine's cache either way.
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


@dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    python: str
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
        for key in ("ssh", "python"):
            if not isinstance(h.get(key), str) or not h[key]:
                raise RuntimeError(f"{p}: [hosts.{name}] needs a string {key!r}")
        workers = h.get("workers")
        if workers is not None:
            workers = _workers(p, f"hosts.{name}", workers)
        hosts.append(
            Host(
                name=str(name),
                ssh=h["ssh"],
                python=h["python"],
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


def read_mode(cache_dir: str | os.PathLike | None) -> str:
    """The mode in force for *cache_dir*; ``local`` when unset or unreadable."""
    if cache_dir is None:
        return "local"
    try:
        mode = _mode_path(cache_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return "local"
    return mode if mode in MODES else "local"


def write_mode(cache_dir: str | os.PathLike, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    _atomic_write(_mode_path(cache_dir), f"{mode}\n".encode())


def capacities(mode: str, local: int, remote: dict[str, int]) -> dict[str, int]:
    """How many tasks each place may run at once under *mode*.

    *remote* maps each reachable host to its capacity.  Every name is
    present in the result, at 0 where the mode excludes it, so a display
    can show what is switched off as well as what is on.
    """
    if mode == "local":
        return {**{name: 0 for name in remote}, "local": local}
    if mode == "all":
        return {**remote, "local": local}
    if mode == "remote":
        if any(remote.values()):
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
