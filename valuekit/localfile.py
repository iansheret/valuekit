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

The *mode* is one line of this file; :mod:`valuekit.modes` says what each
mode means.  The monitor's keys and its ``--mode`` flag edit that line in
place and touch nothing else in the file.

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

from .modes import DEFAULT_MODE, MODES
from .store import _atomic_write

__all__ = [
    "LOCAL_FILE",
    "HostEntry",
    "LocalConfig",
    "load_local",
    "write_mode",
]

LOCAL_FILE = "valuekit.local.toml"
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
