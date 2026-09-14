"""The project's build step, run when its build inputs changed.

A project with a native extension is installed by its lock tool and
rebuilt by its lock tool (for uv, ``uv sync --reinstall-package <name>``),
which runs the project's build backend again.  Nothing rebuilds on
import: a worker imports, and only imports.  On a host the sync runs the
build step when the files it sent include a build input.  On this machine
:func:`build` does the same, for a script that calls it before importing
the project::

    import valuekit
    valuekit.build()          # rebuilds the project if a build input changed
    import mypipeline         # imports the current binary

What counts as a build input is decided once, in
:func:`valuekit.bootstrap.is_build_input`: the project's
``[tool.valuekit] build-inputs`` list, or every file that is not a Python
source.  The hash of the build inputs the environment was last built
from is recorded beside the interpreter, since the build belongs to the
environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import bootstrap
from .project import (
    ProjectError,
    _store_dirs,
    build_inputs,
    distribution_name,
    find_root,
    manifest,
    manifest_hash,
)
from .store import _atomic_write

__all__ = ["build"]

_RECORD = "valuekit-built.json"  # under sys.prefix: {project root: build inputs hash}
_TAIL = 4000  # bytes of the tool's output shown when it fails


def build(path: str | os.PathLike | None = None) -> bool:
    """Run the project's build step if its build inputs changed since the
    last one; return whether it ran.

    The project is the one enclosing *path*, else the calling script.
    Call it before importing the project: an extension already imported
    cannot be replaced.  Raises :class:`ProjectError` when there is no
    project, no lock file valuekit recognises, or no ``[project] name``,
    and :class:`RuntimeError` when the build step fails.
    """
    if path is None:
        caller = sys._getframe(1).f_globals.get("__file__")
        path = caller if caller else os.getcwd()
    path = os.fspath(path)
    # find_root starts at the parent of the path it is given: a directory
    # is searched from itself by naming a file in it.
    root = find_root(os.path.join(path, "pyproject.toml") if os.path.isdir(path) else path)
    if root is None:
        raise ProjectError(f"no project (pyproject.toml or .git) encloses {path}")
    lock = bootstrap.lock_tool(os.listdir(root))
    if lock is None:
        raise ProjectError(
            f"the project at {root} has no lock file valuekit recognises "
            f"({', '.join(bootstrap.KNOWN_LOCKS)})"
        )
    dist = distribution_name(root)
    if not dist:
        raise ProjectError(f"{os.path.join(root, 'pyproject.toml')} has no [project] name")
    patterns = build_inputs(root)
    entries = [(rel, h) for rel, h in manifest(root, exclude=_store_dirs())
               if bootstrap.is_build_input(rel, patterns)]
    inputs_hash = manifest_hash(entries)

    record_path = Path(sys.prefix) / _RECORD
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    key = os.path.realpath(root)
    if record.get(key) == inputs_hash:
        return False

    row = bootstrap._TOOLS[lock]
    exe = bootstrap._find(row["tool"], row["search"])
    if exe is None:
        raise RuntimeError(f"{row['tool']} is not on the PATH")
    cmd = [exe] + [a.format(python=sys.executable, dist=dist) for a in row["rebuild"]]
    print(f"valuekit.build: build inputs of {dist} changed; running {' '.join(cmd)}", file=sys.stderr)
    p = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        tail = p.stdout[-_TAIL:].decode("utf-8", "replace").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed in {root} (exit {p.returncode}):\n{tail}")
    record[key] = inputs_hash
    _atomic_write(record_path, json.dumps(record, indent=1).encode("utf-8"))
    return True
