"""Getting the user's code to the machine that will run it.

A worker must run the code the driver meant, and the driver must not have to
remember to copy it there -- an edit loop that needs a manual sync step is an
edit loop nobody uses.  So the driver describes its project as a *manifest*,
the worker unpacks an immutable copy of it -- a *source tree* -- and imports
from that rather than from whatever happens to be on its own disk.

What gets sent is the user-code partition and nothing else, the same
boundary :func:`valuekit.codehash._classify` already draws: the project's own
files, never libraries.  A dependency is the environment's job on both
machines, exactly as numpy is -- and shipping a locally built extension would
be worse than useless anyway, since it is the wrong architecture as often as
not.  Compiled artefacts are therefore excluded outright rather than by
trusting the project's ignore rules.

A source tree is named by the manifest hash and never modified, so several
versions of a project coexist, a batch cannot have its source changed
underneath it, and re-running an unchanged tree costs one comparison.  It
lives under the cache directory, beside ``objects/`` and ``runs/``: the cache
directory is where valuekit writes, and nothing is written until one is named.

Nothing here trusts that the sync worked.  :mod:`valuekit.worker` audits what
it actually imported afterwards, and the fingerprint handshake checks the
result again -- because a source tree on ``sys.path`` can still lose to an
editable install's meta-path finder, and a silent wrong answer is the one
outcome worth any amount of machinery to avoid.
"""

from __future__ import annotations

import io
import os
import site
import subprocess
import sys
import sysconfig
import tarfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, Iterable

from .values import _frame, _new_hasher

__all__ = [
    "SyncError",
    "manifest",
    "manifest_hash",
    "sync_root",
    "pack_tree",
    "extract_tree",
    "import_roots",
    "user_span_files",
    "is_environment",
]

# Anything whose identity is a build rather than a source.  Excluded
# unconditionally: a project that commits its .so files should still not ship
# them to a machine that may not share this one's architecture.
_SKIP_SUFFIXES = tuple(EXTENSION_SUFFIXES) + (
    # Named explicitly as well: the manifest must be the same on every
    # platform, and EXTENSION_SUFFIXES lists only this platform's.
    ".so", ".pyd",
    ".o", ".a", ".obj", ".lib", ".dylib", ".dll", ".pyc", ".pyo",
)
_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", ".env",
        "build", "dist", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    }
)

MAX_BYTES = 256 << 20  # a project tree, not a data directory
MAX_FILES = 20_000


class SyncError(Exception):
    """The project cannot be described or shipped as it stands."""


# ---------------------------------------------------------------------------
# telling the project apart from the environment
# ---------------------------------------------------------------------------


def _env_prefixes() -> tuple[str, ...]:
    """Directories that hold the interpreter and installed packages.

    Built from sysconfig and site rather than by looking for "site-packages"
    in a path: that substring misses ``pip --target`` and vendored installs,
    and matches any project that happens to live under a directory of that
    name.
    """
    out: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib", "stdlib", "platstdlib"):
        p = paths.get(key)
        if p:
            out.add(os.path.realpath(p))
    for p in (sys.base_prefix, sys.base_exec_prefix):
        if p:
            out.add(os.path.realpath(p))
    try:
        for p in site.getsitepackages():
            out.add(os.path.realpath(p))
    except Exception:
        pass
    try:
        p = site.getusersitepackages()
        if isinstance(p, str):
            out.add(os.path.realpath(p))
    except Exception:
        pass
    return tuple(sorted(out))


_ENV_PREFIXES: tuple[str, ...] | None = None


def is_environment(path: str) -> bool:
    """Whether *path* belongs to the interpreter or an installed package."""
    global _ENV_PREFIXES
    if _ENV_PREFIXES is None:
        _ENV_PREFIXES = _env_prefixes()
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    return any(_under(real, prefix) for prefix in _ENV_PREFIXES)


def _under(path: str, root: str) -> bool:
    """Whether *path* is inside *root*, by path components not characters."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def user_span_files(spans: Iterable[tuple[str, int, int]]) -> list[str]:
    """The real user source files named by a fingerprint's spans.

    Spans record ``co_filename`` unmodified, so they carry synthetic names
    (``<string>`` for a dataclass's generated methods, or anything exec'd),
    the driver script itself, and genuine stdlib or site-packages paths -- a
    user class whose methods came from elsewhere drags those in.  Only what
    survives all three filters is a file worth syncing.
    """
    out: list[str] = []
    for entry in spans:
        name = entry[0]
        if not name or name.startswith("<") or not os.path.isabs(name):
            continue
        real = os.path.realpath(name)
        if real in out or not os.path.exists(real) or is_environment(real):
            continue
        out.append(real)
    return out


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def sync_root(fn: Any) -> str:
    """The project directory enclosing *fn*, by its nearest marker."""
    mod = sys.modules.get(getattr(fn, "__module__", "") or "")
    fname = getattr(mod, "__file__", None)
    if not fname:
        raise SyncError(
            f"cannot locate the project for {getattr(fn, '__qualname__', fn)!r}: "
            "its module has no file. Define it in a module, not a notebook or "
            "an exec'd string."
        )
    here = Path(os.path.realpath(fname)).parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return str(candidate)
    return str(here)


def _file_digest(path: str) -> str | None:
    """Content digest of a file, or None if it cannot be read.

    Deliberately not memoised on ``(mtime, size)``, the way an extension
    binary's digest is in :mod:`valuekit.codehash`.  That key fails in the
    dangerous direction here: a file whose content changes without moving
    its mtime or its size -- a same-size edit within one tick on a
    coarse-mtime filesystem such as HFS+, ext3 or exFAT -- would keep its old
    digest, leave the manifest hash unmoved, and let a worker reuse a
    source tree built from the previous content.  A remote quietly running stale code is
    the worst outcome this library has.

    The cost of not memoising is small for the same reason whole-tree
    transfer is affordable: the boundary rule keeps a project tree to its own
    source, with no libraries, vendored dependencies or build output.  A tree
    small enough to ship every time is small enough to hash every time.

    Reads through symlinks, so a source tree holds real files.
    """
    h = _new_hasher()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None  # staged-then-deleted, or vanished under us
    return h.hexdigest()


def _skip(rel: str) -> bool:
    if rel.endswith(_SKIP_SUFFIXES):
        return True
    return any(part in _SKIP_DIRS for part in Path(rel).parts)


def _git_files(root: str) -> list[str] | None:
    """Tracked plus untracked-not-ignored paths, or None if not a git tree.

    Untracked files are included deliberately: the helper you just wrote and
    have not ``git add``ed is the commonest thing to be editing.
    """
    try:
        out = subprocess.run(
            [
                "git", "-C", root, "ls-files", "-z", "--full-name",
                "--cached", "--others", "--exclude-standard",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [p for p in out.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def _walked_files(root: str) -> list[str]:
    """Everything under *root*, for a project that is not a git tree."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            found.append(os.path.relpath(full, root))
    return found


def manifest(root: str, exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
    """``(relpath, digest)`` for every file to ship, sorted.

    *exclude* names directories to leave out whatever the ignore rules say --
    valuekit's own cache above all, since a cache configured inside the
    project would otherwise be packed into the source tree that lives beside it.

    Missing files are dropped rather than raising: ``git ls-files --cached``
    reads the index, so a path staged and then deleted from the worktree is
    listed and cannot be opened, and there is no flag to exclude them.  The
    same tolerance covers a file that disappears while the manifest is built.
    """
    candidates = _git_files(root)
    if candidates is None:
        candidates = _walked_files(root)

    barred = []
    for d in exclude:
        real = os.path.realpath(d)
        if _under(real, os.path.realpath(root)):
            barred.append(os.path.relpath(real, root))

    entries: list[tuple[str, str]] = []
    total = 0
    biggest: list[tuple[int, str]] = []
    for rel in candidates:
        if _skip(rel) or any(
            rel == b or rel.startswith(b + os.sep) for b in barred
        ):
            continue
        full = os.path.join(root, rel)
        digest = _file_digest(full)
        if digest is None:
            continue  # staged-then-deleted, or vanished under us
        entries.append((rel, digest))
        try:
            size = os.stat(full).st_size
        except OSError:
            size = 0
        total += size
        biggest.append((size, rel))

    if total > MAX_BYTES or len(entries) > MAX_FILES:
        biggest.sort(reverse=True)
        worst = "\n  ".join(f"{s // 1024} KiB  {p}" for s, p in biggest[:10])
        raise SyncError(
            f"the project at {root} is {total // (1 << 20)} MiB over "
            f"{len(entries)} files, past the limit for shipping to a worker. "
            f"The largest entries are:\n  {worst}\n"
            "Data belongs outside the project tree, or behind .gitignore."
        )
    entries.sort()
    return entries


def manifest_hash(entries: list[tuple[str, str]]) -> str:
    """A name for exactly this set of files at exactly these contents."""
    h = _new_hasher()
    for rel, digest in entries:
        _frame(h, b"p", rel.encode("utf-8", "surrogateescape"))
        _frame(h, b"h", digest.encode("ascii"))
    return h.hexdigest()


def import_roots(root: str) -> list[str]:
    """The path entries under *root* that imports currently resolve through.

    Derived from what is loaded rather than assumed to be *root* itself, so a
    src-layout project (where ``import mypkg`` needs ``root/src``) and a tree
    with several roots both come out right.
    """
    roots: set[str] = set()
    real_root = os.path.realpath(root)
    for name, mod in list(sys.modules.items()):
        fname = getattr(mod, "__file__", None)
        if not fname or name.startswith("valuekit"):
            continue
        try:
            real = os.path.realpath(fname)
        except OSError:
            continue
        if not _under(real, real_root) or is_environment(real):
            continue
        # Walk up one directory per package component to reach the entry that
        # `name` was found on.
        depth = name.count(".")
        if os.path.basename(real) == "__init__.py":
            depth += 1
        entry = os.path.dirname(real)
        for _ in range(depth):
            entry = os.path.dirname(entry)
        if _under(entry, real_root):
            roots.add(os.path.relpath(entry, real_root))
    return sorted(roots) or ["."]


# ---------------------------------------------------------------------------
# shipping and materialising
# ---------------------------------------------------------------------------


def pack_tree(root: str, entries: list[tuple[str, str]]) -> bytes:
    """A deterministic tar of the manifest's files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, _ in entries:
            full = os.path.join(root, rel)
            try:
                st = os.stat(full)
                with open(full, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if st.st_mode & 0o100 else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def check_extraction_supported() -> str:
    """"" if tar can be extracted safely here, else why not."""
    if not hasattr(tarfile, "data_filter"):
        return (
            f"this worker runs Python {sys.version.split()[0]}, whose tarfile "
            "cannot filter extraction safely (added in 3.11.4). Upgrade the "
            "worker's interpreter."
        )
    return ""


def extract_tree(data: bytes, dest: str) -> None:
    """Extract a packed tree into *dest*, refusing anything that escapes it."""
    reason = check_extraction_supported()
    if reason:
        raise SyncError(reason)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        # Always explicit: the safe filter is only the default from 3.14.
        tar.extractall(dest, filter="data")
