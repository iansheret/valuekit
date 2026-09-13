"""The user's project: where it is, which files it consists of, and how it
is packed for a host.

A worker must run the code the main process meant, and the main process
must not have to copy it there by hand.  So the main process describes
its project as a *manifest*, the worker unpacks a copy of it -- a *source
tree* -- and imports from that rather than from whatever happens to be on
its own disk.

What gets sent is the user-code partition and nothing else, the same
boundary :func:`valuekit.functionhash._classify` already draws: the project's own
files, never libraries.  A dependency is the environment's job on both
machines, exactly as numpy is -- and shipping a locally built extension would
be worse than useless anyway, since it is the wrong architecture as often as
not.  Compiled artefacts are therefore excluded outright rather than by
trusting the project's ignore rules.

A host keeps one source tree per project, updated in place from the
manifest's difference, so a build directory there persists across edits;
the manifest hash (the *project hash*) says whether the host's copy is current.
On this machine the tree lives under the store directory, beside
``objects/`` and ``events/``: the store directory is where valuekit writes,
and nothing is written until one is named.

What happens to the tree on the host -- unpacking it, building the
environment the project's lock file describes, starting the host process
inside it -- is :mod:`valuekit.bootstrap`'s.  Nothing here trusts that any
of it worked: :mod:`valuekit.worker` checks that what it actually imported came from the tree
afterwards, and the function-hash handshake checks the result again --
because a source tree on ``sys.path`` can still lose to an editable
install's meta-path finder, and a silent wrong answer is the one outcome
worth any amount of machinery to avoid.
"""

from __future__ import annotations

import io
import os
import site
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import warnings
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, Iterable

from .localfile import LOCAL_FILE
from .values import _frame, _new_hasher

__all__ = [
    "ProjectError",
    "Project",
    "manifest",
    "manifest_hash",
    "project_hash",
    "find_root",
    "project_root",
    "pack_tree",
    "import_roots",
    "user_span_files",
    "is_environment",
]

# Anything that is a build product rather than a source.  Excluded
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

# Never shipped and never hashed: the local file says where a computation
# runs, which must not be able to affect a result.
_SKIP_FILES = frozenset({LOCAL_FILE})

MAX_BYTES = 256 << 20  # a project tree, not a data directory
MAX_FILES = 20_000


class ProjectError(Exception):
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
    """The real user source files named by a reachable set's spans.

    Spans record ``co_filename`` unmodified, so they carry synthetic names
    (``<string>`` for a dataclass's generated methods, or anything exec'd),
    the main script itself, and genuine stdlib or site-packages paths -- a
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


def find_root(path: str) -> str | None:
    """The project directory enclosing *path*, by its nearest marker, or None."""
    here = Path(os.path.realpath(path)).parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return str(candidate)
    return None


def project_root(fn: Any) -> str:
    """The project directory enclosing *fn*: its nearest marker, else its directory."""
    mod = sys.modules.get(getattr(fn, "__module__", "") or "")
    fname = getattr(mod, "__file__", None)
    if not fname:
        raise ProjectError(
            f"cannot locate the project for {getattr(fn, '__qualname__', fn)!r}: "
            "its module has no file. Define it in a module, not a notebook or "
            "an exec'd string."
        )
    return find_root(fname) or str(Path(os.path.realpath(fname)).parent)


def _store_dirs() -> list[str]:
    """The current store's directory, if it has one: never part of a tree.

    A cache configured inside the project would otherwise be packed into the
    source tree that lives beside it, and would move the tree's hash every
    time a value was written.
    """
    from .pure import _current_store
    from .store import LocalStore

    store = _current_store()
    return [str(store.root)] if isinstance(store, LocalStore) else []


def project_hash(root: str) -> str:
    """The hash of the project at *root*: its manifest hash, right now.

    Not memoised across calls: the manifest is what says whether the tree
    changed, so a remembered answer is the one thing it must not be.  A
    caller that needs it repeatedly within one operation keeps it for that
    operation (the walk does).
    """
    return manifest_hash(manifest(root, exclude=_store_dirs()))


class Project:
    """The project a function belongs to, as it would be shipped.

    Built once per batch and shared by every host: the manifest walk reads
    every file in the tree, and the answer is the same for all of them.
    ``project_hash`` names exactly this set of files at exactly these contents;
    it is the name of the source tree on every host, and the hash that
    stands for a native extension built from it (see :mod:`valuekit.functionhash`).
    """

    def __init__(self, fn: Any, name: str | None = None):
        self.fn = fn
        self.root = project_root(fn)
        self.name = project_name(self.root, name)
        self.entries = manifest(self.root, exclude=_store_dirs())
        self.project_hash = manifest_hash(self.entries)
        self.roots = import_roots(self.root)
        _warn_if_tracked(self.root)

    def pack(self, entries: list[tuple[str, str]]) -> bytes:
        """A tarball of *entries*, a subset of the manifest."""
        return pack_tree(self.root, entries)

    def refusal(self) -> str:
        """Why no host could take this project, or "".

        Refused here, before anything is sent: a tree without a lock file
        valuekit knows, since no host could build its environment; and a
        dependency in a sibling checkout the tree never contained, which
        would surface on the host as "cannot import X", with nothing to say
        that X lives somewhere the main process never offered to send.
        """
        from . import bootstrap
        from .functionhash import reachable_set

        if bootstrap.lock_tool(rel for rel, _ in self.entries) is None:
            known = ", ".join(bootstrap.KNOWN_LOCKS)
            return (
                f"the project at {self.root} has no lock file valuekit knows how "
                f"to use ({known}), so a host could not build its environment. "
                "Lock the project's dependencies with one of those tools."
            )
        try:
            spans = reachable_set(self.fn).spans
        except Exception:
            return ""
        root = os.path.realpath(self.root)
        outside = [f for f in user_span_files(spans) if not _under(f, root)]
        if outside:
            listed = "\n  ".join(sorted(outside))
            return (
                f"{getattr(self.fn, '__qualname__', self.fn)} depends on user "
                f"code outside its project at {root}, which cannot be sent to a "
                f"worker:\n  {listed}\n"
                "Move it into the project, or install it as a package so both "
                "machines resolve it the same way."
            )
        return ""


def project_name(root: str, override: str | None = None) -> str:
    """The name of the project's directory on a host: *override* (the local
    file's ``project``), else ``[project].name`` from ``pyproject.toml``,
    else the root directory's name.  Made safe for a directory name."""
    from .store import dirname_for

    name = override
    if not name:
        try:
            with open(os.path.join(root, "pyproject.toml"), "rb") as f:
                name = tomllib.load(f).get("project", {}).get("name")
        except (OSError, tomllib.TOMLDecodeError, AttributeError):
            name = None
    if not isinstance(name, str) or not name:
        name = os.path.basename(os.path.realpath(root))
    return dirname_for(name)


_warned: set[str] = set()


def _warn_if_tracked(root: str) -> None:
    """Warn once per project if git tracks the local file, which is meant to
    be ignored: it holds this checkout's machines, not the project's."""
    if root in _warned:
        return
    _warned.add(root)
    for name in _SKIP_FILES:
        try:
            out = subprocess.run(
                ["git", "-C", root, "ls-files", "--cached", "--", name],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return
        if out.returncode == 0 and out.stdout.strip():
            warnings.warn(
                f"{name} in {root} is tracked by git; it is per checkout and "
                "should be in .gitignore",
                stacklevel=3,
            )


def _file_hash(path: str) -> str | None:
    """Content hash of a file, or None if it cannot be read.

    Deliberately not memoised on ``(mtime, size)``, the way an extension
    binary's hash is in :mod:`valuekit.functionhash`.  That key fails in the
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
    if rel.endswith(_SKIP_SUFFIXES) or os.path.basename(rel) in _SKIP_FILES:
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
    """``(relpath, hash)`` for every file to ship, sorted.

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
        h = _file_hash(full)
        if h is None:
            continue  # staged-then-deleted, or vanished under us
        entries.append((rel, h))
        try:
            size = os.stat(full).st_size
        except OSError:
            size = 0
        total += size
        biggest.append((size, rel))

    if total > MAX_BYTES or len(entries) > MAX_FILES:
        biggest.sort(reverse=True)
        worst = "\n  ".join(f"{s // 1024} KiB  {p}" for s, p in biggest[:10])
        raise ProjectError(
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
    for rel, file_hash in entries:
        _frame(h, b"p", rel.encode("utf-8", "surrogateescape"))
        _frame(h, b"h", file_hash.encode("ascii"))
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
            info = tarfile.TarInfo(rel.replace(os.sep, "/"))
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if st.st_mode & 0o100 else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
