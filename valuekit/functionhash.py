"""The function hash: a hash of everything a function reaches by name.

A @pure function's function hash is a content hash of everything **reachable by
name** from its code: its own bytecode and constants, plus — recursively
through user code — every function, class, module, and immutable constant
its names resolve to.  The walk stops at boundaries:

* installed packages contribute ``pkg:<name>==<version>`` (upgrading the
  package invalidates; edits inside site-packages are invisible);
* native extensions whose version cannot describe them -- anything installed
  from a local directory, editable or not, and so rebuilt in place --
  contribute the hash of their binary on the main process;
* the standard library contributes ``std:<module>`` (the Python version is
  already part of the global salt);
* user modules referenced *as modules* (``mymod.helper()``) contribute a hash
  of the module's source file, and a package's submodules are followed
  through attribute access (``mypkg.sub.f()`` depends on sub.py, not only on
  the package's __init__.py);
* module-level constants that are immutable (numbers, strings, bytes,
  tuples, frozensets, read-only arrays) are content-hashed — they are part
  of the function's definition; mutable globals (lists, dicts, sets,
  writeable arrays) are deliberately untracked and silent: @pure is the
  caller's promise that they never change or never matter.

Names are resolved when the function hash is computed — at a @pure function's
first call, once its module is fully loaded — so definition order does not
matter and mutual recursion works.  The decorated function's own code
object is captured at decoration time, before a debugger patches its
bytecode.

The walk's product is the function's *reachable set*: its hash is the code
hash that names the function's call records, and its spans (filename,
first line, last line, one per user code object reached) are what the
debugger hook intersects live breakpoints against to decide when a cache
hit must be bypassed.  The Python major and minor version is a marker in
the hash, since bytecode differs between versions on identical source.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from importlib.machinery import EXTENSION_SUFFIXES
from typing import Any, Callable

import numpy as np

from .values import _frame, _new_hasher, digest, freeze, hash_update

__all__ = ["reachable_set", "ReachableSet", "PYTHON"]

_MISSING = object()

# The running interpreter's major and minor version: a marker in every code
# hash, and what main process and host compare before comparing hashes.
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


class ReachableSet:
    """Everything reachable by name from a function's code: its hash, the
    source spans of the user code objects in it, and the marker of each
    native extension in it (module name -> binary hash), which a worker on
    another machine is given rather than computing."""

    __slots__ = ("hash", "spans", "extensions")

    def __init__(self, hash: str, spans: list[tuple[str, int, int]], extensions: dict[str, str]):
        self.hash = hash
        self.spans = spans
        self.extensions = extensions


# ---------------------------------------------------------------------------
# module classification
# ---------------------------------------------------------------------------

_USER, _PKG, _STD = "user", "pkg", "std"


_dists: dict[str, Any] = {}  # top-level import name -> Distribution or None


def _distribution(top: str) -> Any:
    """Return the installed distribution providing top-level module *top*."""
    if top in _dists:
        return _dists[top]
    import importlib.metadata as md

    try:
        dist = md.distribution(top)
    except md.PackageNotFoundError:
        try:
            # The import name and the distribution name differ often enough
            # (sklearn/scikit-learn) to be worth the slower lookup.
            names = md.packages_distributions().get(top) or []
            dist = md.distribution(names[0]) if names else None
        except Exception:
            dist = None
    except Exception:
        dist = None
    _dists[top] = dist
    return dist


def _dist_version(top: str) -> str | None:
    dist = _distribution(top)
    try:
        return dist.version if dist is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# native extensions
# ---------------------------------------------------------------------------
#
# A compiled extension has no source to walk and no code object to hash.
# Where its version can stand for it -- a distribution installed from a
# released artefact changes only through a reinstall, which moves the
# version -- the version is its marker.  One built from a local directory,
# editable or not, is rebuilt in place under the same version, so its
# marker is the hash of its binary: the code that actually runs, which
# changes exactly when the build did.  Read once per build, memoised on the
# file's size and modification time.
#
# A worker never hashes its own binary, which is built on another machine
# and may differ byte for byte.  The main process sends the markers its walk
# met, one per extension module, and a worker substitutes them wherever its
# own walk meets those modules.  Results the worker computes are therefore
# keyed by the main process's build; the sync guarantees the worker's binary
# was built from the same sources.  An extension the main process never
# reached has no marker on the worker, so the hashes differ and the worker
# is refused rather than trusted.

_markers_here: dict[str, str] | None = None  # set on a worker; None on the main process
_walk = threading.local()  # .extensions: the markers the walk in progress has met
_binary_hashes: dict[str, tuple[tuple[int, int], str]] = {}  # path -> ((size, mtime_ns), hash)


def _is_extension_file(filename: str) -> bool:
    return filename.endswith(tuple(EXTENSION_SUFFIXES))


def _dist_dir(dist: Any) -> str | None:
    """The directory *dist* was installed from, or None if it was not.

    Such an install -- editable or not -- is rebuilt in place under the same
    version, so its version says nothing about its current contents.
    """
    try:
        text = dist.read_text("direct_url.json")
        info = json.loads(text) if text else {}
    except Exception:
        return None
    if "dir_info" not in info:
        return None
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    try:
        return os.path.realpath(url2pathname(urlparse(info["url"]).path))
    except Exception:
        return None


def _is_live_extension(filename: str, top: str) -> bool:
    """Report whether a file is an extension whose version cannot stand for
    its contents."""
    if not _is_extension_file(filename):
        return False
    dist = _distribution(top)
    return dist is None or _dist_dir(dist) is not None


def _binary_hash(filename: str) -> str:
    """The content hash of the file at *filename*, memoised on its size and
    modification time so a build is read once."""
    from . import sync

    try:
        st = os.stat(filename)
        key = (st.st_size, st.st_mtime_ns)
    except OSError:
        return "?"
    cached = _binary_hashes.get(filename)
    if cached is not None and cached[0] == key:
        return cached[1]
    h = sync._file_hash(filename) or "?"
    _binary_hashes[filename] = (key, h)
    return h


def _extension_hash(module_name: str | None, filename: str) -> str:
    """What stands for the live extension *module_name* at *filename*: on the
    main process its binary's hash; on a worker the marker the main process
    sent for it, or a value no marker can equal."""
    name = module_name or ""
    if _markers_here is not None:
        return _markers_here.get(name, "?not-reached-by-the-main-process")
    h = _binary_hash(filename)
    met = getattr(_walk, "extensions", None)
    if met is not None:
        met[name] = h
    return h


def _extension_marker(module_name: str | None) -> str | None:
    """Return the marker for *module_name* if it names a native extension.

    Objects a compiled module defines -- a nanobind function, a Cython class
    -- carry no Python code, so the module they came from stands for them.
    """
    mod = sys.modules.get(module_name or "")
    filename = getattr(mod, "__file__", None)
    if not filename or not _is_extension_file(filename):
        return None
    return _classify(module_name, filename)[1]


def _is_installed(filename: str, top: str) -> bool:
    """Whether a module file belongs to an installed package rather than to
    the user's project.

    By where it lives (the interpreter's own directories, as sysconfig and
    site report them) or by what claims it (a released distribution, wherever
    it was put).  Not by a ``site-packages`` substring: that misses ``pip
    --target`` and vendored installs, and matches any project that happens to
    live under a directory of that name.  A distribution installed from a
    local directory claims nothing here: its files are the user's, edited in
    place under a version that never moves.
    """
    from .sync import is_environment

    if is_environment(filename):
        return True
    dist = _distribution(top)
    return dist is not None and _dist_dir(dist) is None


def _classify(module_name: str | None, filename: str | None) -> tuple[str, str]:
    """Return (kind, marker) for a module: user / pkg / std."""
    top = (module_name or "").split(".")[0]
    if top == "builtins":
        return _STD, "std:builtins"
    if top == "__main__":
        return _USER, "__main__"
    if top == "valuekit":
        # This library is never user code, wherever it is installed from: a
        # user function naming ``log`` or ``ImmutableMap`` must not hash
        # their module-level state.  The store's format version, not a
        # marker here, says when a valuekit change invalidates caches.
        return _PKG, "pkg:valuekit"
    if top and top in sys.stdlib_module_names:
        return _STD, f"std:{top}"
    if filename is None and top:
        mod = sys.modules.get(module_name or "")
        filename = getattr(mod, "__file__", None)
    if filename and _is_live_extension(filename, top):
        # An extension rebuilt in place under a fixed version: its binary,
        # or on a worker the main process's, stands for it.
        return _PKG, f"ext:{module_name}={_extension_hash(module_name, filename)}"
    if filename and _is_installed(filename, top):
        ver = _dist_version(top)
        return _PKG, f"pkg:{top}=={ver or '?'}"
    if filename is None:
        if not top:
            # No module, no file: nothing to anchor to; treat opaquely.
            return _STD, "std:builtins"
        # C extension or namespace pkg with no file: treat as a package.
        ver = _dist_version(top)
        return _PKG, f"pkg:{top}=={ver or '?'}"
    return _USER, filename


def _code_end_line(code: types.CodeType) -> int:
    end = code.co_firstlineno
    try:
        for _, _, line in code.co_lines():
            if line is not None and line > end:
                end = line
    except Exception:
        pass
    return end


def _is_mutable_value(v: Any) -> bool:
    """Mutable-as-it-sits values are outside the purity contract: @pure is
    the caller's promise that they never change or never matter, so they are
    neither hashed nor invalidated on; a snapshot of something that can
    mutate would not reflect later changes."""
    if type(v) in (list, dict, set, bytearray):
        return True
    if isinstance(v, np.ndarray) and v.flags.writeable:
        return True  # set arr.flags.writeable = False to opt a constant in
    return False


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------


class _Walker:
    def __init__(self) -> None:
        self.h = _new_hasher()
        self.spans: list[tuple[str, int, int]] = []
        self.seen: set[int] = set()  # id() of code objects / classes / modules

    # -- helpers -----------------------------------------------------------

    def _mark(self, text: str) -> None:
        _frame(self.h, b"k", text.encode("utf-8", "replace"))

    def _try_digest(self, label: str, v: Any) -> bool:
        if _is_mutable_value(v):
            self._mark(f"untracked:{label}:{type(v).__name__}")
            return False
        try:
            _frame(self.h, b"v", label.encode() + b"=" + digest(v))
            return True
        except Exception:
            # Not hashable either (open handle, logger, RNG, ...): same deal.
            self._mark(f"untracked:{label}:{type(v).__name__}")
            return False

    def _add_value(self, label: str, v: Any) -> None:
        """Content-hash a plain value, falling back to the marker of the
        extension that defines it: a nanobind function or a Cython class has
        no other marker."""
        if self._try_digest(label, v):
            return
        marker = _extension_marker(getattr(v, "__module__", None))
        if marker is not None:
            self._mark(f"{label}:{marker}")

    def _add_dep(self, label: str, v: Any, co_names: tuple = ()) -> None:
        """A default/closure/extra dependency: functions, classes, and modules
        route through global classification (recursing into user code and
        collecting spans); plain values are content-hashed if immutable.

        *co_names* are the referencing function's names, used to follow
        attribute access into a module's submodules (see _add_user_module)."""
        if isinstance(
            v,
            (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type),
        ):
            self._add_global(label, v, co_names)
        else:
            self._add_value(label, v)

    # -- entry points --------------------------------------------------------

    def add_function(
        self, fn: types.FunctionType, code: types.CodeType | None = None
    ) -> None:
        # A @pure wrapper: walk the wrapped function like any other user
        # code.  Cycles (including mutual recursion between @pure functions)
        # are handled by the ordinary code-object seen-set.
        if getattr(fn, "_valuekit_pure", False):
            fn = fn.__wrapped__
        if code is None:
            code = getattr(fn, "__code__", None)
        if code is None:
            self._mark(f"nocode:{getattr(fn, '__qualname__', '?')}")
            return
        if id(code) in self.seen:
            self._mark("cycle")
            return
        # Runtime state that parameterises the function but lives outside
        # its bytecode.  A module arriving this way is attribute-accessed in
        # the body, so the body's names are what resolve its submodules.
        names = code.co_names
        for i, d in enumerate(fn.__defaults__ or ()):
            self._add_dep(f"default[{i}]", d, names)
        for k, d in (fn.__kwdefaults__ or {}).items():
            self._add_dep(f"kwdefault[{k}]", d, names)
        for i, cell in enumerate(fn.__closure__ or ()):
            try:
                self._add_dep(f"closure[{i}]", cell.cell_contents, names)
            except ValueError:  # empty cell
                self._mark(f"emptycell[{i}]")
        self.add_code(code, fn.__globals__)

    def add_code(self, code: types.CodeType, globals_: dict) -> None:
        if id(code) in self.seen:
            self._mark("cycle")
            return
        self.seen.add(id(code))
        self.spans.append((code.co_filename, code.co_firstlineno, _code_end_line(code)))

        _frame(self.h, b"C", code.co_code)
        self._mark("names:" + ",".join(code.co_names))

        # Constants: nested code objects (lambdas, nested defs, comprehensions)
        # recurse with the same globals; plain constants are content-hashed.
        for c in code.co_consts:
            if isinstance(c, types.CodeType):
                self.add_code(c, globals_)
            elif c is not None:
                self._try_digest("const", c)

        # Referenced globals: resolve each name and classify what it points to.
        builtins_ = globals_.get("__builtins__", {})
        if isinstance(builtins_, types.ModuleType):
            builtins_ = vars(builtins_)
        for name in code.co_names:
            obj = globals_.get(name, _MISSING)
            if obj is _MISSING:
                obj = builtins_.get(name, _MISSING)
            if obj is _MISSING:
                # Most commonly an attribute name (LOAD_ATTR shares co_names);
                # nothing to resolve at module scope.  Such a name may still
                # be a submodule of a module resolved here, which is why
                # co_names is handed to _add_global below.
                continue
            self._add_global(name, obj, code.co_names)

    # -- classification of resolved globals ----------------------------------

    def _add_global(self, name: str, obj: Any, co_names: tuple = ()) -> None:
        if isinstance(obj, types.FunctionType):
            if getattr(obj, "_valuekit_pure", False):
                self._mark(f"fn:{name}")
                self.add_function(obj)
                return
            kind, marker = _classify(
                getattr(obj, "__module__", None), obj.__code__.co_filename
            )
            if kind == _USER:
                self._mark(f"fn:{name}")
                self.add_function(obj)
            else:
                self._mark(f"fn:{name}:{marker}")
        elif isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            _, marker = _classify(getattr(obj, "__module__", None), None)
            self._mark(f"cfn:{name}:{marker}")
        elif isinstance(obj, types.ModuleType):
            kind, marker = _classify(obj.__name__, getattr(obj, "__file__", None))
            if kind == _USER:
                self._add_user_module(name, obj, co_names)
            else:
                self._mark(f"mod:{name}:{marker}")
        elif isinstance(obj, type):
            mod = sys.modules.get(getattr(obj, "__module__", "") or "")
            kind, marker = _classify(
                getattr(obj, "__module__", None), getattr(mod, "__file__", None)
            )
            if kind == _USER:
                self._add_user_class(name, obj)
            else:
                self._mark(f"cls:{name}:{marker}")
        else:
            # Module-level constant / object: content-hash if immutable.
            self._add_value(f"global[{name}]", obj)

    def _add_user_module(
        self, name: str, mod: types.ModuleType, co_names: tuple = ()
    ) -> None:
        if id(mod) in self.seen:
            return
        self.seen.add(id(mod))
        fname = getattr(mod, "__file__", None)
        try:
            with open(fname, "rb") as f:  # type: ignore[arg-type]
                src = f.read()
            _frame(self.h, b"m", name.encode() + b"=" + src)
            self.spans.append((fname, 1, 1_000_000_000))  # whole-file span
        except Exception:
            self._mark(f"opaque-module:{name}")

        # A package's source file is only its __init__.py, so stopping here
        # would leave ``mypkg.sub.f()`` depending on nothing that sub.py says:
        # ``sub`` and ``f`` are attribute names, which resolve to nothing at
        # module scope and so never reach add_code's globals lookup.  The
        # referencing code object's names are the candidate attributes; the
        # seen-set makes mutually-importing packages terminate.
        for attr in co_names:
            try:
                sub = getattr(mod, attr, None)
            except Exception:
                continue  # a module-level __getattr__ that raises
            if not isinstance(sub, types.ModuleType):
                continue
            # Classify each submodule in its own right: a package may contain
            # a compiled extension, which is identified by its marker rather
            # than read as source.
            kind, marker = _classify(sub.__name__, getattr(sub, "__file__", None))
            if kind == _USER:
                self._add_user_module(f"{name}.{attr}", sub, co_names)
            else:
                self._mark(f"mod:{name}.{attr}:{marker}")

    def _add_user_class(self, name: str, cls: type) -> None:
        if id(cls) in self.seen:
            return
        self.seen.add(id(cls))
        self._mark(f"cls:{name}:{cls.__module__}.{cls.__qualname__}")
        for k, v in vars(cls).items():
            if isinstance(v, types.FunctionType):
                self.add_function(v)
            elif isinstance(v, (staticmethod, classmethod)):
                self.add_function(v.__func__)
            elif isinstance(v, property):
                for f in (v.fget, v.fset, v.fdel):
                    if isinstance(f, types.FunctionType):
                        self.add_function(f)


def reachable_set(fn: Callable, *, code: types.CodeType | None = None) -> ReachableSet:
    """Walk everything reachable by name from *fn*'s user code.

    *code* optionally overrides the function's own code object (used by
    @pure, which captures it at decoration time, before any debugger patches
    bytecode).
    """
    # The extensions met are collected for the whole walk; a walk nested in
    # another (a function hashed as a value) adds to the outer one's.
    outer = getattr(_walk, "extensions", None)
    if outer is None:
        _walk.extensions = {}
    try:
        w = _Walker()
        w._mark(f"python:{PYTHON}")
        w.add_function(fn, code=code)  # type: ignore[arg-type]
        extensions = dict(_walk.extensions)
    finally:
        if outer is None:
            _walk.extensions = None
    return ReachableSet(w.h.hexdigest(), w.spans, extensions)


# ---------------------------------------------------------------------------
# Registry wiring: functions as *values*
# ---------------------------------------------------------------------------
#
# A function passed as an argument to a @pure function is hashed by its code
# function_hash (including defaults and captured closure values), so lambdas
# work as parameters. Functions are treated as immutable for freezing
# purposes. They have no serialiser: a function may be an input, but cannot
# appear inside a cached return value.

freeze.register(types.FunctionType, lambda v: v)
freeze.register(types.BuiltinFunctionType, lambda v: v)


@hash_update.register
def _h_function(v: types.FunctionType, h: Any) -> None:
    _frame(h, b"L", reachable_set(v).hash.encode("ascii"))


@hash_update.register
def _h_builtin_function(v: types.BuiltinFunctionType, h: Any) -> None:
    _, marker = _classify(getattr(v, "__module__", None), None)
    _frame(h, b"L", f"{marker}:{v.__qualname__}".encode())
