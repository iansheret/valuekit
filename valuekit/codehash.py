"""Recursive code hashing.

A @pure function's identity is a content hash of everything **reachable by
name** from its code: its own bytecode and constants, plus — recursively
through user code — every function, class, module, and immutable constant
its names resolve to.  The walk stops at boundaries:

* installed packages contribute ``pkg:<name>==<version>`` (upgrading the
  package invalidates; edits inside site-packages are invisible);
* the standard library contributes ``std:<module>`` (the Python version is
  already part of the global salt);
* user modules referenced *as modules* (``mymod.helper()``) contribute a hash
  of the module's source file;
* module-level constants that are immutable (numbers, strings, bytes,
  tuples, frozensets, read-only arrays) are content-hashed — they are part
  of the function's definition; mutable globals (lists, dicts, sets,
  writeable arrays) are deliberately untracked and silent: @pure is the
  caller's promise that they never change or never matter.

Names are resolved when the fingerprint is computed — at a @pure function's
first call, once its module is fully loaded — so definition order does not
matter and mutual recursion works.  The decorated function's own code
object is captured at decoration time, before a debugger patches its
bytecode.

The same walk collects (filename, first_line, last_line) *spans* for every
user code object in the closure; the debugger hook intersects live
breakpoints against these to decide when a cache hit must be bypassed.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

import numpy as np

from .values import _frame, _new_hasher, digest, freeze, hash_update

__all__ = ["function_fingerprint"]

_MISSING = object()


def _unit_digest(code: types.CodeType) -> str:
    """Content digest of a single code object — the identity of one function
    body, independent of its name, file, or line number.  Used by the
    dependency index that makes clear_cache(fn) reach fn's callers."""
    h = _new_hasher()
    _frame(h, b"U", code.co_code)
    _frame(h, b"n", ",".join(code.co_names).encode())
    for c in code.co_consts:
        if not isinstance(c, types.CodeType):
            _frame(h, b"c", repr(c).encode("utf-8", "replace"))
    return h.hexdigest()


def _module_unit(mod: types.ModuleType) -> str | None:
    """Content digest of a user module's source file, or None."""
    fname = getattr(mod, "__file__", None)
    if not fname:
        return None
    try:
        with open(fname, "rb") as f:
            src = f.read()
    except OSError:
        return None
    h = _new_hasher()
    _frame(h, b"M", src)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# module classification
# ---------------------------------------------------------------------------

_USER, _PKG, _STD = "user", "pkg", "std"


def _dist_version(top: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(top)
    except md.PackageNotFoundError:
        try:
            dists = md.packages_distributions().get(top) or []
            return md.version(dists[0]) if dists else None
        except Exception:
            return None
    except Exception:
        return None


def _classify(module_name: str | None, filename: str | None) -> tuple[str, str]:
    """Return (kind, marker) for a module: user / pkg / std."""
    top = (module_name or "").split(".")[0]
    if top == "builtins":
        return _STD, "std:builtins"
    if top == "__main__":
        return _USER, "__main__"
    if top and top in sys.stdlib_module_names:
        return _STD, f"std:{top}"
    if filename is None and top:
        mod = sys.modules.get(module_name or "")
        filename = getattr(mod, "__file__", None)
    if filename and ("site-packages" in filename or "dist-packages" in filename):
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
        self.units: set[str] = set()  # closure membership, for clear_cache(fn)
        self.seen: set[int] = set()  # id() of code objects / classes / modules

    # -- helpers -----------------------------------------------------------

    def _mark(self, text: str) -> None:
        _frame(self.h, b"k", text.encode("utf-8", "replace"))

    def _try_digest(self, label: str, v: Any) -> None:
        if _is_mutable_value(v):
            self._mark(f"untracked:{label}:{type(v).__name__}")
            return
        try:
            _frame(self.h, b"v", label.encode() + b"=" + digest(v))
        except Exception:
            # Not hashable either (open handle, logger, RNG, ...): same deal.
            self._mark(f"untracked:{label}:{type(v).__name__}")

    def _add_dep(self, label: str, v: Any) -> None:
        """A default/closure/extra dependency: functions, classes, and modules
        route through global classification (recursing into user code and
        collecting spans); plain values are content-hashed if immutable."""
        if isinstance(
            v,
            (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type),
        ):
            self._add_global(label, v)
        else:
            self._try_digest(label, v)

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
        # its bytecode:
        for i, d in enumerate(fn.__defaults__ or ()):
            self._add_dep(f"default[{i}]", d)
        for k, d in (fn.__kwdefaults__ or {}).items():
            self._add_dep(f"kwdefault[{k}]", d)
        for i, cell in enumerate(fn.__closure__ or ()):
            try:
                self._add_dep(f"closure[{i}]", cell.cell_contents)
            except ValueError:  # empty cell
                self._mark(f"emptycell[{i}]")
        self.add_code(code, fn.__globals__)

    def add_code(self, code: types.CodeType, globals_: dict) -> None:
        if id(code) in self.seen:
            self._mark("cycle")
            return
        self.seen.add(id(code))
        self.spans.append((code.co_filename, code.co_firstlineno, _code_end_line(code)))
        self.units.add(_unit_digest(code))

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
                # nothing to resolve at module scope.
                continue
            self._add_global(name, obj)

    # -- classification of resolved globals ----------------------------------

    def _add_global(self, name: str, obj: Any) -> None:
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
            kind, marker = _classify(getattr(obj, "__module__", None), None)
            self._mark(f"cfn:{name}:{marker}")
        elif isinstance(obj, types.ModuleType):
            kind, marker = _classify(obj.__name__, getattr(obj, "__file__", None))
            if kind == _USER:
                self._add_user_module(name, obj)
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
            self._try_digest(f"global[{name}]", obj)

    def _add_user_module(self, name: str, mod: types.ModuleType) -> None:
        if id(mod) in self.seen:
            return
        self.seen.add(id(mod))
        fname = getattr(mod, "__file__", None)
        try:
            with open(fname, "rb") as f:  # type: ignore[arg-type]
                src = f.read()
            _frame(self.h, b"m", name.encode() + b"=" + src)
            self.spans.append((fname, 1, 1_000_000_000))  # whole-file span
            u = _module_unit(mod)
            if u:
                self.units.add(u)
        except Exception:
            self._mark(f"opaque-module:{name}")

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


def function_fingerprint(
    fn: Callable,
    *,
    code: types.CodeType | None = None,
) -> tuple[str, list[tuple[str, int, int]], list[str]]:
    """Hash *fn* and everything reachable by name from its user code.

    Returns ``(hex_hash, code_spans, unit_digests)``: the units name every
    code object (and user-module source) in the walked closure, and feed the
    on-disk dependency index that lets clear_cache(fn) reach fn's callers.
    *code* optionally overrides the function's own code object (used by
    @pure, which captures it at decoration time, before any debugger patches
    bytecode).
    """
    w = _Walker()
    w.add_function(fn, code=code)  # type: ignore[arg-type]
    return w.h.hexdigest(), w.spans, sorted(w.units)


# ---------------------------------------------------------------------------
# Registry wiring: functions as *values*
# ---------------------------------------------------------------------------
#
# A function passed as an argument to a @pure function is hashed by its code
# fingerprint (including defaults and captured closure values), so lambdas
# work as parameters. Functions are treated as immutable for freezing
# purposes. They have no serialiser: a function may be an input, but cannot
# appear inside a cached return value.

freeze.register(types.FunctionType, lambda v: v)
freeze.register(types.BuiltinFunctionType, lambda v: v)


@hash_update.register
def _h_function(v: types.FunctionType, h: Any) -> None:
    fp, _, _ = function_fingerprint(v)
    _frame(h, b"L", fp.encode("ascii"))


@hash_update.register
def _h_builtin_function(v: types.BuiltinFunctionType, h: Any) -> None:
    kind, marker = _classify(getattr(v, "__module__", None), None)
    _frame(h, b"L", f"{marker}:{v.__qualname__}".encode())
