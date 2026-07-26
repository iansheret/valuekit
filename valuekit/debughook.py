"""Debugger integration.

Caching stays active while a debugger is attached.  A cache hit is bypassed
only when a live breakpoint intersects the code of the function (or of
anything in its user-code dependency closure): setting a breakpoint in a
step, or in a helper it calls, makes that step execute; clearing the
breakpoint restores hits.  Forced runs never write to the store, so nothing
done in a debug session (evaluating expressions, modifying locals, dropping
frames) can enter the cache.

Supported debuggers: pydevd (PyCharm / VS Code's debugpy) and anything built
on ``bdb`` (pdb, ipdb).  The breakpoint tables of both are internal APIs,
so all access is defensive: if a debugger is detected but its table cannot
be read, breakpoints are assumed everywhere, which costs cache hits but
never skips a breakpoint.  An unknown trace function that is not a debugger
(coverage, profilers) never forces execution, so caching works normally
under those tools.

Overrides: every @pure function exposes ``.uncached`` (the raw function),
and ``VALUEKIT_ALWAYS_RUN=1`` forces execution globally.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

__all__ = ["breakpoints_force", "debugger_attached", "EVERYWHERE"]

# Sentinel: "assume a breakpoint on every line".
EVERYWHERE = object()


def _canon(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return path


def _pydevd_breakpoints():
    """Return {canonical_file: set(lines)} from pydevd, EVERYWHERE on failure,
    or None if pydevd is not active."""
    pydevd = sys.modules.get("pydevd")
    if pydevd is None:
        return None
    try:
        dbg = pydevd.get_global_debugger()
        if dbg is None:
            return None
        table = getattr(dbg, "breakpoints", None)
        if table is None:
            return EVERYWHERE
        out: dict[str, set[int]] = {}
        for fname, lines in dict(table).items():
            ls = set(lines.keys()) if isinstance(lines, dict) else set(lines)
            out.setdefault(_canon(str(fname)), set()).update(int(x) for x in ls)
        return out
    except Exception:
        return EVERYWHERE


def _bdb_breakpoints():
    """Return breakpoints from a bdb-based debugger via sys.gettrace(),
    EVERYWHERE on failure, or None if no bdb debugger is tracing."""
    tf = sys.gettrace()
    if tf is None:
        return None
    import bdb

    owner = getattr(tf, "__self__", None)
    if not isinstance(owner, bdb.Bdb):
        return None  # coverage / profiler / unknown tracer: never force
    try:
        return {
            _canon(str(f)): set(int(x) for x in ls)
            for f, ls in dict(owner.breaks).items()
        }
    except Exception:
        return EVERYWHERE


def debugger_attached() -> bool:
    """True when a known debugger (pydevd or anything bdb-based) is
    attached, whether or not any breakpoints are set.  Coverage tools and
    profilers are not debuggers and return False."""
    return _pydevd_breakpoints() is not None or _bdb_breakpoints() is not None


def _merge(a, b):
    if a is EVERYWHERE or b is EVERYWHERE:
        return EVERYWHERE
    if a is None:
        return b
    if b is None:
        return a
    out = {f: set(ls) for f, ls in a.items()}
    for f, ls in b.items():
        out.setdefault(f, set()).update(ls)
    return out


def breakpoints_force(spans: Iterable[tuple[str, int, int]]) -> bool:
    """True if the current debug state requires executing instead of hitting.

    *spans* are (filename, first_line, last_line) for every code object in a
    @pure function's dependency closure (collected at decoration time).
    """
    if os.environ.get("VALUEKIT_ALWAYS_RUN"):
        return True
    bps = _merge(_pydevd_breakpoints(), _bdb_breakpoints())
    if bps is None:
        return False
    if bps is EVERYWHERE:
        return True
    if not bps:
        return False
    for fname, lo, hi in spans:
        lines = bps.get(_canon(fname))
        if lines and any(lo <= ln <= hi for ln in lines):
            return True
    return False
