"""Remove what the current code can no longer reach.

``python -m valuekit.sweep mypipeline.steps mypipeline.batches``

A trace belongs to a fingerprint, and a fingerprint belongs to code.  When
the code changes, its old traces are never consulted again; they only
occupy disk.  This command imports the named modules, takes the fingerprint
of every ``@pure`` and ``@pure_local`` function defined in them, and deletes
every trace and batch record whose fingerprint is not among them, then
every object that no remaining trace or batch names.

Retention is by code version, never by age.  A result from a year ago whose
function has not changed is as current as one from this morning, and is
kept.  What cannot be told apart is a result for an input nobody wants any
more: it looks exactly like one somebody does, so it stays.

The modules named must be every module that defines a memoised function
whose results are wanted.  A function that is not imported here reads as
gone, and its traces go with it -- the worst case is recomputation, as with
every deletion in this library.  The cache directory is the one configured
in those modules if they configure one, else ``--cache`` or ``$VALUEKIT_CACHE``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

from .batches import batches_dir
from .codec import children
from .store import LocalStore

__all__ = ["sweep", "main"]


def live_keys(modules: list[str]) -> set[str]:
    """The fingerprint keys of every memoised function in *modules*."""
    keys: set[str] = set()
    for name in modules:
        mod = importlib.import_module(name)
        for obj in list(vars(mod).values()):
            if getattr(obj, "_valuekit_pure", False):
                keys.add(obj._valuekit_identity()[0])
    return keys


def sweep(cache_dir: str | os.PathLike, modules: list[str], dry_run: bool = False) -> dict:
    """Delete traces, batches and objects the functions in *modules* cannot
    reach.  Returns counts of what was (or would be) removed."""
    store = LocalStore(cache_dir)
    keys = live_keys(modules)
    counts = {"functions": len(keys), "traces": 0, "batches": 0, "objects": 0}

    def remove(path: Path) -> None:
        if dry_run:
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass

    # -- traces of functions that no longer exist in this form ------------
    reachable: set[str] = set()
    for entry in list(store.traces.iterdir()):
        key = entry.name[:-5] if entry.name.endswith(".deps") else entry.name
        if key in keys:
            if entry.is_dir():
                for h, trace in store.get_traces(key):
                    reachable.add(trace["result"])
                    reachable.update(h for _, h in trace.get("logs", []))
            continue
        if entry.is_dir():
            counts["traces"] += sum(1 for _ in entry.glob("*.json"))
        remove(entry)

    # -- batches recorded under those fingerprints --------------------------
    bdir = batches_dir(store)
    if bdir.exists():
        for name_dir in list(bdir.iterdir()):
            if not name_dir.is_dir():
                continue
            kept = []
            for run in list(name_dir.iterdir()):
                if not run.is_dir():
                    continue
                try:
                    header = json.loads((run / "header.json").read_bytes())
                except (OSError, ValueError):
                    header = {}
                if header.get("fn_key") in keys:
                    kept.append(run.name)
                    reachable.update(h for h in header.get("inputs", []) if h)
                else:
                    counts["batches"] += 1
                    remove(run)
            try:
                latest = (name_dir / "latest").read_text().strip()
            except OSError:
                latest = ""
            if not dry_run and latest and latest not in kept:
                remove(name_dir / "latest")
            if not dry_run and not kept:
                remove(name_dir)

    # -- objects nothing above names, transitively ----------------------------
    queue = list(reachable)
    while queue:
        h = queue.pop()
        p = store._find(h)
        if p is None or p.suffix != ".bin":
            continue
        try:
            kids = children(p.read_bytes())
        except (OSError, ValueError):
            continue
        for k in kids:
            if k not in reachable:
                reachable.add(k)
                queue.append(k)
    for p in list(store.objects.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.stem not in reachable:
            counts["objects"] += 1
            remove(p)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m valuekit.sweep",
        description="Delete traces, batches and objects the current code cannot reach.",
    )
    ap.add_argument("modules", nargs="+", help="modules defining the memoised functions")
    ap.add_argument("--cache", help="cache directory (default: $VALUEKIT_CACHE)")
    ap.add_argument("--dry-run", action="store_true", help="report only")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    try:
        keys = live_keys(args.modules)
    except Exception as e:
        print(f"cannot import: {e}", file=sys.stderr)
        return 2
    from .pure import _current_store

    store = _current_store()
    cache = args.cache or getattr(store, "root", None) or os.environ.get("VALUEKIT_CACHE")
    if not cache:
        print("no cache directory: pass --cache or set VALUEKIT_CACHE", file=sys.stderr)
        return 2
    del keys  # recomputed inside sweep(); importing twice is cheap
    counts = sweep(cache, args.modules, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    print(
        f"{counts['functions']} live functions; {verb} {counts['traces']} traces, "
        f"{counts['batches']} batches, {counts['objects']} objects"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
