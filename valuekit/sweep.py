"""Remove what the current code can no longer reach.

``python -m valuekit.sweep mypipeline.steps mypipeline.batches``

A call record belongs to a function hash, and a function hash belongs to code.  When
the code changes, its old call records are never consulted again; they only
occupy disk.  This command imports the named modules, takes the function hash
of every ``@pure`` and ``@pure_local`` function defined in them, and deletes
every call record and batch record whose function_hash is not among them, then
every object that no remaining record or batch names.

Retention is by code version, never by age.  A result from a year ago whose
function has not changed is as current as one from this morning, and
stays.  What cannot be told apart is a result for an input nobody uses any
more: it looks exactly like one somebody does, so it stays.

The modules named must be every module that defines a memoised function
whose results are wanted.  A function that is not imported here reads as
gone, and its call records go with it -- the worst case is recomputation, as with
every deletion in this library.  The store directory is the one configured
in those modules if they configure one, else ``--store`` or ``$VALUEKIT_STORE``.
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
from .runlog import logs_dir
from .store import LocalStore

__all__ = ["sweep", "sweep_store", "main"]


def live_keys(modules: list[str]) -> set[str]:
    """The function hashes of every memoised function in *modules*."""
    keys: set[str] = set()
    for name in modules:
        mod = importlib.import_module(name)
        for obj in list(vars(mod).values()):
            if getattr(obj, "_valuekit_pure", False):
                keys.add(obj._valuekit_reachable().hash)
    return keys


def sweep(store_dir: str | os.PathLike, modules: list[str], dry_run: bool = False) -> dict:
    """Delete call records, batches and objects the functions in *modules* cannot
    reach.  Returns counts of what was (or would be) removed."""
    return sweep_store(LocalStore(store_dir), live_keys(modules), dry_run)


def sweep_store(store: LocalStore, keys: set[str], dry_run: bool = False) -> dict:
    """Keep the call records of the function hashes in *keys* and the
    batches recorded under them; delete every other call record and batch,
    then every object that none of what remains, and no run log entry,
    names.  Returns counts of what was (or would be) removed."""
    counts = {"functions": len(keys), "records": 0, "batches": 0, "objects": 0}

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

    # -- call records of functions that no longer exist in this form ------------
    reachable: set[str] = set()
    for entry in list(store.records.iterdir()):
        key = entry.name
        if key in keys:
            if entry.is_dir():
                for h, record in store.get_records(key):
                    reachable.add(record["result"])
                    for entry in record.get("logs", []):
                        reachable.update(entry[:2])  # the labels and the value
            continue
        if entry.is_dir():
            counts["records"] += sum(1 for _ in entry.glob("*.json"))
        remove(entry)

    # -- batches recorded under those function hashes --------------------------
    bdir = batches_dir(store)
    if bdir.exists():
        for name_dir in list(bdir.iterdir()):
            if not name_dir.is_dir():
                continue
            live_ids = []
            for run in list(name_dir.iterdir()):
                if not run.is_dir():
                    continue
                try:
                    header = json.loads((run / "header.json").read_bytes())
                except (OSError, ValueError):
                    header = {}
                is_live = header.get("function_hash") in keys
                if is_live:
                    live_ids.append(run.name)
                    reachable.update(h for h in header.get("inputs", []) if h)
                else:
                    counts["batches"] += 1
                    remove(run)
            try:
                latest = (name_dir / "latest").read_text().strip()
            except OSError:
                latest = ""
            if not dry_run and latest and latest not in live_ids:
                remove(name_dir / "latest")
            if not dry_run and not live_ids:
                remove(name_dir)

    # -- what the run logs' entries name: a value logged outside any memoised
    # call, or in a forced run, has no call record.  A reference line names
    # a call record, not an object, and roots nothing: a record the sweep
    # removes reads as stale.
    ldir = logs_dir(store)
    if ldir.exists():
        for p in ldir.rglob("*.jsonl"):
            try:
                lines = p.read_bytes().splitlines()
            except OSError:
                continue
            for raw in lines:
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                if "labels" in d and "v" in d:
                    reachable.update((d["labels"], d["v"]))

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
    if not dry_run:
        store._listings.clear()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m valuekit.sweep",
        description="Delete call records, batches and objects the current code cannot reach.",
    )
    ap.add_argument("modules", nargs="+", help="modules defining the memoised functions")
    ap.add_argument("--store", help="store directory (default: $VALUEKIT_STORE)")
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
    cache = args.store or getattr(store, "root", None) or os.environ.get("VALUEKIT_STORE")
    if not cache:
        print("no store directory: pass --store or set VALUEKIT_STORE", file=sys.stderr)
        return 2
    del keys  # recomputed inside sweep(); importing twice is cheap
    counts = sweep(cache, args.modules, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    print(
        f"{counts['functions']} live functions; {verb} {counts['records']} call records, "
        f"{counts['batches']} batches, {counts['objects']} objects"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
