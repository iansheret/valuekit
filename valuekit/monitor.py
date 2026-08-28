"""Watch a running pipeline: ``python -m valuekit.monitor <cache-dir>``.

Reads the event files :mod:`valuekit.events` writes under ``<cache>/runs/``
and redraws a summary a few times a second.  It is a separate process with
its own lifetime, so it can be started twenty minutes into a run, left open
across several, or run over ssh on the machine doing the work.  It only
reads: nothing it does can affect a run.

The number to look at is the hit rate.  It is the one thing the library
promises and the one thing that is otherwise invisible -- a step that ought
to be hitting and silently is not looks exactly like a slow step.

The cache directory is taken as an argument, falling back to
``$VALUEKIT_CACHE``.  There is no remembered location, because remembering
one would mean writing outside the cache directory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

_REFRESH = 0.5  # seconds between redraws
_LIVE_AFTER = 5.0  # a run with no event for longer than this reads as idle
_MAX_FAILURES = 8


def _fmt_dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class _State:
    """Everything seen so far, folded down to what is worth showing."""

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.fns: dict[str, dict] = {}
        self.batches: dict[tuple, dict] = {}
        self.failures: list[tuple] = []

    def apply(self, source: str, e: dict) -> None:
        ev = e.get("ev")
        t = e.get("t", 0.0)
        run = self.runs.setdefault(
            source,
            {"pid": None, "argv": [], "role": "driver", "started": t, "last": t},
        )
        run["last"] = max(run["last"], t)

        if ev == "run":
            run["pid"] = e.get("pid")
            run["argv"] = e.get("argv") or []
            run["role"] = e.get("role", "driver")
            run["started"] = t
            return

        if ev in ("hit", "miss", "forced", "error"):
            f = self.fns.setdefault(source, {}).setdefault(
                e.get("fn", "?"),
                {"hit": 0, "miss": 0, "forced": 0, "error": 0, "time": 0.0},
            )
            f[ev] += 1
            f["time"] += e.get("exec") or e.get("dur") or 0.0
            if ev == "error":
                self._fail(t, e.get("fn", "?"), e.get("exc", "?"), source)
            return

        if ev == "batch":
            self.batches[(source, e.get("id"))] = {
                "fn": e.get("fn", "?"),
                "n": e.get("n", 0),
                "mode": e.get("mode", "parallel"),
                "done": 0,
                "failed": 0,
                "started": t,
                "ended": None,
            }
            return

        if ev == "outcome":
            b = self.batches.get((source, e.get("id")))
            if b is not None:
                b["done"] += 1
                if not e.get("ok", True):
                    b["failed"] += 1
                    self._fail(
                        t, f"{b['fn']}[{e.get('i')}]", e.get("exc", "?"), source
                    )
            return

        if ev == "end":
            b = self.batches.get((source, e.get("id")))
            if b is not None:
                b["ended"] = t

    def _fail(self, t: float, what: str, exc: str, source: str) -> None:
        self.failures.append((t, what, exc, source))
        del self.failures[:-_MAX_FAILURES]

    def current(self) -> set[str]:
        """The sources belonging to the newest run: its driver and the
        workers it spawned.

        Scoping matters for the hit rate.  Aggregated over every run file in
        the directory, one cold first run drags the rate down for good and
        the number stops meaning anything; what a watcher wants is the run
        in front of them.
        """
        drivers = [r for r in self.runs.values() if r["role"] != "worker"]
        if not drivers:
            return set(self.runs)
        # No grace window: a driver writes its batch event before spawning
        # anything, so its own file always predates its workers'. Allowing
        # slack here instead lets the previous run's stragglers leak in and
        # quietly spoil the rate.
        since = max(r["started"] for r in drivers)
        return {s for s, r in self.runs.items() if r["started"] >= since}


class _Tail:
    """Reads whole lines appended to the files in a directory."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.offsets: dict[Path, int] = {}

    def poll(self, state: _State) -> None:
        try:
            paths = sorted(self.dir.glob("*.jsonl"))
        except OSError:
            return
        for p in paths:
            start = self.offsets.get(p, 0)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    fh.seek(start)
                    data = fh.read()
            except OSError:
                continue
            # Only consume up to the last newline: the writer may be midway
            # through a line, and a partial line is not yet an event.
            cut = data.rfind("\n")
            if cut < 0:
                continue
            self.offsets[p] = start + cut + 1
            for line in data[:cut].splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn or corrupt: skip, same as the store does
                if isinstance(e, dict):
                    state.apply(p.name, e)


def _render(state: _State, width: int) -> list[str]:
    now = time.time()
    out: list[str] = []

    live = [r for r in state.runs.values() if now - r["last"] < _LIVE_AFTER]
    drivers = [r for r in live if r["role"] != "worker"]
    workers = len(live) - len(drivers)
    idle = len(state.runs) - len(live)
    extra = f", {workers} worker{'s' if workers != 1 else ''}" if workers else ""
    out.append(f"runs: {len(drivers)} live{extra}, {idle} finished")
    out.append("")

    for r in sorted(drivers, key=lambda r: r["started"]):
        script = os.path.basename(r["argv"][0]) if r["argv"] else "?"
        out.append(
            f"  pid {str(r['pid']):<8} {script:<28} up {_fmt_dur(now - r['started'])}"
        )
    if drivers:
        out.append("")

    scope = state.current()

    active = [
        b
        for (src, _), b in state.batches.items()
        if b["ended"] is None and src in scope
    ]
    if active:
        out.append("batches")
        for b in sorted(active, key=lambda b: b["started"]):
            n = b["n"] or 1
            filled = int(24 * b["done"] / n)
            bar = "#" * filled + "." * (24 - filled)
            fail = f"  {b['failed']} failed" if b["failed"] else ""
            out.append(
                f"  {b['fn']:<24} [{bar}] {b['done']}/{b['n']}"
                f"  {_fmt_dur(now - b['started'])}{fail}"
            )
        out.append("")

    totals: dict[str, dict] = {}
    for src in scope:
        for name, f in state.fns.get(src, {}).items():
            agg = totals.setdefault(
                name, {"hit": 0, "miss": 0, "forced": 0, "error": 0, "time": 0.0}
            )
            for k, v in f.items():
                agg[k] += v

    if totals:
        out.append("this run")
        out.append(
            f"  {'function':<28}{'hits':>8}{'misses':>8}"
            f"{'rate':>8}{'forced':>8}{'errors':>8}{'time':>10}"
        )
        rows = sorted(
            totals.items(),
            key=lambda kv: kv[1]["hit"] + kv[1]["miss"],
            reverse=True,
        )
        t_hit = t_miss = 0
        for name, f in rows:
            looked = f["hit"] + f["miss"]
            rate = f"{100 * f['hit'] / looked:.0f}%" if looked else "-"
            t_hit += f["hit"]
            t_miss += f["miss"]
            # Trim from the front: a qualname's tail is the part that names
            # the function, and its head is enclosing scopes.
            label = name if len(name) <= 28 else "…" + name[-27:]
            out.append(
                f"  {label:<28}{f['hit']:>8}{f['miss']:>8}{rate:>8}"
                f"{f['forced']:>8}{f['error']:>8}{_fmt_dur(f['time']):>10}"
            )
        looked = t_hit + t_miss
        rate = f"{100 * t_hit / looked:.0f}%" if looked else "-"
        out.append(f"  {'':<28}{t_hit:>8}{t_miss:>8}{rate:>8}")
        out.append("")

    shown = [f for f in state.failures if f[3] in scope]
    if shown:
        out.append("failures")
        for t, what, exc, _ in reversed(shown):
            out.append(f"  {_fmt_dur(now - t):>8} ago  {what[:40]:<40} {exc}")

    return [line[:width] for line in out]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = args[0] if args else os.environ.get("VALUEKIT_CACHE")
    if not root:
        print(
            "usage: python -m valuekit.monitor <cache-dir>\n"
            "       (or set VALUEKIT_CACHE)\n\n"
            "The cache directory is the one passed to set_cache_dir(); events\n"
            "are written under its runs/ subdirectory. Nothing is recorded for\n"
            "a program that never configures a cache.",
            file=sys.stderr,
        )
        return 2

    runs = Path(os.path.expanduser(root)) / "runs"
    tail = _Tail(runs)
    state = _State()
    tty = sys.stdout.isatty()
    print(f"watching {runs}", file=sys.stderr)

    try:
        while True:
            tail.poll(state)
            width, height = shutil.get_terminal_size((100, 40))
            lines = _render(state, width)
            if tty:
                # Home the cursor and clear to end of screen, rather than
                # clearing first: no flicker between frames.
                sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines[: height - 1]))
                sys.stdout.write("\n")
            else:
                sys.stdout.write("\n".join(lines) + "\n\n")
            sys.stdout.flush()
            time.sleep(_REFRESH)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
