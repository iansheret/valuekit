"""Watch a running pipeline: ``python -m valuekit.monitor <cache-dir>``.

Reads the event-log files :mod:`valuekit.events` writes under ``<cache>/events/``
and redraws a summary a few times a second.  It is a separate process with
its own lifetime, so it can be started twenty minutes into a run, left open
across several, or run over ssh on the host doing the work.  Watching
has no effect on a run: nothing here is read by the pipeline.

The number to look at is the hit rate.  It is the one thing the library
promises and the one thing that is otherwise invisible -- a step that ought
to be hitting and silently is not looks exactly like a slow step.

The monitor also shows where work is going and lets you change it.  The
*mode* (see :mod:`valuekit.modes`) is a line in the project's local
file, ``valuekit.local.toml``; keys ``l``, ``r`` and ``a`` set it to
``local``, ``remote`` or ``all``, and that line is the only thing the
monitor ever writes.  The main process reads the file whenever it starts a task.  The header
shows the mode and, while a batch runs, what it means for the next task
given which hosts are ready.  ``--mode <mode>`` sets the line and exits,
for scripts.

The store directory is taken as an argument, falling back to
``$VALUEKIT_STORE``.  The project is the one enclosing the current
directory; run the monitor from inside the project, or the mode is shown
as unknown and the keys do nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import localfile, modes

_REFRESH = 0.5  # seconds between redraws
_LIVE_AFTER = 5.0  # a run with no record for longer than this reads as idle
_MAX_FAILURES = 8

_KEYS = {"l": "local", "r": "remote", "a": "all"}


def _consequence(mode: str, remote_caps: dict, syncing: list, names: list) -> str:
    """What the mode means for the next task, given the hosts' states."""
    usable = sorted(n for n, c in remote_caps.items() if c)
    if mode == "local":
        return "everything runs here" + ("; remote hosts are not used" if names else "")
    if mode == "all":
        parts = ["new tasks go to " + ", ".join([*usable, "here"])]
        if syncing:
            parts.append(f"{', '.join(syncing)} still syncing")
        return "; ".join(parts)
    if usable:
        return f"new tasks go to {', '.join(usable)}; none start here"
    if syncing:
        return f"waiting for {', '.join(syncing)} to sync; none start here"
    return "no remote host is usable, so this machine runs the batch"



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
        self.hosts: dict[tuple, dict] = {}  # (source, host) -> the sync's outcome
        self.per_host: dict[str, dict[str, dict]] = {}  # source -> host -> counts

    def apply(self, source: str, e: dict) -> None:
        ev = e.get("ev")
        t = e.get("t", 0.0)
        run = self.runs.setdefault(
            source,
            {"pid": None, "argv": [], "role": "main", "started": t, "last": t},
        )
        run["last"] = max(run["last"], t)

        if ev == "process":
            run["pid"] = e.get("pid")
            run["argv"] = e.get("argv") or []
            run["role"] = e.get("role", "main")
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
                "fn": e.get("name") or e.get("fn", "?"),
                "n": e.get("n", 0),
                "mode": e.get("mode", "parallel"),
                "done": 0,
                "failed": 0,
                "started": t,
                "ended": None,
            }
            return

        if ev == "start":
            self._host_counts(source, e.get("host", "local"))["running"] += 1
            return

        if ev == "requeue":
            # The host went away under this input; it will start again
            # elsewhere and be counted there.
            counts = self._host_counts(source, e.get("host", "local"))
            counts["running"] = max(0, counts["running"] - 1)
            return

        if ev == "outcome":
            counts = self._host_counts(source, e.get("host", "local"))
            counts["running"] = max(0, counts["running"] - 1)
            counts["done"] += 1
            if not e.get("ok", True):
                counts["failed"] += 1
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
            return

        if ev == "host":
            self.hosts[(source, e.get("name", "?"))] = {
                "ok": bool(e.get("ok")),
                "reason": e.get("reason") or "",
                "capacity": e.get("capacity"),
            }

    def _host_counts(self, source: str, host: str) -> dict:
        return self.per_host.setdefault(source, {}).setdefault(
            host, {"running": 0, "done": 0, "failed": 0}
        )

    def _fail(self, t: float, what: str, exc: str, source: str) -> None:
        self.failures.append((t, what, exc, source))
        del self.failures[:-_MAX_FAILURES]

    def current(self) -> set[str]:
        """The sources belonging to the newest run: its main process and the
        workers it spawned.

        Scoping matters for the hit rate.  Aggregated over every run file in
        the directory, one cold first run drags the rate down for good and
        the number stops meaning anything; what a watcher wants is the run
        in front of them.
        """
        mains = [r for r in self.runs.values() if r["role"] != "worker"]
        if not mains:
            return set(self.runs)
        # No grace window: a main process writes its batch record before spawning
        # anything, so its own file always predates its workers'. Allowing
        # slack here instead lets the previous run's stragglers leak in and
        # quietly spoil the rate.
        since = max(r["started"] for r in mains)
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
            # through a line, and a partial line is not yet a record.
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


def _render(
    state: _State,
    width: int,
    mode: str | None = None,
    configured: tuple[str, ...] = (),
    local_workers: int = 0,
    keys: bool = False,
) -> list[str]:
    now = time.time()
    out: list[str] = []

    live = [r for r in state.runs.values() if now - r["last"] < _LIVE_AFTER]
    mains = [r for r in live if r["role"] != "worker"]
    workers = len(live) - len(mains)
    idle = len(state.runs) - len(live)
    extra = f", {workers} worker{'s' if workers != 1 else ''}" if workers else ""
    out.append(f"runs: {len(mains)} live{extra}, {idle} finished")

    scope = state.current()
    batch_live = any(
        b["ended"] is None and src in scope for (src, _), b in state.batches.items()
    )

    # Every remote host that is configured or has done anything, with what
    # the sync said about it.
    names: list[str] = []
    for name in (
        *configured,
        *(h for s, h in state.hosts if s in scope),
        *(h for s in scope for h in state.per_host.get(s, ())),
    ):
        if name not in names and name != "local":
            names.append(name)
    synced = {
        n: h for (s, n), h in state.hosts.items() if s in scope and n in names
    }
    remote_caps = {n: (synced[n]["capacity"] or 0) if n in synced and synced[n]["ok"] else 0 for n in names}
    syncing = [n for n in names if n not in synced] if batch_live else []
    caps = (
        modes.capacities(mode, local_workers, remote_caps, bool(syncing))
        if mode in modes.MODES
        else {}
    )

    if mode is not None:
        line = f"mode: {mode}"
        if keys:
            line += "          l local  r remote  a all  q quit"
        out.append(line)
        if batch_live and mode in modes.MODES:
            out.append("  " + _consequence(mode, remote_caps, syncing, names))
    out.append("")

    for r in sorted(mains, key=lambda r: r["started"]):
        script = os.path.basename(r["argv"][0]) if r["argv"] else "?"
        out.append(
            f"  pid {str(r['pid']):<8} {script:<28} up {_fmt_dur(now - r['started'])}"
        )
    if mains:
        out.append("")

    if names or batch_live:
        out.append("hosts")
        out.append(f"  {'host':<16}{'capacity':>10}{'running':>9}{'done':>7}{'failed':>8}  state")
        for name in (*names, "local"):
            cap = caps.get(name) if caps else None
            counts = {"running": 0, "done": 0, "failed": 0}
            for s in scope:
                c = state.per_host.get(s, {}).get(name)
                if c:
                    for k in counts:
                        counts[k] += c[k]
            h = synced.get(name)
            if name == "local":
                status = "idle under remote" if caps and caps.get("local") == 0 and mode == "remote" else ""
            elif h is None:
                status = "syncing" if batch_live and mode != "local" else "not used"
            elif h["ok"]:
                status = "ready"
            else:
                status = f"dropped: {h['reason'].splitlines()[0][:40]}"
            out.append(
                f"  {name:<16}{'-' if cap is None else cap:>10}"
                f"{counts['running']:>9}{counts['done']:>7}{counts['failed']:>8}  {status}"
            )
        out.append("")

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


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


class _Keys:
    """Non-blocking single keystrokes from the terminal, if it is one.

    Windows reads the console directly; POSIX puts the terminal in cbreak
    mode for the monitor's lifetime and restores it on exit.  Neither is
    used when stdin is not a terminal.
    """

    def __init__(self) -> None:
        self._restore = None
        self._posix = False
        try:
            self.enabled = sys.stdin.isatty()
        except Exception:
            self.enabled = False
        if not self.enabled:
            return
        if os.name == "nt":
            return
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._restore = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
            self._posix = True
        except Exception:
            self.enabled = False

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        try:
            if os.name == "nt":
                import msvcrt

                return msvcrt.getwch() if msvcrt.kbhit() else None
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0)
            return sys.stdin.read(1) if ready else None
        except Exception:
            return None

    def close(self) -> None:
        if self._restore is not None:
            try:
                import termios

                fd, attrs = self._restore
                termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            except Exception:
                pass


def _apply_key(project: str | None, key: str | None) -> bool:
    """Act on one keystroke; return False when the key asks to quit."""
    if key is None:
        return True
    if key in ("q", "\x03"):
        return False
    mode = _KEYS.get(key.lower())
    if mode is not None and project is not None:
        try:
            localfile.write_mode(project, mode)
        except OSError:
            pass
    return True


def _project_here() -> str | None:
    """The project enclosing the current directory, if any."""
    from .project import find_root

    return find_root(os.path.join(os.getcwd(), "pyproject.toml"))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _usage() -> None:
    print(
        "usage: python -m valuekit.monitor [--mode local|remote|all] <cache-dir>\n"
        "       (or set VALUEKIT_STORE)\n\n"
        "The store directory is the one passed to set_store_dir(); the event\n"
        "log is written under its events/ subdirectory. Nothing is recorded for\n"
        "a program that never configures a cache. The mode is a line in the\n"
        "valuekit.local.toml of the project enclosing the current directory:\n"
        "--mode sets it and exits; without it, keys l, r and a set it while\n"
        "watching.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = None
    if "--mode" in args:
        at = args.index("--mode")
        try:
            mode = args[at + 1]
        except IndexError:
            _usage()
            return 2
        del args[at : at + 2]
    root = args[0] if args else os.environ.get("VALUEKIT_STORE")
    if not root or mode is not None and mode not in modes.MODES:
        _usage()
        return 2
    root_path = Path(os.path.expanduser(root))
    project = _project_here()

    if mode is not None:
        if project is None:
            print(f"no project (pyproject.toml or .git) encloses {os.getcwd()}", file=sys.stderr)
            return 2
        localfile.write_mode(project, mode)
        print(f"mode for {project}: {mode}")
        return 0

    runs = root_path / "events"
    tail = _Tail(runs)
    state = _State()
    tty = sys.stdout.isatty()
    keys = _Keys()
    try:
        config = localfile.load_local(project)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        config = localfile.load_local(None)
    configured = tuple(h.name for h in config.hosts)
    if project is None:
        print(f"no project encloses {os.getcwd()}: the mode cannot be shown or set", file=sys.stderr)
    print(f"watching {runs}", file=sys.stderr)

    try:
        while True:
            if not _apply_key(project, keys.poll()):
                return 0
            tail.poll(state)
            width, height = shutil.get_terminal_size((100, 40))
            lines = _render(
                state, width, localfile.read_mode(project, root_path) if project else None,
                configured, config.local_workers, keys.enabled
            )
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
    finally:
        keys.close()


if __name__ == "__main__":
    raise SystemExit(main())
