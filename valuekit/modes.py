"""The mode: which hosts a batch uses.

``all`` uses every reachable remote host and this machine at full
capacity; ``local`` runs everything on this machine; ``remote`` runs as
little here as possible, which means nothing here while any remote host
is ready or still syncing, and everything here when none is.  The mode is
the ``mode`` line of the local file (:mod:`valuekit.localfile`); absent, it
is ``all``: a host named in the file is used unless the mode excludes it.

The scheduler reads the file each time it is about to start a task, so an
edit takes effect for the next task started; tasks already running finish
where they are.
"""

from __future__ import annotations

__all__ = ["MODES", "DEFAULT_MODE", "capacities"]

MODES = ("local", "remote", "all")
DEFAULT_MODE = "all"


def capacities(
    mode: str, local: int, remote: dict[str, int], syncing: bool = False
) -> dict[str, int]:
    """How many tasks each host may run at once under *mode*.

    *remote* maps each remote host to its capacity, 0 until it is ready;
    *syncing* says whether any remote host is still syncing.  Every name is
    present in the result, at 0 where the mode excludes it, so a display
    can show what is switched off as well as what is on.

    Under ``remote`` this machine stays idle while a remote host is still
    syncing: the person who chose that mode wants their machine free, and
    a short batch would otherwise be over before the host was ready.
    """
    if mode == "local":
        return {**{name: 0 for name in remote}, "local": local}
    if mode == "all":
        return {**remote, "local": local}
    if mode == "remote":
        if any(remote.values()) or syncing:
            return {**remote, "local": 0}
        return {**remote, "local": local}
    raise ValueError(f"unknown mode {mode!r}")
