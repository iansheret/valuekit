"""valuekit: an immutable map for pipeline data, plus disk memoisation for
pure functions.

Both parts rest on the same idea (pipeline data as immutable values,
identified by content) and are independent; use either without the other.

The immutable map alone activates nothing else::

    from valuekit import ImmutableMap

    ctx  = ImmutableMap({"raw": signal, "fs": 1000.0})
    ctx2 = ctx | {"scaled": ctx["raw"] * gain}       # derive; ctx unchanged

Memoised pipelines: declare functions pure and set a cache directory::

    from valuekit import ImmutableMap, pure, set_cache_dir

    set_cache_dir("~/.cache/mypipeline")             # nothing caches until this

    @pure
    def calculate_geometry(obs, config):
        order = config["geom_order"]                 # reads are observed
        ...
        return {"az": az, "el": el}                  # the returned dict is the diff

    obs = obs | calculate_geometry(obs, config)      # the same merge as before

The contract: a cache hit returns exactly what executing the current
definition on the current arguments would return.  A result is recomputed
when a fact the call read from its arguments changed, or when anything
reachable by name from the function changed: its code, user functions it
calls (recursively), immutable module constants, defaults and closures, and
package versions.  Mutable globals, dispatch through data, and file
contents are not reachable by name; @pure is the user's promise that they
never change or never matter.  Better, pass them as arguments, where they
are always tracked.
"""

from ._version import __version__
from .values import freeze, content_hash, register_type
from .map import ImmutableMap
from .store import SerializationError
from .pure import pure, set_cache_dir, clear_cache
from .parallel import run_all, BatchResult

__all__ = [
    "ImmutableMap",
    "freeze",
    "content_hash",
    "register_type",
    "pure",
    "set_cache_dir",
    "clear_cache",
    "run_all",
    "BatchResult",
    "SerializationError",
    "__version__",
]
