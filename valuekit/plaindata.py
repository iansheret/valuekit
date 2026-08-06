"""Plain-data dataclasses: cacheable without registration.

A dataclass whose whole meaning is its fields needs no registration.  Its
identity is its qualified name, its dataclass parameters and its ordered
(field, value) pairs, and nothing about such a class can change without
changing that identity: the field *names* are hashed alongside the values,
so adding or renaming one invalidates, and the parameters are hashed too,
because ``order=`` and ``match_args=`` generate behaviour a caller reaches
through the data rather than by name.

Excluding methods is what makes that identity complete.  A method reached
through an argument -- ``obs.magnitude()`` -- is an attribute name, so it
resolves to nothing at module scope and never enters the calling function's
fingerprint; edit it and a stale result is served.  A class defining one is
rejected, and the message points at ``register_type``, where the user takes
on hashing it themselves.

"Plain data" is therefore checked, not documented:

* the class is built by assignment alone -- ``init=True``, no
  ``__post_init__``, every field taken by ``__init__``, and no ``InitVar``
  -- so ``cls(**values)`` provably reproduces the instance;
* neither it nor any base carries behaviour, beyond what the decorator
  itself generates;
* the instance holds exactly its fields and nothing besides.

Every check fails closed: anything unrecognised is not plain data, and the
user is sent to ``register_type`` rather than given a cache entry that
might be wrong.
"""

from __future__ import annotations

import dataclasses
import inspect
import reprlib
import sys
import threading
import types
import weakref
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from .values import _frame, _hash_fallbacks, hash_update

__all__ = [
    "is_dataclass_instance",
    "plain_data_state",
    "plain_data_class",
    "rebuild_plain_data",
]

# Everything about a dataclass declaration that changes generated behaviour.
_PARAM_NAMES = (
    "init",
    "repr",
    "eq",
    "order",
    "unsafe_hash",
    "frozen",
    "match_args",
    "kw_only",
    "slots",
    "weakref_slot",
)

# Where the decorator's own methods come from: it builds most of them with
# exec, so their code carries no user file, while reprlib supplies the
# recursive-repr wrapper around __repr__ and dataclasses itself supplies
# __replace__.  A method written by hand keeps the file it was defined in.
# Should a future Python change the marker, unrecognised code reads as a
# method and the class is rejected.
_GENERATED_FILES = frozenset(
    f
    for f in ("<string>", getattr(dataclasses, "__file__", None), reprlib.__file__)
    if f
)

# Every method the decorator will write, whatever the declaration asks for.
# The file alone would not be enough: a class assembled by exec shares the
# "<string>" marker with generated code, so code under any other name is a
# method however it was compiled.
_GENERATED_NAMES = frozenset(
    {
        "__init__",
        "__repr__",
        "__eq__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__setattr__",
        "__delattr__",
        "__getstate__",
        "__setstate__",
        "__replace__",
    }
)

# The class's lazy annotation function (PEP 649) sits in the class namespace
# carrying the defining file, but is compiler-generated, not a method.
_ANNOTATION_NAMES = frozenset({"__annotate__", "__annotate_func__"})

_spec_cache: "weakref.WeakKeyDictionary[type, Any]" = weakref.WeakKeyDictionary()
_active = threading.local()


def is_dataclass_instance(v: Any) -> bool:
    """Report whether *v* is an instance of a dataclass (plain data or not)."""
    return dataclasses.is_dataclass(v) and not isinstance(v, type)


# ---------------------------------------------------------------------------
# the plain-data checks
# ---------------------------------------------------------------------------


def _reject(cls: type, reason: str) -> TypeError:
    return TypeError(
        f"Cannot cache a {cls.__name__!r}: it {reason}. Register it with "
        "valuekit.register_type(), which takes on hashing it exactly."
    )


def _is_behaviour(attr: str, member: Any) -> bool:
    """Report whether a class attribute is code the decorator did not write."""
    if isinstance(member, (staticmethod, classmethod, property, cached_property)):
        return True
    if isinstance(member, types.FunctionType):
        return (
            attr not in _GENERATED_NAMES
            or member.__code__.co_filename not in _GENERATED_FILES
        )
    return False


def _build_spec(cls: type) -> tuple[str, tuple[str, ...], str]:
    """Return the identity of a plain-data dataclass, or raise saying why it
    is not one."""
    params = cls.__dataclass_params__  # type: ignore[attr-defined]

    # Reject construction that runs user code, or that takes values the
    # instance does not keep: rebuilding must be assignment and nothing more.
    if not params.init:
        raise _reject(cls, "is declared init=False, so nothing rebuilds it")
    if hasattr(cls, "__post_init__"):
        raise _reject(cls, "defines __post_init__, so constructing it runs code")
    fields = dataclasses.fields(cls)
    if not all(f.init for f in fields):
        raise _reject(cls, "has init=False fields, which its constructor omits")
    field_names = tuple(f.name for f in fields)
    taken = tuple(inspect.signature(cls.__init__).parameters)[1:]
    if set(taken) != set(field_names):
        raise _reject(cls, "takes constructor arguments it does not store")

    # Reject behaviour attached anywhere in the class or its bases: it would
    # be invisible to the fingerprint of a function reaching it through an
    # argument.  A field's default sits in the class namespace under the
    # field's own name, and is a value however callable it happens to be.
    ignored = set(field_names) | _ANNOTATION_NAMES
    for base in cls.__mro__:
        if base is object:
            continue
        for attr, member in vars(base).items():
            if attr in ignored or not _is_behaviour(attr, member):
                continue
            where = "" if base is cls else f", inherited from {base.__name__}"
            raise _reject(cls, f"defines {attr}{where}, which is behaviour")

    name = f"{cls.__module__}:{cls.__qualname__}"
    params_key = ",".join(
        f"{n}={int(bool(getattr(params, n, False)))}" for n in _PARAM_NAMES
    )
    return name, field_names, params_key


def _spec(cls: type) -> tuple[str, tuple[str, ...], str]:
    """Return ``(name, field_names, params_key)``, memoised per class.

    The rejection message is memoised too: a class that is not plain data is
    checked once and refused thereafter at the cost of a dictionary lookup.
    """
    cached = _spec_cache.get(cls)
    if cached is None:
        try:
            cached = _build_spec(cls)
        except TypeError as e:
            cached = str(e)
        _spec_cache[cls] = cached
    if isinstance(cached, str):
        raise TypeError(cached)
    return cached


def _check_attributes(v: Any, field_names: tuple[str, ...]) -> None:
    """Reject an instance holding anything other than its declared fields."""
    held = set(getattr(v, "__dict__", ()))
    for base in type(v).__mro__:
        slots = getattr(base, "__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot not in ("__dict__", "__weakref__") and hasattr(v, slot):
                held.add(slot)
    if held != set(field_names):
        raise TypeError(
            f"Cannot cache a {type(v).__name__!r} instance: its attributes "
            f"{sorted(held)} are not its fields {sorted(field_names)}, so its "
            "fields do not describe it. Register it with "
            "valuekit.register_type()."
        )


def plain_data_state(v: Any) -> tuple[str, tuple[str, ...], str, tuple]:
    """Return ``(name, field_names, params_key, values)`` for a plain-data
    dataclass instance, or raise TypeError saying why *v* is not one."""
    name, field_names, params_key = _spec(type(v))
    _check_attributes(v, field_names)
    return name, field_names, params_key, tuple(getattr(v, f) for f in field_names)


# ---------------------------------------------------------------------------
# content hashing
# ---------------------------------------------------------------------------


@contextmanager
def _guard_cycle(v: Any):
    """Refuse a value that contains itself, which would otherwise recurse
    until the stack ran out."""
    ids = getattr(_active, "ids", None)
    if ids is None:
        ids = _active.ids = set()
    if id(v) in ids:
        raise TypeError(
            f"Cannot hash a {type(v).__name__!r}: it contains itself, and a "
            "content hash has to be finite."
        )
    ids.add(id(v))
    try:
        yield
    finally:
        ids.discard(id(v))


def _hash_dataclass(v: Any, h: Any) -> bool:
    """Feed a plain-data dataclass into hasher *h*: identity, then fields."""
    if not is_dataclass_instance(v):
        return False
    name, field_names, params_key, values = plain_data_state(v)
    with _guard_cycle(v):
        _frame(h, b"P", f"{name}|{','.join(field_names)}|{params_key}".encode())
        for value in values:
            hash_update(value, h)
    return True


_hash_fallbacks.append(_hash_dataclass)


# ---------------------------------------------------------------------------
# rebuilding a stored instance
# ---------------------------------------------------------------------------


def plain_data_class(name: str) -> type:
    """Resolve ``"module:qualname"`` to a class already imported in this
    process, raising ValueError if it cannot be reached.

    Nothing is imported on the strength of a stored entry: a module the
    process has not loaded itself reads as a miss.
    """
    module_name, _, qualname = name.partition(":")
    obj: Any = sys.modules.get(module_name)
    if obj is None:
        raise ValueError(f"module {module_name!r} is not imported")
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise ValueError(f"{name} cannot be reached")
    if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
        raise ValueError(f"{name} is no longer a dataclass")
    return obj


def rebuild_plain_data(
    name: str, field_names: tuple[str, ...], params_key: str, values: tuple
) -> Any:
    """Rebuild a stored plain-data instance, raising if the class it names
    has changed since the entry was written."""
    cls = plain_data_class(name)
    if _spec(cls) != (name, field_names, params_key):
        raise ValueError(f"{name} has changed since this entry was stored")
    return cls(**dict(zip(field_names, values)))
