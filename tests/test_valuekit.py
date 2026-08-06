"""valuekit test suite.

Focus: the correctness edges — staleness, granularity, arg binding,
atomic store behaviour — rather than demos.
"""

import dataclasses
import functools
import json
import os
import sys
import types
import warnings
from importlib.machinery import EXTENSION_SUFFIXES

import numpy as np
import pytest

import valuekit as vk
from valuekit import ImmutableMap, pure, freeze, content_hash
from valuekit import codehash
from valuekit.codehash import _classify, function_fingerprint
from valuekit.store import LocalStore, CacheMiss, SerializationError
from valuekit.values import encode_key, decode_key
from valuekit import pure as _pure_mod  # module alias for store poking
from valuekit import debughook


@pytest.fixture()
def cache(tmp_path):
    vk.set_cache_dir(tmp_path / "cache")
    yield tmp_path / "cache"
    vk.set_cache_dir(None)


@pytest.fixture(autouse=True)
def _no_cache_by_default():
    vk.set_cache_dir(None)
    yield
    vk.set_cache_dir(None)


# ===========================================================================
# ImmutableMap fundamentals (behaviour of the original class preserved)
# ===========================================================================


class TestImmutableMap:
    def test_basic_roundtrip(self):
        m = ImmutableMap({"a": 1, "b": "x"})
        assert m["a"] == 1 and m["b"] == "x" and len(m) == 2

    def test_merge_derivation(self):
        m1 = ImmutableMap({"a": 1})
        m2 = m1 | {"b": 2}
        m3 = m2 | {"a": 99}
        assert dict(m1) == {"a": 1}
        assert dict(m2) == {"a": 1, "b": 2}
        assert m3["a"] == 99 and m2["a"] == 1

    def test_dissoc_assoc(self):
        m = ImmutableMap({"a": 1, "b": 2}).dissoc("a").assoc("c", 3)
        assert dict(m) == {"b": 2, "c": 3}

    def test_arrays_frozen_and_copied(self):
        a = np.arange(5.0)
        m = ImmutableMap({"x": a})
        assert not m["x"].flags.writeable
        a[0] = 99  # caller's copy stays writeable; map unaffected
        assert m["x"][0] == 0.0

    def test_prefrozen_array_shared(self):
        a = np.arange(5.0)
        a.flags.writeable = False
        m = ImmutableMap({"x": a})
        assert m["x"] is a

    def test_rejects_unknown_mutable(self):
        with pytest.raises(TypeError):
            ImmutableMap({"x": [1, 2, 3]})

    def test_no_attribute_mutation(self):
        m = ImmutableMap({"a": 1})
        with pytest.raises(AttributeError):
            m._d = {}
        with pytest.raises(TypeError):
            m["b"] = 2  # type: ignore[index]

    def test_equality_with_arrays(self):
        m1 = ImmutableMap({"x": np.arange(3)})
        m2 = ImmutableMap({"x": np.arange(3)})
        assert m1 == m2
        assert m1 != m2 | {"y": 1}

    def test_nested_dict_becomes_map(self):
        m = ImmutableMap({"cfg": {"a": 1}})
        assert isinstance(m["cfg"], ImmutableMap)


# ===========================================================================
# content hashing
# ===========================================================================


class TestHashing:
    def test_stability_and_type_separation(self):
        assert content_hash(1) == content_hash(1)
        assert content_hash(1) != content_hash(1.0)
        assert content_hash(True) != content_hash(1)  # bool is not int here
        assert content_hash("1") != content_hash(1)
        assert content_hash(b"a") != content_hash("a")

    def test_framing_prevents_concat_ambiguity(self):
        assert content_hash(("ab", "c")) != content_hash(("a", "bc"))
        assert content_hash((1, (2, 3))) != content_hash((1, 2, 3))

    def test_array_hash_layout_independent(self):
        a = np.arange(6.0).reshape(2, 3)
        b = np.asfortranarray(a)
        assert content_hash(a) == content_hash(b)
        assert content_hash(a) != content_hash(a.astype(np.float32))
        assert content_hash(a) != content_hash(a.reshape(3, 2))

    def test_map_hash_order_independent_and_cached(self):
        m1 = ImmutableMap({"a": 1, "b": np.arange(4)})
        m2 = ImmutableMap({"b": np.arange(4), "a": 1})
        assert content_hash(m1) == content_hash(m2)
        assert m1._digest is not None  # cached after first computation

    def test_derived_map_hash_differs(self):
        m = ImmutableMap({"a": 1})
        assert content_hash(m) != content_hash(m | {"b": 2})

    def test_frozenset_order_independent(self):
        assert content_hash(frozenset({1, 2, 3})) == content_hash(
            frozenset({3, 1, 2})
        )

    def test_hash_identifies_a_value_exactly(self):
        # A dict is not an ImmutableMap, and nothing converts one to the
        # other, so they are different values.
        assert content_hash({"a": 1}) != content_hash(ImmutableMap({"a": 1}))
        assert content_hash([1, 2]) != content_hash((1, 2))
        assert content_hash({1, 2}) != content_hash(frozenset({1, 2}))
        # Dict order is observable; ImmutableMap order is not.
        assert content_hash({"a": 1, "b": 2}) != content_hash({"b": 2, "a": 1})
        assert content_hash(ImmutableMap({"a": 1, "b": 2})) == content_hash(
            ImmutableMap({"b": 2, "a": 1})
        )

    def test_array_hash_covers_writeability(self):
        a = np.arange(4.0)
        b = a.copy()
        b.flags.writeable = False
        assert content_hash(a) != content_hash(b)

    def test_lambda_hash_covers_closure_and_defaults(self):
        k = 3
        f1 = lambda x, m=2: x * m + k  # noqa: E731
        h1 = content_hash(f1)
        k2 = 4
        f2 = lambda x, m=2: x * m + k2  # noqa: E731  (same bytecode, diff cell)
        # identical source apart from captured value name; force same code:
        def make(kk):
            return lambda x, m=2: x * m + kk

        assert content_hash(make(3)) == content_hash(make(3))
        assert content_hash(make(3)) != content_hash(make(4))  # closure differs
        g1 = lambda x, m=5: x * m  # noqa: E731
        g2 = lambda x, m=6: x * m  # noqa: E731
        assert content_hash(g1) != content_hash(g2)  # defaults differ
        assert isinstance(h1, str)

    def test_key_codec_roundtrip(self):
        for k in [None, True, False, 0, -17, 3.5, 1 + 2j, "héllo", b"\x00\xff",
                  range(1, 10, 2), ("a", (1, 2.0)), frozenset({"x", "y"})]:
            assert decode_key(encode_key(k)) == k


# ===========================================================================
# code hashing / invalidation semantics
# ===========================================================================


def _fp(fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function_fingerprint(fn, **kw)[0]


class TestCodeHash:
    def test_body_change_changes_hash(self):
        def f(x):
            return x + 1

        def g(x):
            return x + 2

        assert _fp(f) != _fp(g)

    def test_rename_preserves_hash(self):
        def f(x):
            return x + 1

        def a_totally_different_name(x):
            return x + 1

        assert _fp(f) == _fp(a_totally_different_name)

    def test_helper_change_invalidates(self):
        # The joblib bug this project exists to fix.
        ns1 = {}
        exec("def helper(x):\n    return x * 2\ndef step(x):\n    return helper(x) + 1", ns1)
        ns2 = {}
        exec("def helper(x):\n    return x * 3\ndef step(x):\n    return helper(x) + 1", ns2)
        assert _fp(ns1["step"]) != _fp(ns2["step"])

    def test_module_level_constant_captured(self):
        ns1, ns2 = {}, {}
        exec("K = 10\ndef f(x):\n    return x + K", ns1)
        exec("K = 11\ndef f(x):\n    return x + K", ns2)
        assert _fp(ns1["f"]) != _fp(ns2["f"])

    def test_nested_lambda_captured(self):
        def f1(xs):
            return sorted(xs, key=lambda v: v * 2)

        def f2(xs):
            return sorted(xs, key=lambda v: v * 3)

        assert _fp(f1) != _fp(f2)

    def test_package_boundary_stops_walk(self):
        def f(x):
            return np.fft.fft(x)

        h = _fp(f)  # should not attempt to hash numpy internals; just work
        assert isinstance(h, str) and len(h) == 40


    def test_constants_never_warn(self):
        # SPEED_OF_LIGHT-style module constants — of every common shape —
        # must decorate in total silence, tracked or not.
        src = (
            "C = 299792458.0\n"
            "NAME = 'L-band'\n"
            "CHANNELS = ['ch1', 'ch2']\n"
            "EDGES = {'L': [1.0, 2.0], 'S': [2.0, 4.0]}\n"
            "STATIONS = {'EL-1', 'EL-2'}\n"
            "GRID = np.linspace(0, 1, 5)\n"
            "def f(x):\n"
            "    return x * C, NAME, CHANNELS[0], EDGES['L'][1], "
            "len(STATIONS), GRID[0]\n"
        )
        ns = {"np": np}
        exec(src, ns)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # ANY warning → test failure
            function_fingerprint(ns["f"])

    def test_immutable_constants_invalidate(self):
        def build(c):
            ns = {}
            exec(f"C = {c!r}\ndef f(x):\n    return x * C", ns)
            return ns["f"]

        assert _fp(build(299792458.0)) == _fp(build(299792458.0))
        assert _fp(build(299792458.0)) != _fp(build(299792459.0))
        assert _fp(build((1, 2))) != _fp(build((1, 3)))

    def test_readonly_array_constant_tracked_writeable_not(self):
        def build(vals, readonly):
            a = np.array(vals)
            if readonly:
                a.flags.writeable = False
            ns = {"GRID": a}
            exec("def f(x):\n    return x + GRID[0]", ns)
            return ns["f"]

        # Read-only array: a true constant — tracked, edits invalidate.
        assert _fp(build([1.0], True)) != _fp(build([2.0], True))
        # Writeable array: mutable → untracked by design, hash is stable.
        assert _fp(build([1.0], False)) == _fp(build([2.0], False))

    def test_mutable_globals_untracked_stable_and_silent(self):
        def build(channels):
            ns = {}
            exec(
                f"CHANNELS = {channels!r}\ndef f(x):\n    return (x, CHANNELS[0])",
                ns,
            )
            return ns["f"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fp1 = function_fingerprint(build(["ch1", "ch2"]))[0]
            fp2 = function_fingerprint(build(["completely", "different"]))[0]
        assert fp1 == fp2  # not our problem, by explicit design

    def test_opaque_globals_silent_and_stable(self):
        def build():
            ns = {"HANDLE": object(), "BUF": bytearray(b"x")}
            exec("def f(x):\n    return x if HANDLE and BUF else x", ns)
            return ns["f"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fp1 = function_fingerprint(build())[0]
            fp2 = function_fingerprint(build())[0]
        assert fp1 == fp2  # distinct opaque objects: untracked, stable


# ===========================================================================
# native extensions
# ===========================================================================


class _NativeCallable:
    """Stands in for a nanobind function: callable, carrying no Python code,
    and attributed to the module that defined it."""

    def __call__(self, x):
        return x


@pytest.fixture
def fake_extension(tmp_path, monkeypatch):
    """A module that looks like a locally built compiled extension: a real
    file with an extension suffix, installed where a wheel would put it, and
    belonging to no installed distribution."""
    installed = tmp_path / "site-packages"
    installed.mkdir()
    path = installed / f"_fake_ext{EXTENSION_SUFFIXES[0]}"
    path.write_bytes(b"compiled bytes, version one")
    mod = types.ModuleType("_fake_ext")
    mod.__file__ = str(path)
    monkeypatch.setitem(sys.modules, "_fake_ext", mod)
    monkeypatch.setattr(_NativeCallable, "__module__", "_fake_ext")
    return mod, path


def _using_global(name, obj):
    """Build a function whose body calls *obj* under the name *name*."""
    ns = {name: obj}
    exec(f"def f(x):\n    return {name}(x)", ns)
    return ns["f"]


class TestNativeExtensions:
    def test_binary_contents_identify_the_extension(self, fake_extension):
        mod, path = fake_extension
        kind, marker = _classify(mod.__name__, mod.__file__)
        assert marker.startswith("ext:_fake_ext=")
        path.write_bytes(b"compiled bytes, version two -- rebuilt")
        assert _classify(mod.__name__, mod.__file__)[1] != marker

    def test_rebuilt_extension_invalidates_its_callers(self, fake_extension):
        # The case this exists for: a @pure function calls into C++, the C++
        # is edited and rebuilt, and the result must not be replayed.
        mod, path = fake_extension
        fn = _using_global("solve", _NativeCallable())
        before = _fp(fn)
        path.write_bytes(b"compiled bytes, version two -- rebuilt")
        assert _fp(fn) != before

    def test_an_identical_build_elsewhere_keeps_the_hash(
        self, fake_extension, tmp_path
    ):
        # The marker is the contents, not the path or the timestamp, so a
        # rebuild that changes nothing costs no recomputation.
        mod, path = fake_extension
        fn = _using_global("solve", _NativeCallable())
        before = _fp(fn)
        moved = tmp_path / f"elsewhere{EXTENSION_SUFFIXES[0]}"
        moved.write_bytes(path.read_bytes())
        mod.__file__ = str(moved)
        assert _fp(fn) == before

    def test_a_released_distribution_keeps_its_version_marker(
        self, fake_extension, monkeypatch
    ):
        # A wheel changes only through a reinstall, which moves its version:
        # there is nothing to gain by reading megabytes of binary.
        mod, path = fake_extension

        class _ReleasedDist:
            version = "1.2.3"

            def read_text(self, name):
                return None  # no direct_url.json: not a local install

        monkeypatch.setattr(codehash, "_distribution", lambda top: _ReleasedDist())
        assert _classify(mod.__name__, mod.__file__)[1] == "pkg:_fake_ext==1.2.3"

    def test_an_extension_module_is_tracked_as_a_module(self, fake_extension):
        mod, path = fake_extension
        fn = _using_global("ext", mod)
        # A module global resolves by name, so the reference is to the module
        # itself rather than to anything it defines.
        before = _fp(fn)
        path.write_bytes(b"compiled bytes, version two -- rebuilt")
        assert _fp(fn) != before

    def test_an_extension_built_in_the_source_tree_is_tracked(self, tmp_path):
        # Built in place rather than installed: there is no version to lean
        # on at all, so the contents are all there is.
        path = tmp_path / f"_intree{EXTENSION_SUFFIXES[0]}"
        path.write_bytes(b"compiled bytes, version one")
        before = _classify("_intree", str(path))[1]
        assert before.startswith("ext:_intree=")
        path.write_bytes(b"compiled bytes, version two -- rebuilt")
        assert _classify("_intree", str(path))[1] != before

    def test_a_missing_binary_is_not_an_error(self, fake_extension):
        mod, path = fake_extension
        fn = _using_global("solve", _NativeCallable())
        path.unlink()
        _fp(fn)  # falls back to the version marker rather than raising


# ===========================================================================
# store
# ===========================================================================


class TestStore:
    def test_roundtrip_all_types(self, tmp_path):
        s = LocalStore(tmp_path)
        vals = [
            None, True, 42, 3.14, 2 - 3j, "héllo", b"\x00\x01", range(5),
            np.float64(1.5), np.int32(7),
            (1, "two", (3.0,)),
            frozenset({1, 2}),
            ImmutableMap({"a": 1, "sub": {"b": np.arange(3)}}),
            np.arange(10.0).reshape(2, 5),
        ]
        for v in vals:
            fv = freeze(v)
            h = s.put_value(fv)
            back = s.get_value(h)
            assert content_hash(back) == content_hash(fv) == h

    def test_roundtrip_preserves_mutable_types(self, tmp_path):
        s = LocalStore(tmp_path)
        vals = [
            [1, "two", [3.0]],
            {"b": 1, "a": [2]},  # order is part of the value
            {1, 2, 3},
            {"nested": [{"deep": (1, {2})}]},
        ]
        for v in vals:
            back = s.get_value(s.put_value(v))
            assert back == v
            assert type(back) is type(v)
            assert content_hash(back) == content_hash(v)
        assert list(s.get_value(s.put_value({"b": 1, "a": 2}))) == ["b", "a"]
        assert s.get_value(s.put_value(np.arange(4.0))).flags.writeable

    def test_arrays_reload_readonly(self, tmp_path):
        s = LocalStore(tmp_path)
        h = s.put_value(freeze(np.arange(100.0)))
        arr = s.get_value(h)
        assert not arr.flags.writeable
        assert freeze(arr) is arr  # zero-copy share on re-freeze

    def test_dedup(self, tmp_path):
        s = LocalStore(tmp_path)
        a = freeze(np.arange(1000.0))
        h1 = s.put_value(freeze((a, 1)))
        h2 = s.put_value(freeze((a, 2)))
        npys = list((tmp_path / "objects").rglob("*.npy"))
        assert len(npys) == 1  # array shared between the two tuples
        assert h1 != h2

    def test_missing_and_corrupt_are_misses(self, tmp_path):
        s = LocalStore(tmp_path)
        with pytest.raises(CacheMiss):
            s.get_value("0" * 40)
        h = s.put_value(freeze((1, 2)))
        path = next((tmp_path / "objects").rglob(f"{h}.bin"))
        path.write_bytes(b"garbage")
        with pytest.raises(CacheMiss):
            s.get_value(h)

    def test_unstorable_rejected(self, tmp_path):
        s = LocalStore(tmp_path)
        with pytest.raises(SerializationError):
            s.put_value(freeze(lambda x: x))

    def test_format_version_guard(self, tmp_path):
        LocalStore(tmp_path)
        (tmp_path / "format").write_text("999\n")
        with pytest.raises(RuntimeError):
            LocalStore(tmp_path)

    def test_trace_dedup(self, tmp_path):
        s = LocalStore(tmp_path)
        t = {"fn": "f", "deps": {}, "result": "0" * 40}
        s.put_trace("k", t)
        s.put_trace("k", dict(t))
        assert len(s.get_traces("k")) == 1

    def test_concurrent_trace_appends_are_not_lost(self, tmp_path):
        # The failure mode of a read-modify-write trace file: parallel
        # writers dropping each other's traces. Appends must all survive.
        from concurrent.futures import ThreadPoolExecutor

        stores = [LocalStore(tmp_path) for _ in range(4)]

        def write(i):
            stores[i % 4].put_trace(
                "k", {"fn": "f", "deps": {"x": {"kind": "value", "hash": str(i)}},
                      "result": "0" * 40}
            )

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(write, range(32)))
        assert len(LocalStore(tmp_path).get_traces("k")) == 32

    def test_torn_trace_line_skipped(self, tmp_path):
        s = LocalStore(tmp_path)
        t = {"fn": "f", "deps": {}, "result": "0" * 40}
        s.put_trace("k", t)
        with open(s._trace_path("k"), "a") as f:
            f.write('{"fn": "g", "trunc')  # simulates a crash mid-append
        assert s.get_traces("k") == [t]

    def test_immutable_map_pickles(self, tmp_path):
        import pickle

        m = ImmutableMap({"x": np.arange(3.0), "sub": {"k": 1}})
        m2 = pickle.loads(pickle.dumps(m))
        assert m2 == m
        assert not m2["x"].flags.writeable  # re-frozen on arrival
        assert isinstance(m2["sub"], ImmutableMap)


# ===========================================================================
# custom-type store codec (reduce/rebuild)
# ===========================================================================


class _Vec:
    """A tiny immutable custom type for codec tests: one read-only array."""

    def __init__(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr.flags.writeable = False
        self.arr = arr

    def __eq__(self, other):
        return isinstance(other, _Vec) and np.array_equal(self.arr, other.arr)


vk.register_type(
    _Vec,
    freeze_fn=lambda v: v,
    hash_fn=lambda v, h: h.update(v.arr.tobytes()),
    reduce_fn=lambda v: v.arr,
    rebuild_fn=lambda arr: _Vec(arr),
)


class TestCustomTypeCodec:
    def test_store_roundtrip_preserves_type_and_content(self, tmp_path):
        s = LocalStore(tmp_path)
        v = freeze(_Vec([1.0, 2.0, 3.0]))
        h = s.put_value(v)
        back = s.get_value(h)
        assert isinstance(back, _Vec)
        assert back == v
        assert content_hash(back) == h

    def test_roundtrips_as_a_pure_return(self, cache):
        calls = []

        @pure
        def make(n):
            calls.append(n)
            return {"v": _Vec(np.arange(n))}

        make(4)
        second = make(4)
        assert calls == [4]  # the second call was a cache hit
        assert isinstance(second["v"], _Vec)
        assert second["v"] == _Vec(np.arange(4))

    def test_requires_both_reduce_and_rebuild(self):
        class _Half:
            pass

        with pytest.raises(TypeError):
            vk.register_type(
                _Half,
                freeze_fn=lambda v: v,
                hash_fn=lambda v, h: h.update(b"x"),
                reduce_fn=lambda v: 0,  # rebuild_fn missing
            )

    def test_codecless_custom_type_is_still_unstorable(self, tmp_path):
        class _NoCodec:
            pass

        vk.register_type(
            _NoCodec, freeze_fn=lambda v: v, hash_fn=lambda v, h: h.update(b"n")
        )
        s = LocalStore(tmp_path)
        with pytest.raises(SerializationError):
            s.put_value(freeze(_NoCodec()))


# ===========================================================================
# plain-data dataclasses
# ===========================================================================


@dataclasses.dataclass
class _Point:
    x: int
    y: float = 0.0


@dataclasses.dataclass(frozen=True)
class _Frozen:
    label: str
    items: tuple = ()


@dataclasses.dataclass(slots=True)
class _Slotted:
    a: int


@dataclasses.dataclass
class _SameFields:
    x: int
    y: float = 0.0


@dataclasses.dataclass
class _Base:
    a: int


@dataclasses.dataclass
class _Derived(_Base):
    b: int = 0


@dataclasses.dataclass
class _KwOnly:
    a: int = 0
    b: int = dataclasses.field(kw_only=True, default=1)


class TestPlainDataHashing:
    def test_identity_includes_the_class(self):
        # Same field names, same values, different class: different values.
        assert content_hash(_Point(1, 2.0)) != content_hash(_SameFields(1, 2.0))

    def test_identity_includes_field_names(self):
        @dataclasses.dataclass
        class Renamed:
            x: int
            z: float = 0.0

        Renamed.__qualname__ = _Point.__qualname__
        Renamed.__module__ = _Point.__module__
        assert content_hash(Renamed(1, 2.0)) != content_hash(_Point(1, 2.0))

    def test_identity_includes_dataclass_params(self):
        # order= generates comparisons a caller reaches through the data, so
        # flipping it has to be a different value.
        @dataclasses.dataclass(order=True)
        class Ordered:
            x: int
            y: float = 0.0

        Ordered.__qualname__ = _Point.__qualname__
        Ordered.__module__ = _Point.__module__
        assert content_hash(Ordered(1, 2.0)) != content_hash(_Point(1, 2.0))

    def test_not_confusable_with_a_dict_of_the_same_fields(self):
        assert content_hash(_Point(1, 2.0)) != content_hash({"x": 1, "y": 2.0})

    def test_class_constants_are_not_behaviour(self):
        @dataclasses.dataclass
        class WithConstant:
            x: int
            SCALE = 2.0

        assert content_hash(WithConstant(1)) == content_hash(WithConstant(1))

    def test_self_reference_is_refused(self):
        @dataclasses.dataclass
        class Node:
            child: object = None

        n = Node()
        n.child = n
        with pytest.raises(TypeError, match="contains itself"):
            content_hash(n)

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: _with_method(), id="method"),
            pytest.param(lambda: _with_property(), id="property"),
            pytest.param(lambda: _with_cached_property(), id="cached_property"),
            pytest.param(lambda: _with_staticmethod(), id="staticmethod"),
            pytest.param(lambda: _with_classmethod(), id="classmethod"),
            pytest.param(lambda: _with_dunder(), id="hand_written_eq"),
            pytest.param(lambda: _with_post_init(), id="post_init"),
            pytest.param(lambda: _with_initvar(), id="initvar"),
            pytest.param(lambda: _with_noninit_field(), id="non_init_field"),
            pytest.param(lambda: _with_inherited_method(), id="inherited_method"),
        ],
    )
    def test_behaviour_beyond_the_fields_is_refused(self, make):
        with pytest.raises(TypeError, match="register_type"):
            content_hash(make())

    def test_stray_instance_attribute_is_refused(self):
        p = _Point(1)
        p.extra = 5  # not a field: the fields no longer describe the instance
        with pytest.raises(TypeError, match="not its fields"):
            content_hash(p)

    def test_a_method_compiled_from_a_string_is_still_behaviour(self):
        # exec'd code shares the "<string>" marker with the decorator's own
        # methods, so the name has to carry the distinction.
        ns = {}
        exec(
            "import dataclasses\n"
            "@dataclasses.dataclass\n"
            "class C:\n"
            "    x: int\n"
            "    def magnitude(self): return self.x\n",
            ns,
        )
        with pytest.raises(TypeError, match="register_type"):
            content_hash(ns["C"](1))

    def test_a_callable_field_default_is_a_value(self):
        # A default sits in the class namespace under the field's own name;
        # it is data, and a function is hashed by its fingerprint.
        @dataclasses.dataclass
        class WithCallableDefault:
            op: object = _default_op

        assert content_hash(WithCallableDefault()) != content_hash(
            WithCallableDefault(lambda n: n + 1)
        )


def _default_op(n):
    return n


def _with_method():
    @dataclasses.dataclass
    class C:
        x: int

        def magnitude(self):
            return self.x

    return C(1)


def _with_property():
    @dataclasses.dataclass
    class C:
        x: int

        @property
        def double(self):
            return 2 * self.x

    return C(1)


def _with_cached_property():
    @dataclasses.dataclass
    class C:
        x: int

        @functools.cached_property
        def double(self):
            return 2 * self.x

    return C(1)


def _with_staticmethod():
    @dataclasses.dataclass
    class C:
        x: int

        @staticmethod
        def helper():
            return 1

    return C(1)


def _with_classmethod():
    @dataclasses.dataclass
    class C:
        x: int

        @classmethod
        def build(cls):
            return cls(1)

    return C(1)


def _with_dunder():
    @dataclasses.dataclass
    class C:
        x: int

        def __eq__(self, other):
            return True

    return C(1)


def _with_post_init():
    @dataclasses.dataclass
    class C:
        x: int

        def __post_init__(self):
            self.x += 1

    return C(1)


def _with_initvar():
    @dataclasses.dataclass
    class C:
        x: int
        scale: dataclasses.InitVar[int] = 1

    return C(1)


def _with_noninit_field():
    @dataclasses.dataclass
    class C:
        x: int
        y: int = dataclasses.field(init=False, default=0)

    return C(1)


def _with_inherited_method():
    class Base:
        def helper(self):
            return 1

    @dataclasses.dataclass
    class C(Base):
        x: int

    return C(1)


class TestPlainDataStore:
    @pytest.mark.parametrize(
        "value",
        [_Point(1, 2.0), _Frozen("a", (1, 2)), _Slotted(3)],
        ids=["plain", "frozen", "slots"],
    )
    def test_roundtrip_preserves_type_and_content(self, tmp_path, value):
        s = LocalStore(tmp_path)
        h = s.put_value(value)
        back = s.get_value(h)
        assert type(back) is type(value) and back == value
        assert content_hash(back) == h

    def test_nested_values_are_shared_by_content(self, tmp_path):
        arr = np.arange(4.0)
        arr.flags.writeable = False
        s = LocalStore(tmp_path)
        h = s.put_value({"a": _Frozen("x", (arr,)), "b": _Frozen("x", (arr,))})
        back = s.get_value(h)
        assert np.array_equal(back["a"].items[0], arr)
        assert content_hash(back["a"]) == content_hash(back["b"])
        assert len(list((tmp_path / "objects").rglob("*.npy"))) == 1  # stored once

    def test_inherited_and_kw_only_fields_roundtrip(self, tmp_path):
        s = LocalStore(tmp_path)
        assert s.get_value(s.put_value(_Derived(1, 2))) == _Derived(1, 2)
        assert content_hash(_Base(1)) != content_hash(_Derived(1))
        assert s.get_value(s.put_value(_KwOnly(1, b=5))) == _KwOnly(1, b=5)

    def test_locally_defined_class_is_hashable_but_not_storable(self, tmp_path):
        value = _with_local_class()
        content_hash(value)  # fine as an argument
        s = LocalStore(tmp_path)
        with pytest.raises(SerializationError, match="module level"):
            s.put_value(value)

    def test_changed_class_reads_as_a_miss(self, tmp_path, monkeypatch):
        s = LocalStore(tmp_path)
        h = s.put_value(_Point(1, 2.0))

        @dataclasses.dataclass
        class Grown:
            x: int
            y: float = 0.0
            z: float = 0.0

        Grown.__qualname__ = _Point.__qualname__
        monkeypatch.setattr(sys.modules[__name__], "_Point", Grown)
        with pytest.raises(CacheMiss):
            s.get_value(h)

    def test_unimported_class_reads_as_a_miss(self, tmp_path, monkeypatch):
        s = LocalStore(tmp_path)
        h = s.put_value(_Point(1, 2.0))
        monkeypatch.delattr(sys.modules[__name__], "_Point")
        with pytest.raises(CacheMiss):
            s.get_value(h)


def _with_local_class():
    @dataclasses.dataclass
    class Local:
        x: int

    return Local(1)


class TestPlainDataPure:
    def test_roundtrips_as_an_argument_and_a_return(self, cache):
        calls = []

        @pure
        def step(cfg):
            calls.append(cfg)
            return _Frozen("out", (cfg.x, cfg.y))

        first = step(_Point(3, 1.5))
        second = step(_Point(3, 1.5))
        assert len(calls) == 1  # the second call was a hit
        assert type(second) is _Frozen and second == first

    def test_a_changed_field_recomputes(self, cache):
        calls = []

        @pure
        def step(cfg):
            calls.append(cfg)
            return cfg.x

        step(_Point(3, 1.5))
        step(_Point(4, 1.5))
        assert len(calls) == 2

    def test_a_dataclass_is_still_rejected_by_the_map(self):
        # Caching does not imply freezing: a map needs a freeze strategy.
        with pytest.raises(TypeError):
            ImmutableMap({"cfg": _Point(1)})


# ===========================================================================
# @pure end-to-end
# ===========================================================================


class TestPure:
    def test_hit_skips_execution(self, cache):
        calls = []

        @pure
        def double(x):
            calls.append(1)
            return x * 2

        assert double(21) == 42
        assert double(21) == 42
        assert len(calls) == 1
        assert double(10) == 20
        assert len(calls) == 2

    def test_persists_across_decorations(self, cache):
        calls = []

        def make():
            @pure
            def step(x):
                calls.append(1)
                return x + 1

            return step

        assert make()(1) == 2
        assert make()(1) == 2  # fresh decoration, same code → same key
        assert len(calls) == 1

    def test_unrelated_key_does_not_invalidate(self, cache):
        calls = []

        @pure
        def geometry(obs):
            calls.append(1)
            return {"out": obs["ra"] + obs["dec"]}

        obs = ImmutableMap({"ra": 1.0, "dec": 2.0})
        r1 = geometry(obs)
        r2 = geometry(obs | {"notes": "run 3", "extra": np.arange(5)})
        assert len(calls) == 1
        assert r1["out"] == r2["out"] == 3.0

    def test_read_value_change_invalidates(self, cache):
        calls = []

        @pure
        def geometry(obs):
            calls.append(1)
            return {"out": obs["ra"] * 2}

        geometry(ImmutableMap({"ra": 1.0}))
        geometry(ImmutableMap({"ra": 5.0}))
        assert len(calls) == 2

    def test_config_granularity_nested(self, cache):
        calls = []

        @pure
        def filt(x, config):
            calls.append(1)
            return x * config["filter"]["order"]

        cfg = ImmutableMap({"filter": {"order": 4, "ripple": 0.1}, "plot": {"dpi": 100}})
        assert filt(2.0, cfg) == 8.0
        # change an unread nested key, and an entirely unread subtree:
        cfg2 = cfg | {"filter": {"order": 4, "ripple": 0.9}, "plot": {"dpi": 300}}
        assert filt(2.0, cfg2) == 8.0
        assert len(calls) == 1
        # change the read leaf:
        cfg3 = cfg | {"filter": {"order": 5, "ripple": 0.1}}
        assert filt(2.0, cfg3) == 10.0
        assert len(calls) == 2

    def test_plain_dict_is_depended_on_whole(self, cache):
        calls = []

        @pure
        def f(config):
            calls.append(1)
            assert type(config) is dict  # passed through untouched
            return config["a"]

        assert f({"a": 1, "b": 2}) == 1
        assert f({"a": 1, "b": 2}) == 1
        assert len(calls) == 1
        assert f({"a": 1, "b": 999}) == 1  # unread key, but nothing observed
        assert len(calls) == 2

    def test_immutable_map_opts_into_granularity(self, cache):
        calls = []

        @pure
        def f(config):
            calls.append(1)
            return config["a"]

        assert f(ImmutableMap({"a": 1, "b": 2})) == 1
        assert f(ImmutableMap({"a": 1, "b": 999})) == 1  # "b" was never read
        assert len(calls) == 1

    def test_derivation_works_inside_a_recorded_call(self, cache):
        # The recording proxy carries all of ImmutableMap's interface, so a
        # @pure function cannot tell it received one.
        @pure
        def f(ctx):
            merged = ctx | {"scaled": 2}  # __or__
            under = {"base": 0} | ctx  # __ror__
            with_k = ctx.assoc("k", 1)
            without = ctx.dissoc("raw")
            return {
                "kinds": tuple(
                    type(m) is ImmutableMap for m in (merged, under, with_k, without)
                ),
                "merged": (merged["scaled"], merged["raw"]),
                "under": (under["base"], under["raw"]),
                "with_k": with_k["k"],
                "dropped": "raw" not in without,
            }

        out = f(ImmutableMap({"raw": 1}))
        assert out["kinds"] == (True,) * 4  # plain maps, not proxies
        assert out["merged"] == (2, 1)
        assert out["under"] == (0, 1)  # self wins on conflict
        assert out["with_k"] == 1
        assert out["dropped"] is True

    def test_derivation_records_the_whole_map(self, cache):
        # Deriving copies every key, so it is a whole-map read: an unrelated
        # key must invalidate, unlike a single-leaf read.
        calls = []

        @pure
        def f(ctx):
            calls.append(1)
            return (ctx | {"b": 2})["a"]

        assert f(ImmutableMap({"a": 1})) == 1
        assert f(ImmutableMap({"a": 1})) == 1
        assert len(calls) == 1
        assert f(ImmutableMap({"a": 1, "unread": 99})) == 1
        assert len(calls) == 2

    def test_recorded_map_pickles_as_a_plain_map(self, cache):
        import pickle

        @pure
        def f(ctx):
            back = pickle.loads(pickle.dumps(ctx))
            return {"kind": type(back) is ImmutableMap, "n": len(back)}

        out = f(ImmutableMap({"a": 1, "b": 2}))
        assert out["kind"] is True and out["n"] == 2

    def test_absence_is_a_dependency(self, cache):
        calls = []

        @pure
        def f(cfg):
            calls.append(1)
            return cfg.get("detrend", 0)

        m = ImmutableMap({"other": 1})
        assert f(m) == 0
        assert f(m | {"unrelated": 5}) == 0
        assert len(calls) == 1
        assert f(m | {"detrend": 7}) == 7  # the probed key appearing invalidates
        assert len(calls) == 2

    def test_contains_presence_dependency(self, cache):
        calls = []

        @pure
        def f(cfg):
            calls.append(1)
            return 1 if "mode" in cfg else 0

        assert f(ImmutableMap({"mode": "a"})) == 1
        assert f(ImmutableMap({"mode": "b"})) == 1  # presence only; value unread
        assert len(calls) == 1
        assert f(ImmutableMap({})) == 0
        assert len(calls) == 2

    def test_iteration_reads_everything(self, cache):
        calls = []

        @pure
        def f(m):
            calls.append(1)
            return sum(m[k] for k in m)

        assert f(ImmutableMap({"a": 1, "b": 2})) == 3
        assert f(ImmutableMap({"a": 1, "b": 2})) == 3
        assert len(calls) == 1
        assert f(ImmutableMap({"a": 1, "b": 2, "c": 3})) == 6  # any change invalidates
        assert len(calls) == 2

    def test_conditional_reads_get_separate_traces(self, cache):
        calls = []

        @pure
        def f(cfg):
            calls.append(1)
            if cfg["mode"] == "a":
                return cfg["x"]
            return cfg["y"]

        m = ImmutableMap({"mode": "a", "x": 1, "y": 2})
        assert f(m) == 1
        assert f(m | {"mode": "b"}) == 2
        assert len(calls) == 2
        # branch "a" must not be invalidated by a change to y (unread there):
        assert f(m | {"y": 99}) == 1
        assert len(calls) == 2

    def test_argument_binding_normalized(self, cache):
        calls = []

        @pure
        def f(a, b=10):
            calls.append(1)
            return a + b

        assert f(1, 2) == 3
        assert f(a=1, b=2) == 3
        assert f(b=2, a=1) == 3
        assert len(calls) == 1
        assert f(1) == 11  # default applied → distinct key
        assert f(1, 10) == 11  # explicit == default → same key
        assert len(calls) == 2

    def test_array_argument_content_keyed(self, cache):
        calls = []

        @pure
        def total(x):
            calls.append(1)
            return float(np.sum(x))

        a = np.arange(5.0)
        assert total(a) == 10.0
        assert total(np.arange(5.0)) == 10.0  # different object, same content
        assert len(calls) == 1
        # boundary freeze: mutating the caller's array cannot poison the key
        a[0] = 100.0
        assert total(a) == 110.0
        assert len(calls) == 2

    def test_returned_dict_merges(self, cache):
        @pure
        def geom(obs):
            return {"sum": obs["a"] + obs["b"]}

        obs = ImmutableMap({"a": 1, "b": 2})
        obs = obs | geom(obs)
        assert obs["sum"] == 3
        obs2 = obs | geom(obs)  # hit; loaded value merges identically
        assert obs2["sum"] == 3

    def test_arrays_in_results_roundtrip(self, cache):
        calls = []

        @pure
        def make(n):
            calls.append(1)
            return {"arr": np.arange(float(n))}

        r1 = make(5)
        r2 = make(5)
        assert len(calls) == 1
        assert np.array_equal(r1["arr"], r2["arr"])
        # The function built a writeable array, so the hit yields one too.
        assert r1["arr"].flags.writeable
        assert r2["arr"].flags.writeable

    def test_readonly_arrays_in_results_stay_readonly(self, cache):
        @pure
        def make(n):
            arr = np.arange(float(n))
            arr.flags.writeable = False
            return {"arr": arr}

        assert not make(5)["arr"].flags.writeable
        assert not make(5)["arr"].flags.writeable  # hit: memory-mapped

    def test_exceptions_not_cached(self, cache):
        calls = []

        @pure
        def flaky(x):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("boom")
            return x

        with pytest.raises(ValueError):
            flaky(1)
        assert flaky(1) == 1  # re-executes; the failure wrote nothing
        assert len(calls) == 2

    def test_code_change_invalidates_but_rename_does_not(self, cache):
        calls = []
        ns = {"calls": calls, "pure": pure}
        exec("@pure\ndef f(x):\n    calls.append(1)\n    return x + 1", ns)
        assert ns["f"](1) == 2
        exec("@pure\ndef renamed(x):\n    calls.append(1)\n    return x + 1", ns)
        assert ns["renamed"](1) == 2  # same code → hit despite the rename
        assert len(calls) == 1
        exec("@pure\ndef f(x):\n    calls.append(1)\n    return x + 2", ns)
        assert ns["f"](1) == 3  # body changed → miss
        assert len(calls) == 2

    def test_targeted_clear(self, cache):
        calls = []

        @pure
        def f(x):
            calls.append("f")
            return x + 1

        @pure
        def g(x):
            calls.append("g")
            return x + 2

        assert f(1) == 2 and g(1) == 3
        vk.clear_cache(f)
        assert f(1) == 2 and g(1) == 3
        # f forgot and recomputed; unrelated g's cache was untouched:
        assert calls == ["f", "g", "f"]

    def test_clear_reaches_callers_transitively(self, cache):
        calls = []

        @pure
        def leaf(x):
            calls.append("c")
            return x + 1

        @pure
        def mid(x):
            calls.append("b")
            return leaf(x) * 2

        @pure
        def top(x):
            calls.append("a")
            return mid(x) + 3

        @pure
        def bystander(x):
            calls.append("z")
            return x * 10

        assert top(1) == 7 and bystander(1) == 10
        calls.clear()

        vk.clear_cache(leaf)  # "leaf has changed"
        assert top(1) == 7
        assert bystander(1) == 10
        # The whole chain through leaf recomputed; the bystander hit:
        assert calls == ["a", "b", "c"]

    def test_clear_reaches_argument_uses(self, cache):
        calls = []

        @pure
        def double(x):
            return x * 2

        @pure
        def apply(x, fn):
            calls.append(1)
            return fn(x)

        assert apply(3, double) == 6
        assert apply(3, double) == 6
        assert len(calls) == 1
        vk.clear_cache(double)  # traces keyed on double-as-argument must go
        assert apply(3, double) == 6
        assert len(calls) == 2

    def test_clear_reaches_module_attribute_callers(self, cache, tmp_path):
        import importlib.util
        import sys as _sys

        modfile = tmp_path / "vk_steps_mod.py"
        modfile.write_text(
            "from valuekit import pure\n"
            "CALLS = []\n"
            "@pure\n"
            "def step(x):\n"
            "    CALLS.append(1)\n"
            "    return x + 5\n"
        )
        spec = importlib.util.spec_from_file_location("vk_steps_mod", modfile)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["vk_steps_mod"] = mod
        spec.loader.exec_module(mod)
        try:
            outer_calls = []
            ns = {"pure": pure, "steps": mod, "outer_calls": outer_calls}
            exec(
                "@pure\n"
                "def outer(x):\n"
                "    outer_calls.append(1)\n"
                "    return steps.step(x) * 2\n",
                ns,
            )
            outer = ns["outer"]
            assert outer(1) == 12 and outer(1) == 12
            assert outer_calls == [1]
            vk.clear_cache(mod.step)
            assert outer(1) == 12
            # outer reached step only through the module: still cleared
            assert outer_calls == [1, 1]
        finally:
            del _sys.modules["vk_steps_mod"]

    
    def test_targeted_clear_rejects_undecorated(self, cache):
        with pytest.raises(TypeError):
            vk.clear_cache(lambda x: x)


    def test_deleting_cache_is_always_safe(self, cache):
        calls = []

        @pure
        def f(x):
            calls.append(1)
            return {"y": np.arange(x)}

        f(4)
        vk.clear_cache()
        r = f(4)
        assert len(calls) == 2 and len(r["y"]) == 4

    def test_corrupt_result_file_recomputes(self, cache):
        calls = []

        @pure
        def f(x):
            calls.append(1)
            return (x, x + 1)

        f(3)
        for p in (cache / "objects").rglob("*.bin"):
            p.write_bytes(b"junk")
        assert f(3) == (3, 4)
        assert len(calls) == 2

    def test_no_store_degrades_to_plain_call(self):
        calls = []

        @pure
        def f(x):
            calls.append(1)
            return x

        f(1)
        f(1)
        assert len(calls) == 2  # no caching configured

    def test_uncached_escape_hatch(self, cache):
        calls = []

        @pure
        def f(x):
            calls.append(1)
            return x

        f(1)
        f.uncached(1)
        f.uncached(1)
        assert len(calls) == 3

    def test_rejects_var_args_and_methods(self):
        with pytest.raises(TypeError):

            @pure
            def f(*args):
                return args

        with pytest.raises(TypeError):

            class C:
                @pure
                def m(self, x):
                    return x

    def test_unregistered_argument_type_rejected(self, cache):
        @pure
        def f(x):
            return 1

        class Weird:
            pass

        with pytest.raises(TypeError):
            f(Weird())

    def test_unstorable_result_rejected(self, cache):
        @pure
        def f(x):
            return lambda: x

        with pytest.raises(SerializationError):
            f(1)

    def test_lambda_argument_is_part_of_key(self, cache):
        calls = []

        @pure
        def apply(x, fn):
            calls.append(1)
            return fn(x)

        assert apply(3, lambda v: v * 2) == 6
        assert apply(3, lambda v: v * 2) == 6
        assert len(calls) == 1
        assert apply(3, lambda v: v * 10) == 30
        assert len(calls) == 2


# ===========================================================================
# transparency: @pure caches, and does not convert
# ===========================================================================


class TestTransparency:
    """@pure must be invisible apart from the skipped execution: same types
    in, same types out, and the same behaviour whether or not a cache is
    configured."""

    def test_argument_objects_are_passed_through(self, cache):
        seen = []

        @pure
        def f(d, items, tags, arr):
            seen.append((d, items, tags, arr))
            return len(d) + len(items) + len(tags) + len(arr)

        d, items, tags = {"a": 1}, [1, 2], {"x"}
        arr = np.arange(3.0)
        assert f(d, items, tags, arr) == 7
        got_d, got_items, got_tags, got_arr = seen[0]
        assert got_d is d and got_items is items and got_tags is tags
        assert got_arr is arr and got_arr.flags.writeable

    def test_mutable_containers_round_trip_as_themselves(self, cache):
        calls = []

        @pure
        def build(n):
            calls.append(1)
            return {"items": [n, n + 1], "tags": {"a"}, "pair": (n, n)}

        miss = build(1)
        hit = build(1)
        assert len(calls) == 1
        assert miss == hit
        assert type(hit["items"]) is list
        assert type(hit["tags"]) is set
        assert type(hit["pair"]) is tuple
        assert type(hit) is dict

    def test_dict_result_keeps_its_order(self, cache):
        @pure
        def build(n):
            return {"b": n, "a": n}

        assert list(build(1)) == ["b", "a"]
        assert list(build(1)) == ["b", "a"]  # hit

    def test_uncached_and_cached_agree(self, tmp_path):
        @pure
        def f(d, items):
            return type(d).__name__, type(items).__name__, d["a"] + sum(items)

        vk.set_cache_dir(None)
        uncached = f({"a": 1}, [2, 3])
        vk.set_cache_dir(tmp_path / "cache")
        try:
            assert f({"a": 1}, [2, 3]) == uncached  # miss
            assert f({"a": 1}, [2, 3]) == uncached  # hit
        finally:
            vk.set_cache_dir(None)

    def test_recorded_map_is_an_immutable_map(self, cache):
        @pure
        def f(ctx):
            return (
                isinstance(ctx, ImmutableMap),
                repr(ctx),
                ctx == ImmutableMap({"a": 1}),
            )

        is_map, text, eq = f(ImmutableMap({"a": 1}))
        assert is_map
        assert text == "ImmutableMap({'a': 1})"  # the proxy does not show
        assert eq

    def test_returned_argument_map_is_plain(self, cache):
        from valuekit.recording import RecordingMap

        @pure
        def identity(ctx):
            return ctx

        out = identity(ImmutableMap({"a": 1}))
        assert not isinstance(out, RecordingMap)
        assert out == ImmutableMap({"a": 1})
        assert identity(ImmutableMap({"a": 1})) == out  # hit agrees

    def test_proxy_nested_in_a_result_does_not_escape(self, cache):
        from valuekit.recording import RecordingMap

        @pure
        def wrap(ctx):
            return {"ctx": ctx, "pair": [ctx, 1]}

        miss = wrap(ImmutableMap({"a": 1}))
        hit = wrap(ImmutableMap({"a": 1}))
        for out in (miss, hit):
            assert type(out["ctx"]) is ImmutableMap
            assert type(out["pair"][0]) is ImmutableMap
            assert not isinstance(out["ctx"], RecordingMap)
        assert miss == hit

    def test_returning_a_map_depends_on_all_of_it(self, cache):
        calls = []

        @pure
        def wrap(ctx):
            calls.append(1)
            return {"ctx": ctx}

        assert wrap(ImmutableMap({"a": 1}))["ctx"] == ImmutableMap({"a": 1})
        assert len(calls) == 1
        # No key was read, but the whole map was returned, so any change to
        # it must invalidate.
        assert wrap(ImmutableMap({"a": 1, "b": 2}))["ctx"] == ImmutableMap(
            {"a": 1, "b": 2}
        )
        assert len(calls) == 2

    def test_trace_records_arguments_as_passed(self, cache):
        # Mutating an argument breaks the purity contract, but the trace is
        # still keyed on what was handed in, so the call is reusable.
        calls = []

        @pure
        def bad(items):
            calls.append(1)
            items.append(99)
            return sum(items)

        assert bad([1, 2]) == 102
        xs = [1, 2]
        assert bad(xs) == 102
        assert len(calls) == 1
        assert xs == [1, 2]  # the hit did not run the body

    def test_hash_only_registration_allows_arguments(self, cache):
        class Tag:
            def __init__(self, name):
                self.name = name

        vk.register_type(Tag, hash_fn=lambda v, h: h.update(v.name.encode()))
        calls = []

        @pure
        def f(tag):
            calls.append(1)
            return tag.name.upper()

        assert f(Tag("a")) == "A"
        assert f(Tag("a")) == "A"
        assert len(calls) == 1
        assert f(Tag("b")) == "B"
        assert len(calls) == 2
        # No freeze strategy, so it still may not enter a map.
        with pytest.raises(TypeError):
            ImmutableMap({"tag": Tag("a")})


# ===========================================================================
# nested @pure
# ===========================================================================


class TestNestedPure:
    def test_map_passed_inward_records_in_both_traces(self, cache):
        calls = []

        @pure
        def inner(ctx):
            calls.append("i")
            return ctx["b"]

        @pure
        def outer(ctx):
            calls.append("o")
            return ctx["a"] + inner(ctx)

        base = {"a": 1, "b": 10, "unread": 0}
        assert outer(ImmutableMap(base)) == 11
        assert calls == ["o", "i"]

        # A key neither function read: both stay valid.
        assert outer(ImmutableMap(base | {"unread": 99})) == 11
        assert calls == ["o", "i"]

        # A key only the *inner* function read: the outer must not shortcut
        # past it, even though the outer never read "b" itself.
        assert outer(ImmutableMap(base | {"b": 20})) == 21
        assert calls == ["o", "i", "o", "i"]

    def test_nested_map_reads_stay_fine_grained(self, cache):
        calls = []

        @pure
        def inner(ctx):
            calls.append("i")
            return ctx["cfg"]["order"]

        @pure
        def outer(ctx):
            calls.append("o")
            return inner(ctx) * 2

        base = {"cfg": {"order": 4, "dpi": 100}}
        assert outer(ImmutableMap(base)) == 8
        assert calls == ["o", "i"]
        # A sibling key inside the sub-map, read by neither: still valid.
        assert outer(ImmutableMap({"cfg": {"order": 4, "dpi": 300}})) == 8
        assert calls == ["o", "i"]
        assert outer(ImmutableMap({"cfg": {"order": 5, "dpi": 100}})) == 10
        assert calls == ["o", "i", "o", "i"]

    def test_inner_edit_invalidates_outer(self, cache):
        def build(inner_body):
            ns = {"pure": pure, "calls": []}
            exec(
                f"def inner(x):\n    calls.append('i')\n    return {inner_body}\n"
                "inner = pure(inner)\n"
                "def outer(x):\n    calls.append('o')\n    return inner(x) + 1\n"
                "outer = pure(outer)",
                ns,
            )
            return ns["outer"], ns["calls"]

        outer1, calls1 = build("x * 2")
        assert outer1(3) == 7
        assert calls1 == ["o", "i"]

        # Identical code, fresh decoration: outer hits, inner never even called.
        outer1b, calls1b = build("x * 2")
        assert outer1b(3) == 7
        assert calls1b == []

        # Edit ONLY the inner function: outer must recompute.
        outer2, calls2 = build("x * 3")
        assert outer2(3) == 10
        assert "o" in calls2

    def test_outer_spans_include_inner(self, cache):
        @pure
        def inner(x):
            return x * 2

        @pure
        def outer(x):
            return inner(x) + 1

        inner_spans = inner._valuekit_identity()[1]
        outer_spans = set(map(tuple, outer._valuekit_identity()[1]))
        inner_files = {f for f, _, _ in inner_spans}
        assert any(
            f in inner_files and lo <= inner_spans[0][1] <= hi
            for f, lo, hi in outer_spans
        )

    def test_breakpoint_in_inner_forces_whole_chain(self, cache, monkeypatch):
        import bdb

        calls = []

        @pure
        def inner(x):
            calls.append("i")
            return x * 2

        @pure
        def outer(x):
            calls.append("o")
            return inner(x) + 1

        # Warm both caches first.
        assert outer(3) == 7
        assert calls == ["o", "i"]

        # Breakpoint in *inner* only: a warm outer must NOT shortcut past it.
        fname, lo, _ = inner._valuekit_identity()[1][0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        assert outer(3) == 7
        assert calls == ["o", "i", "o", "i"]  # both executed for real

        # And those forced runs persisted nothing new:
        dbg.clear_all_breaks()
        assert outer(3) == 7
        assert calls == ["o", "i", "o", "i"]  # original trace still hits

    def test_midrun_force_taints_enclosing_recording(self, cache, monkeypatch):
        import bdb

        calls = []
        dbg = bdb.Bdb()
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        @pure
        def inner(x):
            calls.append("i")
            return x * 2

        fname, lo, _ = inner._valuekit_identity()[1][0]
        armed = [True]

        @pure
        def outer(x):
            calls.append("o")
            if armed[0]:
                # Simulates the user adding a breakpoint while paused mid-run,
                # after outer's own entry check already passed:
                armed[0] = False
                dbg.set_break(fname, lo + 1)
            return inner(x) + 1

        assert outer(3) == 7  # inner was forced inside outer's recording
        dbg.clear_all_breaks()
        assert outer(3) == 7
        # outer's first recording was tainted and discarded, so this second
        # call had to execute again (and could then record cleanly):
        assert calls.count("o") == 2
        assert outer(3) == 7
        assert calls.count("o") == 2  # third call hits the clean recording

    def test_breakpoint_in_one_stage_keeps_sibling_caches(self, cache, monkeypatch):
        import bdb

        calls = []

        @pure
        def stage_a(x):
            calls.append("a")
            return x + 1

        @pure
        def stage_b(x):
            calls.append("b")
            return x + 2

        @pure
        def stage_c(x):
            calls.append("c")
            return x + 3

        @pure
        def process_batch(x):
            calls.append("p")
            return stage_a(x) + stage_b(x) + stage_c(x)

        assert process_batch(1) == 9  # warm everything, cleanly
        assert calls == ["p", "a", "b", "c"]

        fname, lo, _ = stage_c._valuekit_identity()[1][0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        # Only the root-to-breakpoint path is forced; siblings hit.
        calls.clear()
        assert process_batch(1) == 9
        assert calls == ["p", "c"]

        # Sibling caches also POPULATE during the debug session:
        calls.clear()
        assert process_batch(2) == 12
        assert calls == ["p", "a", "b", "c"]  # new input: everything misses once
        calls.clear()
        assert process_batch(2) == 12
        assert calls == ["p", "c"]  # a(2), b(2) recorded despite forced parent

        dbg.clear_all_breaks()

        # The pre-debug recording of process_batch(1) survived untouched:
        calls.clear()
        assert process_batch(1) == 9
        assert calls == []

        # process_batch(2)/stage_c(2) only ever ran forced → record cleanly now:
        calls.clear()
        assert process_batch(2) == 12
        assert calls == ["p", "c"]
        calls.clear()
        assert process_batch(2) == 12
        assert calls == []

    def test_forward_references_tracked(self, cache):
        # Names are resolved at first call, so constants and helpers defined
        # BELOW the @pure function are still part of its identity.
        def build(mult):
            ns = {"pure": pure, "calls": []}
            exec(
                "@pure\n"
                "def f(x):\n"
                "    calls.append(1)\n"
                "    return helper(x)\n"
                f"MULT = {mult}\n"          # defined after decoration
                "def helper(x):\n"           # so is the helper
                "    return x * MULT\n",
                ns,
            )
            return ns["f"], ns["calls"]

        f1, c1 = build(2)
        assert f1(3) == 6
        f1b, c1b = build(2)
        assert f1b(3) == 6 and c1b == []   # identical late defs → hit
        f2, c2 = build(5)
        assert f2(3) == 15 and c2 == [1]   # late-defined constant edit → miss

    def test_mutual_recursion(self, cache):
        calls = []
        ns = {"pure": pure, "calls": calls}
        exec(
            "@pure\n"
            "def even(n):\n"
            "    calls.append('e')\n"
            "    return True if n == 0 else odd(n - 1)\n"
            "@pure\n"
            "def odd(n):\n"
            "    calls.append('o')\n"
            "    return False if n == 0 else even(n - 1)\n",
            ns,
        )
        assert ns["even"](4) is True
        n_first = len(calls)
        assert ns["even"](4) is True
        assert len(calls) == n_first  # full hit; identities were computable

    def test_pure_function_as_argument_unwrapped(self, cache):
        calls = []

        @pure
        def double(x):
            return x * 2

        @pure
        def apply(x, fn):
            calls.append(1)
            return fn(x)

        assert apply(3, double) == 6
        assert apply(3, double) == 6
        assert len(calls) == 1





class TestDebugHook:
    def test_no_debugger_no_forcing(self):
        assert debughook.breakpoints_force([("/x.py", 1, 10)]) is False

    def test_always_run_env(self, monkeypatch):
        monkeypatch.setenv("VALUEKIT_ALWAYS_RUN", "1")
        assert debughook.breakpoints_force([]) is True

    def test_bdb_breakpoint_intersection(self, cache, tmp_path, monkeypatch):
        import bdb

        calls = []

        @pure
        def f(x):
            calls.append(1)
            return x

        fname, lo, hi = f._valuekit_identity()[1][0]

        dbg = bdb.Bdb()
        dbg.set_break(fname, lo)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        f(1)
        f(1)  # breakpoint in span → forced execution, no cache write
        assert len(calls) == 2

        dbg.clear_all_breaks()
        dbg.set_break(fname, hi + 500)  # elsewhere in the file
        f(1)  # miss (nothing was written while forced) → runs and records
        f(1)  # now hits
        assert len(calls) == 3

    def test_forced_runs_write_nothing(self, cache, monkeypatch):
        calls = []

        @pure
        def f(x):
            calls.append(1)
            return x

        monkeypatch.setenv("VALUEKIT_ALWAYS_RUN", "1")
        f(1)
        f(1)
        monkeypatch.delenv("VALUEKIT_ALWAYS_RUN")
        f(1)  # nothing was recorded during forced runs
        f(1)
        assert len(calls) == 3

    def test_unknown_tracer_does_not_force(self, monkeypatch):
        monkeypatch.setattr("sys.gettrace", lambda: (lambda *a: None))
        assert debughook.breakpoints_force([("/x.py", 1, 10)]) is False


# ===========================================================================
# concurrency-ish / atomicity smoke test
# ===========================================================================


def test_two_stores_share_directory(tmp_path):
    s1 = LocalStore(tmp_path)
    s2 = LocalStore(tmp_path)
    h = s1.put_value(freeze((1, 2, 3)))
    assert s2.get_value(h) == (1, 2, 3)
    s2.put_value(freeze((1, 2, 3)))  # idempotent
    t = {"fn": "f", "deps": {}, "result": h}
    s1.put_trace("k", t)
    s2.put_trace("k", t)
    assert len(s1.get_traces("k")) == 1


# ===========================================================================
# run_all: parallel execution
# ===========================================================================


def _write_batch_module(tmp_path):
    """A scenario module written to disk so that spawn workers can import
    the functions by reference. Execution counts go to an append-only log
    (atomic across processes)."""
    log = tmp_path / "runs.log"
    mod = tmp_path / "vk_batch_mod.py"
    mod.write_text(
        "from valuekit import pure\n"
        f"LOG = {str(log)!r}\n"
        "def _note(tag):\n"
        "    with open(LOG, 'a') as f:\n"
        "        f.write(tag + '\\n')\n"
        "@pure\n"
        "def load(sid):\n"
        "    _note(f'L{sid}')\n"
        "    return sid * 10\n"
        "@pure\n"
        "def analyse(sid, x):\n"
        "    _note(f'A{sid}')\n"
        "    if sid == 3:\n"
        "        raise ValueError('bad calibration in scenario 3')\n"
        "    return x + 1\n"
        "def process(sid):\n"
        "    return analyse(sid, load(sid))\n"
        "def hard_death(x):\n"
        "    import os\n"
        "    if x == 1:\n"
        "        os._exit(1)\n"
        "    return x * 2\n"
        "def quick_or_hang(sid):\n"
        "    import time\n"
        "    x = load(sid)\n"
        "    if sid == 9:\n"
        "        _note('H9')\n"
        "        time.sleep(60)\n"
        "        _note('W9')\n"
        "    return x\n"
    )
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("vk_batch_mod", mod)
    m = importlib.util.module_from_spec(spec)
    _sys.modules["vk_batch_mod"] = m
    _sys.path.insert(0, str(tmp_path))  # spawn children inherit sys.path
    spec.loader.exec_module(m)

    def counts():
        try:
            lines = log.read_text().splitlines()
        except OSError:
            lines = []
        return lines

    return m, counts


@pytest.fixture
def debugger_attached(monkeypatch):
    """Simulate an attached debugger with no breakpoints set (bdb-based)."""
    import bdb

    dbg = bdb.Bdb()
    monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)
    return dbg


class TestRunAll:
    # ---- basics -----------------------------------------------------------

    def test_results_in_order_and_workers_cache(self, cache, tmp_path):
        # The cache is configured only via set_cache_dir in this process
        # (the fixture); under spawn, workers see it only through run_all's
        # initialiser. A fully cached second round proves the propagation.
        m, counts = _write_batch_module(tmp_path)
        ids = [7, 5, 6]
        r1 = vk.run_all(m.process, ids, max_workers=2)
        assert r1.values == [71, 51, 61]  # input order preserved
        assert [o.input for o in r1] == ids
        assert r1.failures == []
        n = len(counts())
        assert n == 6  # 3 loads + 3 analyses, all in workers
        r2 = vk.run_all(m.process, ids, max_workers=2)
        assert r2.values == r1.values
        assert len(counts()) == n  # second round: all hits, zero executions

    def test_spontaneous_worker_death_isolated_and_recorded(self, cache, tmp_path):
        # A segfault-like death loses exactly its own input; siblings and
        # the rest of the batch are unaffected.
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.hard_death, [1, 2, 3], max_workers=2)
        assert isinstance(r, vk.BatchResult)
        assert r[1].result() == 4 and r[2].result() == 6  # isolation
        [(x, exc)] = r.failures
        assert x == 1
        assert "died without raising" in str(exc) and "exit code" in str(exc)

    # ---- failure collection --------------------------------------------------

    def test_collect_mode_processes_everything(self, cache, tmp_path):
        m, counts = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2, 3, 4], max_workers=1)
        # scenario 3 failed, but 4 was still processed afterwards:
        c = counts()
        assert "L4" in c and "A4" in c
        assert isinstance(r, vk.BatchResult)
        assert len(r) == 4
        assert [(x, type(e).__name__) for x, e in r.failures] == [(3, "ValueError")]
        assert r[2].input == 3
        assert isinstance(r[2].exception(), ValueError)
        with pytest.raises(ValueError, match="scenario 3"):
            r[2].result()
        assert r[0].result() == 11

    def test_collect_mode_values_raises_exception_group(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 3], max_workers=1)
        with pytest.raises(ExceptionGroup, match="1 of 2 inputs failed") as ei:
            r.values
        (sub,) = ei.value.exceptions
        assert isinstance(sub, ValueError)
        assert "input: 3" in getattr(sub, "__notes__", [])
        with pytest.raises(ExceptionGroup):
            r.values  # a second access must not duplicate the note
        assert getattr(sub, "__notes__", []).count("input: 3") == 1

    # ---- timeout -------------------------------------------------------------

    def test_timeout_is_per_input_and_prompt(self, cache, tmp_path):
        import time

        m, counts = _write_batch_module(tmp_path)
        t0 = time.monotonic()
        r = vk.run_all(m.quick_or_hang, [7, 9, 8], max_workers=3, timeout=1.5)
        assert time.monotonic() - t0 < 20  # the hang did not stall the batch
        # the healthy inputs completed normally:
        assert r[0].result() == 70 and r[2].result() == 80
        [(x, exc)] = r.failures
        assert x == 9 and isinstance(exc, TimeoutError)
        time.sleep(0.5)
        assert "W9" not in counts()  # the hung process was killed, not finished

    def test_capacity_not_degraded_by_timeout(self, cache, tmp_path):
        # After a kill, queued inputs still start: worker capacity is
        # replaced, not consumed, by a timed-out input.
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.quick_or_hang, [9, 1, 2, 3], max_workers=2, timeout=1.5)
        assert len(r) == 4
        assert [x for x, _ in r.failures] == [9]
        assert [o.result() for o in r if o.input != 9] == [10, 20, 30]

    def test_timeout_unbreached_collects_normally(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2, 3], max_workers=2, timeout=60)
        assert [x for x, _ in r.failures] == [3]
        assert r[0].result() == 11

    # ---- an attached debugger is not, by itself, a mode -----------------------

    def test_debugger_attached_changes_nothing(
        self, cache, tmp_path, debugger_attached
    ):
        # Only a *breakpoint* changes how a batch runs. Merely having a
        # debugger attached must not: failures are collected exactly as
        # they are without one, and every input is still processed.
        m, counts = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2, 3, 4], max_workers=2)
        assert isinstance(r, vk.BatchResult)
        assert len(r) == 4
        assert [(x, type(e).__name__) for x, e in r.failures] == [(3, "ValueError")]
        assert r[0].result() == 11
        c = counts()
        assert "L4" in c and "A4" in c  # input 4 ran despite 3 failing
        assert c.count("A3") == 1  # the failure was not replayed

    # ---- sequential mode (breakpoints) ---------------------------------------

    def test_breakpoint_forces_sequential(self, cache, monkeypatch):
        import bdb

        calls = []

        @pure
        def step(x):
            calls.append(x)  # visible only if executed in THIS process
            return x + 1

        def batch(x):
            return step(x)

        fname, lo, _ = step._valuekit_identity()[1][0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        assert vk.run_all(batch, [1, 2, 3]).values == [2, 3, 4]
        assert calls == [1, 2, 3]  # sequential, in-process: breakpoints fire
