"""valuekit test suite.

Focus: the correctness edges — staleness, granularity, arg binding,
atomic store behaviour — rather than demos.
"""

import atexit
import dataclasses
import signal
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import warnings
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path as _Path

import numpy as np
import pytest

import valuekit as vk
from valuekit import ImmutableMap, pure, freeze, content_hash
from valuekit import bootstrap
from valuekit import functionhash
from valuekit import events
from valuekit import parallel
from valuekit import project
from valuekit import protocol
from valuekit.functionhash import PYTHON, _classify, reachable_set
from valuekit.store import LocalStore, CacheMiss, SerializationError, record_hash
from valuekit.values import encode_key, decode_key
from valuekit import pure as _pure_mod  # module alias for store poking
from valuekit import debughook


@pytest.fixture()
def cache(tmp_path):
    vk.set_store_dir(tmp_path / "cache")
    yield tmp_path / "cache"
    vk.set_store_dir(None)


@pytest.fixture(autouse=True)
def _no_cache_by_default(monkeypatch):
    vk.set_store_dir(None)
    # An in-process worker handshake installs the main process's extension
    # markers; no test starts with another test's.
    monkeypatch.setattr(functionhash, "_markers_here", None)
    yield
    vk.set_store_dir(None)


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
        return reachable_set(fn, **kw).hash


class TestFunctionHash:
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
            reachable_set(ns["f"])

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
            fp1 = reachable_set(build(["ch1", "ch2"])).hash
            fp2 = reachable_set(build(["completely", "different"])).hash
        assert fp1 == fp2  # not our problem, by explicit design

    def test_opaque_globals_silent_and_stable(self):
        def build():
            ns = {"HANDLE": object(), "BUF": bytearray(b"x")}
            exec("def f(x):\n    return x if HANDLE and BUF else x", ns)
            return ns["f"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fp1 = reachable_set(build()).hash
            fp2 = reachable_set(build()).hash
        assert fp1 == fp2  # distinct opaque objects: untracked, stable


# ===========================================================================
# submodules reached by attribute access
# ===========================================================================
#
# ``mypkg.sub.f()`` spells ``sub`` and ``f`` as attribute names, and an
# attribute name resolves to nothing at module scope.  A package's source
# file is only its __init__.py, so a walk that stops there depends on
# nothing sub.py says: editing it left the function_hash unchanged and @pure
# served a stale result.  Every test here is a regression guard for that.

_TEST_PKGS = ("vk_sub_pkg", "vk_deep_pkg", "vk_flat_mod", "vk_cyc_a", "vk_cyc_b")


def _write_tree(root, files: dict) -> None:
    """Write {"pkg/mod.py": source, ...} under *root*."""
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)


def _fresh_import(root, name: str):
    """Import *name* from *root*, discarding any copy already imported.

    Each version of a package goes in its own directory, so no stale .pyc
    can be reused when a rewrite happens to leave mtime and size alone.
    """
    import importlib

    for m in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[m]
    importlib.invalidate_caches()
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(root))


def _pkg_versions(tmp_path, files_for, name):
    """Yield the module built from files_for(src) for two leaf sources."""
    for i, leaf in enumerate(("def f(x):\n    return x + 1\n", "def f(x):\n    return x + 555555\n")):
        root = tmp_path / f"v{i}"
        _write_tree(root, files_for(leaf))
        yield _fresh_import(root, name)


class TestSubmoduleWalk:
    @pytest.fixture(autouse=True)
    def _purge(self):
        yield
        for m in [k for k in sys.modules if k.startswith(_TEST_PKGS)]:
            del sys.modules[m]

    @staticmethod
    def _pkg(leaf):
        return {
            "vk_sub_pkg/__init__.py": "from . import leaf\n",
            "vk_sub_pkg/leaf.py": leaf,
        }

    def test_attribute_submodule_invalidates(self, tmp_path):
        # The bug: `import pkg` then `pkg.leaf.f(x)`.
        fps = []
        for mod in _pkg_versions(tmp_path, self._pkg, "vk_sub_pkg"):
            ns = {"pkg": mod}
            exec("def step(x):\n    return pkg.leaf.f(x)", ns)
            fps.append(_fp(ns["step"]))
        assert fps[0] != fps[1]

    def test_explicit_submodule_import_invalidates(self, tmp_path):
        # `import pkg.leaf` binds `pkg`, so the call site is identical.
        fps = []
        for mod in _pkg_versions(tmp_path, self._pkg, "vk_sub_pkg"):
            ns = {"pkg": mod}
            exec("def step(x):\n    return pkg.leaf.f(x)", ns)
            fps.append(_fp(ns["step"]))
        assert fps[0] != fps[1]

    def test_from_import_still_invalidates(self, tmp_path):
        # Control: reached as a name, tracked before this fix and after.
        fps = []
        for mod in _pkg_versions(tmp_path, self._pkg, "vk_sub_pkg"):
            ns = {"f": mod.leaf.f}
            exec("def step(x):\n    return f(x)", ns)
            fps.append(_fp(ns["step"]))
        assert fps[0] != fps[1]

    def test_walk_descends_more_than_one_level(self, tmp_path):
        def files(leaf):
            return {
                "vk_deep_pkg/__init__.py": "from . import inner\n",
                "vk_deep_pkg/inner/__init__.py": "from . import leaf\n",
                "vk_deep_pkg/inner/leaf.py": leaf,
            }

        fps = []
        for mod in _pkg_versions(tmp_path, files, "vk_deep_pkg"):
            ns = {"pkg": mod}
            exec("def step(x):\n    return pkg.inner.leaf.f(x)", ns)
            fps.append(_fp(ns["step"]))
        assert fps[0] != fps[1]  # the walk stopped at the first __init__.py

    def test_submodule_source_appears_in_spans(self, tmp_path):
        mod = next(_pkg_versions(tmp_path, self._pkg, "vk_sub_pkg"))
        ns = {"pkg": mod}
        exec("def step(x):\n    return pkg.leaf.f(x)", ns)
        files = {os.path.basename(s[0]) for s in reachable_set(ns["step"]).spans}
        assert {"__init__.py", "leaf.py"} <= files

    def test_non_user_submodule_not_followed(self, tmp_path):
        # A stdlib module bound inside a package is an attribute like any
        # other; it must stop at the classification boundary, not be read.
        def files(leaf):
            return {
                "vk_sub_pkg/__init__.py": "import json\nfrom . import leaf\n",
                "vk_sub_pkg/leaf.py": leaf,
            }

        mod = next(_pkg_versions(tmp_path, files, "vk_sub_pkg"))
        ns = {"pkg": mod}
        exec("def step(x):\n    return pkg.json.dumps(pkg.leaf.f(x))", ns)
        spans = reachable_set(ns["step"]).spans
        assert not any("json" in os.path.basename(s[0]) for s in spans)

    def test_mutually_importing_packages_terminate(self, tmp_path):
        root = tmp_path / "cyc"
        _write_tree(
            root,
            {
                "vk_cyc_a/__init__.py": "import vk_cyc_b\ndef f(x):\n    return x\n",
                "vk_cyc_b/__init__.py": "import vk_cyc_a\ndef g(x):\n    return x\n",
            },
        )
        mod = _fresh_import(root, "vk_cyc_a")
        ns = {"a": mod}
        exec("def step(x):\n    return a.vk_cyc_b.g(a.f(x))", ns)
        h = _fp(ns["step"])  # must terminate rather than recurse forever
        assert isinstance(h, str) and len(h) == 40

    def test_no_stale_hit_when_submodule_edited(self, cache, tmp_path):
        # The user-visible guarantee, end to end: a hit must equal what
        # executing the current definition would return.
        seen = []
        for mod in _pkg_versions(tmp_path, self._pkg, "vk_sub_pkg"):
            calls = []
            ns = {"pure": pure, "pkg": mod, "calls": calls}
            exec(
                "@pure\ndef step(x):\n"
                "    calls.append(1)\n"
                "    return pkg.leaf.f(x)",
                ns,
            )
            seen.append((ns["step"](10), bool(calls)))
        assert seen[0] == (11, True)
        assert seen[1] == (555565, True)  # executed; previously served 11

    def test_unchanged_submodule_still_hits(self, cache, tmp_path):
        # The fix must not cost hits when nothing changed.
        root = tmp_path / "v"
        _write_tree(root, self._pkg("def f(x):\n    return x + 1\n"))
        mod = _fresh_import(root, "vk_sub_pkg")
        calls = []
        ns = {"pure": pure, "pkg": mod, "calls": calls}
        exec(
            "@pure\ndef step(x):\n    calls.append(1)\n    return pkg.leaf.f(x)",
            ns,
        )
        assert ns["step"](10) == 11 and ns["step"](10) == 11
        assert calls == [1]

    def test_breakpoint_in_submodule_forces_caller(self, cache, tmp_path, monkeypatch):
        # A breakpoint in a submodule must force its @pure callers, or the
        # cached caller would skip straight past it.
        import bdb

        root = tmp_path / "v"
        _write_tree(root, self._pkg("def f(x):\n    return x + 1\n"))
        mod = _fresh_import(root, "vk_sub_pkg")
        calls = []
        ns = {"pure": pure, "pkg": mod, "calls": calls}
        exec(
            "@pure\ndef step(x):\n    calls.append(1)\n    return pkg.leaf.f(x)",
            ns,
        )
        assert ns["step"](10) == 11 and calls == [1]  # recorded

        dbg = bdb.Bdb()
        monkeypatch.setattr(sys, "gettrace", lambda: dbg.trace_dispatch)
        dbg.set_break(mod.leaf.__file__, 2)
        try:
            assert ns["step"](10) == 11
            assert calls == [1, 1]  # forced, not served from the cache
        finally:
            dbg.clear_all_breaks()


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
    """A module that looks like a compiled extension built in place: a real
    file with an extension suffix inside a project tree (a pyproject.toml
    and a C++ source beside it), belonging to no installed distribution."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text("[project]\nname='p'\nversion='0'\n")
    (proj / "native.cpp").write_text("int solve(int x) { return x; }\n")
    path = proj / f"_fake_ext{EXTENSION_SUFFIXES[0]}"
    path.write_bytes(b"compiled bytes, version one")
    mod = types.ModuleType("_fake_ext")
    mod.__file__ = str(path)
    monkeypatch.setitem(sys.modules, "_fake_ext", mod)
    monkeypatch.setattr(_NativeCallable, "__module__", "_fake_ext")
    monkeypatch.setattr(functionhash, "_markers_here", None)
    return mod, path


def _using_global(name, obj):
    """Build a function whose body calls *obj* under the name *name*."""
    ns = {name: obj}
    exec(f"def f(x):\n    return {name}(x)", ns)
    return ns["f"]


_NATIVE = _NativeCallable()


def _ext_user(x):
    # Module level so a worker handshake can resolve it by name.
    return _NATIVE(x)


class TestNativeExtensions:
    def test_the_binary_identifies_the_extension(self, fake_extension):
        mod, path = fake_extension
        kind, marker = _classify(mod.__name__, mod.__file__)
        assert marker == f"ext:_fake_ext={project._file_hash(str(path))}"

    def test_a_rebuild_invalidates_its_callers(self, fake_extension):
        # The case this exists for: a @pure function calls into C++, the C++
        # is edited and rebuilt, and the result must not be served from cache.
        mod, path = fake_extension
        fn = _using_global("solve", _NativeCallable())
        before = _fp(fn)
        (path.parent / "native.cpp").write_text("int solve(int x) { return x + 1; }\n")
        assert _fp(fn) == before  # an edit alone changes nothing: the build is the code that runs
        path.write_bytes(b"compiled bytes, version two")
        assert _fp(fn) != before

    def test_a_walk_records_the_markers_it_met(self, fake_extension):
        mod, path = fake_extension
        fn = _using_global("solve", _NativeCallable())
        reach = reachable_set(fn)
        assert reach.extensions == {"_fake_ext": project._file_hash(str(path))}
        plain = reachable_set(lambda x: x)
        assert plain.extensions == {}

    def test_the_binary_is_read_once_per_build(self, fake_extension, monkeypatch):
        mod, path = fake_extension
        calls = []
        real = project._file_hash
        monkeypatch.setattr(project, "_file_hash", lambda p: calls.append(p) or real(p))
        ns = {"a": _NativeCallable(), "b": _NativeCallable()}
        exec("def f(x):\n    return a(x) + b(x)", ns)
        _fp(ns["f"])
        _fp(ns["f"])
        assert len(calls) == 1
        path.write_bytes(b"compiled bytes, version two")  # the same size
        t = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(t, t))  # a rebuild's mtime moves; a coarse clock may not have
        _fp(ns["f"])
        assert len(calls) == 2

    def test_a_worker_takes_the_markers_from_hello(self, fake_extension, tmp_path, monkeypatch):
        # On a worker an extension's marker is what the main process sent;
        # this worker's own binary is never read.
        m, _ = _write_batch_module(tmp_path)
        mod, path = fake_extension
        calls = []
        real = project._file_hash
        monkeypatch.setattr(project, "_file_hash", lambda p: calls.append(p) or real(p))
        assert _handshake(_hello(m.process, extensions={"_fake_ext": "7" * 40})) == ""
        assert functionhash._markers_here == {"_fake_ext": "7" * 40}
        assert _classify(mod.__name__, mod.__file__)[1] == "ext:_fake_ext=" + "7" * 40
        assert calls == []
        # An extension the main process never reached gets no marker, so the
        # hashes differ and the worker is refused rather than trusted.
        assert _classify("_other_ext", mod.__file__)[1] == "ext:_other_ext=?not-reached-by-the-main-process"

    def test_a_worker_takes_the_project_hash_from_hello(self, tmp_path, monkeypatch):
        # The project hash says which source tree the worker must be in.
        m, _ = _write_batch_module(tmp_path)
        tree = tmp_path / "src" / "p"
        tree.mkdir(parents=True)
        monkeypatch.setenv("VALUEKIT_TREE", str(tree))
        monkeypatch.setenv("VALUEKIT_PROJECT_HASH", "7" * 40)
        assert _handshake(_hello(m.process, project_hash="7" * 40)) == ""
        assert "the main process meant" in _handshake(_hello(m.process, project_hash="8" * 40))

    def test_a_released_distribution_keeps_its_version_marker(
        self, fake_extension, monkeypatch
    ):
        # A wheel changes only through a reinstall, which moves its version.
        mod, path = fake_extension

        class _ReleasedDist:
            version = "1.2.3"

            def read_text(self, name):
                return None  # no direct_url.json: not a local install

        monkeypatch.setattr(functionhash, "_distribution", lambda top: _ReleasedDist())
        assert _classify(mod.__name__, mod.__file__)[1] == "pkg:_fake_ext==1.2.3"

    def test_a_local_install_is_identified_by_its_binary(
        self, fake_extension, tmp_path, monkeypatch
    ):
        # An editable install puts the binary wherever the build backend likes;
        # direct_url.json says it is a local install, so the binary is hashed.
        mod, path = fake_extension
        elsewhere = tmp_path / "site-packages"
        elsewhere.mkdir()
        binary = elsewhere / path.name
        binary.write_bytes(b"compiled elsewhere")
        mod.__file__ = str(binary)
        url = path.parent.as_uri()

        class _LocalDist:
            version = "0"

            def read_text(self, name):
                return json.dumps({"url": url, "dir_info": {"editable": True}})

        monkeypatch.setattr(functionhash, "_distribution", lambda top: _LocalDist())
        assert _classify(mod.__name__, mod.__file__)[1] == f"ext:_fake_ext={project._file_hash(str(binary))}"

    def test_an_extension_module_is_tracked_as_a_module(self, fake_extension):
        mod, path = fake_extension
        fn = _using_global("ext", mod)
        # A module global resolves by name, so the reference is to the module
        # itself rather than to anything it defines.
        before = _fp(fn)
        path.write_bytes(b"compiled bytes, version two")
        assert _fp(fn) != before

    def test_an_extension_with_no_project_is_its_binary(self, tmp_path):
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
        _fp(fn)


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

    def test_record_dedup(self, tmp_path):
        s = LocalStore(tmp_path)
        t = {"fn": "f", "deps": {}, "result": "0" * 40}
        s.put_record("k", t)
        s.put_record("k", dict(t))
        assert len(s.get_records("k")) == 1

    def test_concurrent_record_writes_survive_across_processes(self, tmp_path):
        # Parallel writers used to drop each other's call records on Windows,
        # where an O_APPEND write is a seek followed by a write. Every record
        # is now its own content-named file, so there is nothing shared to
        # lose. Real processes, not threads: the CRT behaviour is per file
        # descriptor and threads in one process did not show it reliably.
        script = (
            "import sys\n"
            "from valuekit.store import LocalStore\n"
            "root, i = sys.argv[1], int(sys.argv[2])\n"
            "s = LocalStore(root)\n"
            "for j in range(50):\n"
            "    s.put_record('k', {'fn': 'f', 'deps': {'x': {'kind': 'value',"
            " 'hash': f'{i}-{j}'}}, 'result': '0' * 40})\n"
        )
        procs = [
            subprocess.Popen([sys.executable, "-c", script, str(tmp_path), str(i)])
            for i in range(8)
        ]
        assert [p.wait() for p in procs] == [0] * 8
        got = LocalStore(tmp_path).get_records("k")
        assert len(got) == 400
        assert all(h == record_hash(t) for h, t in got)

    def test_put_record_returns_hash_and_get_records_pairs(self, tmp_path):
        s = LocalStore(tmp_path)
        t = {"fn": "f", "deps": {}, "result": "0" * 40}
        h = s.put_record("k", t)
        assert h == record_hash(t)
        assert (tmp_path / "records" / "k" / f"{h}.json").exists()
        assert s.get_records("k") == [(h, t)]

    def test_corrupt_record_file_skipped(self, tmp_path):
        # A file that does not parse, and one whose bytes do not hash to its
        # name (a torn write, or an edit), are both ignored: a miss at worst.
        s = LocalStore(tmp_path)
        t = {"fn": "f", "deps": {}, "result": "0" * 40}
        h = s.put_record("k", t)
        d = tmp_path / "records" / "k"
        (d / f"{'1' * 40}.json").write_bytes(b'{"fn": "g", "trunc')
        (d / f"{'2' * 40}.json").write_bytes(b'{"fn": "g", "deps": {}}')
        assert s.get_records("k") == [(h, t)]

    def test_get_records_newest_first(self, tmp_path):
        s = LocalStore(tmp_path)
        old = {"fn": "f", "deps": {}, "result": "0" * 40}
        new = {"fn": "f", "deps": {}, "result": "1" * 40}
        h_old = s.put_record("k", old)
        h_new = s.put_record("k", new)
        d = tmp_path / "records" / "k"
        os.utime(d / f"{h_old}.json", (1_000_000, 1_000_000))
        os.utime(d / f"{h_new}.json", (2_000_000, 2_000_000))
        assert [h for h, _ in s.get_records("k")] == [h_new, h_old]

    def test_listing_sees_another_stores_write(self, tmp_path):
        # The listing is cached per store on the directory's mtime, which
        # any process's write moves.
        a = LocalStore(tmp_path)
        b = LocalStore(tmp_path)
        t1 = {"fn": "f", "deps": {}, "result": "0" * 40}
        t2 = {"fn": "f", "deps": {}, "result": "1" * 40}
        a.put_record("k", t1)
        assert len(a.get_records("k")) == 1
        b.put_record("k", t2)
        assert len(a.get_records("k")) == 2

    def test_a_write_moves_the_directory_mtime_on_a_coarse_clock(self, tmp_path):
        # A write within one tick of a coarse filesystem clock leaves the
        # directory's mtime where it was; put_record then moves it by hand.
        # Setting the mtime ahead of the clock makes the next write's
        # natural mtime read as not moved.
        from valuekit import store as store_mod

        s = LocalStore(tmp_path)
        s.put_record("k", {"fn": "f", "deps": {}, "result": "0" * 40})
        d = tmp_path / "records" / "k"
        ahead = time.time_ns() + 3_600 * 10**9
        os.utime(d, ns=(ahead, ahead))
        ahead = d.stat().st_mtime_ns  # as the filesystem keeps it
        s.put_record("k", {"fn": "f", "deps": {}, "result": "1" * 40})
        assert d.stat().st_mtime_ns == ahead + store_mod._MTIME_STEP

    def test_atomic_write_onto_existing_target_is_success(self, tmp_path, monkeypatch):
        # Windows refuses to replace a file another process has mapped. The
        # target is content-addressed, so if it exists the write is done.
        from valuekit import store as store_mod

        target = tmp_path / "x.bin"
        target.write_bytes(b"same")
        real = os.replace

        def refusing(src, dst):
            if os.path.exists(dst):
                raise PermissionError("Access is denied")
            real(src, dst)

        monkeypatch.setattr(os, "replace", refusing)
        store_mod._atomic_write(target, b"same")
        assert target.read_bytes() == b"same"
        assert not list(tmp_path.glob(".tmp-*"))

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
        # it is data, and a function is hashed by its function_hash.
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

    def test_conditional_reads_get_separate_records(self, cache):
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

    def test_clear_cache_deletes_everything_computed_or_logged(self, cache):
        @pure
        def f(x):
            vk.log({"q": "inside"}, np.arange(x) * 1.5)
            return x * 2

        f(3)
        vk.log({"q": "outside"}, np.ones(4))
        assert list((cache / "objects").rglob("*.npy*"))
        vk.clear_cache()
        assert not any((cache / "records").iterdir())
        assert not list((cache / "objects").rglob("*.npy*"))
        with pytest.raises(LookupError):
            vk.logs()
        assert f(3) == 6  # recomputed, and records again

    def test_a_record_deleted_by_another_process_is_missed_here(self, cache):
        @pure
        def f(x):
            return x

        f(1)
        key = f._valuekit_reachable().hash
        here = sys.modules["valuekit.pure"]._current_store()
        [(h, _)] = here.get_records(key)
        assert here.get_record(key, h)["result"]
        LocalStore(cache).clear()  # another process's clear_cache()
        with pytest.raises(CacheMiss):
            here.get_record(key, h)

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
    """@pure must be invisible apart from the skipped run: same types
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

        vk.set_store_dir(None)
        uncached = f({"a": 1}, [2, 3])
        vk.set_store_dir(tmp_path / "cache")
        try:
            assert f({"a": 1}, [2, 3]) == uncached  # miss
            assert f({"a": 1}, [2, 3]) == uncached  # hit
        finally:
            vk.set_store_dir(None)

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
            return {"labels": ctx, "pair": [ctx, 1]}

        miss = wrap(ImmutableMap({"a": 1}))
        hit = wrap(ImmutableMap({"a": 1}))
        for out in (miss, hit):
            assert type(out["labels"]) is ImmutableMap
            assert type(out["pair"][0]) is ImmutableMap
            assert not isinstance(out["labels"], RecordingMap)
        assert miss == hit

    def test_returning_a_map_depends_on_all_of_it(self, cache):
        calls = []

        @pure
        def wrap(ctx):
            calls.append(1)
            return {"labels": ctx}

        assert wrap(ImmutableMap({"a": 1}))["labels"] == ImmutableMap({"a": 1})
        assert len(calls) == 1
        # No key was read, but the whole map was returned, so any change to
        # it must invalidate.
        assert wrap(ImmutableMap({"a": 1, "b": 2}))["labels"] == ImmutableMap(
            {"a": 1, "b": 2}
        )
        assert len(calls) == 2

    def test_a_record_holds_arguments_as_passed(self, cache):
        # Mutating an argument breaks the purity contract, but the record is
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
    def test_map_passed_inward_records_in_both_records(self, cache):
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

        inner_spans = inner._valuekit_reachable().spans
        outer_spans = set(map(tuple, outer._valuekit_reachable().spans))
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
        fname, lo, _ = inner._valuekit_reachable().spans[0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        assert outer(3) == 7
        assert calls == ["o", "i", "o", "i"]  # both executed for real

        # And those forced runs persisted nothing new:
        dbg.clear_all_breaks()
        assert outer(3) == 7
        assert calls == ["o", "i", "o", "i"]  # original record still hits

    def test_midrun_force_taints_enclosing_recording(self, cache, monkeypatch):
        import bdb

        calls = []
        dbg = bdb.Bdb()
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        @pure
        def inner(x):
            calls.append("i")
            return x * 2

        fname, lo, _ = inner._valuekit_reachable().spans[0]
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
        # outer's first recording contained a forced run and was discarded, so this second
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

        fname, lo, _ = stage_c._valuekit_reachable().spans[0]
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
        # BELOW the @pure function are still part of its function hash.
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





@pure
def _fib(n):
    # Module level: a recursive reference through a closure cell is not
    # walkable by the function_hash, and recursion through a global is the
    # documented shape.
    return n if n < 2 else _fib(n - 1) + _fib(n - 2)


def _record_of(cache, f):
    """The newest stored record of *f*, as ``(hash, doc)``."""
    got = LocalStore(cache).get_records(f._valuekit_reachable().hash)
    assert got, "no record stored"
    return got[0]


class TestCallRecords:
    """What happened inside a call -- the memoised calls it made and
    its log() items -- lives in its record, which is what lets a hit stand
    in for the call."""

    def test_nested_calls_recorded_in_order_with_their_hashes(self, cache):
        @pure
        def a(x):
            return x + 1

        @pure
        def b(x):
            return x * 2

        @pure
        def outer(x):
            return a(b(a(x)))

        assert outer(1) == 5
        _, t = _record_of(cache, outer)
        a_key = a._valuekit_reachable().hash
        b_key = b._valuekit_reachable().hash
        a_hashes = {h for h, _ in LocalStore(cache).get_records(a_key)}
        (hb, _), = LocalStore(cache).get_records(b_key)
        assert [c[:2] for c in t["calls"]] == [
            [a.__qualname__, a_key], [b.__qualname__, b_key], [a.__qualname__, a_key]
        ]
        assert t["calls"][1][2] == hb
        assert {t["calls"][0][2], t["calls"][2][2]} == a_hashes
        assert t["logs"] == []

    def test_a_hit_records_the_record_it_matched(self, cache):
        n = []

        @pure
        def inner(x):
            n.append(x)
            return x + 1

        @pure
        def outer(x):
            return inner(x)

        inner(1)
        h_inner, _ = _record_of(cache, inner)
        outer(1)
        assert n == [1]  # the inner call inside outer was a hit
        _, t = _record_of(cache, outer)
        assert t["calls"] == [[inner.__qualname__, inner._valuekit_reachable().hash, h_inner]]
        assert len(LocalStore(cache).get_records(inner._valuekit_reachable().hash)) == 1

    def test_log_records_labels_and_value_in_the_record(self, cache):
        from valuekit import runlog

        @pure
        def f(x):
            vk.log({"q": "a", "x": x}, x)
            vk.log({"q": "b"}, np.arange(3.0) * x)
            return x

        f(2)
        _, t = _record_of(cache, f)
        s = LocalStore(cache)
        ctx0, v0, k0 = t["logs"][0]
        assert s.get_value(ctx0) == ImmutableMap({"q": "a", "x": 2})
        assert s.get_value(v0) == 2
        assert k0 == runlog.label_hashes(ImmutableMap({"q": "a", "x": 2}))
        np.testing.assert_array_equal(s.get_value(t["logs"][1][1]), [0.0, 2.0, 4.0])

    def test_log_outside_a_call_goes_to_the_log_and_noops_without_a_store(self, tmp_path):
        vk.set_store_dir(tmp_path)
        vk.log({"q": "x"}, 1)
        assert vk.logs().where(q="x").one().value == 1
        vk.set_store_dir(None)
        vk.log({"q": "x"}, 1)  # nothing configured: nothing happens
        with pytest.raises(LookupError):
            vk.logs()

    def test_log_refuses_what_it_cannot_record(self, cache):
        with pytest.raises(TypeError, match="mapping"):
            vk.log("name", 1)
        with pytest.raises(TypeError, match="list"):
            vk.log({"q": [1, 2]}, 1)  # a label value must be freezable

    def test_log_of_a_map_depends_on_all_of_it(self, cache):
        n = []

        @pure
        def f(m):
            vk.log({"q": "m"}, m)
            n.append(1)
            return 1

        m = ImmutableMap({"a": 1, "b": 2})
        f(m)
        f(m | {"b": 3})  # a key f never read: but the log observed the whole map
        assert len(n) == 2

    def test_log_in_a_forced_run_reaches_the_run_log_but_no_record(self, cache, monkeypatch):
        # A run being debugged is one whose logged values are wanted; what
        # a forced run must not do is leave a call record behind.
        monkeypatch.setenv("VALUEKIT_ALWAYS_RUN", "1")

        @pure
        def f(x):
            vk.log({"q": "a"}, np.arange(1000.0) * x)
            return x

        assert f(1) == 1
        assert LocalStore(cache).get_records(f._valuekit_reachable().hash) == []
        [logged] = vk.logs()
        assert logged.labels["q"] == "a" and logged.value[3] == 3.0

    def test_log_in_a_miss_containing_a_forced_run_is_discarded(self, cache, monkeypatch):
        import bdb

        dbg = bdb.Bdb()
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        @pure
        def inner(x):
            return x * 2

        fname, lo, _ = inner._valuekit_reachable().spans[0]
        armed = [True]

        @pure
        def outer(x):
            vk.log({"q": "before"}, x)
            if armed[0]:
                armed[0] = False
                dbg.set_break(fname, lo + 1)
            return inner(x) + 1

        assert outer(3) == 7
        assert LocalStore(cache).get_records(outer._valuekit_reachable().hash) == []

    def test_cached_returns_the_stored_result_without_executing(self, cache):
        n = []

        @pure
        def f(x):
            n.append(x)
            return x + 1

        with pytest.raises(CacheMiss):
            f.cached(1)
        f(1)
        assert f.cached(1) == 2
        assert n == [1]
        with pytest.raises(CacheMiss):
            f.cached(2)
        assert n == [1]

    def test_cached_without_a_store_raises(self):
        @pure
        def f(x):
            return x

        with pytest.raises(CacheMiss):
            f.cached(1)

    def test_cached_inside_a_miss_is_recorded_as_a_call(self, cache):
        @pure
        def inner(x):
            return x + 1

        @pure
        def outer(x):
            return inner.cached(x)

        inner(1)
        h_inner, _ = _record_of(cache, inner)
        outer(1)
        _, t = _record_of(cache, outer)
        assert t["calls"][0][2] == h_inner

    def test_lookup_returns_the_matched_hash_or_none(self, cache):
        @pure
        def f(x):
            return x

        assert f._valuekit_lookup(1) is None
        f(1)
        h, t = _record_of(cache, f)
        assert f._valuekit_lookup(1) == (h, t)
        assert f._valuekit_lookup(2) is None

    def test_pure_local_memoises_identically_and_is_flagged(self, cache):
        n = []

        def body(x):
            n.append(x)
            return x + 1

        f = pure(body)
        g = vk.pure_local(body)
        assert f._valuekit_local is False and g._valuekit_local is True
        f(1)
        assert g(1) == 2 and n == [1]  # same code, same function hash: a hit

    def test_recursion_records_the_calls_it_made(self, cache):
        _fib(3)
        key = _fib._valuekit_reachable().hash
        records = {h: t for h, t in LocalStore(cache).get_records(key)}
        (top,) = [t for t in records.values() if t["deps"]["n"]["hash"] == content_hash(3)]
        assert [c[:2] for c in top["calls"]] == [[_fib.__qualname__, key]] * 2
        assert all(c[2] in records for c in top["calls"])

    def test_threads_record_only_their_own_calls(self, cache):
        import threading

        barrier = threading.Barrier(2)

        @pure
        def inner(x):
            barrier.wait(timeout=5)  # both outers are mid-run together
            return x + 1

        @pure
        def outer(x):
            return inner(x)

        threads = [threading.Thread(target=outer, args=(x,)) for x in (1, 2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        s = LocalStore(cache)
        inner_x = {h: t["deps"]["x"]["hash"] for h, t in s.get_records(inner._valuekit_reachable().hash)}
        outers = s.get_records(outer._valuekit_reachable().hash)
        assert len(outers) == 2
        for _, t in outers:
            [(_, _, h)] = t["calls"]
            assert inner_x[h] == t["deps"]["x"]["hash"]

    def test_valuekit_itself_is_never_user_code(self):
        kind, marker = _classify("valuekit.pure", sys.modules["valuekit.pure"].__file__)
        assert kind == functionhash._PKG and marker == "pkg:valuekit"


def _new_run(monkeypatch, script):
    """Pretend this process is a fresh run of *script*: the next memoised
    call or log begins an run under that name."""
    from valuekit import runlog

    monkeypatch.setattr(sys, "argv", [script])
    runlog._close()
    monkeypatch.setattr(runlog, "_current", None)


class TestRunLog:
    """What an run logged is its complete set of items, as if the
    code had run from scratch; read back by label containment."""

    def test_logged_values_are_selected_by_containment(self, cache):
        @pure
        def f(x):
            vk.log({"q": "r", "x": x, "filter": "kalman"}, np.arange(3.0) * x)
            vk.log({"q": "n", "x": x}, 3)
            vk.log({"q": "n", "x": x}, 3)  # the same again: two items
            return x

        f(1)
        f(2)
        L = vk.logs()
        assert len(L) == 6
        assert len(L.where(filter="kalman")) == 2  # extra keys in the item are fine
        assert len(L.where({"q": "n"})) == 4
        assert len(L.where({"q": "n"}, x=2)) == 2
        it = L.where(q="r", x=2).one()
        assert it.labels == ImmutableMap({"q": "r", "x": 2, "filter": "kalman"})
        np.testing.assert_array_equal(it.value, [0.0, 2.0, 4.0])
        assert isinstance(it.labels, ImmutableMap)
        with pytest.raises(LookupError, match="no logged value"):
            L.where(q="missing").one()
        with pytest.raises(LookupError, match="4 logged values"):
            L.where(q="n").one()
        assert len(L.where(x=1.0)) == 0  # exact: 1.0 is not 1
        assert len(L.where(x=1)) == 3
        assert [dict(i.labels) for i in L.where(q="r").where(x=1)] == [
            {"q": "r", "x": 1, "filter": "kalman"}
        ]

    def test_a_query_value_is_frozen_like_a_logged_one(self, cache):
        vk.log({"cfg": {"order": 3}}, 1)
        assert vk.logs().where(cfg={"order": 3}).one().value == 1
        with pytest.raises(TypeError):
            vk.logs().where(cfg=[1])

    def test_a_hit_emits_what_its_record_holds(self, cache):
        n = []

        @pure
        def inner(x):
            n.append(x)
            vk.log({"q": "inner", "x": x}, x * 10)
            return x

        @pure
        def outer(x):
            vk.log({"q": "outer", "x": x}, x)
            return inner(x) + inner(x)  # the second call is a hit

        outer(1)
        assert n == [1]
        assert len(vk.logs().where(q="inner")) == 2  # once per call, as from scratch
        outer(1)  # a root hit: the whole subtree again
        assert n == [1]
        L = vk.logs()
        assert len(L.where(q="outer")) == 2 and len(L.where(q="inner")) == 4
        assert {it.value for it in L.where(q="inner")} == {10}

    def test_a_new_run_replaces_the_last(self, cache, monkeypatch):
        @pure
        def f(x):
            vk.log({"q": "v"}, x)
            return x

        _new_run(monkeypatch, "physics.py")
        f(1)
        f(2)
        first = vk.logs()._path
        assert len(vk.logs()) == 2
        _new_run(monkeypatch, "physics.py")
        f(2)  # a hit, in a new run
        L = vk.logs("physics")
        assert L._path != first and not first.exists()
        assert [it.value for it in L] == [2]  # nothing from the earlier run

    def test_scripts_keep_separate_logs(self, cache, monkeypatch):
        @pure
        def f(x):
            vk.log({"q": "v"}, x)
            return x

        _new_run(monkeypatch, "/somewhere/physics.py")
        f(1)
        _new_run(monkeypatch, "debug.py")
        f(2)
        assert [it.value for it in vk.logs("physics")] == [1]  # untouched by debug.py
        assert [it.value for it in vk.logs("debug")] == [2]
        with pytest.raises(LookupError, match="debug, physics"):
            vk.logs()  # several scripts: name one
        with pytest.raises(LookupError):
            vk.logs("other")

    def test_a_hit_writes_a_reference_and_the_reader_expands_it(self, cache, monkeypatch):
        n = []

        @pure
        def inner(x):
            n.append(x)
            vk.log({"q": "inner"}, x * 10)
            return x

        @pure
        def outer(x):
            vk.log({"q": "outer"}, x)
            return inner(x)

        _new_run(monkeypatch, "physics.py")
        outer(1)
        assert [set(d) - {"t"} for d in _log_lines(cache)] == [{"labels", "v", "k"}] * 2
        _new_run(monkeypatch, "physics.py")
        outer(1)  # a hit: one line naming the record, no copy of its entries
        [d] = _log_lines(cache)
        assert d["record"] == [outer._valuekit_reachable().hash, outer._valuekit_lookup(1)[0]]
        assert n == [1]
        got = {logged.labels["q"]: logged.value for logged in vk.logs()}
        assert got == {"inner": 10, "outer": 1}

    def test_a_nested_hit_inside_a_running_call_is_expanded(self, cache, monkeypatch):
        @pure
        def inner(x):
            vk.log({"q": "inner"}, x * 10)
            return x

        @pure
        def outer(x, y):
            vk.log({"q": "outer"}, y)
            return inner(x)

        _new_run(monkeypatch, "physics.py")
        outer(1, 1)
        _new_run(monkeypatch, "physics.py")
        outer(1, 2)  # outer runs; inner hits and writes a reference
        assert [("record" in d) for d in _log_lines(cache)] == [False, True]
        got = {logged.labels["q"]: logged.value for logged in vk.logs()}
        assert got == {"inner": 10, "outer": 2}

    def test_a_reference_to_a_record_that_is_gone_is_reported_stale(self, cache, monkeypatch):
        import shutil

        @pure
        def inner(x):
            vk.log({"q": "inner"}, x)
            return x

        @pure
        def outer(x):
            vk.log({"q": "outer"}, x)
            return inner(x)

        _new_run(monkeypatch, "physics.py")
        outer(1)
        _new_run(monkeypatch, "physics.py")
        outer(1)  # a hit: the log is one reference
        shutil.rmtree(cache / "records" / inner._valuekit_reachable().hash)
        with pytest.warns(UserWarning, match="1 logged calls .* no longer in the cache"):
            L = vk.logs()
        assert L.stale == 1
        assert [logged.labels["q"] for logged in L] == ["outer"]  # the readable part stays
        shutil.rmtree(cache / "records")
        with pytest.warns(UserWarning):
            L = vk.logs()
        assert L.stale == 1 and len(L) == 0
        _new_run(monkeypatch, "physics.py")
        outer(1)  # a run repairs it
        assert len(vk.logs()) == 2

    def test_entries_stay_readable_when_the_records_go(self, cache, monkeypatch):
        import shutil

        @pure
        def f(x):
            vk.log({"q": "inside"}, x)
            return x

        _new_run(monkeypatch, "physics.py")
        f(1)
        _new_run(monkeypatch, "physics.py")
        f(1)  # a reference
        f(2)  # an entry: the call ran
        vk.log({"q": "outside"}, 5)  # an entry with no record
        shutil.rmtree(cache / "records")
        with pytest.warns(UserWarning):
            L = vk.logs()
        assert [logged.value for logged in L] == [2, 5]

    def test_a_log_is_readable_while_it_grows(self, cache):
        @pure
        def f(x):
            vk.log({"q": "v"}, x)
            return x

        f(1)
        L = vk.logs()
        assert len(L) == 1
        f(2)
        assert len(L) == 1
        L.refresh()
        assert len(L) == 2 and len(L.where(q="v")) == 2

    def test_logs_without_a_cache_raises(self):
        with pytest.raises(LookupError, match="store"):
            vk.logs()

    def test_a_label_in_the_arguments_recomputes_when_it_changes(self, cache):
        n = []

        @pure
        def f(cfg):
            n.append(1)
            vk.log({"experiment": cfg["experiment"], "q": "v"}, 1)
            return 1

        cfg = ImmutableMap({"experiment": "exp_17", "gain": 2})
        f(cfg)
        f(cfg.assoc("gain", 3))  # unread key: a hit
        f(cfg.assoc("experiment", "exp_18"))  # the label was read: recomputes
        assert len(n) == 2
        assert len(vk.logs().where(experiment="exp_17")) == 2
        assert len(vk.logs().where(experiment="exp_18")) == 1


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

        fname, lo, hi = f._valuekit_reachable().spans[0]

        dbg = bdb.Bdb()
        dbg.set_break(fname, lo)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        f(1)
        f(1)  # breakpoint in span → forced run, no cache write
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
# the build step on this machine
# ===========================================================================


class TestBuild:
    @pytest.fixture
    def built_project(self, tmp_path, monkeypatch):
        """A project with a fake lock tool that records the commands it ran,
        and a record location under a temporary prefix."""
        root = tmp_path / "proj"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'p'\nversion = '0'\n[tool.valuekit]\nbuild-inputs = ['*.c']\n"
        )
        (root / "fake.lock").write_text("")
        (root / "native.c").write_text("int x;")
        (root / "mod.py").write_text("x = 1")
        made = tmp_path / "made.py"
        made.write_text(
            "import sys\n"
            f"open({str(tmp_path / 'commands')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
            "sys.exit(int('fail' in sys.argv[1]))\n"
        )
        row = {
            "tool": sys.executable,
            "install": (str(made), "install"),
            "rebuild": (str(made), "rebuild", "{dist}"),
            "interpreters": (),
            "search": (),
        }
        monkeypatch.setitem(bootstrap._TOOLS, "fake.lock", row)
        monkeypatch.setattr(bootstrap, "KNOWN_LOCKS", ("fake.lock",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
        (tmp_path / "prefix").mkdir()
        return root, tmp_path / "commands"

    def test_build_runs_once_per_change_to_a_build_input(self, built_project, capsys):
        root, commands = built_project
        assert vk.build(root) is True
        assert commands.read_text().splitlines() == ["rebuild p"]
        assert "build inputs of p changed" in capsys.readouterr().err
        assert vk.build(root) is False  # nothing changed
        (root / "mod.py").write_text("x = 2")
        assert vk.build(root) is False  # a Python source is not a build input
        (root / "native.c").write_text("int y;")
        assert vk.build(root) is True
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'p'\nversion = '1'\n[tool.valuekit]\nbuild-inputs = ['*.c']\n"
        )
        assert vk.build(root) is True  # pyproject.toml always counts
        assert len(commands.read_text().splitlines()) == 3

    def test_build_finds_the_project_from_the_calling_script(self, built_project, monkeypatch):
        root, commands = built_project
        monkeypatch.chdir(root)
        ns = {}  # a caller with no __file__, an interactive session say: the cwd is used
        exec("import valuekit\nran = valuekit.build()", ns)
        assert ns["ran"] is True

    def test_a_failed_build_raises_with_the_tools_output(self, built_project, monkeypatch):
        root, commands = built_project
        row = dict(bootstrap._TOOLS["fake.lock"])
        row["rebuild"] = (row["rebuild"][0], "fail-rebuild", "{dist}")
        monkeypatch.setitem(bootstrap._TOOLS, "fake.lock", row)
        with pytest.raises(RuntimeError, match="fail-rebuild p failed"):
            vk.build(root)
        assert vk.build.__module__ == "valuekit.build"

    def test_build_refuses_what_it_cannot_build(self, tmp_path):
        with pytest.raises(project.ProjectError, match="no project"):
            vk.build(tmp_path)
        root = tmp_path / "proj"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname = 'p'\nversion = '0'\n")
        with pytest.raises(project.ProjectError, match="no lock file"):
            vk.build(root)


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
    s1.put_record("k", t)
    s2.put_record("k", t)
    assert len(s1.get_records("k")) == 1


# ===========================================================================
# run_all: parallel run
# ===========================================================================


_LOCKED: dict = {}


def _locked_files(build=True):
    """The files that make a test project something a host can check out and
    run: a pyproject.toml, a uv.lock, and the valuekit wheel the lock points
    at (built from this checkout, so the host runs the code under test).
    Built once per session; ``build=False`` only reports whether they exist."""
    if "dir" in _LOCKED:
        return _LOCKED["dir"]
    if not build:
        return None
    if shutil.which("uv") is None:
        pytest.skip("uv is needed to build a host's environment")
    d = _Path(tempfile.mkdtemp(prefix="vk-locked-"))
    atexit.register(shutil.rmtree, d, True)
    repo = _Path(__file__).resolve().parent.parent
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(d / "wheels"), str(repo)],
        check=True, capture_output=True,
    )
    [whl] = list((d / "wheels").glob("*.whl"))
    # numpy is pinned to the main process's: a package's version is part of the
    # function_hash of every function that uses it, and a host whose numpy
    # differed would refuse the main process's functions as out of sync.  The
    # Python is the main process's minor, which is what a host is built for; a
    # wider range would make uv resolve for Pythons the pin cannot serve.
    from importlib.metadata import version

    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    (d / "pyproject.toml").write_text(
        f"[project]\nname = 'p'\nversion = '0'\nrequires-python = '=={minor}.*'\n"
        f"dependencies = ['valuekit', 'numpy=={version('numpy')}']\n\n"
        f"[tool.uv.sources]\nvaluekit = {{ path = 'wheels/{whl.name}' }}\n"
    )
    subprocess.run(["uv", "lock"], cwd=d, check=True, capture_output=True)
    _LOCKED["dir"] = d
    return d


def _lay_project(root):
    """Give *root* a pyproject.toml, locked if the session has built the
    locked files (a host test's fixture does), plain otherwise."""
    root.mkdir(parents=True, exist_ok=True)
    d = _locked_files(build=False)
    if d is None:
        (root / "pyproject.toml").write_text("[project]\nname='p'\nversion='0'\n")
        return
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(d / name, root / name)
    shutil.copytree(d / "wheels", root / "wheels", dirs_exist_ok=True)


def _local_file(tmp_path, text):
    """Write the test project's valuekit.local.toml (the project is
    tmp_path/proj, which _write_batch_module and _project create)."""
    from valuekit.localfile import LOCAL_FILE

    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    p = root / LOCAL_FILE
    p.write_text((p.read_text() if p.exists() else "") + text)


def _write_batch_module(tmp_path):
    """A scenario module written to disk so that spawn workers can import
    the functions by reference. Each run leaves a file in a log
    directory (atomic across processes, and outside the project so a batch
    does not change the tree it runs from)."""
    log = tmp_path / "runs.log"
    proj = tmp_path / "proj"
    _lay_project(proj)
    mod = proj / "vk_batch_mod.py"
    mod.write_text(
        "from valuekit import pure, pure_local, log as vklog\n"
        "import numpy as np\n"
        f"LOG = {str(log)!r}\n"
        "@pure_local\n"
        "def here(x):\n"
        "    import os\n"
        "    vklog({'q': 'where', 'x': x}, os.getpid())\n"
        "    return os.getpid()\n"
        "@pure\n"
        "def via_local(x):\n"
        "    return here(x)\n"
        "@pure\n"
        "def env_var(name):\n"
        "    import os\n"
        "    return os.environ.get(name, 'absent')\n"
        "@pure\n"
        "def activation(x):\n"
        "    import os, sys\n"
        "    return {'path0': os.environ['PATH'].split(os.pathsep)[0],\n"
        "            'bindir': os.path.dirname(sys.executable),\n"
        "            'venv': os.environ.get('VIRTUAL_ENV'), 'prefix': sys.prefix}\n"
        "@pure\n"
        "def slow(x):\n"
        "    import time\n"
        "    time.sleep(1.5)\n"
        "    return x\n"
        "@pure\n"
        "def with_log(sid):\n"
        "    vklog({'q': 'twice', 'sid': sid}, sid * 2)\n"
        "    vklog({'q': 'twice', 'sid': sid}, sid * 3)\n"
        "    vklog({'q': 'parity', 'sid': sid}, sid % 2)\n"
        "    vklog({'q': 'arr', 'sid': sid}, np.arange(3.0) * sid)\n"
        "    return sid\n"
        "@pure\n"
        "def outer_log(sid):\n"
        "    return with_log(sid) + 1\n"
        "def _note(tag):\n"
        "    import os, tempfile\n"
        "    os.makedirs(LOG, exist_ok=True)\n"
        "    os.close(tempfile.mkstemp(prefix=tag + '.', dir=LOG)[0])\n"
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
        "@pure\n"
        "def process(sid):\n"
        "    return analyse(sid, load(sid))\n"
        "@pure\n"
        "def hard_death(x):\n"
        "    import os\n"
        "    if x == 1:\n"
        "        os._exit(1)\n"
        "    return x * 2\n"
        "@pure\n"
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
    _sys.path.insert(0, str(proj))  # spawn children inherit sys.path
    spec.loader.exec_module(m)

    def counts():
        # One file per run: concurrent appends to one file can lose a
        # line on Windows, where append mode seeks and writes in two steps.
        try:
            return [p.name.split(".")[0] for p in log.iterdir()]
        except OSError:
            return []

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
        # The cache is configured only via set_store_dir in this process
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
        assert "exited without raising" in str(exc) and "exit code" in str(exc)

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
        assert c.count("A3") == 1  # the failure was not re-run

    # ---- sequential mode (breakpoints) ---------------------------------------

    def test_breakpoint_forces_sequential(self, cache, monkeypatch):
        import bdb

        calls = []

        @pure
        def step(x):
            calls.append(x)  # visible only if executed in THIS process
            return x + 1

        @pure
        def batch(x):
            return step(x)

        fname, lo, _ = step._valuekit_reachable().spans[0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)

        assert vk.run_all(batch, [1, 2, 3]).values == [2, 3, 4]
        assert calls == [1, 2, 3]  # sequential, in-process: breakpoints fire


# ===========================================================================
# the run log
# ===========================================================================
#
# The cache's whole promise is that it can stay on, and none of it is
# visible from outside: a step that ought to hit and silently does not looks
# exactly like a slow one. These assert what the event stream records.


class TestBatches:
    """run_all records what it produced under a name; valuekit.batch reads
    it back without importing or running the pipeline."""

    def test_run_all_requires_a_memoised_function(self, cache):
        def plain(x):
            return x

        with pytest.raises(TypeError, match="@pure"):
            vk.run_all(plain, [1])

    def test_a_batch_is_recorded_and_readable(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 2, 3, 4], max_workers=2)
        b = vk.batch("process")
        assert (b.name, b.fn, b.n) == ("process", "process", 4)
        assert b.function_hash == m.process._valuekit_reachable().hash
        assert b.inputs == [1, 2, 3, 4]
        assert b.complete and b.pending == []
        assert b.failures == [(3, "ValueError", "bad calibration in scenario 3")]
        assert [r.input for r in b.rows] == [1, 2, 4]
        row = b[1]
        assert row.result == 11
        assert [c.fn for c in row.calls] == ["load", "analyse"]
        assert [c.result for c in row.calls] == [10, 11]
        with pytest.raises(KeyError, match="failed"):
            b[3]
        with pytest.raises(KeyError, match="not an input"):
            b[99]

    def test_cached_inputs_skip_workers_and_are_still_recorded(self, cache, tmp_path):
        m, counts = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 2], max_workers=2)
        first = vk.batch("process")
        n = len(counts())
        vk.run_all(m.process, [1, 2], max_workers=2)
        assert len(counts()) == n  # nothing executed anywhere
        second = vk.batch("process")
        assert second._path != first._path  # a new batch under the same name
        assert [r.input for r in second.rows] == [1, 2]
        assert second[2].record_hash == first[2].record_hash  # the same call records
        assert [p.name for p in first._path.parent.iterdir() if p.is_dir()] == [
            second._path.name
        ]  # the older record is gone

    def test_batch_names(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1], name="nightly")
        assert vk.batch("nightly").fn == "process"
        with pytest.raises(LookupError):
            vk.batch("process")
        with pytest.raises(LookupError):
            vk.batch("nightly", store_dir=tmp_path / "elsewhere")

    def test_logged_values_are_read_back_from_the_batch_and_the_log(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.with_log, [1, 2, 3, 4], max_workers=2)
        b = vk.batch("with_log")
        assert sorted(it.value for it in b.logs.where(q="twice", sid=2)) == [4, 6]
        assert b.logs.where(q="parity", sid=2).one().value == 0
        np.testing.assert_array_equal(b.logs.where(q="arr", sid=3).one().value, [0.0, 3.0, 6.0])
        assert b[2].logs.where(q="parity").one().value == 0
        assert len(b.logs.where(q="parity")) == 4
        # The workers logged into this run's runlog, one file each.
        L = vk.logs()
        assert len(L.where(q="parity")) == 4
        assert {it.value for it in L.where(q="parity")} == {0, 1}
        assert len(list(L._path.glob("*.jsonl"))) == 4
        # A batch of already-cached inputs emits the items again: the
        # run's runlog says what this run produced, hit or miss.
        vk.run_all(m.with_log, [1, 2], max_workers=2)
        assert len(vk.logs().where(q="parity")) == 6
        # An item logged in a nested call belongs to the root's row too.
        vk.run_all(m.outer_log, [5], max_workers=1)
        assert vk.batch("outer_log")[5].logs.where(q="parity").one().value == 1

    def test_a_batch_is_readable_while_it_runs(self, cache, tmp_path):
        import threading

        m, _ = _write_batch_module(tmp_path)
        done = []
        th = threading.Thread(
            target=lambda: done.append(
                vk.run_all(m.quick_or_hang, [7, 9], max_workers=2, timeout=4)
            )
        )
        th.start()
        b = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                b = vk.batch("quick_or_hang")
            except LookupError:
                time.sleep(0.1)
                continue
            if 7 not in b.pending:
                break
            time.sleep(0.1)
        assert b is not None and not b.complete  # input 9 is still hanging
        assert b[7].result == 70
        th.join()
        b.refresh()
        assert b.complete
        assert [(x, kind) for x, kind, _ in b.failures] == [(9, "TimeoutError")]

    def test_a_batch_inside_a_pure_function_is_recorded_as_calls(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)

        @pure
        def batch_driver(n):
            return vk.run_all(m.process, list(range(1, n + 1))).values

        assert batch_driver(2) == [11, 21]
        _, t = _record_of(cache, batch_driver)
        key = m.process._valuekit_reachable().hash
        assert [c[:2] for c in t["calls"]] == [["process", key]] * 2
        assert {c[2] for c in t["calls"]} == {r.record_hash for r in vk.batch("process")}


class TestLocalFile:
    def test_no_local_file_means_local_only(self, tmp_path):
        from valuekit import localfile

        for root in (tmp_path, None):
            c = localfile.load_local(root)
            assert c.hosts == () and c.local_workers == (os.cpu_count() or 1)
            assert c.mode == "all" and c.project is None

    def test_local_file_parses_with_defaults(self, tmp_path):
        from valuekit import localfile

        p = tmp_path / localfile.LOCAL_FILE
        p.write_text(
            "project = 'exp'\nmode = 'remote'\n"
            "[local]\nworkers = 3\n"
            "[hosts.mac]\nssh = 'ian@mac.local'\npython = '/usr/bin/python3'\n"
            "[hosts.pc]\nssh = 'pc'\npython = 'py'\nworkers = 2\nsource_root = 'D:/vk'\n"
        )
        c = localfile.load_local(tmp_path)
        assert (c.project, c.mode, c.local_workers) == ("exp", "remote", 3)
        mac, pc = c.hosts
        assert (mac.name, mac.ssh, mac.python, mac.workers) == ("mac", "ian@mac.local", "/usr/bin/python3", None)
        assert mac.source_root == localfile.DEFAULT_SOURCE_ROOT
        assert (pc.workers, pc.source_root) == (2, "D:/vk")

    @pytest.mark.parametrize(
        "text",
        [
            "[hosts.mac]\npython = 'p'\n",  # no ssh
            "[hosts.mac]\nssh = 'm'\npython = 'p'\nworkers = -1\n",
            "[local]\nworkers = true\n",
            "not toml at all [[[",
            "mode = 'sideways'\n",
            "project = 7\n",
        ],
    )
    def test_malformed_local_file_names_the_file(self, tmp_path, text):
        from valuekit import localfile

        (tmp_path / localfile.LOCAL_FILE).write_text(text)
        with pytest.raises(RuntimeError, match="valuekit.local.toml"):
            localfile.load_local(tmp_path)

    def test_the_mode_line(self, tmp_path):
        from valuekit import localfile

        # Absent, the mode is all: every configured host is used.
        assert localfile.load_local(tmp_path).mode == "all"
        assert localfile.load_local(None).mode == "all"
        localfile.write_mode(tmp_path, "remote")
        assert localfile.load_local(tmp_path).mode == "remote"
        p = tmp_path / localfile.LOCAL_FILE
        assert p.read_text() == 'mode = "remote"\n'
        # Only the mode line changes; comments and other lines stay.
        p.write_text("# mine\nproject = 'x'\n  mode='local'  # was\n[local]\nworkers = 2\n")
        localfile.write_mode(tmp_path, "all")
        assert p.read_text() == "# mine\nproject = 'x'\nmode = \"all\"\n[local]\nworkers = 2\n"
        assert localfile.load_local(tmp_path).local_workers == 2
        p.write_text("nonsense [[[\n")
        with pytest.raises(RuntimeError, match="cannot read"):
            localfile.load_local(tmp_path)
        with pytest.raises(ValueError):
            localfile.write_mode(tmp_path, "everywhere")

    def test_capacities_per_mode(self):
        from valuekit.modes import capacities

        remote = {"mac": 8, "pc": 4}
        assert capacities("local", 6, remote) == {"mac": 0, "pc": 0, "local": 6}
        assert capacities("all", 6, remote) == {"mac": 8, "pc": 4, "local": 6}
        assert capacities("remote", 6, remote) == {"mac": 8, "pc": 4, "local": 0}
        # As little as possible locally means everything locally when there
        # is nowhere else.
        assert capacities("remote", 6, {}) == {"local": 6}
        assert capacities("remote", 6, {"mac": 0}) == {"mac": 0, "local": 6}
        # ... but a host still syncing is somewhere else: this host
        # waits for it rather than taking the batch itself.
        assert capacities("remote", 6, {"mac": 0}, syncing=True) == {"mac": 0, "local": 0}
        assert capacities("all", 6, {"mac": 0}, syncing=True) == {"mac": 0, "local": 6}

    def test_worker_env_is_an_allowlist(self, monkeypatch):
        from valuekit.hostprocess import worker_env

        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
        monkeypatch.setenv("VALUEKIT_STORE", "y")
        monkeypatch.setenv("PYTHONPATH", "z")
        env = worker_env()
        assert "AWS_SECRET_ACCESS_KEY" not in env and "PYTHONPATH" not in env
        assert env["VALUEKIT_STORE"] == "y" and "PATH" in env


class TestScheduling:
    """Where tasks go: capacities per host, remote hosts first, the mode
    file re-read at every start, and a host that fails or dies dropped."""

    @pytest.fixture(autouse=True)
    def _locked(self):
        _locked_files()

    def _local_workers(self, tmp_path, monkeypatch, n):
        _local_file(tmp_path, f"[local]\nworkers = {n}\n")

    def test_mode_all_fills_remote_hosts_first(self, cache, tmp_path, monkeypatch):
        from valuekit import localfile

        monkeypatch.setattr(
            parallel, "_host_commands", {"h1": (_HOST_CMD, 1), "h2": (_HOST_CMD, 1)}
        )
        self._local_workers(tmp_path, monkeypatch, 2)
        localfile.write_mode(tmp_path / "proj", "all")
        m, _ = _write_batch_module(tmp_path)
        # Enough work that both hosts have joined before it runs out.
        r = vk.run_all(m.slow, list(range(1, 13)))
        assert r.failures == []
        by_host = {}
        for _, e in _records(cache, "outcome"):
            by_host.setdefault(e["host"], []).append(e["i"])
        assert set(by_host) == {"h1", "h2", "local"}
        assert len(by_host["h1"]) >= 1 and len(by_host["h2"]) >= 1
        assert sorted(e["name"] for _, e in _records(cache, "host")) == ["h1", "h2"]
        assert all(e["ok"] for _, e in _records(cache, "host"))

    def test_mode_remote_keeps_local_idle(self, cache, tmp_path, monkeypatch):
        from valuekit import localfile

        monkeypatch.setattr(parallel, "_host_commands", {"h1": (_HOST_CMD, 2)})
        localfile.write_mode(tmp_path / "proj", "remote")
        m, _ = _write_batch_module(tmp_path)
        assert vk.run_all(m.process, [1, 2, 4, 5]).failures == []
        assert {e["host"] for _, e in _records(cache, "outcome")} == {"h1"}
        [(_, h)] = _records(cache, "host")
        assert h["ok"] and h["capacity"] == 2

    def test_max_workers_is_this_machines_capacity(self, cache, tmp_path, monkeypatch):
        from valuekit.parallel import _Hosts

        self._local_workers(tmp_path, monkeypatch, 1)
        monkeypatch.setattr(parallel, "_host_commands", {})
        m, _ = _write_batch_module(tmp_path)
        store = sys.modules["valuekit.pure"]._current_store()
        assert _Hosts(m.process, str(cache), store, 1).capacities()[1] == {"local": 1}
        assert _Hosts(m.process, str(cache), store, 1, 3).capacities()[1] == {"local": 3}
        assert _Hosts(m.process, str(cache), store, 1, 0).capacities()[1] == {"local": 0}

    def test_no_store_means_no_remote_hosts(self, cache, tmp_path, monkeypatch):
        from valuekit.parallel import _Hosts

        monkeypatch.setattr(parallel, "_host_commands", {"h1": (_HOST_CMD, 2)})
        m, _ = _write_batch_module(tmp_path)
        assert _Hosts(m.process, None, None, 1).hosts == []
        assert [h.name for h in _Hosts(m.process, str(cache), None, 1).hosts] == ["h1"]

    def test_an_unreadable_file_leaves_the_last_mode_in_force(self, cache, tmp_path, monkeypatch):
        from valuekit import localfile
        from valuekit.parallel import _Hosts

        monkeypatch.setattr(parallel, "_host_commands", {})
        localfile.write_mode(tmp_path / "proj", "remote")
        m, _ = _write_batch_module(tmp_path)
        hosts = _Hosts(m.process, str(cache), None, 1)
        assert hosts.capacities()[0] == "remote"
        p = tmp_path / "proj" / localfile.LOCAL_FILE
        p.write_text("nonsense [[[\n")
        assert hosts.capacities()[0] == "remote"
        p.write_text('mode = "local"\n')
        assert hosts.capacities()[0] == "local"

    def test_no_capacity_anywhere_runs_in_this_process(self, cache, tmp_path, monkeypatch):
        monkeypatch.setattr(parallel, "_host_commands", {})
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2], max_workers=0)
        assert r.values == [11, 21]
        assert {e["host"] for _, e in _records(cache, "start")} == {"main"}
        assert {e["host"] for _, e in _records(cache, "outcome")} == {"main"}
        r = vk.run_all(m.process, [3, 4], max_workers=0)  # process(3) raises
        assert [x for x, _ in r.failures] == [3] and r[1].result() == 41

    def test_remote_mode_with_no_host_runs_locally_and_says_so(self, cache, tmp_path, monkeypatch):
        from valuekit import localfile

        monkeypatch.setattr(parallel, "_host_commands", {})
        localfile.write_mode(tmp_path / "proj", "remote")
        m, _ = _write_batch_module(tmp_path)
        assert vk.run_all(m.process, [1]).values == [11]
        [(_, o)] = _records(cache, "outcome")
        assert o["host"] == "local" and not _records(cache, "host")

    def test_switching_the_mode_mid_batch_moves_later_tasks(self, cache, tmp_path, monkeypatch):
        import threading

        from valuekit import localfile

        monkeypatch.setattr(parallel, "_host_commands", {"h1": (_HOST_CMD, 2)})
        self._local_workers(tmp_path, monkeypatch, 1)
        localfile.write_mode(tmp_path / "proj", "local")
        m, _ = _write_batch_module(tmp_path)
        threading.Timer(2.0, lambda: localfile.write_mode(tmp_path / "proj", "remote")).start()
        r = vk.run_all(m.slow, [1, 2, 3, 4, 5, 6, 7, 8])
        assert r.failures == []
        outcomes = [e for _, e in _records(cache, "outcome")]
        hosts = [e["host"] for e in sorted(outcomes, key=lambda e: e["t"])]
        assert hosts[0] == "local" and hosts[-1] == "h1"

    def test_a_host_that_dies_loses_nothing(self, cache, tmp_path, monkeypatch):
        # The inputs running there are not done and not failed: they run
        # again on this host, once, and the batch is whole.
        import threading

        from valuekit import hosts as machines_mod
        from valuekit import localfile

        monkeypatch.setattr(parallel, "_host_commands", {"h1": (_HOST_CMD, 2)})
        localfile.write_mode(tmp_path / "proj", "remote")
        real_sync = machines_mod.RemoteHost.sync

        def sync_then_die(self):
            # The host process itself dies a second after it has taken work,
            # as a host going down would; the bootstrap that started it
            # is not the host, so killing the link's process would not do.
            reason = real_sync(self)
            if not reason:
                threading.Timer(1.0, os.kill, (self.pid, signal.SIGTERM)).start()
            return reason

        monkeypatch.setattr(machines_mod.RemoteHost, "sync", sync_then_die)
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.slow, [1, 2, 3, 4, 5, 6])
        assert r.failures == []
        assert r.values == [1, 2, 3, 4, 5, 6]
        moved = [e["i"] for _, e in _records(cache, "requeue")]
        assert 1 <= len(moved) <= 2  # what was running on the host when it dropped
        assert all(e["host"] == "h1" for _, e in _records(cache, "requeue"))
        outcomes = {e["i"]: e["host"] for _, e in _records(cache, "outcome")}
        assert all(outcomes[i] == "local" for i in moved)
        assert any(
            e["name"] == "h1" and not e["ok"] and "closed" in e["reason"]
            for _, e in _records(cache, "host")
        )

    def test_an_input_that_loses_two_hosts_is_a_failure(self, cache, tmp_path, monkeypatch):
        from valuekit import hosts as machines_mod
        from valuekit import localfile

        monkeypatch.setattr(parallel, "_host_commands", {"h1": (_HOST_CMD, 1), "h2": (_HOST_CMD, 1)})
        self._local_workers(tmp_path, monkeypatch, 0)
        localfile.write_mode(tmp_path / "proj", "remote")
        m, _ = _write_batch_module(tmp_path)
        # Every host dies as soon as it is given a task.
        real_start = machines_mod.RemoteHost.start

        def start_and_die(self, x):
            handle = real_start(self, x)
            os.kill(self.pid, signal.SIGTERM)
            return handle

        monkeypatch.setattr(machines_mod.RemoteHost, "start", start_and_die)
        r = vk.run_all(m.process, [1])
        [(x, exc)] = r.failures
        assert x == 1 and "closed" in str(exc)
        assert [e["i"] for _, e in _records(cache, "requeue")] == [0]


class TestMonitor:
    def _state(self, events):
        from valuekit import monitor

        st = monitor._State()
        for e in events:
            st.apply("run.jsonl", {"t": time.time(), **e})
        return st

    def test_state_folds_hosts_and_per_host_counts(self):
        st = self._state(
            [
                {"ev": "process", "pid": 1, "role": "main", "argv": ["drive.py"]},
                {"ev": "host", "id": 1, "name": "mac", "ok": True, "capacity": 8},
                {"ev": "host", "id": 1, "name": "pc", "ok": False, "reason": "ssh failed\nmore"},
                {"ev": "batch", "id": 1, "fn": "process", "name": "nightly", "n": 3, "mode": "parallel"},
                {"ev": "start", "id": 1, "i": 0, "host": "mac"},
                {"ev": "start", "id": 1, "i": 1, "host": "mac"},
                {"ev": "start", "id": 1, "i": 2, "host": "local"},
                {"ev": "outcome", "id": 1, "i": 0, "ok": True, "host": "mac"},
                {"ev": "outcome", "id": 1, "i": 2, "ok": False, "host": "local", "exc": "ValueError"},
                {"ev": "start", "id": 1, "i": 3, "host": "mac"},
                {"ev": "requeue", "id": 1, "i": 3, "host": "mac"},
            ]
        )
        assert st.hosts[("run.jsonl", "mac")]["capacity"] == 8
        assert st.per_host["run.jsonl"]["mac"] == {"running": 1, "done": 1, "failed": 0}
        assert st.per_host["run.jsonl"]["local"] == {"running": 0, "done": 1, "failed": 1}
        assert st.hosts[("run.jsonl", "pc")]["ok"] is False

    def test_render_shows_the_mode_and_the_hosts(self):
        from valuekit import monitor

        events = [
            {"ev": "process", "pid": 1, "role": "main", "argv": ["drive.py"]},
            {"ev": "batch", "id": 1, "fn": "process", "name": "nightly", "n": 3, "mode": "parallel"},
            {"ev": "host", "id": 1, "name": "pc", "ok": False, "reason": "ssh failed\nmore"},
            {"ev": "start", "id": 1, "i": 0, "host": "mac"},
        ]
        st = self._state(events)
        text = "\n".join(monitor._render(st, 120, mode="remote", configured=("mac", "pc"), local_workers=4, keys=True))
        assert "mode: remote" in text and "applied" not in text
        assert "l local  r remote  a all  q quit" in text
        # mac has not answered its sync yet: under remote, nothing starts here.
        assert "waiting for mac to sync; none start here" in text
        mac = next(l for l in text.splitlines() if l.strip().startswith("mac "))
        assert mac.split() == ["mac", "0", "1", "0", "0", "syncing"]
        pc = next(l for l in text.splitlines() if l.strip().startswith("pc"))
        assert "dropped: ssh failed" in pc
        local = next(l for l in text.splitlines() if l.strip().startswith("local"))
        assert local.split() == ["local", "0", "0", "0", "0", "idle", "under", "remote"]
        # Once mac is ready, new tasks go there.
        st = self._state(events + [{"ev": "host", "id": 1, "name": "mac", "ok": True, "capacity": 8}])
        text = "\n".join(monitor._render(st, 120, mode="remote", configured=("mac", "pc"), local_workers=4))
        assert "new tasks go to mac; none start here" in text
        # Under all, this machine works too; under local, hosts are not used.
        text = "\n".join(monitor._render(st, 120, mode="all", configured=("mac", "pc"), local_workers=4))
        assert "new tasks go to mac, here" in text
        text = "\n".join(monitor._render(st, 120, mode="local", configured=("mac", "pc"), local_workers=4))
        assert "everything runs here; remote hosts are not used" in text
        # With no batch running, the mode is shown and nothing is claimed.
        text = "\n".join(monitor._render(monitor._State(), 80, mode="local"))
        assert "mode: local" in text and "runs here" not in text

    def test_keys_set_the_mode_line(self, tmp_path):
        from valuekit import monitor, localfile

        cache = tmp_path / "cache"
        assert monitor._apply_key(str(tmp_path), None) is True
        assert monitor._apply_key(str(tmp_path), "r") is True
        assert localfile.load_local(tmp_path).mode == "remote"
        assert monitor._apply_key(str(tmp_path), "A") is True
        assert localfile.load_local(tmp_path).mode == "all"
        assert monitor._apply_key(str(tmp_path), "x") is True  # unknown keys do nothing
        assert localfile.load_local(tmp_path).mode == "all"
        assert monitor._mode_of(str(tmp_path)) == "all"
        (tmp_path / localfile.LOCAL_FILE).write_text("nonsense [[[\n")
        assert monitor._mode_of(str(tmp_path)).startswith("unreadable (")
        assert monitor._mode_of(None) is None
        assert monitor._apply_key(None, "r") is True  # no project: nothing written
        assert not list(tmp_path.glob("**/placement"))
        assert monitor._apply_key(str(tmp_path), "q") is False

    def test_mode_flag_sets_the_line_and_exits(self, tmp_path, capsys, monkeypatch):
        from valuekit import monitor, localfile

        cache = tmp_path / "cache"
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'p'\nversion = '0'\n")
        monkeypatch.chdir(tmp_path)
        assert monitor.main(["--mode", "remote", str(cache)]) == 0
        assert localfile.load_local(tmp_path).mode == "remote"
        assert monitor.main(["--mode", "sideways", str(cache)]) == 2
        assert monitor.main(["--mode"]) == 2
        monkeypatch.chdir(tmp_path.parent)  # no project here
        assert monitor.main(["--mode", "local", str(cache)]) == 2


class TestSweep:
    def _module(self, tmp_path, body_of_f):
        mod = tmp_path / "vk_sweep_mod.py"
        mod.write_text(
            "from valuekit import pure\n"
            "import numpy as np\n"
            "@pure\n"
            f"def f(x):\n    return {body_of_f}\n"
            "@pure\n"
            "def g(x):\n    return (x, np.arange(4.0) * x)\n"
        )
        sys.modules.pop("vk_sweep_mod", None)
        if str(tmp_path) not in sys.path:
            sys.path.insert(0, str(tmp_path))
        import importlib

        return importlib.import_module("vk_sweep_mod")

    def test_sweep_removes_what_the_current_code_cannot_reach(self, cache, tmp_path):
        from valuekit import sweep

        m = self._module(tmp_path, "np.arange(1000.0) * x")
        vk.run_all(m.f, [1, 2], max_workers=1)
        vk.run_all(m.g, [1], max_workers=1)
        vk.log({"q": "note"}, np.arange(5.0))  # from the main process: no record names it
        old_key = m.f._valuekit_reachable().hash
        g_key = m.g._valuekit_reachable().hash
        n_objects = len([p for p in (cache / "objects").rglob("*") if p.is_file()])

        m = self._module(tmp_path, "np.arange(1000.0) * x + 1")  # f edited
        counts = sweep.sweep(cache, ["vk_sweep_mod"], dry_run=True)
        assert counts["records"] == 2 and counts["batches"] == 1
        assert (cache / "records" / old_key).exists()  # a dry run removes nothing

        counts = sweep.sweep(cache, ["vk_sweep_mod"])
        assert counts == {"functions": 2, "records": 2, "batches": 1, "objects": counts["objects"]}
        assert counts["objects"] >= 2  # f's two old result arrays, at least
        assert not (cache / "records" / old_key).exists()
        assert (cache / "records" / g_key).exists()
        with pytest.raises(LookupError):
            vk.batch("f")
        b = vk.batch("g")
        np.testing.assert_array_equal(b[1].result[1], [0.0, 1.0, 2.0, 3.0])  # still readable
        np.testing.assert_array_equal(vk.logs().where(q="note").one().value, np.arange(5.0))
        assert len([p for p in (cache / "objects").rglob("*") if p.is_file()]) < n_objects

        vk.run_all(m.f, [1], max_workers=1)  # the edited f runs and records afresh
        assert vk.batch("f")[1].result[1] == 2.0
        sys.modules.pop("vk_sweep_mod", None)

    def test_sweep_command(self, cache, tmp_path):
        from valuekit import sweep

        self._module(tmp_path, "x")
        out = subprocess.run(
            [sys.executable, "-m", "valuekit.sweep", "--store", str(cache), "--dry-run",
             "vk_sweep_mod"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        assert "2 live functions" in out.stdout
        sys.modules.pop("vk_sweep_mod", None)


def _log_lines(store_dir):
    """The lines of the newest run's log under *store_dir*, in file order."""
    base = _Path(store_dir) / "logs"
    [name] = [p for p in base.iterdir() if p.is_dir()]
    run = name / (name / "latest").read_text().strip()
    lines = []
    for p in sorted(run.glob("*.jsonl")):
        lines.extend(json.loads(raw) for raw in p.read_bytes().splitlines())
    return lines


def _records(store_dir, ev=None):
    """Every log record under a store directory, oldest first, optionally of
    one kind. Reads the files back and parses them, the same out-of-band shape
    the batch tests already use for run counts."""
    events._flush()  # writes are batched on an interval; force them out
    out = []
    runs = _Path(store_dir) / "events"
    for p in sorted(runs.glob("*.jsonl")) if runs.exists() else []:
        for line in p.read_text().splitlines():
            if line.strip():
                out.append((p.name, json.loads(line)))
    out.sort(key=lambda pe: pe[1]["t"])
    return [(s, e) for s, e in out if ev is None or e["ev"] == ev]


class TestEventLog:
    def test_miss_then_hit(self, cache):
        @pure
        def step(x):
            return x + 1

        assert step(1) == 2 and step(1) == 2
        kinds = [e["ev"] for _, e in _records(cache) if e["ev"] in ("hit", "miss")]
        assert kinds == ["miss", "hit"]
        (_, hit), = _records(cache, "hit")
        # The full qualname, matching what a stored record records.
        assert hit["fn"].endswith("step") and hit["dur"] >= 0

    def test_lookup_and_execution_are_timed_separately(self, cache):
        @pure
        def step(x):
            return x + 1

        step(1)
        (_, miss), = _records(cache, "miss")
        # A miss carries both; a hit never executed, so it carries only dur.
        assert "exec" in miss and "dur" in miss and miss["stored"] is True
        step(1)
        (_, hit), = _records(cache, "hit")
        assert "exec" not in hit

    def test_a_deleted_value_is_not_reported_as_a_hit(self, cache):
        # A record can match and the value still be gone; the lookup falls
        # through to the next candidate, so reporting the match would
        # overcount hits.
        @pure
        def step(x):
            return x + 1

        step(1)
        for obj in (cache / "objects").rglob("*"):
            if obj.is_file():
                obj.unlink()
        assert step(1) == 2  # recomputed
        assert _records(cache, "hit") == []
        assert len(_records(cache, "miss")) == 2

    def test_breakpoint_reports_forced_not_hit(self, cache, monkeypatch):
        import bdb

        @pure
        def step(x):
            return x + 1

        step(1)  # recorded
        fname, lo, _ = step._valuekit_reachable().spans[0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)
        try:
            assert step(1) == 2
        finally:
            dbg.clear_all_breaks()
        assert [e["ev"] for _, e in _records(cache, "forced")] == ["forced"]
        assert _records(cache, "hit") == []

    def test_raising_body_is_reported_and_still_raises(self, cache):
        @pure
        def step(x):
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            step(1)
        (_, err), = _records(cache, "error")
        assert err["exc"] == "ValueError" and err["fn"].endswith("step")
        assert _records(cache, "miss") == []  # nothing was stored

    def test_nothing_is_written_without_a_store_directory(self, tmp_path):
        # The documented rule: the store directory is where valuekit writes,
        # and nothing is written until one is named.
        @pure
        def step(x):
            return x + 1

        assert step(1) == 2
        assert not (tmp_path / "events").exists()

    def test_run_all_reports_batch_outcomes_and_end(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2, 4], max_workers=2)
        assert len(r) == 3

        (_, batch), = _records(cache, "batch")
        assert batch["n"] == 3 and batch["mode"] == "parallel"
        outcomes = [e for _, e in _records(cache, "outcome")]
        assert sorted(o["i"] for o in outcomes) == [0, 1, 2]
        assert all(o["ok"] and o["host"] == "local" for o in outcomes)
        assert [e["id"] for _, e in _records(cache, "end")] == [batch["id"]]

    def test_run_all_reports_a_failure_against_its_input(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 3, 4])
        failed = [e for _, e in _records(cache, "outcome") if not e["ok"]]
        assert len(failed) == 1 and failed[0]["i"] == 1
        assert failed[0]["exc"] == "ValueError"

    def test_workers_write_their_own_files(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 2, 4], max_workers=3)
        by_pid = {}
        for source, e in _records(cache, "process"):
            by_pid.setdefault(e["pid"], set()).add(source)
        # No file is shared between processes: concurrent appends are what
        # does not work on Windows.
        assert all(len(files) == 1 for files in by_pid.values())
        roles = [e["role"] for _, e in _records(cache, "process")]
        assert roles.count("main") == 1 and roles.count("worker") == 3

    def test_sequential_fallback_still_reports(self, cache, tmp_path, monkeypatch):
        import bdb

        m, _ = _write_batch_module(tmp_path)
        fname, lo, _ = reachable_set(m.process).spans[0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)
        try:
            vk.run_all(m.process, [1, 2])
        finally:
            dbg.clear_all_breaks()
        (_, batch), = _records(cache, "batch")
        assert batch["mode"] == "sequential"  # debugging a batch is not dark
        assert len(_records(cache, "outcome")) == 2

    def test_emission_failure_does_not_break_a_run(self, cache):
        # A diagnostic that can break a pipeline is worse than no diagnostic.
        # A plain file where the directory belongs makes the mkdir fail, so
        # this is the real failure rather than a patched one.
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "events").write_text("in the way")

        @pure
        def step(x):
            return x + 1

        assert step(1) == 2 and step(1) == 2  # hit and miss both survive
        assert (cache / "events").is_file()

    def test_existing_cache_opens_with_runs_beside_it(self, cache):
        @pure
        def step(x):
            return x + 1

        step(1)
        assert (cache / "events").is_dir()
        # events/ is additive: the format guard still accepts the directory.
        assert LocalStore(cache).get_records("nothing") == []


# ===========================================================================
# the protocol format and a worker on the other end of a pipe
# ===========================================================================
#
# The remote host here is a host process on this host with no network, which is the point:
# the framing, the value codec, the handshake and the failure mapping all get
# exercised in CI without ssh being configured anywhere.


class TestProtocol:
    def test_every_storable_type_round_trips(self):
        ro = np.arange(4.0)
        ro.flags.writeable = False
        for v in (
            None, True, 42, 3.5, 2 + 3j, "hi", b"raw", range(1, 9, 2),
            (1, "a"), [1, "a"], {1, 2}, frozenset({1, 2}),
            {"b": 1, "a": 2}, ImmutableMap({"k": (1, 2)}),
            np.arange(6.0).reshape(2, 3), np.int64(7), ro,
        ):
            root, objs = protocol.pack(v)
            back = protocol.unpack(root, objs)
            if isinstance(v, np.ndarray):
                assert np.array_equal(back, v)
                # Writeability is part of the content hash, so it is part of
                # the value and has to survive the trip.
                assert back.flags.writeable == v.flags.writeable
            else:
                assert back == v and type(back) is type(v)

    def test_types_the_hash_separates_stay_separate(self):
        assert protocol.pack((1, 2))[0] != protocol.pack([1, 2])[0]
        assert protocol.pack({"a": 1, "b": 2})[0] != protocol.pack({"b": 2, "a": 1})[0]
        w = np.arange(3.0)
        r = np.arange(3.0)
        r.flags.writeable = False
        assert protocol.pack(w)[0] != protocol.pack(r)[0]

    def test_a_shared_object_is_sent_once(self):
        big = np.zeros(100)
        _, objs = protocol.pack([big, big, big])
        assert len(objs) == 2  # the list, and the array once

    def test_objects_the_peer_has_are_not_resent(self):
        v = [np.zeros(10), 1]
        _, first = protocol.pack(v)
        _, again = protocol.pack(v, seen=set(first))
        assert again == {}

    def test_a_message_round_trips(self):
        buf = io.BytesIO()
        protocol.write_message(buf, protocol.TASK, b"payload")
        buf.seek(0)
        assert protocol.read_message(buf) == (protocol.TASK, b"payload")
        assert protocol.read_message(buf) is None  # clean end of stream

    def test_a_truncated_message_is_a_transport_error(self):
        buf = io.BytesIO()
        protocol.write_message(buf, protocol.TASK, b"payload")
        cut = io.BytesIO(buf.getvalue()[:-3])
        with pytest.raises(protocol.ProtocolError):
            protocol.read_message(cut)

    def test_an_absurd_length_is_refused_rather_than_allocated(self):
        # An unbounded length is what lets a corrupt header ask for gigabytes.
        buf = io.BytesIO(protocol.TASK + (1 << 62).to_bytes(8, "little"))
        with pytest.raises(protocol.ProtocolError, match="refusing"):
            protocol.read_message(buf)

    def test_a_missing_object_is_a_transport_error_not_a_cache_miss(self):
        # The store turns corruption into a CacheMiss, which correctly means
        # "recompute" for a cache and would wrongly mean it for a connection.
        root, objs = protocol.pack([1, 2, 3])
        objs.pop(next(h for h in objs if h != root))
        with pytest.raises(protocol.ProtocolError):
            protocol.unpack(root, objs)
        assert not issubclass(protocol.ProtocolError, CacheMiss)

    def test_an_unknown_object_marker_is_refused(self):
        root, objs = protocol.pack(7)
        objs[root] = b"?" + objs[root][1:]
        with pytest.raises(protocol.ProtocolError, match="marker"):
            protocol.unpack(root, objs)


def _hello(fn, python=None, function_hash=None, project_hash="", extensions=None):
    # An empty project hash means "no source tree": the worker imports the way
    # it always did, which is what the handshake tests are about.
    return json.dumps(
        {
            "python": python or PYTHON,
            "module": fn.__module__,
            "qualname": fn.__qualname__,
            "function_hash": function_hash or reachable_set(fn).hash,
            "project_hash": project_hash,
            "extensions": extensions if extensions is not None else reachable_set(fn).extensions,
            "roots": [],
        }
    ).encode()


def _hello_fields(**fields):
    """A HELLO body with the given fields and empty defaults for the rest."""
    hello = {"python": PYTHON, "module": "", "qualname": "", "function_hash": "",
             "project_hash": "", "extensions": {}, "roots": [], **fields}
    return json.dumps(hello).encode()


def _handshake(body):
    """Run a worker's handshake in-process; return the ACCEPTED reason."""
    from valuekit import worker

    rx, tx = io.BytesIO(), io.BytesIO()
    protocol.write_message(rx, protocol.HELLO, body)
    rx.seek(0)
    worker.serve(rx, tx)
    tx.seek(0)
    tag, reason = protocol.read_message(tx)
    assert tag == protocol.ACCEPTED
    return reason.decode()


class TestWorkerHandshake:
    def test_matching_code_hash_is_accepted(self, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        assert _handshake(_hello(m.process)) == ""

    def test_a_differing_code_hash_refuses(self, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        reason = _handshake(_hello(m.process, function_hash="0" * 40))
        assert "differs here" in reason and "the code differs" in reason

    def test_a_python_version_mismatch_names_the_interpreter(self, tmp_path):
        # The function hash covers raw bytecode, so two Python versions differ
        # on identical source; the version is checked first so the message
        # says so instead of showing two opaque hashes.
        m, _ = _write_batch_module(tmp_path)
        reason = _handshake(_hello(m.process, python="3.0"))
        assert "Python 3.0" in reason and "differs here" not in reason

    def test_a_function_in___main___is_refused(self, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        body = _hello_fields(module="__main__", qualname="work", function_hash=reachable_set(m.process).hash)
        reason = _handshake(body)
        assert "__main__" in reason and "Move it to a module" in reason

    def test_an_unimportable_module_refuses(self):
        body = _hello_fields(module="no_such_module_xyz", qualname="f", function_hash="0" * 40)
        assert "cannot import" in _handshake(body)

    def test_a_malformed_hello_is_refused_by_name(self):
        assert "malformed HELLO" in _handshake(b"not json")
        body = json.dumps({"python": PYTHON, "module": "m"}).encode()
        assert "malformed HELLO message: KeyError('qualname')" in _handshake(body)


_HOST_CMD = [sys.executable]  # a Python 3 to bootstrap with, as a host entry names one


def _remote_host(fn, cache, name="h1", completions=None):
    """A RemoteHost over a host process launched on this host."""
    import queue
    from valuekit.hosts import RemoteHost

    from valuekit.hosts import ProcessConnection

    _locked_files()
    return RemoteHost(
        project.Project(fn), str(cache), completions or queue.Queue(), name,
        lambda: ProcessConnection(bootstrap.local_command(sys.executable)), str(cache / "source"),
    )


def _settle(handle, completions, timeout=30):
    """Feed *handle* from *completions* until it has an answer or the time is up."""
    import queue

    deadline = time.monotonic() + timeout
    while handle.finished() is None and time.monotonic() < deadline:
        try:
            h, payload = completions.get(timeout=0.2)
        except queue.Empty:
            continue
        h.feed(payload)


class TestRemoteHost:
    """Batches through a host process on this host, in remote mode."""

    @pytest.fixture(autouse=True)
    def _use_host(self, monkeypatch, cache, tmp_path):
        from valuekit import localfile

        _locked_files()
        monkeypatch.setattr(parallel, "_host_commands", {"h1": _HOST_CMD})
        _local_file(tmp_path, 'mode = "remote"\n')

    def test_results_come_back_in_input_order(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 2, 4], max_workers=2)
        assert r.values == [11, 21, 41]

    def test_a_raised_exception_is_attributed_to_its_input(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [1, 3, 4])
        assert [x for x, _ in r.failures] == [3]
        # The protocol cannot carry an exception object, so a remote failure is a
        # RuntimeError naming the original -- unlike the local host, which
        # pickles the exception itself.
        (_, exc), = r.failures
        assert isinstance(exc, RuntimeError)
        assert "bad calibration" in str(exc)
        assert "ValueError" in str(exc.__cause__) or "ValueError" in str(exc)

    def test_a_worker_that_dies_is_recorded_against_its_input(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.hard_death, [0, 1, 2])
        failed = [x for x, _ in r.failures]
        assert failed == [1]
        assert r[0].result() == 0 and r[2].result() == 4  # neighbours unharmed

    def test_a_non_storable_input_names_the_type(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.process, [lambda z: z])
        (_, exc), = r.failures
        assert "function" in str(exc)  # not a pickle error

    def test_the_cache_is_shared_with_workers(self, cache, tmp_path):
        m, counts = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 2])
        n = len(counts())
        vk.run_all(m.process, [1, 2])
        assert len(counts()) == n  # second round: all hits, zero executions

    def test_workers_write_nothing_and_report_through_the_main_process(self, cache, tmp_path):
        # A worker's store is the main process's: its hits, misses, call records and
        # values all arrive here, and it opens no run file of its own.
        m, _ = _write_batch_module(tmp_path)
        vk.run_all(m.process, [1, 2])
        roles = [e["role"] for _, e in _records(cache, "process")]
        assert roles == ["main"]
        misses = [e["fn"] for _, e in _records(cache, "miss")]
        assert sorted(misses) == ["analyse", "analyse", "load", "load", "process", "process"]
        assert LocalStore(cache).get_records(m.load._valuekit_reachable().hash)

    def test_a_pure_local_call_in_a_worker_runs_in_the_main_process(self, cache, tmp_path):
        m, _ = _write_batch_module(tmp_path)
        r = vk.run_all(m.via_local, [1, 2])
        assert r.values == [os.getpid()] * 2  # this process, not a worker
        b = vk.batch("via_local")
        assert b[1].logs.where(q="where").one().value == os.getpid()
        assert [c.result for c in b[1].calls] == [os.getpid()]
        assert vk.logs().where(q="where", x=1).one().value == os.getpid()
        # The main process stored the call's record; the worker's row names it.
        assert LocalStore(cache).get_records(m.here._valuekit_reachable().hash)

    def test_values_logged_in_a_worker_reach_the_main_process_log(self, cache, tmp_path):
        m, counts = _write_batch_module(tmp_path)
        vk.run_all(m.with_log, [1, 2])
        assert [e["host"] for _, e in _records(cache, "start")] == ["h1", "h1"]
        L = vk.logs()
        np.testing.assert_array_equal(L.where(q="arr", sid=2).one().value, [0.0, 2.0, 4.0])
        assert len(L.where(q="twice")) == 4
        assert len(list(L._path.glob("*.jsonl"))) == 1  # the main process's file; no worker wrote
        # A hit inside a worker asks the main process to emit what the record
        # recorded: outer_log misses there, with_log hits.
        vk.run_all(m.outer_log, [1])
        assert len(vk.logs().where(q="parity", sid=1)) == 2
        assert vk.batch("outer_log")[1].logs.where(q="arr").one().value[1] == 1.0

    def test_the_worker_environment_is_an_allowlist(self, cache, tmp_path, monkeypatch):
        m, _ = _write_batch_module(tmp_path)
        monkeypatch.setenv("VK_UNRELATED_SECRET", "hunter2")
        assert vk.run_all(m.env_var, ["VK_UNRELATED_SECRET"]).values == ["absent"]

    def test_the_environment_is_activated_for_workers(self, cache, tmp_path):
        # A tool the lock installed beside the interpreter (cmake, for an
        # extension that rebuilds on import) must be findable by name; no
        # shell activated anything on the host, so the bootstrap does.
        m, _ = _write_batch_module(tmp_path)
        seen = vk.run_all(m.activation, [1]).values[0]
        assert os.path.normcase(seen["path0"]) == os.path.normcase(seen["bindir"])
        assert os.path.normcase(seen["venv"]) == os.path.normcase(seen["prefix"])

    def test_a_timeout_kills_one_input_and_spares_the_rest(self, cache, tmp_path):
        # A worker speaks before it finishes -- a HELLO message, then the result's
        # objects -- so "readable" is not "done". Reading until done would sit
        # inside a task that has already blown its deadline and never come
        # back to enforce it.
        m, _ = _write_batch_module(tmp_path)
        t0 = time.monotonic()
        r = vk.run_all(m.quick_or_hang, [7, 9, 8], max_workers=3, timeout=1.5)
        assert time.monotonic() - t0 < 20  # the hang did not stall the batch
        failed = [(x, type(e).__name__) for x, e in r.failures]
        assert failed == [(9, "TimeoutError")]
        assert r[0].result() == 70 and r[2].result() == 80

    def test_a_breakpoint_still_short_circuits_before_any_host(
        self, cache, tmp_path, monkeypatch
    ):
        import bdb

        m, _ = _write_batch_module(tmp_path)
        fname, lo, _ = reachable_set(m.process).spans[0]
        dbg = bdb.Bdb()
        dbg.set_break(fname, lo + 1)
        monkeypatch.setattr("sys.gettrace", lambda: dbg.trace_dispatch)
        try:
            assert vk.run_all(m.process, [1, 2]).values == [11, 21]
        finally:
            dbg.clear_all_breaks()
        (_, batch), = _records(cache, "batch")
        assert batch["mode"] == "sequential"


# ===========================================================================
# the project as sent to a host
# ===========================================================================
#
# The worker runs on this host, so the main process's live tree is genuinely
# reachable. That is exactly why these tests matter: without the source tree and
# the import check, a worker could import from the live tree and the whole feature
# would look like it worked while proving nothing.


_WORK = "from valuekit import pure\n@pure\ndef work(x):\n    return x + 100\n"


def _project(tmp_path, body=_WORK, extra=None):
    """A small project tree with a marker file, as a real one would have."""
    root = tmp_path / "proj"
    _lay_project(root)
    (root / "vk_sync_mod.py").write_text(body)
    for name, text in (extra or {}).items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _load(root, name="vk_sync_mod"):
    import importlib.util

    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec.loader.exec_module(mod)
    return mod


class TestProject:
    def test_build_inputs_come_from_pyproject(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname = 'p'\nversion = '0'\n")
        assert project.build_inputs(str(root)) is None
        assert project.distribution_name(str(root)) == "p"
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'p'\nversion = '0'\n[tool.valuekit]\nbuild-inputs = ['src/**/*.c']\n"
        )
        assert project.build_inputs(str(root)) == ["src/**/*.c"]
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'p'\nversion = '0'\n[tool.valuekit]\nbuild-inputs = 'src'\n"
        )
        with pytest.raises(project.ProjectError, match="build-inputs"):
            project.build_inputs(str(root))

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for name in [k for k in sys.modules if k.startswith("vk_sync_mod")]:
            del sys.modules[name]

    def test_the_local_file_is_not_in_the_manifest(self, tmp_path):
        root = _project(tmp_path, extra={"valuekit.local.toml": "mode = 'remote'\n"})
        names = [rel for rel, _ in project.manifest(str(root))]
        assert "vk_sync_mod.py" in names and "valuekit.local.toml" not in names

    def test_a_tracked_local_file_warns_once(self, tmp_path, monkeypatch):
        import warnings

        root = _project(tmp_path, extra={"valuekit.local.toml": "mode = 'remote'\n"})
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "valuekit.local.toml"], cwd=root, check=True)
        m = _load(root)
        monkeypatch.setattr(project, "_warned", set())
        with pytest.warns(UserWarning, match="valuekit.local.toml .* tracked"):
            project.Project(m.work)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            project.Project(m.work)  # once per project

    def test_untracked_files_are_included(self, tmp_path):
        # The commonest edit-loop case: a helper written and not yet added.
        root = _project(tmp_path, extra={"helper.py": "X = 1\n"})
        rels = {rel for rel, _ in project.manifest(str(root))}
        assert {"vk_sync_mod.py", "helper.py", "pyproject.toml"} <= rels

    def test_build_artefacts_are_never_shipped(self, tmp_path):
        root = _project(
            tmp_path,
            extra={
                "_core.so": "not really a binary",
                "thing.o": "object file",
                "__pycache__/x.cpython-311.pyc": "bytecode",
            },
        )
        rels = {rel for rel, _ in project.manifest(str(root))}
        assert not any(r.endswith((".so", ".o", ".pyc")) for r in rels)
        assert not any("__pycache__" in r for r in rels)

    def test_the_store_directory_is_not_packed_into_its_own_source_tree(self, tmp_path):
        root = _project(tmp_path)
        (root / "cache").mkdir()
        (root / "cache" / "junk").write_text("x" * 100)
        rels = {rel for rel, _ in project.manifest(str(root), exclude=[root / "cache"])}
        assert not any(r.startswith("cache") for r in rels)

    def test_a_file_deleted_from_the_worktree_does_not_break_the_manifest(
        self, tmp_path
    ):
        # git ls-files --cached reads the index, so a staged-then-deleted path
        # is still listed and cannot be opened. There is no flag for it.
        root = _project(tmp_path, extra={"gone.py": "x = 1\n"})
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        (root / "gone.py").unlink()
        rels = {rel for rel, _ in project.manifest(str(root))}
        assert "vk_sync_mod.py" in rels and "gone.py" not in rels

    def test_the_hash_is_stable_and_moves_with_content(self, tmp_path):
        root = _project(tmp_path)
        first = project.manifest_hash(project.manifest(str(root)))
        assert first == project.manifest_hash(project.manifest(str(root)))
        (root / "vk_sync_mod.py").write_text("from valuekit import pure\n@pure\ndef work(x):\n    return x + 999\n")
        assert project.manifest_hash(project.manifest(str(root))) != first

    def test_content_is_hashed_even_when_mtime_and_size_do_not_move(
        self, tmp_path
    ):
        # Memoising a file hash on (mtime, size) fails in the dangerous
        # direction: a same-size edit within one tick on a coarse-mtime
        # filesystem would keep the old hash, leave the manifest hash
        # unmoved, and let a worker reuse a source tree from the previous content.
        root = _project(tmp_path)
        f = root / "vk_sync_mod.py"
        before = os.stat(f)
        first = project.manifest_hash(project.manifest(str(root)))

        f.write_text("from valuekit import pure\n@pure\ndef work(x):\n    return x + 999\n")  # same length
        os.utime(f, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = os.stat(f)
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns

        assert project.manifest_hash(project.manifest(str(root))) != first

    def test_the_environment_is_not_user_code(self):
        assert project.is_environment(np.__file__)
        assert not project.is_environment(__file__)

    def test_spans_that_are_not_files_are_ignored(self):
        # Spans carry <string> for generated code, and stdlib paths for a user
        # class whose methods came from elsewhere.
        spans = [("<string>", 1, 2), ("relative.py", 1, 2), (np.__file__, 1, 2)]
        assert project.user_span_files(spans) == []

    def test_a_packed_tree_round_trips(self, tmp_path):
        root = _project(tmp_path, extra={"pkg/__init__.py": "", "pkg/a.py": "A = 2\n"})
        entries = project.manifest(str(root))
        dest = tmp_path / "out"
        assert bootstrap._extract(project.pack_tree(str(root), entries), str(dest)) == ""
        assert (dest / "pkg" / "a.py").read_text() == "A = 2\n"
        assert {p.name for p in dest.rglob("*.py")} == {
            "vk_sync_mod.py", "__init__.py", "a.py"
        }


def _trees(cache):
    """The source trees under a store directory (not their markers)."""
    return sorted(p for p in (cache / "source").iterdir() if p.is_dir())


class TestBootstrap:
    """The host's half: a project directory becomes an environment through
    its lock tool, and is updated in place."""

    @pytest.fixture
    def fake_tool(self, monkeypatch, tmp_path):
        # A lock tool that "syncs" by writing a marker script where the
        # interpreter would be, so the table can be exercised without uv.
        made = tmp_path / "made.py"
        made.write_text(
            "import os, sys\n"
            "os.makedirs('env', exist_ok=True)\n"
            "open('env/python', 'w').write(sys.argv[1])\n"
            "open('env/commands', 'a').write(' '.join(sys.argv[2:]) + '\\n')\n"
        )
        row = {
            "tool": sys.executable,
            "install": (str(made), "{python}", "install"),
            "rebuild": (str(made), "{python}", "rebuild", "{dist}"),
            "interpreters": ("env/python",),
            "search": (),
        }
        monkeypatch.setitem(bootstrap._TOOLS, "fake.lock", row)
        monkeypatch.setattr(bootstrap, "KNOWN_LOCKS", ("fake.lock",))
        return row

    def _tree(self, tmp_path, **files):
        """A main process-side tree: its hash, its manifest and a tar of all of it."""
        root = tmp_path / "src"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        for name, text in files.items():
            (root / name).write_text(text)
        entries = project.manifest(str(root))
        return project.manifest_hash(entries), dict(entries), project.pack_tree(str(root), entries)

    def _prepare(self, root, tree, delete=(), data=None, entries=None, build_inputs=None):
        th, files, tar = tree
        return bootstrap._prepare(
            str(root), "p", th, "3.99", list(delete), files, tar if data is None else data,
            "pdist", build_inputs,
        )

    def _commands(self, root):
        return (root / "p" / "env" / "commands").read_text().splitlines()

    def test_the_project_is_rebuilt_when_a_build_input_changed(self, tmp_path, fake_tool):
        root = tmp_path / "source"
        first = self._tree(tmp_path, **{"a.py": "1", "native.c": "int x;", "fake.lock": ""})
        assert self._prepare(root, first)[1] == ""
        assert self._commands(root) == ["install"]  # the first install builds everything
        edited_py = self._tree(tmp_path, **{"a.py": "2", "native.c": "int x;", "fake.lock": ""})
        assert self._prepare(root, edited_py)[1] == ""
        assert self._commands(root)[-1] == "install"  # a Python source needs no build
        edited_c = self._tree(tmp_path, **{"a.py": "2", "native.c": "int y;", "fake.lock": ""})
        assert self._prepare(root, edited_c)[1] == ""
        assert self._commands(root)[-1] == "rebuild pdist"
        # With a build-inputs list, only what it names counts.
        edited_txt = self._tree(tmp_path, **{"a.py": "2", "native.c": "int y;", "fake.lock": "", "n.txt": "x"})
        assert self._prepare(root, edited_txt, build_inputs=["*.c"])[1] == ""
        assert self._commands(root)[-1] == "install"
        edited_c2 = self._tree(tmp_path, **{"a.py": "2", "native.c": "int z;", "fake.lock": "", "n.txt": "x"})
        assert self._prepare(root, edited_c2, build_inputs=["*.c"])[1] == ""
        assert self._commands(root)[-1] == "rebuild pdist"

    def test_a_directory_without_a_manifest_is_rebuilt(self, tmp_path, fake_tool):
        root = tmp_path / "source"
        first = self._tree(tmp_path, **{"a.py": "1", "fake.lock": ""})
        assert self._prepare(root, first)[1] == ""
        (root / "p.manifest").unlink()  # a failed update leaves no manifest
        second = self._tree(tmp_path, **{"a.py": "2", "fake.lock": ""})
        assert self._prepare(root, second)[1] == ""
        assert self._commands(root)[-1] == "rebuild pdist"

    def test_build_inputs(self):
        is_input = bootstrap.is_build_input
        assert is_input("native.c") and is_input("CMakeLists.txt") and is_input("data/x.bin")
        assert not is_input("pkg/a.py") and not is_input("pkg/a.pyi")
        assert is_input("pyproject.toml") and is_input("uv.lock")
        assert is_input("pyproject.toml", ["*.c"]) and is_input("uv.lock", ["*.c"])
        assert is_input("src/a/b.c", ["src/**/*.c"]) and is_input("src/b.c", ["src/**/*.c"])
        assert not is_input("other/b.c", ["src/**/*.c"]) and not is_input("src/a/b.h", ["src/**/*.c"])
        assert is_input("a/b/c.txt", ["**/*.txt"]) and not is_input("a/b/c.txt", ["*.txt"])
        assert is_input("x.c", ["?.c"]) and not is_input("xy.c", ["?.c"])

    def test_a_project_becomes_an_environment(self, tmp_path, fake_tool):
        tree = self._tree(tmp_path, **{"fake.lock": "", "a.py": "A = 1\n"})
        root = tmp_path / "source"
        python, reason = self._prepare(root, tree)
        assert reason == "" and _Path(python).read_text() == "3.99"
        assert (root / "p" / "a.py").read_text() == "A = 1\n"
        manifest = json.loads((root / "p.manifest").read_text())
        assert manifest["project_hash"] == tree[0] and manifest["python"] == python
        assert manifest["files"] == tree[1]
        assert not (root / "p.lock").exists()

    def test_an_update_changes_only_what_the_manifest_says(self, tmp_path, fake_tool):
        root = tmp_path / "source"
        first = self._tree(tmp_path, **{"fake.lock": "", "a.py": "A = 1\n", "old.py": "x\n"})
        assert self._prepare(root, first)[1] == ""
        build = root / "p" / "build" / "cache.txt"  # what a build leaves behind
        build.parent.mkdir()
        build.write_text("keep me")
        second = self._tree(tmp_path, **{"fake.lock": "", "a.py": "A = 2\n", "new.py": "y\n"})
        # The main process sends only what differs, and names what went.
        changed = [(rel, h) for rel, h in second[1].items() if first[1].get(rel) != h]
        tar = project.pack_tree(str(tmp_path / "src"), changed)
        assert sorted(rel for rel, _ in changed) == ["a.py", "new.py"]
        python, reason = self._prepare(root, second, delete=["old.py"], data=tar)
        assert reason == ""
        assert (root / "p" / "a.py").read_text() == "A = 2\n"
        # Written files are newer than anything built from the old tree, so
        # a backend that rebuilds on import rebuilds.
        assert (root / "p" / "a.py").stat().st_mtime > build.stat().st_mtime
        assert (root / "p" / "new.py").exists() and not (root / "p" / "old.py").exists()
        assert build.read_text() == "keep me"
        assert json.loads((root / "p.manifest").read_text())["project_hash"] == second[0]

    def test_a_deletion_outside_the_tree_is_ignored(self, tmp_path, fake_tool):
        root = tmp_path / "source"
        (tmp_path / "precious").write_text("x")
        tree = self._tree(tmp_path, **{"fake.lock": ""})
        assert self._prepare(root, tree, delete=["../../precious", "/etc/passwd"])[1] == ""
        assert (tmp_path / "precious").exists()

    def test_the_sync_names_this_interpreter_when_its_minor_matches(self, tmp_path, fake_tool):
        # A host that already has the main process's minor uses it: no download,
        # and none of the lock tool's search through managed installations.
        mine = "%d.%d" % sys.version_info[:2]
        th, files, tar = self._tree(tmp_path, **{"fake.lock": ""})
        python, reason = bootstrap._prepare(str(tmp_path / "source"), "p", th, mine, [], files, tar)
        assert reason == "" and _Path(python).read_text() == sys.executable
        assert bootstrap._python_request("3.99") == "3.99"

    def test_a_failed_sync_keeps_the_tree_and_drops_the_manifest(self, tmp_path, fake_tool):
        root = tmp_path / "source"
        tree = self._tree(tmp_path, **{"fake.lock": "", "a.py": "A = 1\n"})
        assert self._prepare(root, tree)[1] == ""
        (tmp_path / "made.py").write_text("raise SystemExit('no compiler here')\n")
        second = self._tree(tmp_path, **{"fake.lock": "", "a.py": "A = 2\n"})
        python, reason = self._prepare(root, second)
        assert python == "" and "no compiler here" in reason
        assert (root / "p" / "a.py").read_text() == "A = 2\n"  # the tree stays, updated
        assert not (root / "p.manifest").exists() and not (root / "p.lock").exists()

    def test_a_tree_without_a_known_lock_is_refused(self, tmp_path, fake_tool):
        tree = self._tree(tmp_path, **{"other.lock": ""})
        python, reason = self._prepare(tmp_path / "source", tree)
        assert python == "" and "fake.lock" in reason

    def test_a_missing_tool_says_where_it_looked(self, tmp_path, fake_tool):
        fake_tool["tool"] = "no-such-tool-xyz"
        fake_tool["search"] = ("~/nowhere",)
        tree = self._tree(tmp_path, **{"fake.lock": ""})
        python, reason = self._prepare(tmp_path / "source", tree)
        assert "no-such-tool-xyz" in reason and "~/nowhere" in reason

    def test_extraction_refuses_anything_outside_the_tree(self, tmp_path):
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("../escape.py")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
        assert "escape.py" in bootstrap._extract(buf.getvalue(), str(tmp_path / "out"))
        assert not (tmp_path / "escape.py").exists()

    def test_a_stale_lock_is_broken_and_a_live_one_waited_for(self, tmp_path, fake_tool, monkeypatch):
        root = tmp_path / "source"
        root.mkdir()
        lock = root / "p.lock"
        lock.write_text("")
        os.utime(lock, (time.time() - 2 * bootstrap._STALE,) * 2)
        tree = self._tree(tmp_path, **{"fake.lock": ""})
        assert self._prepare(root, tree)[1] == ""  # the stale lock did not block
        lock.write_text("")  # a live one does, until the wait runs out
        monkeypatch.setattr(bootstrap, "_WAIT", 1)
        python, reason = self._prepare(root, self._tree(tmp_path, **{"fake.lock": "", "b.py": ""}))
        assert python == "" and "another main process" in reason

    def test_stage0_is_shell_safe(self):
        assert not set(bootstrap.STAGE0) & set("$\\%^&|<>\"")
        # The one-liner runs this module off stdin, as ssh will feed it.
        script = _Path(bootstrap.__file__).read_bytes()
        out = subprocess.run(
            bootstrap.local_command(sys.executable),
            input=script + b"\0" + b"not json\n",
            capture_output=True,
        )
        assert out.returncode != 0 and b"Traceback" in out.stderr

    def test_stage_zero_exits_when_its_stream_ends_early(self):
        # A main process that dies before sending the script closes stage 0's
        # stdin. Reading one byte at a time, an EOF is an empty read; it must
        # end the process, not be joined forever (this once left an orphan
        # spinning at full CPU and growing without bound on both platforms).
        p = subprocess.run(
            bootstrap.local_command(sys.executable), input=b"", capture_output=True, timeout=30
        )
        assert p.returncode == 1


class TestSourceTree:
    @pytest.fixture(autouse=True)
    def _use_host(self, monkeypatch, cache, tmp_path):
        from valuekit import localfile

        _locked_files()
        monkeypatch.setattr(parallel, "_host_commands", {"h1": _HOST_CMD})
        _local_file(tmp_path, 'mode = "remote"\n')
        yield
        for name in [k for k in sys.modules if k.startswith("vk_sync_mod")]:
            del sys.modules[name]

    def test_a_batch_runs_from_a_source_tree(self, cache, tmp_path):
        m = _load(_project(tmp_path))
        assert vk.run_all(m.work, [1, 2]).values == [101, 102]
        [tree] = _trees(cache)
        assert tree.name == "p" and (tree / "vk_sync_mod.py").exists()  # the project's name
        # The environment the host built lives in the tree, and the manifest
        # beside it names the interpreter and the project hash.
        assert (tree / ".venv").is_dir()
        manifest = json.loads(tree.with_name("p.manifest").read_text())
        assert _Path(manifest["python"]).exists()
        assert manifest["project_hash"] == project.Project(m.work).project_hash

    def test_the_worker_imports_the_source_tree_not_the_live_tree(self, cache, tmp_path):
        import queue

        root = _project(tmp_path)
        m = _load(root)
        completions = queue.Queue()
        host = _remote_host(m.work, cache, completions=completions)
        assert host.sync() == ""
        # Delete the source outright. If the worker were resolving imports
        # against the live tree this cannot survive.
        (root / "vk_sync_mod.py").unlink()
        handle = host.start(7)
        _settle(handle, completions)
        try:
            done = handle.finished()
            assert done.kind == "ok" and done.value == 107
        finally:
            handle.release()
            host.close()

    def test_an_edit_updates_the_tree_in_place(self, cache, tmp_path):
        root = _project(tmp_path)
        m = _load(root)
        assert vk.run_all(m.work, [1]).values == [101]
        [tree] = _trees(cache)
        kept = tree / "build" / "CMakeCache.txt"  # what an extension build leaves
        kept.parent.mkdir()
        kept.write_text("keep")
        (root / "vk_sync_mod.py").write_text(
            "from valuekit import pure\n@pure\ndef work(x):\n    return x + 999999\n"  # a different length: a
        )                                            # same-length edit within
        m = _load(root)                              # one second reloads the
        assert vk.run_all(m.work, [1]).values == [1000000]  # stale .pyc
        assert _trees(cache) == [tree]  # the same directory, updated
        assert kept.read_text() == "keep"
        assert json.loads(tree.with_name("p.manifest").read_text())["project_hash"] == project.Project(m.work).project_hash

    def test_the_local_file_is_never_sent(self, cache, tmp_path):
        root = _project(tmp_path)
        m = _load(root)
        assert (root / "valuekit.local.toml").exists()  # the fixture wrote the mode there
        assert vk.run_all(m.work, [1]).values == [101]
        [tree] = _trees(cache)
        assert not (tree / "valuekit.local.toml").exists()

    def test_an_unchanged_tree_is_not_resent(self, cache, tmp_path):
        m = _load(_project(tmp_path))
        first = _remote_host(m.work, cache)
        assert first.sync() == ""
        first.close()
        before = (cache / "source" / "p.manifest").stat().st_mtime_ns
        # A second host over the same tree finds the source tree already there
        # and asks for nothing.
        second = _remote_host(m.work, cache)
        assert second.sync() == ""
        second.close()
        assert (cache / "source" / "p.manifest").stat().st_mtime_ns == before
        assert len(_trees(cache)) == 1

    def test_a_tree_without_a_manifest_is_resent_whole(self, cache, tmp_path):
        m = _load(_project(tmp_path))
        host = _remote_host(m.work, cache)
        # A directory with no manifest (a failed or interrupted update) is
        # not trusted: every file is sent and written over what is there.
        half = cache / "source" / "p"
        half.mkdir(parents=True)
        (half / "vk_sync_mod.py").write_text(
            "from valuekit import pure\n@pure\ndef work(x):\n    return 'WRONG'\n"
        )
        assert host.sync() == ""
        host.close()
        assert vk.run_all(m.work, [1]).values == [101]

    def test_a_host_in_use_refuses_a_run_with_different_code(self, cache, tmp_path):
        root = _project(tmp_path)
        m = _load(root)
        first = _remote_host(m.work, cache)
        assert first.sync() == ""
        # The same code again joins: nothing to update.
        same = _remote_host(m.work, cache)
        assert same.sync() == ""
        same.close()
        # Different code is refused while the first run holds the host, and
        # the refusal says which run: the person stops it or waits.
        (root / "vk_sync_mod.py").write_text(
            "from valuekit import pure\n@pure\ndef work(x):\n    return x + 999999\n"
        )
        m2 = _load(root)
        second = _remote_host(m2.work, cache)
        reason = second.sync()
        assert "in use by a run of" in reason and "different version" in reason
        assert "999999" not in (cache / "source" / "p" / "vk_sync_mod.py").read_text()
        first.close()
        # Once the first run has gone its marker goes, and the update proceeds.
        deadline = time.monotonic() + 10
        while list((cache / "source").glob("p.busy-*")) and time.monotonic() < deadline:
            time.sleep(0.1)
        third = _remote_host(m2.work, cache)
        assert third.sync() == ""
        third.close()
        assert "999999" in (cache / "source" / "p" / "vk_sync_mod.py").read_text()

    def test_a_project_without_a_lock_is_refused_before_anything_is_sent(
        self, cache, tmp_path
    ):
        root = _project(tmp_path)
        (root / "uv.lock").unlink()
        m = _load(root)
        assert vk.run_all(m.work, [1]).values == [101]  # locally
        [(_, host)] = _records(cache, "host")
        assert host["ok"] is False
        assert "no lock file" in host["reason"] and "uv.lock" in host["reason"]
        assert not (cache / "source").exists()

    def test_a_dependency_outside_the_project_drops_the_host(self, cache, tmp_path):
        outside = tmp_path / "sibling"
        outside.mkdir()
        (outside / "vk_sync_mod_far.py").write_text("def helper(x):\n    return x\n")
        sys.path.insert(0, str(outside))
        try:
            _load(outside, "vk_sync_mod_far")
            m = _load(
                _project(
                    tmp_path,
                    body="from valuekit import pure\nimport vk_sync_mod_far\n"
                    "@pure\ndef work(x):\n    return vk_sync_mod_far.helper(x)\n",
                )
            )
            # The host is refused before any worker starts, once, as a fact
            # about the host; the batch then runs where it can.
            assert vk.run_all(m.work, [1]).values == [1]
            [(_, host)] = _records(cache, "host")
            assert host["ok"] is False and "outside its project" in host["reason"]
            [(_, outcome)] = _records(cache, "outcome")
            assert outcome["host"] == "local"
        finally:
            sys.path.remove(str(outside))
            sys.modules.pop("vk_sync_mod_far", None)

    def test_a_failed_sync_is_one_reason_not_one_per_input(self, cache, tmp_path):
        m = _load(_project(tmp_path))
        host = _remote_host(m.work, cache)
        host._hello = json.dumps({**json.loads(host._hello), "python": "wrong-python"}).encode()
        reason = host.sync()
        host.close()
        assert "wrong-python" in reason
        assert host.sync() == reason  # remembered, not retried
