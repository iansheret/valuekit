"""A worker's store: the driver's store, reached over the connection.

A worker keeps no cache of its own.  Every value and trace it produces is
sent to the driver, every lookup asks the driver, and every run-log record
goes there too, so the driver's directory is the one place a batch's
results exist wherever the work ran.  This is also what lets a
``@pure_local`` function called in a worker run on the driver instead:
the call is a request like any other, and the answer is a value.

The conversation is strictly sequential on this side -- one request, then
its reply -- so nothing here multiplexes.  Replies are read off the same
stream the task arrived on; an OBJECT frame at any point is one more
object the driver has sent, and is kept.

Objects sent and objects received are both remembered by hash: the driver
has all of them, so none is sent twice, and a reply can name any of them
without repeating it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, BinaryIO

from . import wire
from .store import CacheMiss, trace_hash

__all__ = ["WireStore"]


class WireStore:
    """:class:`~valuekit.store.CacheStore` over a framed connection."""

    def __init__(self, rx: BinaryIO, tx: BinaryIO):
        self._rx = rx
        self._tx = tx
        self._objects: dict[str, bytes] = {}  # everything sent or received
        self._seen: set[str] = set()  # the same, as the hashes wire.pack skips

    # -- objects ----------------------------------------------------------

    def receive(self, body: bytes) -> None:
        """Keep one OBJECT frame the driver sent."""
        wire.recv_object(body, self._objects)
        self._seen.add(body[:20].hex())

    def unpack(self, root: str) -> Any:
        return wire.unpack(root, self._objects)

    def put_value(self, v: Any) -> str:
        root, objects = wire.pack(v, self._seen)
        for h, payload in objects.items():
            wire.write_frame(self._tx, wire.OBJECT, bytes.fromhex(h) + payload)
            self._objects[h] = payload
            self._seen.add(h)
        return root

    def get_value(self, h: str) -> Any:
        if h not in self._objects:
            wire.write_frame(self._tx, wire.GET_VALUE, bytes.fromhex(h))
            reason = self._reply(wire.VALUE)
            if reason:
                raise CacheMiss(f"{h}: {reason.decode('utf-8', 'replace')}")
        try:
            return self.unpack(h)
        except wire.WireError as e:
            raise CacheMiss(f"{h}: {e}") from e

    # -- traces -------------------------------------------------------------

    def get_traces(self, fn_key: str) -> list[tuple[str, dict]]:
        wire.write_frame(self._tx, wire.GET_TRACES, fn_key.encode())
        body = self._reply(wire.TRACES)
        return [(h, t) for h, t in json.loads(body)]

    def put_trace(self, fn_key: str, trace: dict, units: Sequence[str] = ()) -> str:
        wire.write_frame(
            self._tx, wire.TRACE, wire.strings(fn_key, json.dumps(trace), *units)
        )
        return trace_hash(trace)

    # -- the driver's side of the run log ------------------------------------

    def emit(self, record: dict) -> None:
        wire.write_frame(self._tx, wire.EVENT, json.dumps(record).encode())

    # -- a call that must run on the driver ----------------------------------

    def local_call(self, module: str, qualname: str, args: tuple, kwargs: dict):
        """Run ``module:qualname(*args, **kwargs)`` on the driver.

        Returns ``(value, trace_hash)``; the hash is empty if the driver
        stored no trace for the call.  A failure there raises here, with
        the driver's traceback as the message.
        """
        root = self.put_value((args, kwargs))
        wire.write_frame(self._tx, wire.CALL, wire.strings(module, qualname, root))
        body = self._reply(wire.CALLED)
        if body[:1] == b"o":
            result_root, h = wire.unstrings(body[1:])
            return self.unpack(result_root), h
        kind, text, tb = wire.unstrings(body[1:])
        raise RuntimeError(f"{qualname} failed on the driver: {kind}: {text}\n{tb}")

    # -- replies ----------------------------------------------------------------

    def _reply(self, tag: bytes) -> bytes:
        """Read frames until the reply tagged *tag*; keep objects on the way."""
        while True:
            frame = wire.read_frame(self._rx)
            if frame is None:
                raise wire.WireError("the driver went away mid-request")
            got, body = frame
            if got == wire.OBJECT:
                self.receive(body)
            elif got == tag:
                return body
            else:
                raise wire.WireError(f"expected {tag!r}, got {got!r}")
