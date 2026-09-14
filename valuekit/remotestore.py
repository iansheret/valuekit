"""A worker's store: the main process's store, reached over the connection.

A worker keeps no cache of its own.  Every value and record it produces is
sent to the main process, every lookup asks the main process, and every event
goes there too, so the main process's directory is the one place a batch's
results exist wherever the work ran.  This is also what lets a
``@pure_local`` function called in a worker run on the main process instead:
the call is a request like any other, and the reply is a value.

The conversation is strictly sequential on this side -- one request, then
its reply -- so nothing here multiplexes.  Replies are read off the same
stream the task arrived on; an OBJECT message at any point is one more
object the main process has sent, and is stored.

Objects sent and objects received are both indexed by hash: the main process
has all of them, so none is sent twice, and a reply can name any of them
without repeating it.
"""

from __future__ import annotations

import json
from typing import Any, BinaryIO

from . import protocol
from .store import CacheMiss, record_hash

__all__ = ["RemoteStore"]


class RemoteStore:
    """:class:`~valuekit.store.CacheStore` over a framed connection."""

    def __init__(self, rx: BinaryIO, tx: BinaryIO):
        self._rx = rx
        self._tx = tx
        self._objects: dict[str, bytes] = {}  # everything sent or received
        self._seen: set[str] = set()  # the same, as the hashes protocol.pack skips

    # -- objects ----------------------------------------------------------

    def receive(self, body: bytes) -> None:
        """Keep one OBJECT message the main process sent."""
        protocol.recv_object(body, self._objects)
        self._seen.add(body[:20].hex())

    def unpack(self, root: str) -> Any:
        return protocol.unpack(root, self._objects)

    def put_value(self, v: Any) -> str:
        root, objects = protocol.pack(v, self._seen)
        for h, payload in objects.items():
            protocol.write_message(self._tx, protocol.OBJECT, bytes.fromhex(h) + payload)
            self._objects[h] = payload
            self._seen.add(h)
        return root

    def get_value(self, h: str) -> Any:
        if h not in self._objects:
            protocol.write_message(self._tx, protocol.GET_VALUE, bytes.fromhex(h))
            reason = self._reply(protocol.VALUE)
            if reason:
                raise CacheMiss(f"{h}: {reason.decode('utf-8', 'replace')}")
        try:
            return self.unpack(h)
        except protocol.ProtocolError as e:
            raise CacheMiss(f"{h}: {e}") from e

    # -- call records -------------------------------------------------------------

    def get_records(self, function_hash: str) -> list[tuple[str, dict]]:
        protocol.write_message(self._tx, protocol.GET_RECORDS, function_hash.encode())
        body = self._reply(protocol.RECORDS)
        return [(h, t) for h, t in json.loads(body)]

    def put_record(self, function_hash: str, record: dict) -> str:
        body = json.dumps({"function_hash": function_hash, "record": record}).encode()
        protocol.write_message(self._tx, protocol.RECORD, body)
        return record_hash(record)

    # -- the main process's side of the run log ------------------------------------

    def emit(self, record: dict) -> None:
        protocol.write_message(self._tx, protocol.EVENT, json.dumps(record).encode())

    # -- the main process's side of the run's log ------------------------------------

    def emit_line(self, line: str) -> None:
        protocol.write_message(self._tx, protocol.LOGGED, line.encode())

    # -- a call that must run on the main process ----------------------------------

    def local_call(self, module: str, qualname: str, args: tuple, kwargs: dict):
        """Run ``module:qualname(*args, **kwargs)`` on the main process.

        Returns ``(value, record_hash)``; the hash is empty if the main process
        stored no call record for the call.  A failure there raises here, with
        the main process's traceback as the message.
        """
        root = self.put_value((args, kwargs))
        protocol.write_message(self._tx, protocol.CALL, protocol.strings(module, qualname, root))
        body = self._reply(protocol.CALLED)
        if body[:1] == b"o":
            result_root, h = protocol.unstrings(body[1:])
            return self.unpack(result_root), h
        kind, text, tb = protocol.unstrings(body[1:])
        raise RuntimeError(f"{qualname} failed on the main process: {kind}: {text}\n{tb}")

    # -- replies ----------------------------------------------------------------

    def _reply(self, tag: bytes) -> bytes:
        """Read messages until the reply tagged *tag*; keep objects on the way."""
        while True:
            message = protocol.read_message(self._rx)
            if message is None:
                raise protocol.ProtocolError("the main process went away mid-request")
            got, body = message
            if got == protocol.OBJECT:
                self.receive(body)
            elif got == tag:
                return body
            else:
                raise protocol.ProtocolError(f"expected {tag!r}, got {got!r}")
