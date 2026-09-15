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
    """:class:`~valuekit.store.CacheStore` over a framed connection.

    Events, run-log lines and ``@pure_local`` calls always go to the main
    process.  Values and call records do too, unless *direct* is a
    :class:`~valuekit.store.LocalStore` on the main process's store
    directory, which a worker on the main process's machine is given so
    that it reads and writes them itself.
    """

    def __init__(self, rx: BinaryIO, tx: BinaryIO, direct=None):
        self._rx = rx
        self._tx = tx
        self._direct = direct
        self._objects: dict[str, bytes] = {}  # everything sent or received
        self._seen: set[str] = set()  # the same, as the hashes protocol.pack skips

    # -- objects ----------------------------------------------------------

    def receive(self, body: bytes) -> None:
        """Keep one OBJECT message the main process sent."""
        protocol.recv_object(body, self._objects)
        self._seen.add(body[:20].hex())

    def unpack(self, root: str) -> Any:
        fallback = self._direct.get_value if self._direct is not None else None
        return protocol.unpack(root, self._objects, fallback)

    def put_value(self, v: Any) -> str:
        if self._direct is not None:
            return self._direct.put_value(v)
        root, objects = protocol.pack(v, self._seen)
        for h, payload in objects.items():
            protocol.write_message(self._tx, protocol.OBJECT, bytes.fromhex(h) + payload)
            self._objects[h] = payload
            self._seen.add(h)
        return root

    def get_value(self, h: str) -> Any:
        if self._direct is not None:
            return self._direct.get_value(h)
        if h not in self._objects:
            protocol.write_message(self._tx, protocol.GET_VALUE, bytes.fromhex(h))
            reply = json.loads(self._reply())
            if "reason" in reply:
                raise CacheMiss(f"{h}: {reply['reason']}")
        try:
            return self.unpack(h)
        except protocol.ProtocolError as e:
            raise CacheMiss(f"{h}: {e}") from e

    # -- call records -------------------------------------------------------------

    def get_records(self, function_hash: str) -> list[tuple[str, dict]]:
        if self._direct is not None:
            return self._direct.get_records(function_hash)
        protocol.write_message(self._tx, protocol.GET_RECORDS, function_hash.encode())
        return [(h, t) for h, t in json.loads(self._reply())]

    def put_record(self, function_hash: str, record: dict) -> str:
        if self._direct is not None:
            return self._direct.put_record(function_hash, record)
        body = json.dumps({"function_hash": function_hash, "record": record}).encode()
        protocol.write_message(self._tx, protocol.RECORD, body)
        return record_hash(record)

    # -- the event log and the run's log, written by the main process -------------

    def event(self, record: dict) -> None:
        protocol.write_message(self._tx, protocol.EVENT, json.dumps(record).encode())


    def log_line(self, line: dict) -> None:
        protocol.write_message(self._tx, protocol.LOGGED, json.dumps(line).encode())

    # -- a call that must run on the main process ----------------------------------

    def local_call(self, module: str, qualname: str, args: tuple, kwargs: dict):
        """Run ``module:qualname(*args, **kwargs)`` on the main process.

        Returns ``(value, record_hash)``; the hash is empty if the main process
        stored no call record for the call.  A failure there raises here, with
        the main process's traceback as the message.
        """
        root = self.put_value((args, kwargs))
        call = json.dumps({"module": module, "qualname": qualname, "root": root}).encode()
        protocol.write_message(self._tx, protocol.CALL, call)
        reply = json.loads(self._reply())
        if "failed" in reply:
            f = reply["failed"]
            raise RuntimeError(
                f"{qualname} failed on the main process: {f['type']}: {f['message']}\n{f['traceback']}"
            )
        return self.unpack(reply["root"]), reply["record_hash"]

    # -- replies ----------------------------------------------------------------

    def _reply(self) -> bytes:
        """Read messages until the reply to the request just sent; keep the
        objects that arrive before it."""
        while True:
            message = protocol.read_message(self._rx)
            if message is None:
                raise protocol.ProtocolError("the main process went away mid-request")
            got, body = message
            if got == protocol.OBJECT:
                self.receive(body)
            elif got == protocol.REPLY:
                return body
            else:
                raise protocol.ProtocolError(f"expected a reply, got {got!r}")
