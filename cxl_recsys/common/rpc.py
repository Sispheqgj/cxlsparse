"""Shared gRPC unary RPCs with UTF-8 JSON envelopes (no pickle).

Only request tokens and control metadata cross gRPC. KV payloads never do.
See docs/system.md for the public wire contract and interface definitions.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import grpc
import orjson

SERVICE = "cxlrecsys.v1.Control"
logger = logging.getLogger(__name__)
OPTIONS = [
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ("grpc.max_send_message_length", 8 * 1024 * 1024),
]


class ValidatedEnvelope(dict):
    """Internal marker for SubmitBatch envelopes built from validated requests."""


def encode(value):
    if isinstance(value, ValidatedEnvelope):
        try:
            encoded = orjson.dumps(value)
            # Retain strict nonfinite rejection even if the internal marker is
            # misused. Native JSON maps nonfinite floats to null.
            if b"null" not in encoded:
                return encoded
        except TypeError:
            # Preserve arbitrary-size integer metadata and stdlib semantics.
            pass
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def decode_submit_batch(raw):
    """Fast token-array parsing; validation remains mandatory in Worker.

    Native JSON can round oversized integers to floats. Tokens must be int32,
    but metadata permits larger integers, so reparse those envelopes exactly.
    """
    try:
        value = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        return json.loads(raw)
    for request in value["requests"]:
        if not isinstance(request, dict) or any(
            type(request.get(key)) is float for key in ("timestamp", "visit_index")
        ):
            return json.loads(raw)
    return value


class RPC:
    def __init__(self, target):
        self.channel = grpc.insecure_channel(target, options=OPTIONS)
        self.methods = {}

    def _method(self, method):
        # Reuse gRPC callables; the channel remains shared and thread safe.
        if method not in self.methods:
            self.methods[method] = self.channel.unary_unary(
                f"/{SERVICE}/{method}", request_serializer=encode,
                response_deserializer=json.loads,
            )
        return self.methods[method]

    def call(self, method, value=None, timeout=10):
        return self._method(method)(value or {}, timeout=timeout)

    def call_async(self, method, value=None, timeout=10):
        return self._method(method).future(value or {}, timeout=timeout)

    def close(self):
        self.channel.close()


def serve(address, methods):
    server = grpc.server(ThreadPoolExecutor(max_workers=16), options=OPTIONS)

    def wrap(method, fn):
        def handle(request, context):
            try:
                return fn(request)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "rpc_rejected handler=%s error_type=%s",
                    fn.__name__,
                    type(exc).__name__,
                )
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:
                logger.exception("rpc_handler_failed handler=%s", fn.__name__)
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return grpc.unary_unary_rpc_method_handler(
            handle, request_deserializer=(decode_submit_batch if method == "SubmitBatch" else json.loads), response_serializer=encode
        )

    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE, {k: wrap(k, v) for k, v in methods.items()}
            ),
        )
    )
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Cannot bind {address}")
    server.start()
    return server
