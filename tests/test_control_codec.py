import datetime
import json
import uuid

import pytest

from cxl_recsys.common.batch import validate_batch
from cxl_recsys.common.request import placeholder, validate
from cxl_recsys.common.rpc import ValidatedEnvelope, decode_submit_batch, encode


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"visit_index": 2**100},
        {"timestamp": 2**100},
        {"timestamp": 1.25},
        {"visit_index": 2**64 - 1},
    ],
)
@pytest.mark.parametrize("user", ["normal", "中文用户", 'quote"\\\n\t'])
def test_native_valid_envelope_equivalence(metadata, user):
    request = placeholder(user_id=user, history_length=8192)
    request.update(metadata)
    original = {"batch_id": "batch", "requests": [request], "expected_epoch": "epoch"}
    decoded = decode_submit_batch(encode(ValidatedEnvelope(original)))
    assert decoded == json.loads(json.dumps(original))
    assert validate_batch(decoded, max_history=8192) == validate_batch(
        original, max_history=8192
    )
    for key in metadata:
        assert type(decoded["requests"][0][key]) is type(metadata[key])


@pytest.mark.parametrize("token", [True, False, 1.0, -1, 2**31, 2**100, None, "1"])
def test_invalid_tokens_still_rejected(token):
    request = placeholder()
    request["history_token_ids"] = [token]
    decoded = decode_submit_batch(
        json.dumps({"batch_id": "batch", "requests": [request]}).encode()
    )
    with pytest.raises(ValueError):
        validate_batch(decoded)
    with pytest.raises(ValueError):
        validate(request)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_strict_nonfinite_encoding(value):
    for envelope in [dict, ValidatedEnvelope]:
        with pytest.raises(ValueError):
            encode(envelope(timestamp=value))


@pytest.mark.parametrize("value", [uuid.uuid4(), datetime.datetime(2020, 1, 1)])
def test_generic_json_rejects_extended_types(value):
    with pytest.raises(TypeError):
        encode({"value": value})


@pytest.mark.parametrize(
    "raw", [b"null", b"[]", b'{"requests":[null]}', b'{"requests":[{"timestamp":NaN}]}']
)
def test_invalid_shape_keeps_validation_rejection(raw):
    decoded = decode_submit_batch(raw)
    with pytest.raises(ValueError):
        validate_batch(decoded)


def test_token_copy_and_exact_types():
    request = placeholder()
    parsed = validate(request)
    request["history_token_ids"][0] = 9
    assert parsed["history_token_ids"][0] == 1

    class Int(int):
        pass

    request["history_token_ids"] = [Int(1)]
    with pytest.raises(ValueError):
        validate(request)
