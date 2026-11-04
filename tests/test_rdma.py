import ctypes

import pytest
import torch

from cxl_recsys.mempool.rdma import Endpoint, Region


def test_endpoint_roundtrip_preserves_read_capabilities():
    endpoint = Endpoint()
    endpoint.addr = 123456
    endpoint.max_rd_atomic = 16
    endpoint.max_dest_rd_atomic = 8
    endpoint.gid[:] = list(range(16))
    restored = Endpoint.parse(endpoint.as_dict())
    assert restored.as_dict() == endpoint.as_dict()
    assert restored.max_rd_atomic == 16
    assert restored.max_dest_rd_atomic == 8


@pytest.mark.rdma
def test_real_verbs_read_write_loopback():
    server = Region(4096)
    client = Region(128)
    try:
        server.connect(client.endpoint.as_dict())
        client.connect(server.endpoint.as_dict())
        assert client.endpoint.max_rd_atomic > 0
        assert server.endpoint.max_dest_rd_atomic > 0
        source = torch.arange(64).to(torch.bfloat16)
        target = torch.empty_like(source)
        client.transfer(256, source, True)
        assert ctypes.string_at(server.endpoint.addr + 256, 128) == ctypes.string_at(
            source.data_ptr(), 128
        )
        client.transfer(256, target, False)
        assert torch.equal(source, target)
        with pytest.raises(RuntimeError, match="bounds"):
            client.transfer(4096, target, False)
    finally:
        client.close()
        server.close()
