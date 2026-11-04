"""Benchmark Worker with CXL backup/restore under the pool's device lock."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from cxl_recsys import cli
from cxl_recsys.mempool import MemoryPool

CHUNK = 8 * 1024**2


def main():
    output = Path(sys.argv[1])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    original_init, original_close = MemoryPool.__init__, MemoryPool.close
    backups = {}

    def save(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.backend != "cxl":
            return
        manifest = {"restored": False, "regions": []}
        backups[id(self)] = manifest
        for index, region in enumerate(self.cxl.regions):
            if region.mapping is None:
                continue
            path = output / f"dax-backup-{index}.bin"
            digest = hashlib.sha256()
            with path.open("xb") as target:
                for begin in range(0, len(region.mapping), CHUNK):
                    data = region.mapping[begin : begin + CHUNK]
                    target.write(data)
                    digest.update(data)
                target.flush()
                os.fsync(target.fileno())
            manifest["regions"].append(
                dict(
                    device=region.device,
                    offset=region.offset,
                    bytes=len(region.mapping),
                    file=str(path.resolve()),
                    sha256=digest.hexdigest(),
                    restored=False,
                )
            )
        (output / "dax-backup.json").write_text(json.dumps(manifest, indent=2))

    def restore(self):
        manifest = backups.get(id(self))
        if manifest is not None and not manifest["restored"]:
            torch.cuda.synchronize()
            for entry, region in zip(manifest["regions"], self.cxl.regions):
                path = Path(entry["file"])
                with path.open("rb") as source:
                    assert (
                        hashlib.file_digest(source, "sha256").hexdigest()
                        == entry["sha256"]
                    )
                    source.seek(0)
                    for begin in range(0, entry["bytes"], CHUNK):
                        data = source.read(min(CHUNK, entry["bytes"] - begin))
                        region.mapping[begin : begin + len(data)] = data
                digest = hashlib.sha256()
                for begin in range(0, entry["bytes"], CHUNK):
                    digest.update(region.mapping[begin : begin + CHUNK])
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError("CXL restore checksum mismatch; retain backup")
                entry["restored"] = True
            manifest["restored"] = True
            (output / "dax-backup.json").write_text(json.dumps(manifest, indent=2))
        original_close(self)

    MemoryPool.__init__ = save
    MemoryPool.close = restore
    cli.main()


if __name__ == "__main__":
    main()
