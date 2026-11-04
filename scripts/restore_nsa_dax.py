"""Restore saved devdax bytes after the failed Worker has exited."""

import fcntl
import hashlib
import json
import mmap
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
chunk = 8 * 1024**2
for region in manifest["regions"]:
    source = Path(region["file"])
    assert source.stat().st_size == region["bytes"]
    with source.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == region["sha256"]
    fd = os.open(region["device"], os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mapping = mmap.mmap(
            fd,
            region["bytes"],
            offset=region["offset"],
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            with source.open("rb") as file:
                for start in range(0, region["bytes"], chunk):
                    data = file.read(min(chunk, region["bytes"] - start))
                    mapping[start : start + len(data)] = data
            digest = hashlib.sha256()
            for start in range(0, region["bytes"], chunk):
                digest.update(mapping[start : start + chunk])
            assert digest.hexdigest() == region["sha256"]
            region["restored"] = True
        finally:
            mapping.close()
    finally:
        os.close(fd)
manifest["restored"] = True
manifest["recovery"] = "independent CPU-only process after Worker CUDA context failure"
path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
