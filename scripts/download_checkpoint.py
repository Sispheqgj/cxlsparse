"""Pinned checkpoint transfers with range resume and LFS checksums."""

import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request

FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def download_checkpoint(output, repository, revision, files, *, workers=1):
    """Fetch fixed files with resumable transfers and LFS checksum validation."""
    output.mkdir(parents=True, exist_ok=True)
    url = f"{ENDPOINT}/api/models/{repository}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    for name in files:
        target = output / name
        entry = entries[name]
        digest = entry.get("lfs", {}).get("sha256")
        if target.exists():
            if target.stat().st_size != entry["size"]:
                raise ValueError(f"Existing file has the wrong size: {target}")
            if digest:
                with target.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                        raise ValueError(f"Existing file checksum mismatch: {target}")
            print(f"Already present: {target}", flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"{ENDPOINT}/{repository}/resolve/{revision}/{name}"
        if workers > 1 and entry["size"] > 32 * 1024**2:
            download_parallel(url, partial, entry["size"], workers)
        failures = 0
        while not partial.exists() or partial.stat().st_size < entry["size"]:
            offset = partial.stat().st_size if partial.exists() else 0
            end = min(offset + 32 * 1024**2, entry["size"]) - 1
            request = urllib.request.Request(
                f"{url}?download=true&offset={offset}&end={end}&attempt={failures}",
                headers={"Range": f"bytes={offset}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if offset and not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError("Server did not honor the resume offset")
                    with partial.open("ab") as destination:
                        while chunk := response.read(1024 * 1024):
                            destination.write(chunk)
                if partial.stat().st_size == offset:
                    raise OSError("Download made no progress")
                print(f"{name}: {partial.stat().st_size}/{entry['size']}", flush=True)
                failures = 0
            except OSError as exc:
                failures += 1
                print(f"Retry {failures} for {name} at {offset}: {exc}", flush=True)
                if failures >= 5:
                    raise
                time.sleep(2)
        if partial.stat().st_size != entry["size"]:
            raise ValueError(f"Incomplete download: {partial}")
        if digest:
            with partial.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(f"Download checksum mismatch: {partial}")
        partial.rename(target)
        print(f"Downloaded: {target} ({entry['size']} bytes)", flush=True)
    (output / "source.json").write_text(
        json.dumps({"repository": repository, "revision": revision}, indent=2) + "\n"
    )


def download_parallel(url, partial, size, workers):
    """Resume disjoint ranges, then append in order; final LFS hash is mandatory."""
    offset = partial.stat().st_size if partial.exists() else 0
    parts = partial.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    ranges = [
        (start, min(start + 4 * 1024**2, size))
        for start in range(offset, size, 4 * 1024**2)
    ]

    def fetch(bounds):
        start, stop = bounds
        target = parts / str(start)
        if target.exists() and target.stat().st_size == stop - start:
            return
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    f"{url}?download=true&range={start}-{stop - 1}&try={attempt}",
                    headers={"Range": f"bytes={start}-{stop - 1}"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    if (
                        response.headers.get("Content-Range")
                        != f"bytes {start}-{stop - 1}/{size}"
                    ):
                        raise ValueError("Server did not honor the requested range")
                    with target.open("wb") as destination:
                        while chunk := response.read(1024 * 1024):
                            destination.write(chunk)
                if target.stat().st_size != stop - start:
                    raise OSError("Incomplete range")
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for done, _ in enumerate(executor.map(fetch, ranges), 1):
            if done % 16 == 0 or done == len(ranges):
                print(
                    f"{partial.name}: completed ranges {done}/{len(ranges)}", flush=True
                )
    with partial.open("ab") as destination:
        for start, _ in ranges:
            part = parts / str(start)
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            destination.flush()
            part.unlink()
    parts.rmdir()
