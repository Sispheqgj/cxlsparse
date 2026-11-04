"""Unambiguous byte quantities shared by YAML, CLI and backing-pool setup."""

import re


def parse_bytes(value) -> int:
    """Bare integers mean bytes; explicit SI/IEC suffixes are also accepted."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]+)\s*(B|[KMGT]i?B)?\s*", value, re.IGNORECASE)
        if match:
            number, suffix = match.groups()
            suffix = (suffix or "B").upper()
            power = 0 if suffix == "B" else "KMGT".index(suffix[0]) + 1
            return int(number) * (1024 if "I" in suffix else 1000) ** power
    raise ValueError(
        "Size must be nonnegative bytes or an integer with units, e.g. 128GiB"
    )


def dax_values(name, value):
    """Normalize legacy scalar options and new list options to the same shape."""
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{name} must be a nonempty list")
    if name != "dax_device":
        return [parse_bytes(item) for item in values]
    devices = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise ValueError("dax_device must contain device names or absolute paths")
        if re.fullmatch(r"dax[0-9]+\.[0-9]+", item):
            item = "/dev/" + item
        if not item.startswith("/"):
            raise ValueError("dax_device must be a daxN.M name or an absolute path")
        devices.append(item)
    return devices


def dax_regions(devices, offsets, sizes) -> list[tuple[str, int, int]]:
    """Return ordered (device, offset bytes, reserved bytes) region definitions."""
    devices = dax_values("dax_device", devices)
    offsets = dax_values("dax_offset", offsets)
    sizes = dax_values("dax_bytes", sizes)
    if devices is None or offsets is None or sizes is None:
        raise ValueError("CXL requires dax_device, dax_offset and dax_bytes")
    if not len(devices) == len(offsets) == len(sizes):
        raise ValueError("dax_device, dax_offset and dax_bytes must have equal lengths")
    if any(size <= 0 for size in sizes):
        raise ValueError("Every dax_bytes region size must be positive")
    return [
        (str(device), parse_bytes(offset), parse_bytes(size))
        for device, offset, size in zip(devices, offsets, sizes)
    ]
