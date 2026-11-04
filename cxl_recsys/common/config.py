"""YAML defaults validated against the actual argparse option definitions."""

import argparse
import math
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT
from .sizes import dax_values

DEFAULT_CONFIG = PROJECT_ROOT / "config.yml"
# Build the effective Worker profile unless the build command overrides a field.
# Inheritance is resolved AFTER merging --config, not just in the default YAML.
BUILD_WORKER_DEFAULTS = (
    "kernel_dir",
    "hot",
    "dim",
    "backend",
    "model",
    "model_path",
)
NULLABLE = {
    ("scheduler", "input"),
    ("scheduler", "save_requests"),
    ("scheduler", "gr_heat_path"),
    ("scheduler", "gr_curve_field"),
    ("scheduler", "gr_text_catalog_path"),
    ("scheduler", "gr_user_probabilities"),
    ("scheduler", "gr_item_probabilities"),
    ("worker", "dax_device"),
    ("worker", "dax_offset"),
    ("worker", "dax_bytes"),
    ("worker", "pool_target"),
    ("worker", "model_path"),
    ("build-kernels", "model_path"),
}


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        if key in mapping:
            raise ValueError(f"Duplicate configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def read_config(path):
    try:
        with Path(path).open() as source:
            data = yaml.load(source, Loader=UniqueSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def configure(parsers, override=None):
    data = read_config(DEFAULT_CONFIG)
    schema = {}
    for name, parser in parsers.items():
        options = {}
        for action in parser._actions:
            if action.dest not in ("help", "config"):
                options[action.dest] = action
        schema[name] = options

    def check_keys(values, complete):
        if set(values) - set(schema):
            raise ValueError(f"Unknown config sections: {set(values) - set(schema)}")
        if complete and set(values) != set(schema):
            raise ValueError("Default config must contain every subcommand section")
        for section, options in values.items():
            if not isinstance(options, dict):
                raise ValueError(f"{section} must be a mapping")
            unknown = set(options) - set(schema[section])
            missing = set(schema[section]) - set(options)
            if section == "build-kernels":
                missing -= set(BUILD_WORKER_DEFAULTS)
            if unknown or (complete and missing):
                raise ValueError(
                    f"{section}: unknown keys={unknown}, missing defaults={missing if complete else set()}"
                )

    check_keys(data, True)
    if override and Path(override).resolve() != DEFAULT_CONFIG.resolve():
        extra = read_config(override)
        check_keys(extra, False)
        for name, values in extra.items():
            data[name].update(values)
    for key in BUILD_WORKER_DEFAULTS:
        data["build-kernels"].setdefault(key, data["worker"][key])
    for name, options in data.items():
        for key, value in options.items():
            action = schema[name][key]
            label = f"{name}.{key}"
            if name == "worker" and key in ("dax_device", "dax_offset", "dax_bytes"):
                options[key] = dax_values(key, value)
                continue
            if value is None:
                if (name, key) not in NULLABLE:
                    raise ValueError(f"{label} cannot be null")
                continue
            values = value if action.nargs == "+" else [value]
            if action.nargs == "+" and (not isinstance(value, list) or not value):
                raise ValueError(f"{label} must be a nonempty list")
            expected = (
                bool
                if isinstance(action, argparse.BooleanOptionalAction)
                else action.type or str
            )
            for item in values:
                valid = type(item) is expected
                if expected is float:
                    valid = type(item) in (int, float) and math.isfinite(item)
                if not valid:
                    raise ValueError(f"{label} must contain {expected.__name__} values")
                if action.choices and item not in action.choices:
                    raise ValueError(f"{label} must be one of {action.choices}")
        parsers[name].set_defaults(**options)
    return data
