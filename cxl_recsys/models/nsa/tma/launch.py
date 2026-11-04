"""Keep explicit cooperative launches through TileLang's host/device split."""

_installed = False
_MARKER = "nsa.cooperative_launch"


def cooperative_launch(function):
    """Tag this kernel and preserve its launch contract after device lowering."""
    global _installed
    if not _installed:
        from tilelang.backend.pass_pipeline import (
            PassPipeline,
            get_pipeline,
            register_pipeline,
        )

        original = get_pipeline("cuda")

        def lower(module, target):
            enabled = any(
                f.attrs.get(_MARKER, False) for f in module.functions.values()
            )
            lowered = original.lower(module, target)
            if enabled:
                for symbol, func in lowered.functions.items():
                    if func.attrs.get("tirx.is_global_func", False):
                        lowered.update_func(
                            symbol, func.with_attr("use_cooperative_groups", True)
                        )
            return lowered

        register_pipeline(PassPipeline("cuda", lower))
        _installed = True
    # The marker also changes TileLang's compilation cache key.
    return function.with_attr(_MARKER, True)
