from __future__ import annotations


class BboxArrangerDiffLow:
    """Compatibility stub for a removed legacy bbox layout path."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args[0] if args else kwargs.get("args")

    def put_objects(self) -> None:
        raise NotImplementedError(
            "BboxArrangerDiffLow depends on removed legacy bbox_put_utils code. "
            "Use pat3d.providers.legacy_layout_bridge or the current layout providers instead."
        )
