from __future__ import annotations

from pat3d.providers._sam3d_image_to_3d_impl import SAM3DImageTo3DProvider

# Backward-compatible import path for older code/config/tests.
SAM3DTextTo3DProvider = SAM3DImageTo3DProvider
