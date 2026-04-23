# PAT3D scripts

Scripts are grouped by their runtime responsibility:

- `dashboard/`: dashboard server helpers and runtime/metrics extraction entrypoints.
- `environment/`: local environment bootstrap, prerequisite checks, and auxiliary environment setup.
- `object_generation/`: subprocess runners for depth and SAM 3D object asset generation.
- `physics/`: DiffSim subprocess runners and scene export utilities.
- `segmentation/`: subprocess runners for SAM 3 segmentation.
