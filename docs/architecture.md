# Architecture Overview

The repository is organized into data, modeling, evaluation, and deployment layers.

- `data`: dataset abstraction and signer-aware splitting
- `landmarks`: extraction and preprocessing shims
- `features`: normalization and sequence preparation
- `models`: baseline, recurrent, and transformer models
- `training`: training loops and checkpoints
- `api`: FastAPI service
- `scripts`: operational entry points

The design keeps research logic distinct from deployment concerns and permits phased implementation.
