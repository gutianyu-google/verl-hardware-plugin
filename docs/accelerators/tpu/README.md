# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin`.

The plugin registers the TPU platform (`PlatformTPU`: device metadata, Ray resource configuration,
and the PJRT worker environment), the TorchTitan TPU training engine (`TorchTitanTPUEngineWithLMHead`
under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`), a vLLM rollout for TPU, and two checkpoint
engines that send the trainer's weights to it (`ray-plasma-store` and `tpu-sync`).

## Directory Structure

```text
verl_hardware_plugin/accelerators/tpu/
├── platform_tpu.py                  # TPU platform settings
├── engines/
│   ├── ray_weight_registry.py        # Ray actor holding the `ray-plasma-store` weight ObjectRefs
│   ├── torchtitan_tpu.py             # TorchTitan TPU training engine (FSDP2 / SPMD)
│   ├── tpu_checkpoint_engine.py      # `ray-plasma-store` weight sync: through the Ray object store
│   ├── tpu_sync_checkpoint_engine.py # `tpu-sync` weight sync: peer to peer to the vLLM rollout (Raiden)
│   ├── tpu_sync_parity.py            # Optional `tpu-sync` parity check (trainer vs rollout weight norms)
│   ├── tpu_sync_rendezvous.py        # Per-job actor: Raiden controller address, tensor shapes, parity stats
│   ├── tpu_utils.py                  # Sequence bucketing and TorchTitan TPU config/input helpers
│   ├── weight_export.py              # Exports the trainer's weights as local FSDP shards or full tensors
│   └── weight_sync_backends.py       # Weight sync backend names and their deprecated aliases
└── rollout/
    ├── tpu_raiden.py                 # vLLM worker extension that receives Raiden weight syncs
    ├── tpu_vllm.py                   # vLLM rollout replica and server for TPU
    └── tpu_vllm_patches.py           # vLLM / vllm-torchtpu patches for multi-host TPU
```

```text
docs/accelerators/tpu/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
├── quick_start.md                    # Selecting and verifying the platform
├── rollout.md                        # Running the vLLM rollout on TPU
└── weight_sync.md                    # Choosing and configuring the weight sync (ray-plasma-store / tpu-sync)
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — select the TPU platform and verify it resolves
- [vLLM Rollout](./rollout.md) — run GRPO with the vLLM rollout on TPU
- [Weight Sync](./weight_sync.md) — send the trainer's weights to the rollout peer to peer with `tpu-sync`
- [GRPO example](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo) — end-to-end GRPO recipes in verl (`Dockerfile`, KubeRay `RayCluster`, Qwen3-0.6B / Qwen3-4B scripts)

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
| Communication backend | `tpu_dist` (registered by `torch_tpu`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` (see note below) |
| Ray resource name | `TPU` |
| IPC support | No |
| Colocated worker groups | Not supported — a chip belongs to a single process |

### Why the visibility env var is `CUDA_VISIBLE_DEVICES`

This is intentional. verl assigns `os.environ[<this key>]` when launching vLLM servers. Pointing it
at `TPU_VISIBLE_CHIPS` would overwrite the per-worker chip index that the platform sets through
`get_worker_env_vars()` and reads back through `ray_local_rank_override()`, breaking rank mapping.
The two variables serve different purposes and must not be merged.

## Chip Support

| Generation | HBM per device | Topologies with a built-in mapping |
|------------|----------------|------------------------------------|
| v6e | 32 GB per chip | `v6e-4`, `v6e-8`, `v6e-32`, and by chip count 1 to 256 |
| TPU 7x (`tpu7x`) | 96 GB per TensorCore (192 GB per dual-core chip) | by device count: 8 (one `2x2x1` host), 16, 32, 64, 128 (`4x4x4`) |

On TPU 7x a chip has two TensorCores that `torch_tpu` drives as two devices, one process each: a host is
8 devices, a slice is a 4-D mesh `x,y,z,cores`, and libtpu lays the mesh out by process (`CLOUD_TPU_TASK_ID`
is the process's global rank, `TPU_VISIBLE_DEVICES` the process's core). A pool that spans its GKE slice
takes the mesh from the slice shape the TPU webhook puts in the pod (`TPU_HOST_BOUNDS` times
`TPU_CHIPS_PER_HOST_BOUNDS`, plus the core axis, as torchtitan's launcher computes it); a pool that uses
part of a slice takes the whole-host shape for its device count from `TPU_V7X_TOPOLOGY_MAP` in
`platform_tpu.py`. The generation comes from the `ray.io/tpu-pod-type` node label of the pool's hosts or
from `TPU_ACCELERATOR_TYPE`; set `TORCH_TPU_TOPOLOGY='x,y,z,cores'` to override. Rollout engines on TPU 7x
use one worker per TensorCore, e.g. a TP=8 engine per host; on every generation an engine's mesh is its
own worker count. When the generation cannot be determined, the platform falls back to the v6e HBM figure.

## Related Documentation

- [verl plugin system](../../development.md)
