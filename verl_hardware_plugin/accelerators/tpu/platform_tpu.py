# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Google TPU platform implementation.

Registers the ``tpu`` platform and provides the device abstraction, the Ray resource
options, and the worker-environment plumbing a TPU process needs to join a slice. The
training engine, the generation engine and the weight-transfer path live elsewhere.

``torch_tpu`` mirrors part of the NVIDIA device API surface, so ``TPUDeviceModuleProxy``
adapts it to the memory-management calls verl expects. The platform class itself derives
from ``PlatformBase``: every device query, Ray resource option and environment variable is
defined here rather than inherited from another vendor's platform.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import ray
import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry
from verl_hardware_plugin.accelerators.tpu import device_api as tpu_device_api
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import extend_torchtitan_engine_config
from verl_hardware_plugin.accelerators.tpu.patches import reduce_avg_allreduce_patch

extend_torchtitan_engine_config()

logger = logging.getLogger(__name__)


def _ensure_torch_tpu() -> bool:
    """Try to import torch_tpu, which registers the ``tpu_dist`` distributed backend.

    Returns True if the TPU runtime is usable after the attempt.
    """
    if hasattr(torch, "tpu"):
        return True
    try:
        import torch_tpu  # noqa: F401

        return hasattr(torch, "tpu")
    except Exception as e:
        logger.debug("The current machine has no torch.tpu, because: %s", e)
    return False


_ensure_torch_tpu()  # Attempt at module load time so availability checks are faster later

# Base port for the TPU distributed slice builder mesh. Each local rank takes ``base + local_rank``.
TPU_PROCESS_BASE_PORT = 8471

# HBM capacity in bytes of one addressable device (a chip, or a TensorCore on TPU 7x)
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
# A TPU 7x chip has 192 GB and two TensorCores; a process drives one core, so this is what a process sees.
HBM_BYTES_TPU_V7X = 96 * 1024 * 1024 * 1024  # 96 GB per TensorCore device

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "v7x": HBM_BYTES_TPU_V7X,
    "tpu7x": HBM_BYTES_TPU_V7X,
}

# Fallback HBM capacity when the chip generation cannot be determined.
HBM_BYTES_TPU_DEFAULT = HBM_BYTES_TPU_V6E

# TPU slice topology (an ``x,y,z`` chip mesh) keyed by the number of chips in the slice. The vLLM
# rollout (``rollout/tpu_vllm_patches.py``) resolves its mesh here too, so the trainer and the
# rollout agree on the physical layout of a given slice size.
DEFAULT_TPU_TOPOLOGY_MAP = {
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}

# Pod types are consulted first: a pod type names the physical slice exactly, whereas the chip
# count is only the number of chips this job was given and can be a subset of the slice.
TPU_POD_TYPE_TOPOLOGY_MAP = {
    "v6e-4": "2,2,1",
    "v6e-8": "2,4,1",
    "v6e-16": "4,4,1",
    "v6e-32": "4,8,1",
}

# Slices at or below this many chips live on a single host, so the chips are addressed within
# the host bounds rather than across them.
TPU_SINGLE_HOST_MAX_CHIPS = 4

# TPU 7x (Ironwood): a chip has two TensorCores that torch_tpu drives as two devices, one process each,
# so a slice is a 4-D mesh ``x,y,z,cores`` keyed by its number of DEVICES (what ``world_size`` counts).
# A host holds 4 chips (``2x2x1``) = 8 devices; the multi-host entries follow the GKE slice shapes
# (``2x2x2`` = 2 hosts, ``2x2x4`` = 4, ``2x4x4`` = 8, ``4x4x4`` = 16). Only whole hosts (and a single
# device) are listed: a standalone process can open only a chip's first core, so sub-host meshes have
# no validated layout.
TPU_V7X_DEVICES_PER_HOST = 8
TPU_V7X_TOPOLOGY_MAP = {
    1: "1,1,1,1",
    8: "2,2,1,2",
    16: "2,2,2,2",
    32: "2,2,4,2",
    64: "2,4,4,2",
    128: "4,4,4,2",
}
# One process per TensorCore: every process is a ``1,1,1,1`` cell of the mesh.
TPU_V7X_PROCESS_BOUNDS = "1,1,1,1"


def tpu_accelerator_name(pod_type: str = "") -> str:
    """Lower-cased TPU generation or pod type: ``pod_type`` (what Ray reports, see ``ray_tpu_accelerator_name``)
    or, failing that, the ``TPU_ACCELERATOR_TYPE`` / ``ACCELERATOR_TYPE`` / ``TPU_TYPE`` environment variables."""
    return (
        pod_type
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.environ.get("ACCELERATOR_TYPE")
        or os.environ.get("TPU_TYPE")
        or ""
    ).lower()


def is_tpu_v7x(pod_type: str = "") -> bool:
    """Whether the slice is TPU 7x (see ``tpu_accelerator_name``).

    The generation appears as ``tpu7x`` (GKE's accelerator label and ``TPU_ACCELERATOR_TYPE``), ``tpu7x-<n>``
    (Ray's pod type) or ``v7x``; no other generation's name contains ``7x``.
    """
    return "7x" in tpu_accelerator_name(pod_type)


def tpu_v7x_process_env(topology: str, rank: int, local_rank: int) -> dict[str, str]:
    """The per-process variables of a TPU 7x mesh with one process per TensorCore.

    libtpu lays the mesh out by process: the process bounds are the whole topology, every process holds one
    device, and ``CLOUD_TPU_TASK_ID`` is the process's global rank (on v6e it is the index of its host).
    ``TPU_VISIBLE_DEVICES`` names the device the way ``TPU_VISIBLE_CHIPS`` names the chip. The chip stays the
    one Ray assigned: pinning ``TPU_VISIBLE_CHIPS`` to the rank with ``RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS``
    makes Ray 2.56's core worker abort, and the Ray assignment formed the mesh correctly in every run.
    """
    return {
        "TPU_PROCESS_BOUNDS": topology,
        "TPU_CHIPS_PER_PROCESS_BOUNDS": TPU_V7X_PROCESS_BOUNDS,
        "CLOUD_TPU_TASK_ID": str(rank),
        "TPU_VISIBLE_DEVICES": str(local_rank),
    }


def ray_tpu_accelerator_name(node_ips: Optional[list[str]] = None) -> str:
    """The pod type or accelerator type Ray reports for a TPU node (``""`` without Ray or TPU nodes).

    Reads ``ray.io/tpu-pod-type``, else ``ray.io/accelerator-type``, of one alive TPU node: a node whose
    address is in ``node_ips`` (the hosts of the pool being configured) when any is, so a cluster that
    mixes generations is read per pool, otherwise the first TPU node.
    """
    try:
        if not ray.is_initialized():
            return ""
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
    except Exception as e:
        logger.debug("Could not read the TPU node labels from Ray: %s", e)
        return ""
    if node_ips:
        tpu_nodes = [node for node in tpu_nodes if node.get("NodeManagerAddress") in node_ips] or tpu_nodes
    if not tpu_nodes:
        return ""
    labels = tpu_nodes[0].get("Labels", {})
    return labels.get("ray.io/tpu-pod-type") or labels.get("ray.io/accelerator-type") or ""


def tpu_mesh_env(total_chips: int, num_nodes: int, rank: int, local_rank: int, pod_type: str = "") -> dict[str, str]:
    """The complete mesh environment of one process of a slice.

    The topology and bounds of ``resolve_tpu_topology_bounds`` and, on TPU 7x, the per-process variables of
    ``tpu_v7x_process_env``. Callers apply it after their generic defaults, so that on TPU 7x the task id is
    the process's global rank rather than its host's index.
    """
    topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
        total_chips=total_chips, num_nodes=num_nodes, pod_type=pod_type
    )
    env = {
        "TORCH_TPU_TOPOLOGY": topology,
        "TPU_HOST_BOUNDS": host_bounds,
        "TPU_CHIPS_PER_HOST_BOUNDS": chips_per_host_bounds,
        "CHIPS_PER_HOST": chips_per_host,
    }
    if is_tpu_v7x(pod_type):
        env.update(tpu_v7x_process_env(topology, rank=rank, local_rank=local_rank))
    return env


def resolve_tpu_topology_bounds(
    total_chips: int,
    num_nodes: int,
    pod_type: str = "",
) -> tuple[str, str, str, str]:
    """Resolve ``(topology, host_bounds, chips_per_host_bounds, chips_per_host)`` for a TPU slice.

    Resolution order for the topology is: the ``TORCH_TPU_TOPOLOGY`` escape hatch, then the pod
    type reported by Ray, then the chip count.

    On TPU 7x (``pod_type`` or the accelerator-type environment names it) ``total_chips`` counts
    TensorCore devices, the topology is the 4-D entry of ``TPU_V7X_TOPOLOGY_MAP`` and every process
    is its own cell of the mesh, whatever the slice size.

    Raises:
        ValueError: if the topology cannot be determined. Guessing here is worse than failing:
            a wrong mesh does not error, it silently trains on part of the slice.
    """
    inferred_chips_per_host = max(1, total_chips // max(1, num_nodes))
    chips_per_host = str(os.environ.get("VERL_TPU_CHIPS_PER_HOST", inferred_chips_per_host))

    if is_tpu_v7x(pod_type):
        topology = os.environ.get("TORCH_TPU_TOPOLOGY") or TPU_V7X_TOPOLOGY_MAP.get(total_chips)
        if not topology:
            raise ValueError(
                f"Cannot determine the TPU 7x topology for {total_chips} devices (pod_type={pod_type!r}). "
                "Set TORCH_TPU_TOPOLOGY='x,y,z,cores' to override. "
                f"Known device counts: {sorted(TPU_V7X_TOPOLOGY_MAP)}."
            )
        return topology, topology, TPU_V7X_PROCESS_BOUNDS, chips_per_host

    topology = (
        os.environ.get("TORCH_TPU_TOPOLOGY")
        or TPU_POD_TYPE_TOPOLOGY_MAP.get(pod_type.lower())
        or DEFAULT_TPU_TOPOLOGY_MAP.get(total_chips)
    )
    if not topology:
        raise ValueError(
            f"Cannot determine the TPU topology for total_chips={total_chips} "
            f"(pod_type={pod_type!r}). Set TORCH_TPU_TOPOLOGY='x,y,z' to override. "
            f"Known chip counts: {sorted(DEFAULT_TPU_TOPOLOGY_MAP)}."
        )

    if total_chips <= TPU_SINGLE_HOST_MAX_CHIPS:
        host_bounds = "1,1,1"
        chips_per_host_bounds = topology if num_nodes == 1 else "1,1,1"
    else:
        host_bounds = topology
        chips_per_host_bounds = "1,1,1"

    return topology, host_bounds, chips_per_host_bounds, chips_per_host


def get_tpu_chip_hbm_bytes() -> int:
    """Detect the TPU chip generation from Ray node labels or env vars and return its HBM capacity."""
    tpu_type = tpu_accelerator_name(ray_tpu_accelerator_name())

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate the PyTorch NVIDIA memory management APIs.

    Provides default fallback implementations for memory tracking methods
    (e.g. memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's
    codebase but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "memory_stats":
            return lambda *args, **kwargs: {}
        elif name == "get_device_name":
            return lambda *args, **kwargs: "Google TPU"
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=HBM_BYTES_TPU_DEFAULT):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        return tpu_device_api.is_available()

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        try:
            return tpu_device_api.current_device()
        except Exception as e:
            logger.warning(f"TPU current_device() failed: {e}")
            return 0

    def device_count(self) -> int:
        try:
            return tpu_device_api.device_count()
        except Exception as e:
            logger.warning(f"TPU device_count() failed: {e}")
            return 0

    def synchronize(self, device_index: Optional[int] = None) -> None:
        try:
            tpu_device_api.synchronize()
        except Exception as e:
            logger.warning(f"TPU synchronize() failed: {e}")

    # torch_tpu >= 0.2.0 makes these torch.tpu functions internal (they raise when called from
    # outside torch / torch_tpu), so they are served through torch.accelerator instead of being
    # forwarded to torch.tpu by __getattr__.
    def current_stream(self, device: Any = None):
        return tpu_device_api.current_stream(device)

    def set_stream(self, stream: Any) -> None:
        tpu_device_api.set_stream(stream)

    def device(self, device_index: Any):
        return tpu_device_api.device(device_index)

    def get_rng_state(self, device: Any = None) -> torch.Tensor:
        return tpu_device_api.get_rng_state(device)

    def initial_seed(self) -> int:
        return tpu_device_api.initial_seed()

    def manual_seed_all(self, seed: int) -> None:
        # torch_tpu does not always expose manual_seed_all; the global torch seed is the
        # right fallback because a TPU process owns exactly one chip.
        fn = getattr(self._original_module, "manual_seed_all", None)
        if fn is not None:
            fn(seed)
        else:
            torch.manual_seed(seed)

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


def configure_torchrun_tpu_env() -> None:
    """Populate PJRT multi-process environment variables when launched via ``torchrun``.

    In Ray single-controller runs (``sft_trainer_ray`` / ``main_ppo``), ``get_worker_env_vars``
    populates ``TORCH_TPU_SLICEBUILDER_ADDRESSES`` and the topology bounds on each worker actor.
    In ``torchrun`` multi-controller runs (``verl.trainer.sft_trainer``), ``PlatformTPU()`` is
    instantiated inside ``auto_set_device`` / ``initialize_global_process_group`` before
    ``torch.distributed.init_process_group(backend="cpu:gloo,tpu:tpu_dist")`` runs. Setting
    missing PJRT variables from ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` lets ``torchrun``
    work out of the box without a wrapper script.
    """
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        return
    try:
        if ray.is_initialized():
            return
    except Exception:
        pass

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("RANK", local_rank))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    except ValueError:
        return

    if world_size < 1 or local_world_size < 1:
        return

    num_nodes = max(1, world_size // local_world_size)
    raw_hosts = os.environ.get("TPU_WORKER_HOSTNAMES")
    if raw_hosts:
        hostnames = [h.strip() for h in raw_hosts.split(",") if h.strip()]
    elif num_nodes == 1:
        hostnames = [os.environ.get("MASTER_ADDR", "127.0.0.1")]
    else:
        return

    if len(hostnames) != num_nodes:
        return

    sb_addresses = [
        f"{hostnames[r // local_world_size]}:{TPU_PROCESS_BASE_PORT + (r % local_world_size)}"
        for r in range(world_size)
    ]

    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TORCH_TPU_SLICEBUILDER_ADDRESSES", ",".join(sb_addresses))
    os.environ.setdefault("TPU_PROCESS_ADDRESSES", ",".join(sb_addresses))
    os.environ.setdefault("TPU_PROCESS_PORT", str(TPU_PROCESS_BASE_PORT + local_rank))
    os.environ.setdefault("CLOUD_TPU_TASK_ID", str(rank // local_world_size))
    os.environ.setdefault("TPU_WORKER_HOSTNAMES", ",".join(hostnames))
    os.environ.setdefault("TPU_VISIBLE_CHIPS", str(local_rank))

    pod_type = (
        os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("ACCELERATOR_TYPE") or os.environ.get("TPU_TYPE") or ""
    )
    try:
        mesh_env = tpu_mesh_env(
            total_chips=world_size, num_nodes=num_nodes, rank=rank, local_rank=local_rank, pod_type=pod_type
        )
    except ValueError:
        return
    if is_tpu_v7x(pod_type):
        # The GKE TPU webhook pre-sets 3-D host bounds and a per-host task id on TPU 7x pods; the mesh is
        # laid out by process here, so those are replaced rather than kept.
        os.environ.update(mesh_env)
    else:
        for name, value in mesh_env.items():
            os.environ.setdefault(name, value)


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformBase):
    """Platform backend for Google TPUs.

    Answers device queries, builds the Ray resource options for TPU actors, and supplies
    the PJRT environment a worker needs. The parts of ``PlatformBase`` modelled on a CUDA
    device are answered explicitly here rather than inherited.
    """

    def __init__(self):
        super().__init__()
        reduce_avg_allreduce_patch.apply()
        configure_torchrun_tpu_env()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)
        if hasattr(torch, "tpu"):
            for attr in (
                "reset_peak_memory_stats",
                "empty_cache",
                "memory_stats",
                "get_device_name",
                "get_device_properties",
            ):
                if not hasattr(torch.tpu, attr):
                    setattr(torch.tpu, attr, getattr(self._device_module, attr))
        self._claimed_slices: set[str] = set()

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def empty_cache(self) -> None:
        # Delegate when torch.tpu provides it. A TPU runtime that manages HBM itself has no
        # CUDA-style caching allocator to drain, in which case there is nothing to do.
        empty = getattr(self.device_module, "empty_cache", None)
        if empty is not None:
            empty()

    def set_allocator_settings(self, settings: str) -> None:
        # TPU exposes no configurable caching allocator equivalent to CUDA's.
        pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # TPU does not use the CUDA compute capability model.
        return (None, None)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            return tpu_device_api.is_available()
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        # The env vars come first because a CPU-only Ray driver process has no chip attached
        # yet still belongs to a TPU job. Falling back to the runtime lets a plain TPU VM,
        # where nothing sets TPU_NAME, detect itself instead of defaulting to CUDA.
        return self.is_available()

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def visible_devices_envvar(self) -> str:
        # Deliberately CUDA_VISIBLE_DEVICES, not TPU_VISIBLE_CHIPS. verl assigns
        # os.environ[this key] when launching vLLM servers; pointing it at
        # TPU_VISIBLE_CHIPS would overwrite the chip index managed by Ray.
        return "CUDA_VISIBLE_DEVICES"

    def is_ipc_supported(self) -> bool:
        # TPU has no CUDA-style IPC handle. verl derives ``use_shm = not is_support_ipc()``,
        # so returning False routes weight transfer through shared memory instead.
        return False

    def cudart(self) -> Any:
        # There is no CUDA runtime on a TPU host; PlatformBase documents None for "not applicable".
        return None

    # No TPU profiler or NVTX equivalent. PlatformBase asks platforms without profiling
    # support to make these no-ops: yield immediately, and start/stop do nothing.
    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def ray_noset_envvars(self) -> list[str]:
        return []

    def supports_colocated_worker_groups(self) -> bool:
        # A TPU chip belongs to one process and cannot be shared between colocated WorkerGroups.
        return False

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Pin the TPU platform inside Ray workers."""
        return {
            "runtime_env": {
                "env_vars": {"VERL_PLATFORM": "tpu"},
            }
        }

    def auto_assign_accelerator_type(self, name_prefix: str, accelerator_type: Optional[str]) -> Optional[str]:
        """Pin a resource pool to its own TPU slice on multi-slice clusters.

        Called from ``verl.single_controller.ray.base.RayResourcePool.get_placement_groups``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        A KubeRay TPU cluster advertises one ``tpu-group-<n>`` custom resource per slice. Without
        an affinity label a pool can straddle two slices, which the TPU mesh cannot span. Each pool
        gets the first slice that no earlier pool claimed, so the trainer and rollout pools land on
        different slices; once every slice is claimed, further pools share the last one.
        """
        if accelerator_type is not None:
            return accelerator_type

        try:
            if ray.is_initialized():
                tpu_slices = set()
                for node in ray.nodes():
                    if node.get("Alive"):
                        for res in node.get("Resources", {}):
                            if res.startswith("tpu-group-"):
                                tpu_slices.add(res)
                slices = sorted(tpu_slices)
                for tpu_slice in slices:
                    if tpu_slice not in self._claimed_slices:
                        self._claimed_slices.add(tpu_slice)
                        return tpu_slice
                if slices:
                    return slices[-1]
        except Exception as e:
            logger.debug("Could not auto-assign a TPU slice for %r: %s", name_prefix, e)

        return accelerator_type

    def configure_placement_group_bundle(
        self,
        bundle: dict,
        use_gpu: bool,
        device_name: str,
        name_prefix: str,
        accelerator_type: Optional[str] = None,
    ) -> None:
        """Shape a placement-group bundle for GKE TPU.

        Called unconditionally from ``verl.single_controller.ray.base.RayResourcePool``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        The slice affinity is requested as a fractional amount so it acts as a label rather
        than a real reservation.
        """
        if use_gpu:
            bundle[device_name] = 1
        if accelerator_type is not None:
            bundle[accelerator_type] = 1e-4

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
    ) -> dict[str, str]:
        """Generate TPU-specific distributed environment variables for PJRT mesh initialization.

        ``pgs`` are the placement groups of the worker's resource pool. A pool is pinned to one
        slice (see ``auto_assign_accelerator_type``), so its bundles, in placement-group and
        bundle order, are exactly the processes of the PJRT mesh. This runs in the process that
        creates the workers (the trainer's TaskRunner), so nothing here may depend on the IP of
        the calling process -- it can be the head node or any TPU host.
        """
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []

        for pg in pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        base_port = TPU_PROCESS_BASE_PORT
        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": str(local_world_size),
        }

        # Derive the mesh from the pod type Ray reports for this pool's hosts, falling back to the chip count.
        tpu_type = ray_tpu_accelerator_name(node_ips=unique_hostnames)
        env_vars.update(
            tpu_mesh_env(
                total_chips=world_size,
                num_nodes=len(unique_hostnames),
                rank=rank,
                local_rank=local_rank,
                pod_type=tpu_type,
            )
        )

        return env_vars

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {}
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
        )
        env_vars.update(tpu_env)
        return env_vars
