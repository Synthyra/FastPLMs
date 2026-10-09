"""Process-group setup and collectives for multi-GPU training.

Every helper is a passthrough when the run is a single process, so a trainer calls the same code
path either way.
"""

from __future__ import annotations

import os
import sys
import torch
import torch.distributed as dist

from dataclasses import dataclass
from typing import Any
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedState:
    """Who this process is among the ranks of a run, and which device it owns."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_distributed: bool = False
    is_main_process: bool = True
    device: torch.device | None = None
    owns_process_group: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cpu")


def init_distributed(backend: str = "") -> DistributedState:
    """Read the environment `torchrun` sets and, when it asks for several ranks on CUDA, start the process group.

    An empty `backend` is `gloo` on Windows and `nccl` elsewhere. A group this call started is
    destroyed by `cleanup_distributed`; one that already existed is left alone.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    launched_with_torchrun = local_rank != -1
    should_distribute = launched_with_torchrun and world_size > 1 and torch.cuda.is_available()

    owns_process_group = False
    if should_distribute and not dist.is_initialized():
        if not backend:
            backend = "gloo" if sys.platform == "win32" else "nccl"
        init_kwargs: dict[str, Any] = {"backend": backend, "init_method": "env://"}
        if backend == "nccl" and local_rank >= 0:
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = local_rank
        dist.init_process_group(**init_kwargs)
        owns_process_group = True

    is_distributed = dist.is_initialized() and world_size > 1

    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    state = DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        is_main_process=(rank == 0),
        device=device,
        owns_process_group=owns_process_group,
    )
    print(
        "[pipeline] init_distributed() completed: "
        f"is_distributed={state.is_distributed}, world_size={state.world_size}, "
        f"rank={state.rank}, local_rank={state.local_rank}, device={state.device}"
    )
    return state


def cleanup_distributed(state: DistributedState) -> None:
    """Destroy the process group `init_distributed` started, after every rank arrives."""
    if state.owns_process_group and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(state: DistributedState) -> None:
    """Wait for every rank; nothing to wait for in a single process."""
    if state.is_distributed:
        dist.barrier()


def wrap_model_ddp(model: nn.Module, local_rank: int) -> nn.Module:
    """`model` under DistributedDataParallel on the device of `local_rank`."""
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """The module under any DistributedDataParallel and `torch.compile` wrappers."""
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def reduce_scalar(value: float, device: torch.device, is_distributed: bool, op: str = "avg") -> float:
    """`value` averaged (`avg`) or summed (`sum`) over the ranks."""
    if not is_distributed:
        return value
    tensor = torch.tensor([value], device=device)  # (1,)
    if op == "avg":
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unknown reduction op '{op}', expected 'avg' or 'sum'")
    return tensor.item()


def broadcast_value(value: Any, device: torch.device, is_distributed: bool, src: int = 0) -> Any:
    """`value` as rank `src` holds it, for an int or a float; any other type passes through unchanged."""
    if not is_distributed:
        return value
    if isinstance(value, (int, float)):
        tensor = torch.tensor([value], device=device)  # (1,)
        dist.broadcast(tensor, src=src)
        return type(value)(tensor.item())
    return value


def gather_objects(local_obj: Any, world_size: int, is_distributed: bool) -> list[Any]:
    """One entry per rank, in rank order: each rank's `local_obj`."""
    if not is_distributed:
        return [local_obj]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_obj)
    return gathered


def broadcast_object(value: Any, is_distributed: bool, src: int = 0) -> Any:
    """Any picklable `value` as rank `src` holds it."""
    if not is_distributed:
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_tensors(
    *tensors: Tensor,
    device: torch.device,
    world_size: int,
    is_distributed: bool,
) -> tuple[Tensor, ...]:
    """Gather variable-length tensors from all ranks, flattened, on the CPU.

    Ranks with different sample counts are padded to the longest, and the padding is stripped. Every
    input tensor must have the same first-dimension length.
    """
    # tensors: (n, ...) for each of the k tensors, the same n in all
    if not is_distributed or len(tensors) == 0:
        return tensors  # (n, ...) each

    num_tensors = len(tensors)
    local_size = tensors[0].shape[0]

    stacked = torch.stack([tensor.flatten().to(device) for tensor in tensors], dim=0)  # (k, n_local)

    size_tensor = torch.tensor([local_size], device=device, dtype=torch.long)  # (1,)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]  # world_size * (1,)
    dist.all_gather(size_list, size_tensor)

    max_size = max(size.item() for size in size_list)

    if local_size < max_size:
        padding = torch.zeros(num_tensors, max_size - local_size, device=device, dtype=stacked.dtype)  # (k, n_max - n_local)
        stacked = torch.cat([stacked, padding], dim=1)  # (k, n_max)

    gathered_list = [torch.zeros_like(stacked) for _ in range(world_size)]  # world_size * (k, n_max)
    dist.all_gather(gathered_list, stacked)

    results = []
    for tensor_index in range(num_tensors):
        parts = []
        for gathered, size in zip(gathered_list, size_list, strict=True):
            actual = int(size.item())
            parts.append(gathered[tensor_index, :actual])  # (n_rank,)
        results.append(torch.cat(parts).cpu())  # (sum_rank n_rank,)

    return tuple(results)  # (sum_rank n_rank,) each
