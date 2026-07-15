"""Small geometry, augmentation, activation, and EMA utilities for Boltz2.

The quaternion behavior follows PyTorch3D's public rotation convention. License
and attribution records are retained in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LinearNoBias = partial(nn.Linear, bias=False)


def exists(value: object) -> bool:
    """Return whether a value is not ``None``."""

    return value is not None


def default(value: Any, fallback: Any) -> Any:
    """Return ``value`` unless it is ``None``."""

    return fallback if value is None else value


def log(tensor: Tensor, eps: float = 1e-20) -> Tensor:
    """Compute a finite logarithm by applying a scalar lower bound."""

    return torch.log(tensor.clamp(min=eps))


class SwiGLU(nn.Module):
    """Split X in half and apply a SiLU-gated linear unit."""

    def forward(self, x: Tensor) -> Tensor:
        values, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * values


def _masked_center(coordinates: Tensor, mask: Tensor) -> Tensor:
    weights = mask[:, :, None]
    return (coordinates * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1,
        keepdim=True,
    )


def center(atom_coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Center coordinate tensor X with shape ``(b, n_atoms, 3)``."""

    return atom_coords - _masked_center(atom_coords, atom_mask)


def _copysign(magnitudes: Tensor, signs: Tensor) -> Tensor:
    """Apply the elementwise sign of S to magnitude tensor M."""

    return torch.where((magnitudes < 0) != (signs < 0), -magnitudes, magnitudes)


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert real-first quaternion tensor Q from ``(..., 4)`` to ``(..., 3, 3)``."""

    real, i_axis, j_axis, k_axis = torch.unbind(quaternions, dim=-1)
    scale = 2.0 / (quaternions * quaternions).sum(dim=-1)
    entries = (
        1 - scale * (j_axis * j_axis + k_axis * k_axis),
        scale * (i_axis * j_axis - k_axis * real),
        scale * (i_axis * k_axis + j_axis * real),
        scale * (i_axis * j_axis + k_axis * real),
        1 - scale * (i_axis * i_axis + k_axis * k_axis),
        scale * (j_axis * k_axis - i_axis * real),
        scale * (i_axis * k_axis - j_axis * real),
        scale * (j_axis * k_axis + i_axis * real),
        1 - scale * (i_axis * i_axis + j_axis * j_axis),
    )
    return torch.stack(entries, dim=-1).reshape((*quaternions.shape[:-1], 3, 3))


def random_quaternions(
    n: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw ``n`` uniformly distributed unit quaternions with nonnegative real part."""

    resolved_device = torch.device(device) if isinstance(device, str) else device
    samples = torch.randn((n, 4), dtype=dtype, device=resolved_device)
    squared_norm = (samples * samples).sum(dim=1)
    signed_norm = _copysign(torch.sqrt(squared_norm), samples[:, 0])
    return samples / signed_norm[:, None]


def random_rotations(
    n: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw rotation tensor R with shape ``(n, 3, 3)``."""

    return quaternion_to_matrix(random_quaternions(n, dtype=dtype, device=device))


def compute_random_augmentation(
    multiplicity: int,
    s_trans: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Draw independent rotations R and translations T for a replicated batch."""

    rotations = random_rotations(multiplicity, dtype=dtype, device=device)
    translations = (
        torch.randn(
            (multiplicity, 1, 3),
            dtype=dtype,
            device=device,
        )
        * s_trans
    )
    return rotations, translations


def randomly_rotate(
    coords: Tensor,
    return_second_coords: bool = False,
    second_coords: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor | None]:
    """Rotate X and optionally Y using the same sampled rotation tensor R."""

    rotations = random_rotations(len(coords), coords.dtype, coords.device)
    rotated = torch.einsum("bmd,bds->bms", coords, rotations)
    if not return_second_coords:
        return rotated
    rotated_second = (
        None if second_coords is None else torch.einsum("bmd,bds->bms", second_coords, rotations)
    )
    return rotated, rotated_second


def center_random_augmentation(
    atom_coords: Tensor,
    atom_mask: Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor | None]:
    """Center and rigidly augment coordinate tensors X and optional Y."""

    primary = atom_coords
    secondary = second_coords
    if centering:
        centroid = _masked_center(primary, atom_mask)
        primary = primary - centroid
        if secondary is not None:
            secondary = secondary - centroid

    if augmentation:
        primary, secondary = randomly_rotate(
            primary,
            return_second_coords=True,
            second_coords=secondary,
        )
        translation = torch.randn_like(primary[:, 0:1, :]) * s_trans
        primary = primary + translation
        if secondary is not None:
            secondary = secondary + translation

    return (primary, secondary) if return_second_coords else primary


class ExponentialMovingAverage:
    """Maintain detached exponential moving averages of trainable parameters."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float,
        use_num_updates: bool = True,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must lie in [0, 1]")
        self.decay = decay
        self.num_updates: int | None = 0 if use_num_updates else None
        self.shadow_params = [
            parameter.clone().detach() for parameter in parameters if parameter.requires_grad
        ]
        self.collected_params: list[Tensor] = []

    @staticmethod
    def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
        return [parameter for parameter in parameters if parameter.requires_grad]

    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        trainable = self._trainable(parameters)
        if len(trainable) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed")
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        update_weight = 1.0 - decay
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow_params, trainable, strict=True):
                shadow.sub_(update_weight * (shadow - parameter))

    def compatible(self, parameters: Sequence[Tensor]) -> bool:
        """Return whether parameter count and tensor shapes match the EMA state."""

        return len(self.shadow_params) == len(parameters) and all(
            shadow.shape == parameter.shape
            for shadow, parameter in zip(self.shadow_params, parameters, strict=True)
        )

    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        trainable = self._trainable(parameters)
        if len(trainable) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed")
        for shadow, parameter in zip(self.shadow_params, trainable, strict=True):
            parameter.data.copy_(shadow.data)

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        self.collected_params = [parameter.clone() for parameter in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        current = list(parameters)
        if len(current) != len(self.collected_params):
            raise ValueError("stored EMA parameter count changed")
        for collected, parameter in zip(self.collected_params, current, strict=True):
            parameter.data.copy_(collected.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        device: torch.device | str,
    ) -> None:
        self.decay = float(state_dict["decay"])
        updates = state_dict["num_updates"]
        self.num_updates = None if updates is None else int(updates)
        self.shadow_params = [tensor.to(device) for tensor in state_dict["shadow_params"]]

    def to(self, device: torch.device | str) -> None:
        self.shadow_params = [tensor.to(device) for tensor in self.shadow_params]
