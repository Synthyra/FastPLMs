"""Differentiable rigid rotations and affine transforms for ESMFold2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
from torch.nn import functional as F

from .esmfold2_misc import fp32_autocast_context


def _index_tuple(index: Any) -> tuple[Any, ...]:
    if isinstance(index, int) or index is None:
        return (index,)
    return tuple(index)


def _sqrt_subgradient(values: torch.Tensor) -> torch.Tensor:
    """Square root with a zero subgradient for non-positive inputs."""

    result = torch.zeros_like(values)
    positive = values > 0
    result[positive] = torch.sqrt(values[positive])
    return result


def _quat_invert(quaternion: torch.Tensor) -> torch.Tensor:
    conjugate_sign = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * conjugate_sign


def _quat_mult(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for real-first quaternion tensors."""

    aw, ax, ay, az = torch.unbind(left, -1)
    bw, bx, by, bz = torch.unbind(right, -1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def _quat_rotation(
    quaternion: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Rotate points using normalized real-first quaternions."""

    aw, ax, ay, az = torch.unbind(quaternion, -1)
    bx, by, bz = torch.unbind(points, -1)
    product = torch.stack(
        (
            -ax * bx - ay * by - az * bz,
            aw * bx + ay * bz - az * by,
            aw * by - ax * bz + az * bx,
            aw * bz + ax * by - ay * bx,
        ),
        -1,
    )
    return _quat_mult(product, _quat_invert(quaternion))[..., 1:]


def _graham_schmidt(
    x_axis: torch.Tensor,
    xy_plane: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Construct a right-handed orthonormal frame from two directions."""

    with fp32_autocast_context(x_axis.device.type):
        e1 = xy_plane
        denominator = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
        x_axis = x_axis / denominator
        projection = (x_axis * e1).sum(dim=-1, keepdim=True)
        e1 = e1 - x_axis * projection
        denominator = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
        e1 = e1 / denominator
        e2 = torch.cross(x_axis, e1, dim=-1)
        return torch.stack([x_axis, e1, e2], dim=-1)


class Rotation:
    """Common interface for matrix-backed and quaternion-backed rotations."""

    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self: ...

    @classmethod
    def random(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self: ...

    def __getitem__(self, idx: Any) -> Self: ...

    @property
    def tensor(self) -> torch.Tensor: ...

    @property
    def shape(self) -> torch.Size: ...

    def as_matrix(self) -> RotationMatrix: ...

    def as_quat(self, normalize: bool = False) -> RotationQuat: ...

    def compose(self, other: Self) -> Self: ...

    def convert_compose(self, other: Self) -> Self: ...

    def apply(self, points: torch.Tensor) -> torch.Tensor: ...

    def invert(self) -> Self: ...

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad

    @classmethod
    def _from_tensor(cls, tensor: torch.Tensor) -> Self:
        return cls(tensor)  # type: ignore[call-arg]

    def to(self, **kwargs) -> Self:
        return self._from_tensor(self.tensor.to(**kwargs))

    def detach(self, *args, **kwargs) -> Self:
        return self._from_tensor(self.tensor.detach(**kwargs))

    def tensor_apply(self, func) -> Self:
        transformed = [func(component) for component in self.tensor.unbind(dim=-1)]
        return self._from_tensor(torch.stack(transformed, dim=-1))


class RotationQuat(Rotation):
    """A rotation represented by a real-first quaternion."""

    def __init__(self, quats: torch.Tensor, normalized: bool = False):
        assert quats.shape[-1] == 4
        self._normalized = normalized
        if normalized:
            quats = F.normalize(quats.to(torch.float32), dim=-1)
            self._quats = quats.where(quats[..., :1] >= 0, -quats)
        else:
            self._quats = quats.to(torch.float32)

    @property
    def tensor(self) -> torch.Tensor:
        return self._quats

    @property
    def shape(self) -> torch.Size:
        return self._quats.shape[:-1]

    @classmethod
    def identity(cls, shape, **tensor_kwargs) -> RotationQuat:
        quaternions = torch.ones((*shape, 4), **tensor_kwargs)
        selector = torch.tensor([1, 0, 0, 0], device=quaternions.device)
        return cls(quaternions * selector)

    @classmethod
    def random(cls, shape, **tensor_kwargs) -> RotationQuat:
        return cls(torch.randn((*shape, 4), **tensor_kwargs), normalized=True)

    def __getitem__(self, idx: Any) -> RotationQuat:
        indices = _index_tuple(idx)
        return RotationQuat(self._quats[(*indices, slice(None))])

    def normalized(self) -> RotationQuat:
        if self._normalized:
            return self
        return RotationQuat(self._quats, normalized=True)

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        return self

    def as_matrix(self) -> RotationMatrix:
        quaternion = self.normalized().tensor
        r, i, j, k = torch.unbind(quaternion, -1)
        scale = 2.0 / torch.linalg.norm(quaternion, dim=-1)
        elements = torch.stack(
            (
                1 - scale * (j * j + k * k),
                scale * (i * j - k * r),
                scale * (i * k + j * r),
                scale * (i * j + k * r),
                1 - scale * (i * i + k * k),
                scale * (j * k - i * r),
                scale * (i * k - j * r),
                scale * (j * k + i * r),
                1 - scale * (i * i + j * j),
            ),
            -1,
        )
        return RotationMatrix(elements.reshape((*quaternion.shape[:-1], 3, 3)))

    def compose(self, other: RotationQuat) -> RotationQuat:
        with fp32_autocast_context(self.device.type):
            return RotationQuat(_quat_mult(self._quats, other._quats))

    def convert_compose(self, other: Rotation) -> RotationQuat:
        return self.compose(other.as_quat())

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return _quat_rotation(self.normalized()._quats, points)

    def invert(self) -> RotationQuat:
        return RotationQuat(_quat_invert(self._quats))


class RotationMatrix(Rotation):
    """A rotation represented by a dense FP32 matrix."""

    def __init__(self, rots: torch.Tensor):
        if rots.shape[-1] == 9:
            rots = rots.unflatten(-1, (3, 3))
        assert rots.shape[-2:] == (3, 3)
        self._rots = rots.to(torch.float32)

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    @property
    def shape(self) -> torch.Size:
        return self._rots.shape[:-2]

    @classmethod
    def identity(cls, shape, **tensor_kwargs) -> RotationMatrix:
        matrix = torch.eye(3, **tensor_kwargs)
        matrix = matrix.view(*(1 for _ in shape), 3, 3)
        return cls(matrix.expand(*shape, -1, -1))

    @classmethod
    def random(cls, shape, **tensor_kwargs) -> RotationMatrix:
        return RotationQuat.random(shape, **tensor_kwargs).as_matrix()

    @staticmethod
    def from_graham_schmidt(
        x_axis: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-12,
    ) -> RotationMatrix:
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))

    def __getitem__(self, idx: Any) -> RotationMatrix:
        indices = _index_tuple(idx)
        return RotationMatrix(self._rots[(*indices, slice(None), slice(None))])

    def as_matrix(self) -> RotationMatrix:
        return self

    def to_3x3(self) -> torch.Tensor:
        return self._rots

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            self._rots.flatten(-2),
            dim=-1,
        )
        q_abs = _sqrt_subgradient(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            )
        )
        products = torch.stack(
            (
                q_abs[..., 0] ** 2,
                m21 - m12,
                m02 - m20,
                m10 - m01,
                m21 - m12,
                q_abs[..., 1] ** 2,
                m10 + m01,
                m02 + m20,
                m02 - m20,
                m10 + m01,
                q_abs[..., 2] ** 2,
                m12 + m21,
                m10 - m01,
                m20 + m02,
                m21 + m12,
                q_abs[..., 3] ** 2,
            ),
            dim=-1,
        ).unflatten(-1, (4, 4))
        floor = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        candidates = products / (2.0 * q_abs[..., None].max(floor))
        best = torch.zeros_like(q_abs, dtype=torch.bool)
        best.scatter_(-1, q_abs.argmax(dim=-1, keepdim=True), True)
        quaternion = candidates[best, :].reshape(q_abs.shape)
        return RotationQuat(quaternion)

    def compose(self, other: RotationMatrix) -> RotationMatrix:
        with fp32_autocast_context(self.device.type):
            return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation) -> RotationMatrix:
        return self.compose(other.as_matrix())

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            if self._rots.shape[-3] == 1:
                return points @ self._rots.transpose(-1, -2).squeeze(-3)
            return torch.einsum("...ij,...j", self._rots, points)

    def invert(self) -> RotationMatrix:
        return RotationMatrix(self._rots.transpose(-1, -2))


@dataclass(frozen=True)
class Affine3D:
    """A rigid transform with translation and rotation components."""

    trans: torch.Tensor
    rot: Rotation

    def __post_init__(self) -> None:
        assert self.trans.shape[:-1] == self.rot.shape

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    @property
    def requires_grad(self) -> bool:
        return self.trans.requires_grad

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat((self.rot.tensor, self.trans), dim=-1)

    @staticmethod
    def identity(
        shape_or_affine: tuple[int, ...] | Affine3D,
        rotation_type: type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ) -> Affine3D:
        if isinstance(shape_or_affine, Affine3D):
            kwargs = {
                "dtype": shape_or_affine.dtype,
                "device": shape_or_affine.device,
            }
            kwargs.update(tensor_kwargs)
            shape = shape_or_affine.shape
            rotation_type = type(shape_or_affine.rot)
        else:
            kwargs = tensor_kwargs
            shape = shape_or_affine
        return Affine3D(
            torch.zeros((*shape, 3), **kwargs),
            rotation_type.identity(shape, **kwargs),
        )

    @staticmethod
    def random(
        shape: tuple[int, ...],
        std: float = 1,
        rotation_type: type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ) -> Affine3D:
        translation = torch.randn((*shape, 3), **tensor_kwargs).mul(std)
        rotation = rotation_type.random(shape, **tensor_kwargs)
        return Affine3D(trans=translation, rot=rotation)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> Affine3D:
        width = tensor.shape[-1]
        if width == 4:
            translation = tensor[..., :3, 3]
            rotation: Rotation = RotationMatrix(tensor[..., :3, :3])
        elif width == 6:
            translation = tensor[..., -3:]
            rotation = RotationQuat(F.pad(tensor[..., :3], (1, 0), value=1))
        elif width == 7:
            translation = tensor[..., -3:]
            rotation = RotationQuat(tensor[..., :4])
        elif width == 12:
            translation = tensor[..., -3:]
            rotation = RotationMatrix(tensor[..., :-3].unflatten(-1, (3, 3)))
        else:
            raise RuntimeError(
                f"Cannot detect rotation fromat from {tensor.shape[-1] - 3}-d flat vector"
            )
        return Affine3D(translation, rotation)

    @staticmethod
    def from_tensor_pair(
        translation: torch.Tensor,
        rotation: torch.Tensor,
    ) -> Affine3D:
        return Affine3D(translation, RotationMatrix(rotation))

    @staticmethod
    def from_graham_schmidt(
        neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-10,
    ) -> Affine3D:
        x_axis = origin - neg_x_axis
        plane_direction = xy_plane - origin
        rotation = RotationMatrix.from_graham_schmidt(
            x_axis,
            plane_direction,
            eps,
        )
        return Affine3D(trans=origin, rot=rotation)

    @staticmethod
    def cat(affines: list[Affine3D], dim: int = 0) -> Affine3D:
        if dim < 0:
            dim = len(affines[0].shape) + dim
        return Affine3D.from_tensor(torch.cat([affine.tensor for affine in affines], dim=dim))

    def __getitem__(self, idx: Any) -> Affine3D:
        indices = _index_tuple(idx)
        translation = self.trans[(*indices, slice(None))]
        return Affine3D(trans=translation, rot=self.rot[idx])

    def to(self, **kwargs) -> Affine3D:
        return Affine3D(self.trans.to(**kwargs), self.rot.to(**kwargs))

    def detach(self, *args, **kwargs) -> Affine3D:
        return Affine3D(
            self.trans.detach(**kwargs),
            self.rot.detach(**kwargs),
        )

    def tensor_apply(self, func) -> Affine3D:
        components = [func(value) for value in self.tensor.unbind(dim=-1)]
        return Affine3D.from_tensor(torch.stack(components, dim=-1))

    def as_matrix(self) -> Affine3D:
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def as_quat(self, normalize: bool = False) -> Affine3D:
        return Affine3D(
            trans=self.trans,
            rot=self.rot.as_quat(normalize),
        )

    def compose(
        self,
        other: Affine3D,
        autoconvert: bool = False,
    ) -> Affine3D:
        compose_rotation = self.rot.convert_compose if autoconvert else self.rot.compose
        rotation = compose_rotation(other.rot)
        translation = self.rot.apply(other.trans) + self.trans
        return Affine3D(trans=translation, rot=rotation)

    def compose_rotation(
        self,
        other: Rotation,
        autoconvert: bool = False,
    ) -> Affine3D:
        compose = self.rot.convert_compose if autoconvert else self.rot.compose
        return Affine3D(trans=self.trans, rot=compose(other))

    def scale(self, value: torch.Tensor | float) -> Affine3D:
        return Affine3D(self.trans * value, self.rot)

    def mask(self, mask: torch.Tensor, with_zero: bool = False) -> Affine3D:
        if with_zero:
            masked = torch.zeros_like(self.tensor).where(
                mask[..., None],
                self.tensor,
            )
            return Affine3D.from_tensor(masked)
        identity = self.identity(
            self.shape,
            rotation_type=type(self.rot),
            device=self.device,
            dtype=self.dtype,
        ).tensor
        return Affine3D.from_tensor(identity.where(mask[..., None], self.tensor))

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(points) + self.trans

    def invert(self) -> Affine3D:
        rotation = self.rot.invert()
        return Affine3D(trans=-rotation.apply(self.trans), rot=rotation)


def build_affine3d_from_coordinates(
    coords: torch.Tensor,
) -> tuple[Affine3D, torch.Tensor]:
    """Build residue frames from X with shape (b, l, 3, 3)."""

    maximum_distance = 1e6
    coord_mask = torch.all(
        torch.all(
            torch.isfinite(coords) & (coords < maximum_distance),
            dim=-1,
        ),
        dim=-1,
    )

    def backbone_affine(positions: torch.Tensor) -> Affine3D:
        n, ca, c = positions.unbind(dim=-2)
        return Affine3D.from_graham_schmidt(c, ca, n)

    coords = coords.clone().float()
    coords[~coord_mask] = 0
    average = coords.masked_fill(~coord_mask[..., None, None], 0).sum(1) / (
        coord_mask.sum(-1)[..., None, None] + 1e-8
    )
    average_affine = backbone_affine(average.float()).as_matrix()

    b, length, _, _ = coords.shape
    assert isinstance(b, int)
    assert isinstance(length, int)
    rotation = average_affine.rot.tensor[..., None, :].expand(b, length, 9)
    translation = average_affine.trans[..., None, :].expand(b, length, 3)
    identity = RotationMatrix.identity(
        (b, length),
        dtype=torch.float32,
        device=coords.device,
        requires_grad=False,
    )
    rotation = rotation.where(
        coord_mask.any(-1)[..., None, None],
        identity.tensor,
    )
    missing_frame = Affine3D(translation, RotationMatrix(rotation))

    residue_frame = backbone_affine(coords.float())
    residue_frame = Affine3D.from_tensor(
        residue_frame.tensor.where(
            coord_mask[..., None],
            missing_frame.tensor,
        )
    )
    return residue_frame, coord_mask
