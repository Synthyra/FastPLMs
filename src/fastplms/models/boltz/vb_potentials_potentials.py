"""Differentiable structure-steering potentials for Boltz2 diffusion.

Each potential follows the same mechanism: derive constraint indices and
bounds from features, evaluate a geometric variable, map that variable to an
energy, and optionally scatter its analytic derivative back to atom
coordinates.  This module is maintained independently from the pinned parity
oracle while preserving the public class names used by converted checkpoints.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from . import vb_const as const
from .vb_loss_diffusionv2 import weighted_rigid_align
from .vb_potentials_schedules import (
    ExponentialInterpolation,
    ParameterSchedule,
    PiecewiseStepFunction,
)


Parameter = ParameterSchedule | float | int | bool
ParameterMap = dict[str, Parameter]


@dataclass(frozen=True, slots=True)
class _PreparedPotential:
    coordinates: torch.Tensor  # (..., a, 3)
    index: torch.Tensor  # (q, n)
    function_args: tuple[Any, ...]
    com_index: torch.Tensor | None  # (a_input,)
    atom_pad_mask: torch.Tensor | None  # (a_input,)
    ref_coords: torch.Tensor | None  # (n_ref, n, 3)
    ref_mask: torch.Tensor | None  # (n_ref, n)
    ref_token_index: torch.Tensor | None  # (a_input,)
    negation_mask: torch.Tensor | None  # (n,)
    union_index: torch.Tensor | None  # (n,)


def _reduce_atoms_to_centers(
    coordinates: torch.Tensor,
    com_index: torch.Tensor,
    atom_pad_mask: torch.Tensor,
) -> torch.Tensor:
    # coordinates: (..., a, 3); com_index/atom_pad_mask: (a,).
    unpadded_index = com_index[atom_pad_mask]  # (a_valid,)
    unpadded_coordinates = coordinates[..., atom_pad_mask, :]  # (..., a_valid, 3)
    center_shape = (*unpadded_coordinates.shape[:-2], unpadded_index.max() + 1, 3)
    return torch.zeros(center_shape, device=coordinates.device).scatter_reduce(
        -2,
        unpadded_index.unsqueeze(-1).expand_as(unpadded_coordinates),
        unpadded_coordinates,
        "mean",
    )  # (..., n_center, 3)


def _union_weights(
    energy: torch.Tensor,
    union_index: torch.Tensor,
    union_lambda: float,
) -> torch.Tensor:
    # energy: (..., n); union_index: (n,).
    unnormalized = torch.exp(-union_lambda * energy)  # (..., n)
    partition = torch.zeros(
        (*energy.shape[:-1], union_index.max() + 1),
        device=union_index.device,
    ).scatter_reduce(
        -1,
        union_index.expand_as(unnormalized),
        unnormalized,
        "sum",
    )  # (..., n_union)
    weights = unnormalized / partition[..., union_index]  # (..., n)
    weights[partition[..., union_index] == 0] = 0  # (..., n)
    return weights  # (..., n)


def _scatter_constraint_gradients(
    coefficients: torch.Tensor,
    variable_gradient: torch.Tensor,
    index: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    # coefficients: (..., n); variable_gradient: (..., q, n, 3).
    # index: (q, n); coordinates: (b, a, 3).
    product = coefficients.tile(variable_gradient.shape[-3]).unsqueeze(
        -1
    ) * variable_gradient.flatten(start_dim=-3, end_dim=-2)  # (..., q * n, 3)
    if product.dim() > 3:
        product = product.sum(dim=list(range(1, product.dim() - 2)))  # (b, q * n, 3)
    scatter_index = (
        index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coordinates.shape[:-2], -1, 3))
    )  # (b, q * n, 3)
    return torch.zeros_like(coordinates).scatter_reduce(
        -2,
        scatter_index,
        product,
        "sum",
    )  # (b, a, 3)


class Potential(ABC):
    """Base contract for energy and analytic-gradient steering potentials."""

    def __init__(self, parameters: ParameterMap | None = None) -> None:
        self.parameters = parameters

    def _prepare(
        self,
        coordinates: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
        computed_args: tuple[Any, ...] | None = None,
    ) -> _PreparedPotential:
        # coordinates: (..., a_input, 3).
        if computed_args is None:
            computed_args = self.compute_args(feats, parameters)
        index, args, com_args, ref_args, operator_args = computed_args  # index: (q, n)
        com_index = atom_pad_mask = None
        if com_args is not None:
            com_index, atom_pad_mask = com_args  # each (a_input,)
            coordinates = _reduce_atoms_to_centers(
                coordinates,
                com_index,
                atom_pad_mask,
            )  # (..., n_center, 3)

        ref_coords = ref_mask = ref_token_index = None
        if ref_args is not None:
            # ref_coords: (n_ref, n, 3); ref_mask: (n_ref, n).
            # ref_atom_index: (n,); ref_token_index: (a_input,).
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coordinates = coordinates[..., ref_atom_index, :]  # (..., n, 3)

        negation_mask = union_index = None
        if operator_args is not None:
            negation_mask, union_index = operator_args  # each (n,)
        return _PreparedPotential(
            coordinates=coordinates,
            index=index,
            function_args=args,
            com_index=com_index,
            atom_pad_mask=atom_pad_mask,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            ref_token_index=ref_token_index,
            negation_mask=negation_mask,
            union_index=union_index,
        )

    def compute(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
    ) -> torch.Tensor:
        """Evaluate one energy per coordinate sample."""

        # coords: (..., a, 3).
        computed_args = self.compute_args(feats, parameters)
        if computed_args[0].shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)  # (...)
        prepared = self._prepare(coords, feats, parameters, computed_args)
        value = self.compute_variable(
            prepared.coordinates,
            prepared.index,
            ref_coords=prepared.ref_coords,
            ref_mask=prepared.ref_mask,
            compute_gradient=False,
        )  # (..., n)
        energy = self.compute_function(
            value,
            *prepared.function_args,
            negation_mask=prepared.negation_mask,
            compute_derivative=False,
        )  # (..., n)
        if prepared.union_index is not None:
            weights = _union_weights(
                energy,
                prepared.union_index,
                parameters["union_lambda"],
            )  # (..., n)
            return (energy * weights).sum(dim=-1)  # (...)
        return energy.sum(dim=tuple(range(1, energy.dim())))  # (b,)

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
    ) -> torch.Tensor:
        """Return the analytic coordinate gradient of the potential energy."""

        # coords: (b, a, 3).
        computed_args = self.compute_args(feats, parameters)
        if computed_args[0].shape[1] == 0:
            return torch.zeros_like(coords)  # (b, a, 3)
        prepared = self._prepare(coords, feats, parameters, computed_args)
        value, variable_gradient = self.compute_variable(
            prepared.coordinates,
            prepared.index,
            ref_coords=prepared.ref_coords,
            ref_mask=prepared.ref_mask,
            compute_gradient=True,
        )  # value: (..., n); variable_gradient: (..., q, n, 3)
        energy, energy_derivative = self.compute_function(
            value,
            *prepared.function_args,
            negation_mask=prepared.negation_mask,
            compute_derivative=True,
        )  # each (..., n)
        if prepared.union_index is not None:
            weights = _union_weights(
                energy,
                prepared.union_index,
                parameters["union_lambda"],
            )  # (..., n)
            union_energy = torch.zeros(
                (*energy.shape[:-1], prepared.union_index.max() + 1),
                device=prepared.union_index.device,
            ).scatter_reduce(
                -1,
                prepared.union_index.expand_as(energy),
                energy * weights,
                "sum",
            )  # (..., n_union)
            coefficients = (
                energy_derivative
                * weights
                * (
                    1
                    + parameters["union_lambda"]
                    * (energy - union_energy[..., prepared.union_index])
                )
            )  # (..., n)
        else:
            coefficients = energy_derivative  # (..., n)

        atom_gradient = _scatter_constraint_gradients(
            coefficients,
            variable_gradient,
            prepared.index,
            prepared.coordinates,
        )  # (b, a_prepared, 3)
        if prepared.com_index is not None:
            atom_gradient = atom_gradient[..., prepared.com_index, :]  # (b, a, 3)
        elif prepared.ref_token_index is not None:
            atom_gradient = atom_gradient[..., prepared.ref_token_index, :]  # (b, a, 3)
        return atom_gradient  # (b, a, 3)

    def compute_parameters(self, t: float) -> dict[str, Any] | None:
        """Resolve scheduled parameters at diffusion time ``t``."""

        if self.parameters is None:
            return None
        return {
            name: parameter.compute(t) if isinstance(parameter, ParameterSchedule) else parameter
            for name, parameter in self.parameters.items()
        }

    @abstractmethod
    def compute_function(
        self,
        value: torch.Tensor,
        *args: Any,
        negation_mask: torch.Tensor | None = None,
        compute_derivative: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # value: (..., n); negation_mask: (n,).
        raise NotImplementedError

    @abstractmethod
    def compute_variable(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        ref_coords: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        compute_gradient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # coords: (..., a, 3); index: (q, n).
        raise NotImplementedError

    @abstractmethod
    def compute_args(
        self,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def get_reference_coords(
        self,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
    ) -> tuple[None, None]:
        del feats, parameters
        return None, None


class FlatBottomPotential(Potential):
    """Linear penalty outside an allowed lower-to-upper interval."""

    def compute_function(
        self,
        value: torch.Tensor,
        k: torch.Tensor,
        lower_bounds: torch.Tensor | None,
        upper_bounds: torch.Tensor | None,
        negation_mask: torch.Tensor | None = None,
        compute_derivative: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # value: (..., n); k/lower_bounds/upper_bounds: (n,) or (..., n).
        # negation_mask: (n,) or a shape broadcastable to (..., n).
        lower = (
            (torch.full_like(value, float("-inf")) if lower_bounds is None else lower_bounds)
            .expand_as(value)
            .clone()
        )  # (..., n)
        upper = (
            (torch.full_like(value, float("inf")) if upper_bounds is None else upper_bounds)
            .expand_as(value)
            .clone()
        )  # (..., n)

        if negation_mask is not None:
            if not torch.is_tensor(negation_mask):
                raise TypeError(
                    f"negation_mask must be a boolean tensor, got {type(negation_mask).__name__}."
                )
            if negation_mask.dtype != torch.bool:
                raise TypeError(
                    f"negation_mask must be a boolean tensor, got {negation_mask.dtype}."
                )
            if negation_mask.device != value.device:
                raise ValueError(
                    "negation_mask must be on the same device as value; "
                    f"got {negation_mask.device} and {value.device}."
                )
            try:
                expanded_negation_mask = negation_mask.expand_as(value)  # (..., n)
            except RuntimeError as error:
                raise ValueError(
                    "negation_mask must be broadcastable to value shape "
                    f"{tuple(value.shape)}, got {tuple(negation_mask.shape)}."
                ) from error
            unbounded_below = torch.isneginf(lower)  # (..., n)
            unbounded_above = torch.isposinf(upper)  # (..., n)
            # (..., n)
            valid_negation = unbounded_below | unbounded_above | expanded_negation_mask
            if not bool(torch.all(valid_negation).item()):
                raise ValueError(
                    "negation_mask may be false only where at least one bound is infinite."
                )
            select_upper = ~unbounded_above & ~expanded_negation_mask  # (..., n)
            lower[select_upper] = upper[select_upper]  # (..., n)
            upper[select_upper] = float("inf")  # (..., n)
            select_lower = ~unbounded_below & ~expanded_negation_mask  # (..., n)
            upper[select_lower] = lower[select_lower]  # (..., n)
            lower[select_lower] = float("-inf")  # (..., n)

        below = value < lower  # (..., n)
        above = value > upper  # (..., n)
        energy = torch.zeros_like(value)  # (..., n)
        energy[below] = (k * (lower - value))[below]  # (..., n)
        energy[above] = (k * (value - upper))[above]  # (..., n)
        if not compute_derivative:
            return energy  # (..., n)
        derivative = torch.zeros_like(value)  # (..., n)
        derivative[below] = -k.expand_as(below)[below]  # (..., n)
        derivative[above] = k.expand_as(above)[above]  # (..., n)
        return energy, derivative  # each (..., n)


class ReferencePotential(Potential):
    """Measure atom displacement after weighted rigid alignment."""

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        ref_coords: torch.Tensor,
        ref_mask: torch.Tensor,
        compute_gradient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # coords: (b, a, 3); index: (1, n).
        # ref_coords: (n_ref, n, 3); ref_mask: (n_ref, n).
        aligned_reference = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )  # (b, n_ref, n, 3)
        displacement = coords[:, index] - aligned_reference  # (b, n_ref, n, 3)
        distance = torch.linalg.norm(displacement, dim=-1)  # (b, n_ref, n)
        if not compute_gradient:
            return distance  # (b, n_ref, n)
        unit_displacement = displacement / distance.unsqueeze(-1)  # (b, n_ref, n, 3)
        # (b, 1, n_ref, n, 3)
        gradient = (unit_displacement * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return distance, gradient  # (b, n_ref, n); (b, 1, n_ref, n, 3)


class DistancePotential(Potential):
    """Measure Euclidean distances for indexed atom pairs."""

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        ref_coords: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        compute_gradient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # coords: (..., a, 3); index: (2, n).
        del ref_coords, ref_mask
        displacement = coords.index_select(-2, index[0]) - coords.index_select(
            -2,
            index[1],
        )  # (..., n, 3)
        distance = torch.linalg.norm(displacement, dim=-1)  # (..., n)
        if not compute_gradient:
            return distance  # (..., n)
        unit_displacement = displacement / distance.unsqueeze(-1)  # (..., n, 3)
        # Returns (..., n); (..., 2, n, 3).
        return distance, torch.stack((unit_displacement, -unit_displacement), dim=1)


class DihedralPotential(Potential):
    """Measure signed torsion angles for indexed atom quartets."""

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        ref_coords: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        compute_gradient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # coords: (..., a, 3); index: (4, n).
        del ref_coords, ref_mask
        # Each displacement has shape (..., n, 3).
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)  # (..., n, 3)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)  # (..., n, 3)
        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)  # (..., n)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)  # (..., n)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)  # (..., n)

        orientation = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)  # (..., n)
        cosine = (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2) / (
            n_ijk_norm * n_jkl_norm
        )  # (..., n)
        # (..., n)
        angle = orientation * torch.arccos(torch.clamp(cosine, -1 + 1e-8, 1 - 1e-8))
        if not compute_gradient:
            return angle  # (..., n)

        projection_i = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)  # (..., n, 1)
        projection_l = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)  # (..., n, 1)
        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)  # (..., n, 3)
        grad_l = -n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)  # (..., n, 3)
        grad_j = (projection_i - 1) * grad_i - projection_l * grad_l  # (..., n, 3)
        grad_k = (projection_l - 1) * grad_l - projection_i * grad_i  # (..., n, 3)
        # Returns (..., n); (..., 4, n, 3).
        return angle, torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)


class AbsDihedralPotential(DihedralPotential):
    """Measure the unsigned magnitude of indexed torsion angles."""

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        ref_coords: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        compute_gradient: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # coords: (..., a, 3); index: (4, n).
        del ref_coords, ref_mask
        if not compute_gradient:
            return torch.abs(super().compute_variable(coords, index))  # (..., n)
        angle, gradient = super().compute_variable(
            coords,
            index,
            compute_gradient=True,
        )  # (..., n); (..., 4, n, 3)
        gradient[(angle < 0)[..., None, :, None].expand_as(gradient)] *= -1  # same
        return torch.abs(angle), gradient  # (..., n); (..., 4, n, 3)


def _element_radii(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    # feats["ref_element"]: (1, a, n_element).
    element_radii = torch.zeros(
        const.num_elements,
        dtype=torch.float32,
        device=feats["ref_element"].device,
    )  # (n_element,)
    element_radii[1:119] = torch.tensor(
        const.vdw_radii,
        dtype=torch.float32,
        device=element_radii.device,
    )  # (n_element,)
    # (a,)
    return (feats["ref_element"].float() @ element_radii.unsqueeze(-1)).squeeze(-1)[0]


def _atom_chain_ids(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    # atom_to_token: (1, a, l); asym_id: (1, l).
    return (
        torch.bmm(
            feats["atom_to_token"].float(),
            feats["asym_id"].unsqueeze(-1).float(),
        )
        .squeeze(-1)
        .long()
    )[0]  # (a,)


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        # rdkit_bounds_index: (1, 2, n); bound and mask features: (1, n).
        pair_index = feats["rdkit_bounds_index"][0]  # (2, n)
        lower = feats["rdkit_lower_bounds"][0].clone()  # (n,)
        upper = feats["rdkit_upper_bounds"][0].clone()  # (n,)
        bond = feats["rdkit_bounds_bond_mask"][0]  # (n,)
        angle = feats["rdkit_bounds_angle_mask"][0]  # (n,)
        lower[bond * ~angle] *= 1.0 - parameters["bond_buffer"]  # (n,)
        upper[bond * ~angle] *= 1.0 + parameters["bond_buffer"]  # (n,)
        lower[~bond * angle] *= 1.0 - parameters["angle_buffer"]  # (n,)
        upper[~bond * angle] *= 1.0 + parameters["angle_buffer"]  # (n,)
        shared_buffer = min(parameters["bond_buffer"], parameters["angle_buffer"])
        lower[bond * angle] *= 1.0 - shared_buffer  # (n,)
        upper[bond * angle] *= 1.0 + shared_buffer  # (n,)
        lower[~bond * ~angle] *= 1.0 - parameters["clash_buffer"]  # (n,)
        upper[~bond * ~angle] = float("inf")  # (n,)

        atom_radii = _element_radii(feats)  # (a,)
        bond_cutoff = 0.35 + atom_radii[pair_index].mean(dim=0)  # (n,)
        lower[~bond] = torch.max(lower[~bond], bond_cutoff[~bond])  # (n,)
        upper[bond] = torch.min(upper[bond], bond_cutoff[bond])  # (n,)
        # Returns index (2, n) and three function arguments of shape (n,).
        return pair_index, (torch.ones_like(lower), lower, upper), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        # connected_atom_index: (1, 2, n).
        pair_index = feats["connected_atom_index"][0]  # (2, n)
        upper = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            device=pair_index.device,
        )  # (n,)
        # Returns index (2, n), k (n,), and upper bound (n,).
        return pair_index, (torch.ones_like(upper), None, upper), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        atom_chain_id = _atom_chain_ids(feats)  # (a,)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()  # (a,)
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])  # (n_chain,)
        nonion_atom = (chain_sizes > 1)[atom_chain_id]  # (a,)
        atom_radii = _element_radii(feats)  # (a,)
        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )  # (2, n_pair)
        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)  # (n_pair,)
        # (n_pair,)
        pair_ion_mask = nonion_atom[pair_index[0]] * nonion_atom[pair_index[1]]

        num_chains = atom_chain_id.max() + 1  # ()
        connected = feats["connected_chain_index"][0]  # (2, n_connection)
        connected_matrix = torch.eye(
            num_chains,
            device=atom_chain_id.device,
            dtype=torch.bool,
        )  # (n_chain, n_chain)
        connected_matrix[connected[0], connected[1]] = True  # (n_chain, n_chain)
        connected_matrix[connected[1], connected[0]] = True  # (n_chain, n_chain)
        connected_pair = connected_matrix[
            atom_chain_id[pair_index[0]],
            atom_chain_id[pair_index[1]],
        ]  # (n_pair,)
        # (2, n)
        pair_index = pair_index[:, pair_pad_mask * pair_ion_mask * ~connected_pair]
        # (n,)
        lower = atom_radii[pair_index].sum(dim=0) * (1.0 - parameters["buffer"])
        # Returns index (2, n), k (n,), and lower bound (n,).
        return pair_index, (torch.ones_like(lower), lower, None), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        atom_chain_id = _atom_chain_ids(feats)  # (a,)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()  # (a,)
        nonion_chain = torch.bincount(atom_chain_id[atom_pad_mask]) > 1  # (n_chain,)
        pair_index = feats["symmetric_chain_index"][0]  # (2, n_candidate)
        pair_index = pair_index[
            :,
            nonion_chain[pair_index[0]] * nonion_chain[pair_index[1]],
        ]  # (2, n)
        lower = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )  # (n,)
        # Returns center index (2, n), k/lower (n,), and atom-to-center maps (a,).
        return (
            pair_index,
            (torch.ones_like(lower), lower, None),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


def _oriented_bounds(
    orientations: torch.Tensor,
    positive_lower: float,
    negative_upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # orientations: (n,).
    lower = torch.zeros(orientations.shape, device=orientations.device)  # (n,)
    upper = torch.zeros(orientations.shape, device=orientations.device)  # (n,)
    lower[orientations] = positive_lower  # (n,)
    upper[orientations] = float("inf")  # (n,)
    lower[~orientations] = float("-inf")  # (n,)
    upper[~orientations] = negative_upper  # (n,)
    return lower, upper  # each (n,)


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        # stereo_bond_index: (1, 4, n); orientations: (1, n).
        index = feats["stereo_bond_index"][0]  # (4, n)
        orientation = feats["stereo_bond_orientations"][0].bool()  # (n,)
        lower, upper = _oriented_bounds(
            orientation,
            torch.pi - parameters["buffer"],
            parameters["buffer"],
        )  # each (n,)
        # Returns index (4, n) and three function arguments of shape (n,).
        return index, (torch.ones_like(lower), lower, upper), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        # chiral_atom_index: (1, 4, n); orientations: (1, n).
        index = feats["chiral_atom_index"][0]  # (4, n)
        orientation = feats["chiral_atom_orientations"][0].bool()  # (n,)
        lower, upper = _oriented_bounds(
            orientation,
            parameters["buffer"],
            -parameters["buffer"],
        )  # each (n,)
        # Returns index (4, n) and three function arguments of shape (n,).
        return index, (torch.ones_like(lower), lower, upper), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        # The feature stores two atom-index rows for each six-entry bond pattern.
        bond_index = feats["planar_bond_index"][0].T  # (2, n_bond_entry)
        improper_pattern = torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 3]],
            device=bond_index.device,
        ).T  # (4, 2)
        # (4, n_improper)
        improper_index = bond_index[:, improper_pattern].swapaxes(0, 1).flatten(start_dim=1)
        upper = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )  # (n_improper,)
        # Returns index (4, n_improper), k (n_improper,), and upper (n_improper,).
        return (
            improper_index,
            (torch.ones_like(upper), None, upper),
            None,
            None,
            None,
        )


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        del parameters
        if "template_mask_cb" not in feats or "template_force" not in feats:
            # Empty sentinel index: (1, 0).
            return torch.empty((1, 0)), None, None, None, None
        # template_mask_cb/template_force: (b, n_template, n)/(b, n_template).
        template_mask = feats["template_mask_cb"][feats["template_force"]]  # (n_ref, n)
        if template_mask.shape[0] == 0:
            # Empty sentinel index: (1, 0).
            return torch.empty((1, 0)), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()  # (n_ref, n, 3)
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()  # (n_ref, n)
        atom_indices = torch.arange(
            feats["atom_pad_mask"].shape[1],
            device=feats["atom_pad_mask"].device,
            dtype=torch.float32,
        )[None, :, None]  # (1, a, 1)
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                atom_indices,
            )
            .squeeze(-1)
            .long()[0]  # (n,)
        )
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()[0]  # (a,)
        )

        index = torch.arange(
            template_mask.shape[-1],
            dtype=torch.long,
            device=template_mask.device,
        )[None]  # (1, n)
        upper = torch.full(
            template_mask.shape,
            float("inf"),
            device=index.device,
            dtype=torch.float32,
        )  # (n_ref, n)
        reference_indices = torch.argwhere(template_mask).T  # (2, n_active)
        upper[reference_indices.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][reference_indices[0]]  # (n_ref, n)
        # Returns index (1, n), function args (n_ref, n), and reference mappings.
        return (
            index,
            (torch.ones_like(upper), None, upper),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    """Contact-union potential retaining the historical checkpoint name."""

    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        del parameters
        # pair index: (1, 2, n); threshold/operator features: (1, n).
        index = feats["contact_pair_index"][0]  # (2, n)
        upper = feats["contact_thresholds"][0].clone()  # (n,)
        # Returns index (2, n), k/upper (n,), and two operator tensors (n,).
        return (
            index,
            (torch.ones_like(upper), None, upper),
            None,
            None,
            (
                feats["contact_negation_mask"][0],
                feats["contact_union_index"][0],
            ),
        )


ContactPotential = ContactPotentital


def get_potentials(
    steering_args: dict[str, bool],
    boltz2: bool = False,
) -> list[Potential]:
    """Build the ordered potential stack for requested steering modes."""

    use_fk = steering_args["fk_steering"]
    use_physical = steering_args["physical_guidance_update"]
    use_contacts = steering_args.get("contact_guidance_update", False)
    potentials: list[Potential] = []
    if use_fk or use_physical:
        potentials.extend(
            (
                SymmetricChainCOMPotential(
                    {
                        "guidance_interval": 4,
                        "guidance_weight": 0.5 if use_physical else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(1.0, 5.0, -2.0),
                    }
                ),
                VDWOverlapPotential(
                    {
                        "guidance_interval": 5,
                        "guidance_weight": PiecewiseStepFunction(
                            [0.4],
                            [0.125, 0.0],
                        )
                        if use_physical
                        else 0.0,
                        "resampling_weight": PiecewiseStepFunction(
                            [0.6],
                            [0.01, 0.0],
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    {
                        "guidance_interval": 1,
                        "guidance_weight": 0.15 if use_physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    {
                        "guidance_interval": 1,
                        "guidance_weight": 0.01 if use_physical else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    {
                        "guidance_interval": 1,
                        "guidance_weight": 0.1 if use_physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    {
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if use_physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    {
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if use_physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            )
        )
    if boltz2 and (use_fk or use_contacts):
        potentials.extend(
            (
                ContactPotentital(
                    {
                        "guidance_interval": 4,
                        "guidance_weight": PiecewiseStepFunction(
                            [0.25, 0.75],
                            [0.0, 0.5, 1.0],
                        )
                        if use_contacts
                        else 0.0,
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(8.0, 0.0, -2.0),
                    }
                ),
                TemplateReferencePotential(
                    {
                        "guidance_interval": 2,
                        "guidance_weight": 0.1 if use_contacts else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            )
        )
    return potentials
