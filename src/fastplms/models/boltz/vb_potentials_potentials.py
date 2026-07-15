"""Differentiable structure-steering potentials for Boltz2 diffusion.

Each potential follows the same mechanism: derive constraint indices and
bounds from features, evaluate a geometric variable, map that variable to an
energy, and optionally scatter its analytic derivative back to atom
coordinates.  This module is maintained independently from the pinned parity
oracle while preserving the public class names used by converted checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

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
    coordinates: torch.Tensor
    index: torch.Tensor
    function_args: tuple[Any, ...]
    com_index: torch.Tensor | None
    atom_pad_mask: torch.Tensor | None
    ref_coords: torch.Tensor | None
    ref_mask: torch.Tensor | None
    ref_token_index: torch.Tensor | None
    negation_mask: torch.Tensor | None
    union_index: torch.Tensor | None


def _reduce_atoms_to_centers(
    coordinates: torch.Tensor,
    com_index: torch.Tensor,
    atom_pad_mask: torch.Tensor,
) -> torch.Tensor:
    unpadded_index = com_index[atom_pad_mask]
    unpadded_coordinates = coordinates[..., atom_pad_mask, :]
    center_shape = (*unpadded_coordinates.shape[:-2], unpadded_index.max() + 1, 3)
    return torch.zeros(center_shape, device=coordinates.device).scatter_reduce(
        -2,
        unpadded_index.unsqueeze(-1).expand_as(unpadded_coordinates),
        unpadded_coordinates,
        "mean",
    )


def _union_weights(
    energy: torch.Tensor,
    union_index: torch.Tensor,
    union_lambda: float,
) -> torch.Tensor:
    unnormalized = torch.exp(-union_lambda * energy)
    partition = torch.zeros(
        (*energy.shape[:-1], union_index.max() + 1),
        device=union_index.device,
    ).scatter_reduce(
        -1,
        union_index.expand_as(unnormalized),
        unnormalized,
        "sum",
    )
    weights = unnormalized / partition[..., union_index]
    weights[partition[..., union_index] == 0] = 0
    return weights


def _scatter_constraint_gradients(
    coefficients: torch.Tensor,
    variable_gradient: torch.Tensor,
    index: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    product = coefficients.tile(variable_gradient.shape[-3]).unsqueeze(
        -1
    ) * variable_gradient.flatten(start_dim=-3, end_dim=-2)
    if product.dim() > 3:
        product = product.sum(dim=list(range(1, product.dim() - 2)))
    scatter_index = (
        index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coordinates.shape[:-2], -1, 3))
    )
    return torch.zeros_like(coordinates).scatter_reduce(
        -2,
        scatter_index,
        product,
        "sum",
    )


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
        if computed_args is None:
            computed_args = self.compute_args(feats, parameters)
        index, args, com_args, ref_args, operator_args = computed_args
        com_index = atom_pad_mask = None
        if com_args is not None:
            com_index, atom_pad_mask = com_args
            coordinates = _reduce_atoms_to_centers(
                coordinates,
                com_index,
                atom_pad_mask,
            )

        ref_coords = ref_mask = ref_token_index = None
        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coordinates = coordinates[..., ref_atom_index, :]

        negation_mask = union_index = None
        if operator_args is not None:
            negation_mask, union_index = operator_args
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

        computed_args = self.compute_args(feats, parameters)
        if computed_args[0].shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)
        prepared = self._prepare(coords, feats, parameters, computed_args)
        value = self.compute_variable(
            prepared.coordinates,
            prepared.index,
            ref_coords=prepared.ref_coords,
            ref_mask=prepared.ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value,
            *prepared.function_args,
            negation_mask=prepared.negation_mask,
            compute_derivative=False,
        )
        if prepared.union_index is not None:
            weights = _union_weights(
                energy,
                prepared.union_index,
                parameters["union_lambda"],
            )
            return (energy * weights).sum(dim=-1)
        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any],
    ) -> torch.Tensor:
        """Return the analytic coordinate gradient of the potential energy."""

        computed_args = self.compute_args(feats, parameters)
        if computed_args[0].shape[1] == 0:
            return torch.zeros_like(coords)
        prepared = self._prepare(coords, feats, parameters, computed_args)
        value, variable_gradient = self.compute_variable(
            prepared.coordinates,
            prepared.index,
            ref_coords=prepared.ref_coords,
            ref_mask=prepared.ref_mask,
            compute_gradient=True,
        )
        energy, energy_derivative = self.compute_function(
            value,
            *prepared.function_args,
            negation_mask=prepared.negation_mask,
            compute_derivative=True,
        )
        if prepared.union_index is not None:
            weights = _union_weights(
                energy,
                prepared.union_index,
                parameters["union_lambda"],
            )
            union_energy = torch.zeros(
                (*energy.shape[:-1], prepared.union_index.max() + 1),
                device=prepared.union_index.device,
            ).scatter_reduce(
                -1,
                prepared.union_index.expand_as(energy),
                energy * weights,
                "sum",
            )
            coefficients = (
                energy_derivative
                * weights
                * (
                    1
                    + parameters["union_lambda"]
                    * (energy - union_energy[..., prepared.union_index])
                )
            )
        else:
            coefficients = energy_derivative

        atom_gradient = _scatter_constraint_gradients(
            coefficients,
            variable_gradient,
            prepared.index,
            prepared.coordinates,
        )
        if prepared.com_index is not None:
            atom_gradient = atom_gradient[..., prepared.com_index, :]
        elif prepared.ref_token_index is not None:
            atom_gradient = atom_gradient[..., prepared.ref_token_index, :]
        return atom_gradient

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
        lower = (
            (torch.full_like(value, float("-inf")) if lower_bounds is None else lower_bounds)
            .expand_as(value)
            .clone()
        )
        upper = (
            (torch.full_like(value, float("inf")) if upper_bounds is None else upper_bounds)
            .expand_as(value)
            .clone()
        )

        if negation_mask is not None:
            unbounded_below = torch.isneginf(lower)
            unbounded_above = torch.isposinf(upper)
            assert torch.all(unbounded_below + unbounded_above + negation_mask)
            select_upper = ~unbounded_above * ~negation_mask
            lower[select_upper] = upper[select_upper]
            upper[select_upper] = float("inf")
            select_lower = ~unbounded_below * ~negation_mask
            upper[select_lower] = lower[select_lower]
            lower[select_lower] = float("-inf")

        below = value < lower
        above = value > upper
        energy = torch.zeros_like(value)
        energy[below] = (k * (lower - value))[below]
        energy[above] = (k * (value - upper))[above]
        if not compute_derivative:
            return energy
        derivative = torch.zeros_like(value)
        derivative[below] = -k.expand_as(below)[below]
        derivative[above] = k.expand_as(above)[above]
        return energy, derivative


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
        aligned_reference = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )
        displacement = coords[:, index] - aligned_reference
        distance = torch.linalg.norm(displacement, dim=-1)
        if not compute_gradient:
            return distance
        unit_displacement = displacement / distance.unsqueeze(-1)
        gradient = (unit_displacement * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return distance, gradient


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
        del ref_coords, ref_mask
        displacement = coords.index_select(-2, index[0]) - coords.index_select(
            -2,
            index[1],
        )
        distance = torch.linalg.norm(displacement, dim=-1)
        if not compute_gradient:
            return distance
        unit_displacement = displacement / distance.unsqueeze(-1)
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
        del ref_coords, ref_mask
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)
        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        orientation = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        cosine = (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2) / (
            n_ijk_norm * n_jkl_norm
        )
        angle = orientation * torch.arccos(torch.clamp(cosine, -1 + 1e-8, 1 - 1e-8))
        if not compute_gradient:
            return angle

        projection_i = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        projection_l = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (projection_i - 1) * grad_i - projection_l * grad_l
        grad_k = (projection_l - 1) * grad_l - projection_i * grad_i
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
        del ref_coords, ref_mask
        if not compute_gradient:
            return torch.abs(super().compute_variable(coords, index))
        angle, gradient = super().compute_variable(
            coords,
            index,
            compute_gradient=True,
        )
        gradient[(angle < 0)[..., None, :, None].expand_as(gradient)] *= -1
        return torch.abs(angle), gradient


def _element_radii(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    element_radii = torch.zeros(
        const.num_elements,
        dtype=torch.float32,
        device=feats["ref_element"].device,
    )
    element_radii[1:119] = torch.tensor(
        const.vdw_radii,
        dtype=torch.float32,
        device=element_radii.device,
    )
    return (feats["ref_element"].float() @ element_radii.unsqueeze(-1)).squeeze(-1)[0]


def _atom_chain_ids(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.bmm(
            feats["atom_to_token"].float(),
            feats["asym_id"].unsqueeze(-1).float(),
        )
        .squeeze(-1)
        .long()
    )[0]


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        pair_index = feats["rdkit_bounds_index"][0]
        lower = feats["rdkit_lower_bounds"][0].clone()
        upper = feats["rdkit_upper_bounds"][0].clone()
        bond = feats["rdkit_bounds_bond_mask"][0]
        angle = feats["rdkit_bounds_angle_mask"][0]
        lower[bond * ~angle] *= 1.0 - parameters["bond_buffer"]
        upper[bond * ~angle] *= 1.0 + parameters["bond_buffer"]
        lower[~bond * angle] *= 1.0 - parameters["angle_buffer"]
        upper[~bond * angle] *= 1.0 + parameters["angle_buffer"]
        shared_buffer = min(parameters["bond_buffer"], parameters["angle_buffer"])
        lower[bond * angle] *= 1.0 - shared_buffer
        upper[bond * angle] *= 1.0 + shared_buffer
        lower[~bond * ~angle] *= 1.0 - parameters["clash_buffer"]
        upper[~bond * ~angle] = float("inf")

        atom_radii = _element_radii(feats)
        bond_cutoff = 0.35 + atom_radii[pair_index].mean(dim=0)
        lower[~bond] = torch.max(lower[~bond], bond_cutoff[~bond])
        upper[bond] = torch.min(upper[bond], bond_cutoff[bond])
        return pair_index, (torch.ones_like(lower), lower, upper), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        pair_index = feats["connected_atom_index"][0]
        upper = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            device=pair_index.device,
        )
        return pair_index, (torch.ones_like(upper), None, upper), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        atom_chain_id = _atom_chain_ids(feats)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        nonion_atom = (chain_sizes > 1)[atom_chain_id]
        atom_radii = _element_radii(feats)
        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )
        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = nonion_atom[pair_index[0]] * nonion_atom[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected = feats["connected_chain_index"][0]
        connected_matrix = torch.eye(
            num_chains,
            device=atom_chain_id.device,
            dtype=torch.bool,
        )
        connected_matrix[connected[0], connected[1]] = True
        connected_matrix[connected[1], connected[0]] = True
        connected_pair = connected_matrix[
            atom_chain_id[pair_index[0]],
            atom_chain_id[pair_index[1]],
        ]
        pair_index = pair_index[:, pair_pad_mask * pair_ion_mask * ~connected_pair]
        lower = atom_radii[pair_index].sum(dim=0) * (1.0 - parameters["buffer"])
        return pair_index, (torch.ones_like(lower), lower, None), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        atom_chain_id = _atom_chain_ids(feats)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        nonion_chain = torch.bincount(atom_chain_id[atom_pad_mask]) > 1
        pair_index = feats["symmetric_chain_index"][0]
        pair_index = pair_index[
            :,
            nonion_chain[pair_index[0]] * nonion_chain[pair_index[1]],
        ]
        lower = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
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
    lower = torch.zeros(orientations.shape, device=orientations.device)
    upper = torch.zeros(orientations.shape, device=orientations.device)
    lower[orientations] = positive_lower
    upper[orientations] = float("inf")
    lower[~orientations] = float("-inf")
    upper[~orientations] = negative_upper
    return lower, upper


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        index = feats["stereo_bond_index"][0]
        orientation = feats["stereo_bond_orientations"][0].bool()
        lower, upper = _oriented_bounds(
            orientation,
            torch.pi - parameters["buffer"],
            parameters["buffer"],
        )
        return index, (torch.ones_like(lower), lower, upper), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        index = feats["chiral_atom_index"][0]
        orientation = feats["chiral_atom_orientations"][0].bool()
        lower, upper = _oriented_bounds(
            orientation,
            parameters["buffer"],
            -parameters["buffer"],
        )
        return index, (torch.ones_like(lower), lower, upper), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats: dict[str, torch.Tensor], parameters: dict[str, Any]):
        bond_index = feats["planar_bond_index"][0].T
        improper_pattern = torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 3]],
            device=bond_index.device,
        ).T
        improper_index = bond_index[:, improper_pattern].swapaxes(0, 1).flatten(start_dim=1)
        upper = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
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
            return torch.empty((1, 0)), None, None, None, None
        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty((1, 0)), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        atom_indices = torch.arange(
            feats["atom_pad_mask"].shape[1],
            device=feats["atom_pad_mask"].device,
            dtype=torch.float32,
        )[None, :, None]
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                atom_indices,
            )
            .squeeze(-1)
            .long()[0]
        )
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()[0]
        )

        index = torch.arange(
            template_mask.shape[-1],
            dtype=torch.long,
            device=template_mask.device,
        )[None]
        upper = torch.full(
            template_mask.shape,
            float("inf"),
            device=index.device,
            dtype=torch.float32,
        )
        reference_indices = torch.argwhere(template_mask).T
        upper[reference_indices.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][reference_indices[0]]
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
        index = feats["contact_pair_index"][0]
        upper = feats["contact_thresholds"][0].clone()
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
