"""Denoise Boltz2 atom coordinates with optional structure-steering potentials."""

# Based on https://github.com/lucidrains/alphafold3-pytorch.
# MIT License, Copyright (c) 2024 Phil Wang.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from math import sqrt
from typing import Any
from einops import rearrange
from torch import nn
from torch.nn import Module
from tqdm.auto import tqdm

from . import vb_const as const
from . import vb_layers_initialize as init
from .vb_loss_diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from .vb_modules_encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from .vb_modules_transformersv2 import (
    DiffusionTransformer,
)
from .vb_modules_utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from .vb_potentials_potentials import get_potentials


class DiffusionModule(Module):
    """Predict coordinate updates from token and local atom attention."""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # Token conditioning shares trunk features across repeated diffusion samples.
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs: torch.Tensor,  # (b, t, d_s)
        s_trunk: torch.Tensor,  # (b, t, d_s)
        r_noisy: torch.Tensor,  # (b * m, a, 3)
        times: torch.Tensor,  # (b * m,)
        feats: dict[str, torch.Tensor],
        diffusion_conditioning: dict[str, Any],
        multiplicity: int = 1,
    ) -> torch.Tensor:
        # b: base batch; m: multiplicity; t: tokens; a: atoms; d_s: token width.
        if self.activation_checkpointing:
            s, _normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
                use_reentrant=False,
            )  # s: (b * m, t, 2 * d_s); Fourier features: (b * m, d_fourier)
        else:
            s, _normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )  # s: (b * m, t, 2 * d_s); Fourier features: (b * m, d_fourier)

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            multiplicity=multiplicity,
        )  # a: (b * m, t, 2 * d_s); q_skip/c_skip: (b * m, n_atom, d_a)

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)  # (b * m, t, 2 * d_s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)  # (b * m, t)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )  # (b * m, t, 2 * d_s)
        a = self.a_norm(a)  # (b * m, t, 2 * d_s)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )  # (b * m, n_atom, 3)

        return r_update  # (b * m, n_atom, 3)


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args: dict[str, Any],
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        # Mean of the log-normal training-noise distribution.
        P_mean: float = -1.2,
        # Standard deviation of the log-normal training-noise distribution.
        P_std: float = 1.5,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list[float] | None = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference: bool | None = None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
    ) -> None:
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(self.score_model, dynamic=False, fullgraph=False)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (...); coefficient preserves shape (...).
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (...); coefficient preserves shape (...).
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (...); coefficient preserves shape (...).
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (...); log-noise condition preserves shape (...).
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords: torch.Tensor,  #: Float['b m 3'],
        sigma: torch.Tensor | float,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # noised_atom_coords: (s, a, 3); sigma: float or (s,).
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)  # (s,) after a scalar input is expanded

        padded_sigma = rearrange(sigma, "b -> b 1 1")  # (s, 1, 1)

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )  # (s, a, 3)

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords + self.c_out(padded_sigma) * r_update
        )  # (s, a, 3)
        return denoised_coords

    def sample_schedule(self, num_sampling_steps: int | None = None) -> torch.Tensor:
        # n_step is the requested sampling-step count; the returned schedule has n_step + 1 values.
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sampling_steps, device=self.device, dtype=torch.float32)  # (n_step,)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_sampling_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho  # (n_sigma,); schedule or batch axis is determined by the caller

        sigmas = sigmas * self.sigma_data  # (n_sigma,); schedule or batch axis is determined by the caller

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask: torch.Tensor,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        max_parallel_samples: int | None = None,
        steering_args: dict[str, Any] | None = None,
        verbose: bool = False,
        **network_condition_kwargs: Any,
    ) -> dict[str, Any]:
        # a is atom count; m is multiplicity, including steering particles.
        # s is the current sample axis after batch repetition or particle resampling.
        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=True)

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)  # (m, n_evaluation)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )  # (n_group, n_particle)
        if steering_args["physical_guidance_update"] or steering_args["contact_guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )  # (s, a, 3)
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)  # (s, a); s expands on repetition and can shrink on resampling

        shape = (*atom_mask.shape, 3)

        # Pair each schedule value with the next sigma and gamma.
        sigmas = self.sample_schedule(num_sampling_steps)  # (n_sigma,); schedule or batch axis is determined by the caller
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)  # (n_step + 1,)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:], strict=True))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # atom position is noise at the beginning
        init_sigma = sigmas[0]  # ()
        atom_coords = init_sigma * torch.randn(shape, device=self.device)  # (s, a, 3)
        token_repr = None  # None; the current sampler does not produce token features
        atom_coords_denoised = None  # (s, a, 3) or None before the first denoising step

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(
            tqdm(
                sigmas_and_gammas,
                desc="Boltz2: Diffusion sampling",
                unit="step",
                disable=not verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )  # (m, 3, 3), (m, 1, 3)
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)  # (s, a, 3)
            atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr  # (s, a, 3)
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)  # (s, a, 3) or None before the first denoising step
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R) + random_tr
                )  # (s, a, 3) or None before the first denoising step
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )  # (s, a, 3)

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()  # three scalar tensors () become Python floats

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)  # (s, a, 3)
            atom_coords_noisy = atom_coords + eps  # (s, a, 3)

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)  # (s, a, 3) or None before the first denoising step
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)  # (m,)
                sample_ids_chunks = sample_ids.chunk(multiplicity % max_parallel_samples + 1)  # tuple of index tensors, each (n_chunk,)

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )  # (n_chunk, a, 3)
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk  # (n_chunk, a, 3)

                if steering_args["fk_steering"] and (
                    (step_idx % steering_args["fk_resampling_interval"] == 0 and noise_var > 0)
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)  # (m,)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )  # (m,)
                            energy += parameters["resampling_weight"] * component_energy  # (m,)
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)  # (m, n_evaluation)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy  # (m,)
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]  # (m,)

                    # Compute ll difference between guided and unguided transition distribution
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ) and noise_var > 0:
                        ll_difference = (eps**2 - (eps + scaled_guidance_update) ** 2).sum(
                            dim=(-1, -2)
                        ) / (2 * noise_var)  # (m,)
                    else:
                        ll_difference = torch.zeros_like(energy)  # (m,)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )  # (n_group, n_particle)

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and step_idx < num_sampling_steps - 1:
                    guidance_update = torch.zeros_like(atom_coords_denoised)  # (s, a, 3)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)  # (s, a, 3)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"] == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )  # (s, a, 3)
                        guidance_update -= energy_gradient  # (s, a, 3)
                    atom_coords_denoised += guidance_update  # (s, a, 3) or None before the first denoising step
                    scaled_guidance_update = (
                        guidance_update * -1 * self.step_scale * (sigma_t - t_hat) / t_hat
                    )  # (s, a, 3)

                if steering_args["fk_steering"] and (
                    (step_idx % steering_args["fk_resampling_interval"] == 0 and noise_var > 0)
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1] if step_idx < num_sampling_steps - 1 else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()  # (s_next,); final resampling keeps one particle per group

                    atom_coords = atom_coords[resample_indices]  # (s, a, 3)
                    atom_coords_noisy = atom_coords_noisy[resample_indices]  # (s, a, 3)
                    atom_mask = atom_mask[resample_indices]  # (s, a); s expands on repetition and can shrink on resampling
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]  # (s, a, 3) or None before the first denoising step
                    energy_traj = energy_traj[resample_indices]  # (s_next, n_evaluation)
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ):
                        scaled_guidance_update = scaled_guidance_update[resample_indices]  # (s, a, 3)
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]  # (s_next, *feature_shape)

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )  # (s, a, 3)

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)  # (s, a, 3)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat  # (s, a, 3)
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )  # (s, a, 3)

            atom_coords = atom_coords_next  # (s, a, 3)

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (...); weight preserves shape (...).
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size: int) -> torch.Tensor:
        # Return one noise scale per example, shape (batch_size,).
        return (
            self.sigma_data
            * (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()
        )

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        feats: dict[str, torch.Tensor],
        diffusion_conditioning: dict[str, Any],
        multiplicity: int = 1,
    ) -> dict[str, Any]:
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(multiplicity, 0)  # (n_sigma,); schedule or batch axis is determined by the caller
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)  # (n_sigma,); schedule or batch axis is determined by the caller
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")  # (b * m, 1, 1)

        atom_coords = feats["coords"]  # (s, a, 3)

        atom_mask = feats["atom_pad_mask"]  # (s, a); s expands on repetition and can shrink on resampling
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)  # (s, a); s expands on repetition and can shrink on resampling

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )  # (s, a, 3)

        noise = torch.randn_like(atom_coords)  # (b * m, a, 3)
        noised_atom_coords = atom_coords + padded_sigmas * noise  # (b * m, a, 3)

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )  # (b * m, a, 3)

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats: dict[str, torch.Tensor],
        out_dict: dict[str, torch.Tensor],
        add_smooth_lddt_loss: bool = True,
        nucleotide_loss_weight: float = 5.0,
        ligand_loss_weight: float = 10.0,
        multiplicity: int = 1,
        filter_by_plddt: float = 0.0,
    ) -> dict[str, Any]:
        # Coordinates: (b * m, a, 3); base feature masks: (b, a).
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()  # (b * m, a, 3)
            sigmas = out_dict["sigmas"].float()  # (n_sigma,); schedule or batch axis is determined by the caller

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()  # (b, a)

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt  # (b, a)
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()  # (b, a)

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(multiplicity, 0)  # (b * m, a)

            align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])  # (b * m, a)
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )  # (b, a)
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)  # (b * m, a)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
                ).float()
            )  # (b * m, a)

            atom_coords = out_dict["aligned_true_atom_coords"].float()  # (s, a, 3)
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )  # (b * m, a, 3)

            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )  # (b * m, a, 3)

            # weighted MSE loss of denoised atom positions
            mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)  # (b * m, a)
            mse_loss = torch.sum(mse_loss * align_weights * resolved_atom_mask, dim=-1) / (
                torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5
            )  # (b * m,)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)  # (b * m,)
            mse_loss = (mse_loss * loss_weights).mean()  # ()

            total_loss = mse_loss  # ()

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero  # ()
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )  # ()

                total_loss = total_loss + lddt_loss  # ()

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
