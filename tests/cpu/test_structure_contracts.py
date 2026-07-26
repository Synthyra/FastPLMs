"""Portable CPU contracts around public structure and binder helpers."""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from tests.integration import test_binder_design as binder_contracts
from tests.structure import test_esmfold2_complex_identity as complex_identity
from tests.structure import test_structure_public_helpers as structure_contracts
from tests.structure.test_structure_public_helpers import (
    test_boltz_atom_confidence_is_finite_for_short_uneven_batches_and_multiplicity as _finite,
)
from tests.structure.test_structure_public_helpers import (
    test_boltz_atom_confidence_mapping_preserves_batch_multiplicity_and_atom_order as _order,
)
from tests.unit import test_boltz_checkpoint_io as boltz_checkpoint_contracts
from tests.unit import test_esmfold2_reimplemented_leaves as esmfold2_leaf_contracts
from tests.unit import test_esmfold_api as esmfold_contracts
from tests.unit import test_structure_output_contracts as output_contracts
from tests.unit.test_esmfold2_reimplemented_leaves import (
    test_experimental_top_level_kernel_backend_validates_before_zero_layer_dispatch as _kernel,
)


test_boltz_atom_confidence_is_finite_for_short_uneven_batches_and_multiplicity = _finite
test_boltz_atom_confidence_mapping_preserves_batch_multiplicity_and_atom_order = _order
test_experimental_top_level_kernel_backend_validates_before_zero_layer_dispatch = _kernel

test_boltz_public_helper_is_seeded_and_restores_ambient_rng = (
    structure_contracts.test_boltz_public_helper_is_seeded_and_restores_ambient_rng
)
test_boltz_public_helper_rejects_coerced_seed_types_before_rng_mutation = (
    structure_contracts.test_boltz_public_helper_rejects_coerced_seed_types_before_rng_mutation
)
test_boltz_public_helper_owns_cuda_bf16_autocast_policy = (
    structure_contracts.test_boltz_public_helper_owns_cuda_bf16_autocast_policy
)
test_boltz_public_helper_rejects_non_finite_coordinates = (
    structure_contracts.test_boltz_public_helper_rejects_non_finite_coordinates
)
test_boltz_indexing_matrix_rejects_invalid_public_dimensions = (
    structure_contracts.test_boltz_indexing_matrix_rejects_invalid_public_dimensions
)
test_boltz_piecewise_schedule_validates_and_owns_its_configuration = (
    structure_contracts.test_boltz_piecewise_schedule_validates_and_owns_its_configuration
)
test_boltz_flat_bottom_potential_rejects_invalid_negation_masks = (
    structure_contracts.test_boltz_flat_bottom_potential_rejects_invalid_negation_masks
)
test_boltz_real_features_flow_through_tiny_core_and_structure_loss = (
    structure_contracts.test_boltz_real_features_flow_through_tiny_core_and_structure_loss
)
test_binder_structure_loss_is_finite_and_differentiable = (
    structure_contracts.test_binder_structure_loss_is_finite_and_differentiable
)
test_esmfold_fold_single_uses_linker_masked_mean_plddt = (
    structure_contracts.test_esmfold_fold_single_uses_linker_masked_mean_plddt
)
test_esmfold2_real_features_flow_through_tiny_core_and_tm_loss = (
    structure_contracts.test_esmfold2_real_features_flow_through_tiny_core_and_tm_loss
)
test_pocket_conditioning_is_rejected_instead_of_silently_dropped = (
    esmfold2_leaf_contracts.test_pocket_conditioning_is_rejected_instead_of_silently_dropped
)
test_distogram_conditioning_is_rejected_instead_of_silently_dropped = (
    esmfold2_leaf_contracts.test_distogram_conditioning_is_rejected_instead_of_silently_dropped
)
test_boltz_save_pretrained_defaults_to_safetensors_and_round_trips = (
    boltz_checkpoint_contracts.test_save_pretrained_defaults_to_safetensors_and_round_trips
)
test_boltz_public_forward_honors_output_controls_backward_and_reload = (
    output_contracts.test_boltz_public_forward_honors_output_controls_backward_and_reload
)
test_esmfold_forward_uses_official_plddt_scale = (
    esmfold_contracts.test_forward_uses_official_plddt_scale
)
test_esmfold_infer_preserves_official_multimer_contract = (
    esmfold_contracts.test_infer_preserves_official_multimer_contract
)
test_fast_esmfold_public_forward_honors_output_controls_and_backward = (
    output_contracts.test_fast_esmfold_public_forward_honors_output_controls_and_backward
)
test_fast_esmfold_output_attentions_uses_masked_per_call_eager_fallback = (
    output_contracts.test_fast_esmfold_output_attentions_uses_masked_per_call_eager_fallback
)
test_fast_esmfold_tiny_model_saves_and_reloads_exact_state = (
    output_contracts.test_fast_esmfold_tiny_model_saves_and_reloads_exact_state
)
test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides = (
    output_contracts.test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides
)
test_molecular_round_trip_preserves_identity_and_repeated_chain_boundaries = (
    complex_identity.test_molecular_round_trip_preserves_identity_and_repeated_chain_boundaries
)
test_backbone_state_dict_does_not_mutate_source_atom_mask = (
    complex_identity.test_backbone_state_dict_does_not_mutate_source_atom_mask
)
test_binder_model_identity_records_selected_kernel_and_mixed_parameter_dtypes = (
    binder_contracts.test_binder_model_identity_records_selected_kernel_and_mixed_parameter_dtypes
)


def test_public_binder_workflow_pads_heterogeneous_prepared_atoms_without_truncation() -> None:
    """Run real preparation, batching, and forward wiring through an injected tiny core."""

    from examples import binder_design_fastplms as binder

    class FakeFoldModel:
        input_types = SimpleNamespace(
            ProteinInput=lambda **values: SimpleNamespace(**values),
            StructurePredictionInput=lambda **values: SimpleNamespace(**values),
        )

        def __init__(self) -> None:
            self.prepared_sizes: list[int] = []
            self.forward_batches: list[dict[str, torch.Tensor]] = []

        def prepare_structure_input(
            self,
            input_data: object,
            *,
            seed: int | None,
        ) -> tuple[dict[str, torch.Tensor], list[str]]:
            del input_data, seed
            count = (32, 65)[len(self.prepared_sizes)]
            marker = float(len(self.prepared_sizes) + 1)
            self.prepared_sizes.append(count)
            return (
                {
                    "ref_pos": torch.full((1, count, 3), marker),  # (1, n_atom, xyz=3)
                    "atom_attention_mask": torch.ones(  # (1, n_atom)
                        (1, count),
                        dtype=torch.bool,
                    ),
                },
                [f"chain-{int(marker)}"],
            )

        def forward(self, **inputs: object) -> dict[str, torch.Tensor]:
            ref_pos = inputs["ref_pos"]
            atom_attention_mask = inputs["atom_attention_mask"]
            res_type_soft = inputs["res_type_soft"]
            assert isinstance(ref_pos, torch.Tensor)
            assert isinstance(atom_attention_mask, torch.Tensor)
            assert isinstance(res_type_soft, torch.Tensor)
            self.forward_batches.append(
                {
                    "ref_pos": ref_pos.detach().clone(),
                    "atom_attention_mask": atom_attention_mask.detach().clone(),
                }
            )
            batch_size, token_count = res_type_soft.shape[:2]
            marker = ref_pos[:, 0, 0]  # (b,)
            logits = marker[:, None, None, None].expand(  # (b, l, l, bins=128)
                batch_size,
                token_count,
                token_count,
                128,
            )
            return {"distogram_logits": logits}

        def __call__(self, **inputs: object) -> dict[str, torch.Tensor]:
            return self.forward(**inputs)

    model = FakeFoldModel()
    design = torch.zeros((2, 2, binder.AA_DIMS), dtype=torch.float32)  # (b=2, l=2, aa)
    design[0, :, 0] = 1
    design[1, :, 1] = 1
    result = binder.fold_and_get_distogram(
        model,
        "ACD",
        binder.sequence_to_one_hot("ACD", device="cpu"),
        design,
        seed=7,
    )

    assert model.prepared_sizes == [32, 65]
    assert len(model.forward_batches) == 1
    batch = model.forward_batches[0]
    assert batch["ref_pos"].shape == (2, 96, 3)
    assert batch["atom_attention_mask"].shape == (2, 96)
    torch.testing.assert_close(batch["ref_pos"][0, :32], torch.ones((32, 3)))
    torch.testing.assert_close(batch["ref_pos"][1, :65], torch.full((65, 3), 2.0))
    assert not batch["ref_pos"][0, 32:].any()
    assert not batch["ref_pos"][1, 65:].any()
    assert batch["atom_attention_mask"][0, :32].all()
    assert batch["atom_attention_mask"][1, :65].all()
    assert not batch["atom_attention_mask"][0, 32:].any()
    assert not batch["atom_attention_mask"][1, 65:].any()
    assert result["inputs"]["ref_pos"].shape == (2, 96, 3)
    assert result["chain_info_list"] == [["chain-1"], ["chain-2"]]
    assert result["distogram_logits"][:, 0, 0, 0].tolist() == [1.0, 2.0]


def test_binder_example_rejects_prepared_feature_schema_drift() -> None:
    from examples import binder_design_fastplms as binder

    class StrictModel:
        def forward(self, token_index: torch.Tensor) -> torch.Tensor:
            return token_index

    with pytest.raises(TypeError, match="unexpected_feature"):
        binder._prepare_model_forward_kwargs(
            StrictModel(),
            {
                "token_index": torch.zeros(1),
                "unexpected_feature": torch.zeros(1),
            },
            {"seed": 7},
        )


def test_binder_example_main_wires_explicit_offline_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from examples import binder_design_fastplms as binder

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        binder,
        "run_local",
        lambda arguments: observed.update(vars(arguments)),
    )
    output = tmp_path / "binder-output"

    result = binder.main(
        [
            "--target-sequence",
            "ACD",
            "--binder-sequence",
            "###",
            "--seed",
            "9",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--output-dir",
            str(output),
            "--inversion-model",
            "Custom/inversion",
            "--critic-model",
            "Custom/critic",
            "--lm-model",
            "Custom/lm",
            "--model-revision",
            f"Custom/inversion={'a' * 40}",
            "--model-revision",
            f"Custom/critic={'b' * 40}",
            "--model-revision",
            f"Custom/lm={'c' * 40}",
            "--local-files-only",
            "--kernel-backend",
            "sdpa",
            "--compile-model",
            "--not-antibody",
        ]
    )

    assert result == 0
    assert observed == {
        "target_name": None,
        "target_sequence": "ACD",
        "binder_name": None,
        "binder_sequence": "###",
        "seed": 9,
        "batch_size": 2,
        "steps": 1,
        "output_dir": str(output),
        "inversion_model_names": ["Custom/inversion"],
        "critic_model_names": ["Custom/critic"],
        "lm_model": "Custom/lm",
        "model_revisions": {
            "Custom/inversion": "a" * 40,
            "Custom/critic": "b" * 40,
            "Custom/lm": "c" * 40,
        },
        "local_files_only": True,
        "kernel_backend": "sdpa",
        "compile_model": True,
        "is_antibody": False,
    }


def test_binder_documentation_exposes_pinned_and_offline_loading_contract() -> None:
    documentation = (Path(__file__).resolve().parents[2] / "docs" / "binder_design.md").read_text(
        encoding="utf-8"
    )

    for required_text in (
        "src/fastplms/models.toml",
        "--inversion-model",
        "--critic-model",
        "--lm-model",
        "--model-revision",
        "--local-files-only",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "fastplms_weights_revision",
        "fastplms_runtime_revision",
    ):
        assert required_text in documentation


def test_esmfold2_public_esmc_loaders_propagate_instance_offline_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastplms.models.esmfold2 import modeling_esmfold2 as release_module
    from fastplms.models.esmfold2 import modeling_esmfold2_experimental as experimental_module

    observed: list[tuple[object, str, dict[str, object]]] = []

    def fake_install(model, source: str, **kwargs: object) -> None:
        observed.append((model, source, kwargs))

    monkeypatch.setattr(release_module, "_install_esmc_backbone", fake_install)
    monkeypatch.setattr(experimental_module, "_install_esmc_backbone", fake_install)
    release_stub = SimpleNamespace()
    experimental_stub = SimpleNamespace()

    ESMFold2Model.load_esmc(
        release_stub,
        "Synthyra/ESMplusplus_6B",
        precision="bf16",
        device="cpu",
        local_files_only=True,
    )
    ESMFold2ExperimentalModel.load_esmc(
        experimental_stub,
        "Synthyra/ESMplusplus_6B",
        precision="bf16",
        device="cpu",
        local_files_only=True,
    )

    assert observed == [
        (
            release_stub,
            "Synthyra/ESMplusplus_6B",
            {
                "precision": "bf16",
                "device": "cpu",
                "local_files_only": True,
            },
        ),
        (
            experimental_stub,
            "Synthyra/ESMplusplus_6B",
            {
                "precision": "bf16",
                "device": "cpu",
                "local_files_only": True,
            },
        ),
    ]


def _tiny_esmfold2_config(model_type: str) -> ESMFold2Config:
    atom_token_width = 8
    input_feature_width = atom_token_width // 2 + 2 * NUM_RES_TYPES + 1
    return ESMFold2Config(
        type=model_type,
        d_single=8,
        d_pair=8,
        num_loops=0,
        num_diffusion_samples=1,
        lm_d_model=8,
        lm_num_layers=1,
        inputs={
            "d_inputs": input_feature_width,
            "atom_encoder": {
                "d_atom": 8,
                "d_token": atom_token_width,
                "n_blocks": 0,
                "n_heads": 2,
                "swa_window_size": 32,
                "expansion_ratio": 2,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 1,
            },
        },
        folding_trunk={"n_layers": 0, "n_heads": 2, "dropout": 0.0},
        structure_head={
            "diffusion_module": {
                "c_atom": 8,
                "c_token": 8,
                "c_z": 8,
                "c_s_inputs": input_feature_width,
                "fourier_dim": 8,
                "atom_num_blocks": 0,
                "atom_num_heads": 2,
                "token_num_blocks": 0,
                "token_num_heads": 2,
                "transition_multiplier": 2,
            },
            "distogram_bins": 8,
            "inference_num_steps": 1,
        },
        confidence_head={
            "enabled": False,
            "folding_trunk": {"n_layers": 0, "n_heads": 2, "dropout": 0.0},
            "num_plddt_bins": 4,
            "num_pde_bins": 4,
            "num_pae_bins": 4,
            "distogram_bins": 8,
        },
        msa_encoder={"enabled": False},
        lm_encoder={"enabled": False, "n_layers": 0},
        parcae={"enabled": True, "min_steps": 1, "max_steps": 1, "coda_n_layers": 0},
    )


@pytest.mark.parametrize(
    ("model_class", "model_type"),
    (
        (ESMFold2Model, "release"),
        (ESMFold2ExperimentalModel, "experimental"),
    ),
)
def test_esmfold2_advertised_models_tiny_init_backward_and_save_reload(
    model_class: type[ESMFold2Model] | type[ESMFold2ExperimentalModel],
    model_type: str,
    tmp_path: Path,
) -> None:
    model = model_class(_tiny_esmfold2_config(model_type))
    pair_state = torch.randn(1, 2, 2, 8, requires_grad=True)  # (b=1, l=2, l, d_pair=8)
    logits = model.distogram_head(  # (b, l, l, distogram_bins=8)
        pair_state + pair_state.transpose(1, 2)
    )
    logits.square().mean().backward()

    assert torch.isfinite(logits).all()
    assert pair_state.grad is not None and torch.isfinite(pair_state.grad).all()
    save_dir = tmp_path / model_class.__name__
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(
        save_dir,
        local_files_only=True,
        load_esmc=False,
    )
    assert set(reloaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0.0, atol=0.0)
