from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from fastplms.registry import load_model_registry
from tools.conversion import (
    StateTransformError,
    apply_state_transform,
    available_state_transforms,
)
from tools.conversion.extract_esmfold2_geometry import extract_geometry
from tools.conversion.state_validation import assert_state_dict_equal

ROOT = Path(__file__).resolve().parents[2]


def test_every_manifest_state_transform_has_a_pure_implementation() -> None:
    registry = load_model_registry()
    declared = {family.state_transform for family in registry.families.values()}
    assert declared == set(available_state_transforms())


def test_conversion_tools_contain_no_hub_mutation_or_authentication_code() -> None:
    forbidden = re.compile(
        r"push_to_hub|create_repo|delete_repo|upload_(?:file|folder)|"
        r"\blogin\s*\(|\bHfApi\b|update_HF|snapshot_download|hf_hub_download"
    )
    files = sorted((ROOT / "tools" / "conversion").glob("*.py"))
    assert {path.name for path in files} == {
        "__init__.py",
        "extract_esmfold2_geometry.py",
        "state_transforms.py",
        "state_validation.py",
    }
    for path in files:
        assert forbidden.search(path.read_text(encoding="utf-8")) is None, path


def test_esmfold2_geometry_extractor_is_literal_only_and_reproducible(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "vendor"
        / "upstream"
        / "biohub-transformers"
        / "src"
        / "transformers"
        / "models"
        / "esmfold2"
        / "protein_utils.py"
    )
    expected_path = (
        ROOT / "src" / "fastplms" / "models" / "esmfold2" / "protein_reference_geometry.json"
    )
    generated = (
        json.dumps(
            extract_geometry(source),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert generated == expected_path.read_text(encoding="utf-8")

    marker = tmp_path / "executed"
    inert_source = tmp_path / "inert.py"
    inert_source.write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "PROTEIN_REF_POS: dict = {'UNK': {'CA': (0.0, 0.0, 0.0)}}\n",
        encoding="utf-8",
    )
    assert extract_geometry(inert_source)["residues"] == {"UNK": {"CA": (0.0, 0.0, 0.0)}}
    assert not marker.exists()

    dynamic_source = tmp_path / "dynamic.py"
    dynamic_source.write_text(
        "PROTEIN_REF_POS: dict = make_geometry()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a Python literal"):
        extract_geometry(dynamic_source)


@pytest.mark.parametrize(
    "transform_id",
    [
        "identity",
        "e1_to_fastplms_v1",
        "dplm_to_fastplms_v1",
        "dplm2_to_fastplms_v1",
        "ankh_t5_to_fastplms_v1",
    ],
)
def test_identity_key_transforms_are_value_exact_and_non_aliasing(
    transform_id: str,
) -> None:
    source = {
        "decoder.block.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "encoder.embed_tokens.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "lm_head.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
    }
    transformed = apply_state_transform(
        transform_id,
        source,
        expected_keys=source,
    )

    assert_state_dict_equal(source, transformed, context=transform_id)
    assert list(transformed) == sorted(source)
    for key in source:
        assert transformed[key].data_ptr() != source[key].data_ptr()


def test_esm2_transform_matches_parity_mapping_and_is_idempotent() -> None:
    from tests.parity.support.state_transforms import transform_state

    source = {
        "embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0]]),
        "layers.0.self_attn.rot_emb.inv_freq": torch.tensor([0.5, 0.25]),
        "layers.0.self_attn.out_proj.bias": torch.tensor([3.0]),
        "layers.0.self_attn_layer_norm.weight": torch.tensor([4.0]),
        "layers.0.fc1.weight": torch.tensor([[5.0]]),
        "layers.0.fc2.bias": torch.tensor([6.0]),
        "layers.0.final_layer_norm.weight": torch.tensor([7.0]),
        "emb_layer_norm_after.weight": torch.tensor([8.0]),
        "contact_head.regression.weight": torch.tensor([[9.0]]),
        "lm_head.dense.weight": torch.tensor([[10.0]]),
        "lm_head.weight": torch.tensor([[11.0]]),
        "lm_head.bias": torch.tensor([12.0]),
    }
    expected = transform_state("esm2_hf_to_fastplms_v1", source)
    transformed = apply_state_transform(
        "esm2_hf_to_fastplms_v1",
        source,
        expected_keys=expected,
    )

    assert_state_dict_equal(expected, transformed, context="ESM2 conversion")
    assert torch.equal(
        transformed["esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"],
        source["layers.0.self_attn.rot_emb.inv_freq"],
    )
    assert transformed["lm_head.bias"].data_ptr() != source["lm_head.bias"].data_ptr()
    assert transformed["lm_head.decoder.bias"].data_ptr() != source["lm_head.bias"].data_ptr()
    assert transformed["lm_head.bias"].data_ptr() != transformed["lm_head.decoder.bias"].data_ptr()

    canonical = apply_state_transform(
        "esm2_hf_to_fastplms_v1",
        transformed,
        expected_keys=transformed,
    )
    assert_state_dict_equal(transformed, canonical, context="canonical ESM2 conversion")


def test_esmc_transform_maps_official_keys_exactly() -> None:
    source = {
        "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight": torch.tensor([1.0, 2.0]),
        "esmc.transformer.blocks.0.ffn.fc1_weight": torch.tensor([[3.0, 4.0]]),
        "esmc.transformer.blocks.0.rotary._extra_state": torch.tensor(1),
        "lm_head.weight": torch.tensor([[5.0, 6.0]]),
    }
    expected = {
        "sequence_head.weight": source["lm_head.weight"],
        "transformer.blocks.0.attn.layernorm_qkv.0.weight": source[
            "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight"
        ],
        "transformer.blocks.0.ffn.1.weight": source["esmc.transformer.blocks.0.ffn.fc1_weight"],
    }

    transformed = apply_state_transform(
        "esmc_to_fastplms_v1",
        source,
        expected_keys=expected,
    )
    assert_state_dict_equal(expected, transformed, context="ESMC conversion")


def test_esmfold_transform_maps_native_state_and_removes_untrained_heads() -> None:
    from tests.parity.support.state_transforms import transform_state

    source = {
        "esm.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "esm.layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0]]),
        "esm.contact_head.regression.weight": torch.tensor([[2.0]]),
        "esm.contact_head.regression.bias": torch.tensor([2.5]),
        "esm.lm_head.weight": torch.tensor([[3.0, 4.0]]),
        "trunk.blocks.0.weight": torch.tensor([5.0]),
        "trunk.structure_module.atom_mask": torch.tensor([True]),
        "positional_encoding._float_tensor": torch.tensor([6.0]),
        "lm_head.weight": torch.tensor([[7.0, 8.0]]),
    }
    expected = {
        "esm.embeddings.word_embeddings.weight": source["esm.embed_tokens.weight"],
        "esm.encoder.layer.0.attention.self.query.weight": source[
            "esm.layers.0.self_attn.q_proj.weight"
        ],
        "trunk.blocks.0.weight": source["trunk.blocks.0.weight"],
        "lm_head.weight": source["lm_head.weight"],
    }
    transformed = apply_state_transform(
        "esmfold_meta_to_fastplms_v1",
        source,
        expected_keys=expected,
    )
    assert_state_dict_equal(expected, transformed, context="native ESMFold conversion")
    assert_state_dict_equal(
        expected,
        transform_state("esmfold_meta_to_fastplms_v1", source),
        context="independent native ESMFold parity transform",
    )

    canonical_with_obsolete_head = {
        **transformed,
        "mlm_head.weight": torch.tensor([[9.0]]),
        "esm.contact_head.regression.weight": torch.tensor([[10.0]]),
        "esm.contact_head.regression.bias": torch.tensor([11.0]),
        "trunk.structure_module.atom_mask": torch.tensor([True]),
    }
    canonical = apply_state_transform(
        "esmfold_meta_to_fastplms_v1",
        canonical_with_obsolete_head,
        expected_keys=expected,
    )
    assert_state_dict_equal(expected, canonical, context="canonical ESMFold conversion")
    assert_state_dict_equal(
        expected,
        transform_state("esmfold_meta_to_fastplms_v1", canonical_with_obsolete_head),
        context="independent canonical ESMFold parity transform",
    )
    assert not any(name.startswith("esm.contact_head.") for name in transformed)
    assert not any(name.startswith("esm.contact_head.") for name in canonical)


def test_esm3_transform_adds_only_the_fastplms_wrapper_prefix() -> None:
    source = {"encoder.sequence_embed.weight": torch.arange(6).reshape(2, 3)}
    expected = {"esm3.encoder.sequence_embed.weight": source["encoder.sequence_embed.weight"]}
    transformed = apply_state_transform(
        "esm3_to_fastplms_v1",
        source,
        expected_keys=expected,
    )
    assert_state_dict_equal(expected, transformed, context="ESM3 conversion")


def test_boltz2_transform_selects_only_the_declared_inference_core() -> None:
    source = {
        "model.module.input_embedder.weight": torch.tensor([1.0]),
        "model.template_module.weight": torch.tensor([2.0]),
        "ema.input_embedder.weight": torch.tensor([3.0]),
    }
    expected = {"core.input_embedder.weight": source["model.module.input_embedder.weight"]}
    transformed = apply_state_transform(
        "boltz2_inference_core_v1",
        source,
        expected_keys=expected,
    )
    assert_state_dict_equal(expected, transformed, context="Boltz2 conversion")

    with pytest.raises(StateTransformError, match="undeclared non-inference"):
        apply_state_transform(
            "boltz2_inference_core_v1",
            {**source, "model.training_only.weight": torch.tensor([4.0])},
            expected_keys=expected,
        )


@pytest.mark.parametrize(
    "candidate",
    [
        {"weight": torch.tensor([1.0]), "extra": torch.tensor([1.0])},
        {},
        {"weight": torch.tensor([1], dtype=torch.int64)},
        {"weight": torch.tensor([2.0])},
    ],
)
def test_exact_state_validation_rejects_schema_or_value_drift(
    candidate: dict[str, torch.Tensor],
) -> None:
    reference = {"weight": torch.tensor([1.0])}
    with pytest.raises(AssertionError, match="state_dict parity failed"):
        assert_state_dict_equal(reference, candidate, context="exact conversion")
