"""Actual offline Transformers AutoClass dispatch from tiny remote-code artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import pytest
from pathlib import Path
from typing import Any

from fastplms.models.ankh.modeling_ankh import FastAnkhConfig
from fastplms.models.boltz.modeling_boltz2 import Boltz2Config
from fastplms.models.dplm.modeling_dplm import DPLMConfig
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config
from fastplms.models.e1.modeling_e1 import E1Config
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig
from fastplms.models.esm3.modeling_esm3 import FastESM3Config
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmFoldConfig
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.registry import ModelFamily, get_model_registry
from tools.artifacts.build import (
    ArtifactError,
    _artifact_auto_map,
    _runtime_source_entries,
    _write_bootstrap,
    _write_runtime_bundle,
    _write_runtime_snapshot,
)
from tools.artifacts.offline_probe import _CPU_CONTRACT_MARKER, _runtime_site_packages


_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "tools" / "artifacts" / "offline_probe.py"
_STRUCTURE_FAMILIES = frozenset({"boltz2", "esmfold", "esmfold2"})
_REMOTE_RESAVE_FAMILIES = frozenset({"ankh", "dplm", "dplm2", "esm2", "esm3", "esm_plusplus"})


def _transformer_values(vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 16,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": min(7, vocab_size - 1),
        "position_embedding_type": "rotary",
        "attn_backend": "eager",
        "num_labels": 3,
    }


def _tiny_esmfold_config() -> FastEsmFoldConfig:
    return FastEsmFoldConfig(
        vocab_size=33,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        is_folding_model=True,
        attn_backend="eager",
        esmfold_config={
            "fp16_esm": False,
            "bypass_lm": True,
            "lddt_head_hid_dim": 4,
            "trunk": {
                "num_blocks": 1,
                "sequence_state_dim": 8,
                "pairwise_state_dim": 4,
                "sequence_head_width": 4,
                "pairwise_head_width": 2,
                "position_bins": 4,
                "max_recycles": 1,
                "chunk_size": None,
                "structure_module": {
                    "sequence_dim": 8,
                    "pairwise_dim": 4,
                    "ipa_dim": 2,
                    "resnet_dim": 4,
                    "num_heads_ipa": 2,
                    "num_qk_points": 1,
                    "num_v_points": 1,
                    "dropout_rate": 0.0,
                    "num_blocks": 1,
                    "num_transition_layers": 1,
                    "num_resnet_blocks": 1,
                    "num_angles": 7,
                },
            },
        },
    )


def _tiny_esmfold2_config() -> ESMFold2Config:
    atom_token_width = 8
    input_feature_width = atom_token_width // 2 + 2 * NUM_RES_TYPES + 1
    return ESMFold2Config(
        type="release",
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


def _tiny_config(family_id: str) -> Any:
    if family_id == "esm2":
        values = _transformer_values(16)
        values["position_embedding_type"] = "absolute"
        return FastEsmConfig(**values)
    if family_id == "esm_plusplus":
        return ESMplusplusConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            dropout=0.0,
            pad_token_id=1,
            mask_token_id=7,
            num_labels=3,
            attn_backend="eager",
        )
    if family_id == "esm3":
        return FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    if family_id == "e1":
        config = E1Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_num_sequences=4,
            max_num_positions_within_seq=16,
            max_num_positions_global=16,
            attn_backend="sdpa",
            dtype="float32",
            num_labels=3,
        )
        config.use_cache = False
        return config
    if family_id == "dplm":
        return DPLMConfig(**_transformer_values(16))
    if family_id == "dplm2":
        values = _transformer_values(64)
        values["attn_backend"] = "sdpa"
        return DPLM2Config(**values)
    if family_id == "ankh":
        return FastAnkhConfig(
            vocab_size=16,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_heads=2,
            num_layers=1,
            num_decoder_layers=1,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            attn_backend="eager",
            use_cache=False,
            num_labels=3,
        )
    if family_id == "boltz2":
        return Boltz2Config(core_kwargs={"width": 3})
    if family_id == "esmfold":
        return _tiny_esmfold_config()
    if family_id == "esmfold2":
        return _tiny_esmfold2_config()
    raise AssertionError(f"Missing tiny configuration for manifest family {family_id!r}.")


def _write_cpu_artifact(root: Path, family: ModelFamily) -> Path:
    registry = get_model_registry()
    spec = registry[family.representative]
    if dict(spec.auto_map) != dict(family.auto_map):
        raise AssertionError(f"Representative {spec.id} has a model-specific AutoMap override.")

    artifact = root / family.id / "artifact"
    artifact.mkdir(parents=True)
    config = _tiny_config(family.id)
    config.auto_map = _artifact_auto_map(spec)
    config.fastplms_cpu_contract_only = True
    config.save_pretrained(artifact)
    config_path = artifact / "config.json"
    artifact_config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_config["auto_map"] = _artifact_auto_map(spec)
    config_path.write_text(
        json.dumps(artifact_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact / _CPU_CONTRACT_MARKER).write_text(
        json.dumps(
            {"release_artifact": False, "schema_version": 1, "scope": "tests/cpu"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    runtime_root = root / family.id / "runtime" / "fastplms"
    payloads = {
        target.as_posix(): source.read_bytes()
        for source, target in _runtime_source_entries(_ROOT, spec)
    }
    _write_runtime_snapshot(runtime_root, payloads)
    runtime_hash = _write_runtime_bundle(artifact / "fastplms_bundle.py", runtime_root)
    _write_bootstrap(artifact / "modeling_fastplms.py", spec, runtime_hash)
    return artifact


def _case_payload(family: ModelFamily) -> list[dict[str, object]]:
    return [
        {
            "auto_class": auto_class,
            "class_path": class_path,
            "expected_missing_key_prefixes": [],
            "expected_unexpected_key_prefixes": [],
        }
        for auto_class, class_path in sorted(family.auto_map.items())
    ]


def _run_family_probe(
    root: Path,
    family: ModelFamily,
    artifact: Path,
) -> subprocess.CompletedProcess[str]:
    family_root = root / family.id
    cases_path = family_root / "cases.json"
    output_path = family_root / "results.json"
    cases_path.write_text(
        json.dumps(_case_payload(family), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        str(_PROBE),
        "--artifact",
        str(artifact),
        "--family",
        family.id,
        "--bf16-execution",
        family.bf16_execution,
        "--cases-file",
        str(cases_path),
        "--implementation",
        "artifact",
        "--output",
        str(output_path),
        "--tiny-cpu-contract",
    ]
    for site_packages in _runtime_site_packages():
        command.extend(("--runtime-site-package", str(site_packages)))
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(family_root / "hf-home"),
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(family_root / "modules"),
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return subprocess.run(
        command,
        cwd=family_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


_FAMILY_IDS = tuple(sorted(get_model_registry().families))


@pytest.mark.parametrize("family_id", _FAMILY_IDS)
def test_every_family_dispatches_all_advertised_remote_autoclasses_offline(
    family_id: str,
    tmp_path: Path,
) -> None:
    registry = get_model_registry()
    family = registry.families[family_id]
    artifact = _write_cpu_artifact(tmp_path, family)
    completed = _run_family_probe(tmp_path, family, artifact)
    output_path = tmp_path / family_id / "results.json"
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output_path.is_file()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(output) == set(family.auto_map)

    for auto_class, result in output.items():
        expected = family.auto_map[auto_class]
        assert result["class"] == expected.rsplit(".", maxsplit=1)[1]
        if auto_class != "AutoConfig" and family_id not in _STRUCTURE_FAMILIES:
            assert result["resized_vocab"] >= 9
            assert result["tuple_fields"] >= 1
        if family_id in _STRUCTURE_FAMILIES and auto_class != "AutoConfig":
            assert result["structure_forward_delegated"] is True
    if family_id in _REMOTE_RESAVE_FAMILIES:
        assert output["AutoModel"]["resaved"] is True
    marker = json.loads((artifact / _CPU_CONTRACT_MARKER).read_text(encoding="utf-8"))
    assert marker == {
        "release_artifact": False,
        "schema_version": 1,
        "scope": "tests/cpu",
    }
    assert not (artifact / "artifact-manifest.json").exists()
    assert not (artifact / "source-record.json").exists()
    assert not (artifact / "runtime-attestation.json").exists()


def test_structure_auto_dispatch_is_complemented_by_public_forward_contracts() -> None:
    from tests.cpu import test_structure_contracts as structure_contracts

    for test_name in (
        "test_boltz_public_forward_honors_output_controls_backward_and_reload",
        "test_fast_esmfold_public_forward_honors_output_controls_and_backward",
        "test_fast_esmfold_tiny_model_saves_and_reloads_exact_state",
        "test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides",
        "test_esmfold2_advertised_models_tiny_init_backward_and_save_reload",
    ):
        assert callable(getattr(structure_contracts, test_name))


def test_grouped_remote_dispatch_covers_exactly_37_family_entries() -> None:
    registry = get_model_registry()
    assert sum(len(family.auto_map) for family in registry.families.values()) == 37


def test_isolated_probe_blocks_reference_reads_before_remote_code_exec(
    tmp_path: Path,
) -> None:
    registry = get_model_registry()
    family = registry.families["esm2"]
    artifact = _write_cpu_artifact(tmp_path, family)
    bootstrap = artifact / "modeling_fastplms.py"
    forbidden = _ROOT / "vendor" / "upstream" / "forbidden-cpu-probe-read"
    bootstrap.write_text(
        "from pathlib import Path\n"
        f"Path({str(forbidden)!r}).read_bytes()\n" + bootstrap.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    completed = _run_family_probe(tmp_path, family, artifact)

    assert completed.returncode != 0
    assert "may not access submodule/reference path" in completed.stdout + completed.stderr
