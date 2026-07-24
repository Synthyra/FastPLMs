"""Parity checks for independently implemented ESMFold2 source utilities."""

from __future__ import annotations

import importlib.util
import random
import sys
import types
import numpy as np
import pytest
import torch
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastplms.models.esmfold2 import configuration_esmfold2 as local_configuration
from fastplms.models.esmfold2 import esmfold2_affine3d as local_affine
from fastplms.models.esmfold2 import esmfold2_conformers as local_conformers
from fastplms.models.esmfold2 import esmfold2_constants as local_constants
from fastplms.models.esmfold2 import esmfold2_metrics as local_metrics
from fastplms.models.esmfold2 import esmfold2_misc as local_misc
from fastplms.models.esmfold2 import esmfold2_molecular_complex as local_complex
from fastplms.models.esmfold2 import esmfold2_output as local_output
from fastplms.models.esmfold2 import esmfold2_paired_msa as local_paired_msa
from fastplms.models.esmfold2 import esmfold2_prepare_input as local_prepare_input
from fastplms.models.esmfold2 import esmfold2_processor as local_processor
from fastplms.models.esmfold2 import esmfold2_protein_structure as local_structure
from fastplms.models.esmfold2 import esmfold2_residue_constants as local_residue_constants
from fastplms.models.esmfold2 import esmfold2_types as local_types
from fastplms.models.esmfold2.esmfold2_msa import MSA
from fastplms.models.esmfold2.esmfold2_parsing import FastaEntry


pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
BIOHUB_ESM = ROOT / "vendor/upstream/biohub-esm/esm"
BIOHUB_TRANSFORMERS = ROOT / "vendor/upstream/biohub-transformers/src/transformers/models/esmfold2"
_MISSING = object()


def _package(name: str) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


@contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def _load_source(
    module_name: str,
    path: Path,
    aliases: dict[str, types.ModuleType],
) -> types.ModuleType:
    assert path.is_file(), f"pinned parity source is missing: {path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _temporary_modules({**aliases, module_name: module}):
        spec.loader.exec_module(module)
    return module


def _biohub_packages() -> dict[str, types.ModuleType]:
    return {
        "esm": _package("esm"),
        "esm.models": _package("esm.models"),
        "esm.models.esmfold2": _package("esm.models.esmfold2"),
        "esm.utils": _package("esm.utils"),
        "esm.utils.msa": _package("esm.utils.msa"),
        "esm.utils.structure": _package("esm.utils.structure"),
    }


def _msa_compatibility_module() -> types.ModuleType:
    module = types.ModuleType("esm.utils.msa.msa")
    module.MSA = MSA
    module.is_a3m_insertion = lambda character: character == "." or character.islower()
    return module


def _load_official_configuration() -> types.ModuleType:
    import transformers.configuration_utils as configuration_utils

    root_name = "_fastplms_pinned_transformers"
    aliases = {
        root_name: _package(root_name),
        f"{root_name}.models": _package(f"{root_name}.models"),
        f"{root_name}.models.esmfold2": _package(f"{root_name}.models.esmfold2"),
        f"{root_name}.configuration_utils": configuration_utils,
    }
    return _load_source(
        f"{root_name}.models.esmfold2.configuration_esmfold2",
        BIOHUB_TRANSFORMERS / "configuration_esmfold2.py",
        aliases,
    )


def _load_official_structure() -> types.ModuleType:
    aliases = {
        **_biohub_packages(),
        "esm.utils.residue_constants": local_residue_constants,
        "esm.utils.misc": local_misc,
        "esm.utils.structure.affine3d": local_affine,
    }
    return _load_source(
        "_fastplms_pinned_biohub_protein_structure",
        BIOHUB_ESM / "utils/structure/protein_structure.py",
        aliases,
    )


def _load_official_metrics(official_structure: types.ModuleType) -> types.ModuleType:
    aliases = {
        **_biohub_packages(),
        "esm.utils.residue_constants": local_residue_constants,
        "esm.utils.misc": local_misc,
        "esm.utils.structure.protein_structure": official_structure,
    }
    return _load_source(
        "_fastplms_pinned_biohub_metrics",
        BIOHUB_ESM / "utils/structure/metrics.py",
        aliases,
    )


def _load_official_paired_msa() -> types.ModuleType:
    aliases = {
        **_biohub_packages(),
        "esm.models.esmfold2.constants": local_constants,
        "esm.utils.msa.msa": _msa_compatibility_module(),
    }
    return _load_source(
        "_fastplms_pinned_biohub_paired_msa",
        BIOHUB_ESM / "models/esmfold2/paired_msa.py",
        aliases,
    )


def _load_official_processor() -> types.ModuleType:
    aliases = {
        **_biohub_packages(),
        "esm.models.esmfold2.conformers": local_conformers,
        "esm.models.esmfold2.output": local_output,
        "esm.models.esmfold2.prepare_input": local_prepare_input,
        "esm.models.esmfold2.types": local_types,
        "esm.utils.structure.molecular_complex": local_complex,
    }
    return _load_source(
        "_fastplms_pinned_biohub_processor",
        BIOHUB_ESM / "models/esmfold2/processor.py",
        aliases,
    )


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # actual: (...), expected: (...)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_configuration_matches_pinned_biohub_schema() -> None:
    official = _load_official_configuration()
    kwargs = {
        "type": "release",
        "d_single": 320,
        "d_pair": 192,
        "esmc_id": "Synthyra/ESMplusplus_6B",
        "inputs": {"d_inputs": 777, "atom_encoder": {"n_blocks": 5}},
        "folding_trunk": {"n_layers": 8, "n_heads": 4},
        "structure_head": {"diffusion_module": {"token_num_blocks": 3}},
        "confidence_head": {"folding_trunk": {"n_layers": 2}},
        "msa_encoder": {"enabled": True, "d_msa": 64},
        "parcae": {"max_steps": None},
        "lm_encoder": {"per_loop_lm_dropout": False},
        "msa_encoder_overwrite": False,
    }
    actual = local_configuration.ESMFold2Config(**kwargs)
    expected = official.ESMFold2Config(**kwargs)

    scalar_fields = (
        "type",
        "d_single",
        "d_pair",
        "n_relative_residx_bins",
        "n_relative_chain_bins",
        "num_loops",
        "num_diffusion_samples",
        "disable_msa_features",
        "lm_dropout",
        "force_lm_dropout_during_inference",
        "lm_d_model",
        "lm_num_layers",
        "esmc_id",
        "msa_encoder_overwrite",
    )
    assert {name: getattr(actual, name) for name in scalar_fields} == {
        name: getattr(expected, name) for name in scalar_fields
    }
    nested_fields = (
        "inputs",
        "folding_trunk",
        "structure_head",
        "confidence_head",
        "msa_encoder",
        "parcae",
        "lm_encoder",
    )
    assert {name: asdict(getattr(actual, name)) for name in nested_fields} == {
        name: asdict(getattr(expected, name)) for name in nested_fields
    }


def test_protein_geometry_matches_pinned_biohub_on_h100() -> None:
    assert torch.cuda.is_available(), "the ESMFold2 compliance suite requires CUDA"
    official = _load_official_structure()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260714)
    # mobile: (3, 12, 3)
    mobile = torch.randn((3, 12, 3), generator=generator, device=device)
    target = mobile + 0.05 * torch.randn(
        mobile.shape,
        generator=generator,
        device=device,
    )
    # mask: (3, 12)
    mask = torch.tensor(
        [[1] * 12, [1] * 9 + [0] * 3, [1, 0] * 6],
        dtype=torch.bool,
        device=device,
    )

    actual_alignment = local_structure.compute_alignment_tensors(mobile, target, mask)
    expected_alignment = official.compute_alignment_tensors(mobile, target, mask)
    for actual, expected in zip(actual_alignment, expected_alignment, strict=True):
        _assert_equal(actual, expected)

    actual_affine, actual_rmsd = local_structure.compute_affine_and_rmsd(mobile, target, mask)
    expected_affine, expected_rmsd = official.compute_affine_and_rmsd(mobile, target, mask)
    _assert_equal(actual_affine.tensor, expected_affine.tensor)
    _assert_equal(actual_rmsd, expected_rmsd)
    _assert_equal(
        local_structure.compute_gdt_ts_no_alignment(mobile, target, mask),
        official.compute_gdt_ts_no_alignment(mobile, target, mask),
    )


def test_structure_metrics_match_pinned_biohub_on_h100() -> None:
    assert torch.cuda.is_available(), "the ESMFold2 compliance suite requires CUDA"
    official = _load_official_metrics(_load_official_structure())
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(712)
    # predicted: (2, 10, 3)
    predicted = torch.randn((2, 10, 3), generator=generator, device=device)
    target = predicted + 0.2 * torch.randn(
        predicted.shape,
        generator=generator,
        device=device,
    )
    # atom_mask: (2, 10)
    atom_mask = torch.tensor(
        [[1] * 10, [1] * 7 + [0] * 3],
        dtype=torch.float32,
        device=device,
    )
    # sequence_id: (2, 10)
    sequence_id = torch.tensor(
        [[0] * 5 + [1] * 5, [0] * 4 + [1] * 6],
        device=device,
    )
    for per_residue in (False, True):
        _assert_equal(
            local_metrics.compute_lddt(
                predicted,
                target,
                atom_mask,
                per_residue=per_residue,
                sequence_id=sequence_id,
            ),
            official.compute_lddt(
                predicted,
                target,
                atom_mask,
                per_residue=per_residue,
                sequence_id=sequence_id,
            ),
        )

    # predictions: (2, 14, 14)
    predictions = torch.rand((2, 14, 14), generator=generator, device=device)
    # targets: (...)
    targets = torch.randint(0, 2, (2, 14, 14), generator=generator, device=device).float()
    targets[1, 11:] = -1
    # lengths: (2,)
    lengths = torch.tensor([14, 11], device=device)
    actual_contacts = local_metrics.contact_precision(
        predictions, targets, lengths, minsep=3, maxsep=10
    )
    expected_contacts = official.contact_precision(
        predictions, targets, lengths, minsep=3, maxsep=10
    )
    for name, expected in expected_contacts.items():
        _assert_equal(actual_contacts[name], expected)


def test_paired_msa_matches_pinned_biohub() -> None:
    official = _load_official_paired_msa()
    msa_a = MSA(
        [
            FastaEntry("query", "ACD-E"),
            FastaEntry("a key=10", "AqCD-E"),
            FastaEntry("b key=20", "ACD-E"),
            FastaEntry("unpaired", "A-CDX"),
        ]
    )
    msa_b = MSA(
        [
            FastaEntry("query", "FGHI"),
            FastaEntry("a key=10", "FGHI"),
            FastaEntry("b key=20", "FgGHI"),
            FastaEntry("extra key=10", "F-HI"),
        ]
    )
    arguments = {
        "chain_msas": {1: msa_a, 2: msa_b, 3: None},
        "chain_query_res_types": {
            1: np.asarray([0, 1, 2, 3, 4]),
            2: np.asarray([5, 6, 7, 8]),
            3: np.asarray([9, 10]),
        },
        "token_asym_ids": np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
        "token_res_ids": np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1]),
        "max_pairs": 8,
        "max_total": 12,
        "max_seqs": 10,
    }
    actual = local_paired_msa.construct_paired_msa(**arguments)
    official_arguments = {
        **arguments,
        "chain_msas": {
            chain_id: (
                None
                if msa is None
                else types.SimpleNamespace(
                    entries=msa.entries,
                    depth=msa.depth,
                    deletions=None,
                )
            )
            for chain_id, msa in arguments["chain_msas"].items()
        },
    }
    expected = official.construct_paired_msa(**official_arguments)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def _random_trace(module: types.ModuleType) -> tuple[Any, Any]:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with module._seed_context(19):
        inside = (random.random(), float(np.random.random()), torch.rand(3, device="cuda"))
    after = (random.random(), float(np.random.random()), torch.rand(3, device="cuda"))
    return inside, after


def test_processor_cleaning_and_rng_match_pinned_biohub_on_h100() -> None:
    assert torch.cuda.is_available(), "the ESMFold2 compliance suite requires CUDA"
    official = _load_official_processor()
    source_msa = MSA.from_sequences(["AAA-AAA-BBB", "AAA-AAA-BBB"])
    input_value = local_types.StructurePredictionInput(
        sequences=[
            local_types.ProteinInput(
                id=["entity"],
                sequence="AAA|AAA|BBB",
                modifications=[local_types.Modification(position=1, ccd="MSE")],
                msa=source_msa,
            )
        ]
    )
    assert local_processor.clean_esmfold2_input(input_value) == official.clean_esmfold2_input(
        input_value
    )

    actual_trace = _random_trace(local_processor)
    expected_trace = _random_trace(official)
    assert actual_trace[0][:2] == expected_trace[0][:2]
    assert actual_trace[1][:2] == expected_trace[1][:2]
    _assert_equal(actual_trace[0][2], expected_trace[0][2])
    _assert_equal(actual_trace[1][2], expected_trace[1][2])
