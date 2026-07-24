"""Fast ESMFold2 schema and protein-feature regression gates."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import torch

from fastplms.models.esmfold2 import esmfold2_constants as molecular_schema
from fastplms.models.esmfold2 import esmfold2_constants_esm3 as token_schema
from fastplms.models.esmfold2.esmfold2_parsing import parse_fasta, read_sequences
from fastplms.models.esmfold2.protein_utils import prepare_protein_features


_OFFICIAL_FEATURE_DIGEST = "255bab0048c0c8984e03eb878b7a9c88dfa0ad083983e2641d6a4a0eb8e39fc5"
_OFFICIAL_SCHEMA_DIGEST = "34b14af14c0f640034eec8174f80081234a98d617a5dbf9682a06a04465dc024"
_DIGEST_SEQUENCES = ("ACGX", "ARNDCQEGHILKMFPSTWYV", "M" * 31, "M" * 33)

_MOLECULAR_SCHEMA_NAMES = (
    "MOL_TYPE_PROTEIN",
    "MOL_TYPE_DNA",
    "MOL_TYPE_RNA",
    "MOL_TYPE_NONPOLYMER",
    "PROTEIN_RESIDUE_TO_RES_TYPE",
    "PROTEIN_UNK_RES_TYPE",
    "RNA_RESIDUE_TO_RES_TYPE",
    "RNA_UNK_RES_TYPE",
    "DNA_RESIDUE_TO_RES_TYPE",
    "DNA_UNK_RES_TYPE",
    "GAP_RES_TYPE",
    "PROTEIN_3TO1",
    "PROTEIN_1TO3",
    "DNA_1TO3",
    "RNA_1TO3",
    "ESM_PROTEIN_VOCAB",
    "DNA_RNA_LIGAND_INPUT_ID",
    "MSA_PAD_TOKEN_ID",
    "MSA_GAP_TOKEN_ID",
    "RES_TYPE_TO_CCD",
    "ELEMENT_TO_ATOMIC_NUM",
    "ELEMENT_NUMBER_TO_SYMBOL",
    "PROTEIN_HEAVY_ATOMS",
    "DNA_HEAVY_ATOMS",
    "RNA_HEAVY_ATOMS",
    "DNA_BACKBONE_ATOMS",
    "RNA_BACKBONE_ATOMS",
)
_TOKEN_SCHEMA_NAMES = (
    "SEQUENCE_BOS_TOKEN",
    "SEQUENCE_PAD_TOKEN",
    "SEQUENCE_EOS_TOKEN",
    "SEQUENCE_CHAINBREAK_TOKEN",
    "SEQUENCE_MASK_TOKEN",
    "VQVAE_CODEBOOK_SIZE",
    "VQVAE_SPECIAL_TOKENS",
    "VQVAE_DIRECTION_LOSS_BINS",
    "VQVAE_PAE_BINS",
    "VQVAE_MAX_PAE_BIN",
    "VQVAE_PLDDT_BINS",
    "STRUCTURE_MASK_TOKEN",
    "STRUCTURE_BOS_TOKEN",
    "STRUCTURE_EOS_TOKEN",
    "STRUCTURE_PAD_TOKEN",
    "STRUCTURE_CHAINBREAK_TOKEN",
    "STRUCTURE_UNDEFINED_TOKEN",
    "SASA_PAD_TOKEN",
    "SS8_PAD_TOKEN",
    "INTERPRO_PAD_TOKEN",
    "RESIDUE_PAD_TOKEN",
    "CHAIN_BREAK_STR",
    "SEQUENCE_BOS_STR",
    "SEQUENCE_EOS_STR",
    "MASK_STR_SHORT",
    "SEQUENCE_MASK_STR",
    "SASA_MASK_STR",
    "SS8_MASK_STR",
    "SEQUENCE_VOCAB",
    "SEQUENCE_STANDARD_AA_MIN_TOKEN",
    "SEQUENCE_STANDARD_AA_MAX_TOKEN",
    "SSE_8CLASS_VOCAB",
    "SSE_3CLASS_VOCAB",
    "SSE_8CLASS_TO_3CLASS_MAP",
    "SASA_DISCRETIZATION_BOUNDARIES",
    "MAX_RESIDUE_ANNOTATIONS",
    "TFIDF_VECTOR_SIZE",
    "FUNCTION_TOKENS_DEPTH",
)


def _feature_digest() -> str:
    digest = hashlib.sha256()
    for sequence in _DIGEST_SEQUENCES:
        for name, tensor in sorted(prepare_protein_features(sequence).items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(sequence.encode())
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _schema_digest() -> str:
    payload = {name: getattr(molecular_schema, name) for name in _MOLECULAR_SCHEMA_NAMES}
    payload["CHARGED_ATOMS"] = sorted(
        [*key, value] for key, value in molecular_schema.CHARGED_ATOMS.items()
    )
    payload["tokens"] = {name: getattr(token_schema, name) for name in _TOKEN_SCHEMA_NAMES}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def test_protein_features_match_pinned_official_tensor_digest() -> None:
    """The digest was generated from Biohub Transformers at its manifest revision."""

    assert _feature_digest() == _OFFICIAL_FEATURE_DIGEST


def test_generated_schemas_match_pinned_official_semantic_digest() -> None:
    assert _schema_digest() == _OFFICIAL_SCHEMA_DIGEST


def test_protein_features_preserve_padding_and_unknown_residues() -> None:
    # Feature dimensions are b=1, l=2 residues, a=32 reference atoms, xyz=3.
    features = prepare_protein_features("GX")
    assert features["res_type"].tolist() == [[9, 22]]
    assert features["input_ids"].tolist() == [[6, 3]]
    assert features["ref_pos"].shape == (1, 32, 3)
    assert features["atom_attention_mask"].sum().item() == 8
    assert torch.equal(features["msa"], features["res_type"].unsqueeze(1))


def test_generated_token_and_molecular_schemas_keep_checkpoint_indices() -> None:
    assert len(token_schema.SEQUENCE_VOCAB) == 33
    assert token_schema.SEQUENCE_VOCAB[4:24] == list("LAGVSERTIDPKQNFYMHWC")
    assert token_schema.VQVAE_SPECIAL_TOKENS == {
        "MASK": 4096,
        "EOS": 4097,
        "BOS": 4098,
        "PAD": 4099,
        "CHAINBREAK": 4100,
    }
    assert molecular_schema.PROTEIN_RESIDUE_TO_RES_TYPE["MSE"] == 14
    assert molecular_schema.RES_TYPE_TO_CCD[32] == "DN"
    assert molecular_schema.ELEMENT_TO_ATOMIC_NUM["U"] == 92
    assert 2 not in molecular_schema.ELEMENT_NUMBER_TO_SYMBOL


def test_fasta_stream_order_and_ownership() -> None:
    source = io.StringIO(">first\nAC\n>second\nGX\n")
    assert list(read_sequences(source)) == [("first", "AC"), ("second", "GX")]
    assert not source.closed
    assert list(parse_fasta("# note\n>x\nAA\n")) == [("x", "AA")]


def test_esmfold2_package_init_is_lazy() -> None:
    probe = """
import sys
import fastplms.models.esmfold2 as package

assert package.__all__ == [
    "ESMFold2Config",
    "ESMFold2ExperimentalModel",
    "ESMFold2Model",
]
assert "fastplms.models.esmfold2.modeling_esmfold2" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
