"""Token schemas needed by the ESMC encoder inside ESMFold2.

The values implement the published Biohub ESM sequence-token contract pinned by
``models.toml``. They are generated from ordered schemas so token position and
special-token relationships are explicit and independently testable. This
module performs no downloads and resolves no model assets at import time.
"""

from __future__ import annotations

from types import MappingProxyType


def _words(value: str) -> list[str]:
    return value.split()


SEQUENCE_VOCAB = _words(
    "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - | <mask>"
)

_sequence_token_ids = MappingProxyType({token: index for index, token in enumerate(SEQUENCE_VOCAB)})
SEQUENCE_BOS_TOKEN = _sequence_token_ids["<cls>"]
SEQUENCE_PAD_TOKEN = _sequence_token_ids["<pad>"]
SEQUENCE_EOS_TOKEN = _sequence_token_ids["<eos>"]
SEQUENCE_CHAINBREAK_TOKEN = _sequence_token_ids["|"]
SEQUENCE_MASK_TOKEN = _sequence_token_ids["<mask>"]
SEQUENCE_STANDARD_AA_MIN_TOKEN = _sequence_token_ids["L"]
SEQUENCE_STANDARD_AA_MAX_TOKEN = _sequence_token_ids["X"]

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    name: VQVAE_CODEBOOK_SIZE + offset
    for offset, name in enumerate(("MASK", "EOS", "BOS", "PAD", "CHAINBREAK"))
}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50

STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
STRUCTURE_UNDEFINED_TOKEN = 955

SASA_PAD_TOKEN = 0
SS8_PAD_TOKEN = 0
INTERPRO_PAD_TOKEN = 0
RESIDUE_PAD_TOKEN = 0

CHAIN_BREAK_STR = "|"
SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"
MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"
SASA_MASK_STR = "<unk>"
SS8_MASK_STR = "<unk>"

SSE_8CLASS_VOCAB = "GHITEBSC"
SSE_3CLASS_VOCAB = "HEC"
SSE_8CLASS_TO_3CLASS_MAP = dict(zip(SSE_8CLASS_VOCAB, "HHHCEECC", strict=True))

SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]
MAX_RESIDUE_ANNOTATIONS = 16
TFIDF_VECTOR_SIZE = 58_641
FUNCTION_TOKENS_DEPTH = 8


def _validate_schema() -> None:
    if len(SEQUENCE_VOCAB) != len(set(SEQUENCE_VOCAB)):
        raise RuntimeError("The ESM sequence vocabulary contains duplicate tokens.")
    if SEQUENCE_STANDARD_AA_MAX_TOKEN - SEQUENCE_STANDARD_AA_MIN_TOKEN != 20:
        raise RuntimeError("The canonical residue interval must contain 20 tokens.")
    if tuple(VQVAE_SPECIAL_TOKENS.values()) != tuple(range(4096, 4101)):
        raise RuntimeError("The structure special-token interval is not contiguous.")


_validate_schema()

__all__ = [name for name in globals() if name.isupper()]
