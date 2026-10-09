"""FastPLMs' ESM3 computes what Biohub's native ESM3 computes, module by module, on CPU.

The official side is Biohub's `esm` package at the commit `models.toml` pins as `biohub-esm`,
imported unchanged from its pinned tree by `tests.parity.support.pinned_oracles`; without that
tree these tests skip. ESM3 builds a small network with random weights, and FastPLMs receives its
state through the transform the manifest declares for ESM3, `esm3_to_fastplms_v1`.

Every track is fed: sequence, structure tokens and backbone coordinates, secondary structure,
solvent accessibility, function and residue annotations, and pLDDT. The first block carries
ESM3's geometric attention, which reads the backbone frames and the chain ids.

Every comparison is exact (`rtol=0, atol=0`): both sides lay activations out as (b, l, d) and run
the same float32 operations in the same order. FastPLMs' SDPA backend makes ESM3's SDPA call, and
its eager backend computes the attention products ESM3 computes when it returns weights, so each
backend is compared with the ESM3 path it mirrors.

The smallest official checkpoint has 1.4B parameters, beyond a CPU unit test, so no test here
loads one.

Shape symbols: `b` batch, `l` sequence length, `d` hidden width, `h` attention heads, `d_h`
per-head width, `c` a track's vocabulary, `n` layers, `k` function-annotation depth, `m` residue
annotations per position.
"""

from __future__ import annotations

import pytest
import torch

from types import SimpleNamespace
from tests.parity.support.pinned_oracles import (
    OfficialPackage,
    assert_equal_to_rounding,
    assert_identical,
    official_package,
    randomize_parameters,
)
from torch import Tensor, nn

from fastplms.models.esm3 import modeling_esm3
from fastplms.models.esm3.modeling_esm3 import (
    EsmSequenceTokenizer,
    FastESM3Config,
    FastESM3Model,
    build_affine3d_from_coordinates,
)
from fastplms.registry import get_model_spec
from tools.conversion import apply_state_transform


STATE_TRANSFORM = get_model_spec("esm3_small").family.state_transform
LAYERS = 3  # n; the first block also carries geometric attention
HIDDEN = 64  # d, a multiple of the eight function-annotation slots
HEADS = 4  # h, so d_h = 16
VECTOR_HEADS = 4
SEED = 0
WEIGHT_SCALE = 0.3
# ESM3 normalizes queries and keys before their product, so the gain of those two norms sets how
# far attention is from uniform. At this gain a key the mask should hide would show.
QUERY_KEY_NORM_GAIN = 8.0
BACKENDS = ("eager", "sdpa")
# Right-padded to unequal lengths, with masked residues, the rarer residue letters, and a chain
# break.
SEQUENCES = ("MKTAYIAK<mask>RQISFVKSHF", "GSUZ<mask>BOX|MKV", "MA<mask>")
OFFICIAL_MODULES = (
    "esm.models.esm3",
    "esm.tokenization",
    "esm.utils.constants.esm3",
    "esm.utils.structure.affine3d",
)
# The track constants both sides fill absent tracks and special positions with.
TRACK_CONSTANTS = (
    "SEQUENCE_BOS_TOKEN",
    "SEQUENCE_PAD_TOKEN",
    "SEQUENCE_EOS_TOKEN",
    "SEQUENCE_CHAINBREAK_TOKEN",
    "SEQUENCE_MASK_TOKEN",
    "STRUCTURE_MASK_TOKEN",
    "STRUCTURE_BOS_TOKEN",
    "STRUCTURE_EOS_TOKEN",
    "STRUCTURE_PAD_TOKEN",
    "STRUCTURE_CHAINBREAK_TOKEN",
    "SASA_PAD_TOKEN",
    "SS8_PAD_TOKEN",
    "INTERPRO_PAD_TOKEN",
    "RESIDUE_PAD_TOKEN",
    "MAX_RESIDUE_ANNOTATIONS",
    "FUNCTION_TOKENS_DEPTH",
    "SEQUENCE_VOCAB",
)


@pytest.fixture(scope="module")
def biohub_esm() -> OfficialPackage:
    return official_package("biohub-esm", "esm", OFFICIAL_MODULES)


def build_official(biohub_esm: OfficialPackage) -> nn.Module:
    """ESM3 at a small width, with random weights."""
    official = biohub_esm["esm.models.esm3"].ESM3(
        d_model=HIDDEN,
        n_heads=HEADS,
        v_heads=VECTOR_HEADS,
        n_layers=LAYERS,
        # ESM3 builds its structure and function decoders only for the SDK's decoding.
        structure_encoder_fn=None,
        structure_decoder_fn=None,
        function_decoder_fn=None,
        # Its forward reads only the sequence tokenizer, for the mask token that fills an absent
        # sequence track. The other tracks' tokenizers read InterPro and keyword files from the
        # gated weights repository.
        tokenizers=SimpleNamespace(sequence=biohub_esm["esm.tokenization"].EsmSequenceTokenizer()),
    ).eval()
    randomize_parameters(official, SEED, WEIGHT_SCALE)
    with torch.no_grad():
        for block in official.transformer.blocks:
            block.attn.q_ln.weight.mul_(QUERY_KEY_NORM_GAIN)
            block.attn.k_ln.weight.mul_(QUERY_KEY_NORM_GAIN)
    return official


def build_fastplms(official: nn.Module, backend: str) -> FastESM3Model:
    """FastPLMs' ESM3 holding the official state, mapped by the declared transform."""
    config = FastESM3Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_vector_heads=VECTOR_HEADS,
        num_hidden_layers=LAYERS,
        attn_backend=backend,
    )
    fastplms = FastESM3Model(config).eval()
    state = apply_state_transform(
        STATE_TRANSFORM, official.state_dict(), expected_keys=fastplms.state_dict()
    )
    fastplms.load_state_dict(state, strict=True)
    return fastplms


@pytest.fixture(scope="module")
def official(biohub_esm: OfficialPackage) -> nn.Module:
    return build_official(biohub_esm)


@pytest.fixture(scope="module", params=BACKENDS)
def networks(request, official: nn.Module) -> tuple[nn.Module, FastESM3Model, str]:
    return official, build_fastplms(official, request.param), request.param


def tokenize(official: nn.Module, sequences: tuple[str, ...]) -> Tensor:
    """Sequence-track ids from ESM3's own tokenizer: cls, residues, eos, then padding."""
    tokenizer = official.tokenizers.sequence
    return tokenizer(list(sequences), padding=True, return_tensors="pt")["input_ids"]  # (b, l)


def track_inputs(official: nn.Module, sequences: tuple[str, ...]) -> dict[str, Tensor]:
    """Every ESM3 input track, drawn at random where the sequence track leaves it free."""
    tokens = tokenize(official, sequences)  # (b, l)
    b, l = tokens.shape
    generator = torch.Generator().manual_seed(SEED)
    tokenizer = official.tokenizers.sequence
    residues = ~torch.isin(  # (b, l)
        tokens,
        torch.tensor([tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]),
    )
    # A chain of backbones along one axis, 3.8 angstroms apart, with jittered N, CA, and C atoms.
    coordinates = torch.randn(b, l, 3, 3, generator=generator) * 2  # (b, l, 3, 3)
    coordinates += 3.8 * torch.arange(l, dtype=torch.float32)[None, :, None, None]
    coordinates[~residues] = float("nan")
    coordinates[0, 5] = float("nan")  # a residue without coordinates
    # Residues after a chain break belong to the next chain.
    chain_id = (tokens == tokenizer.chain_break_token_id).long().cumsum(dim=-1)  # (b, l)
    annotations = torch.randint(0, 1478, (b, l, 16), generator=generator)  # (b, l, m)
    return {
        "sequence_tokens": tokens,
        "structure_tokens": torch.randint(0, 4096, (b, l), generator=generator),  # (b, l)
        "ss8_tokens": torch.randint(0, 11, (b, l), generator=generator),  # (b, l)
        "sasa_tokens": torch.randint(0, 19, (b, l), generator=generator),  # (b, l)
        "function_tokens": torch.randint(0, 260, (b, l, 8), generator=generator),  # (b, l, k)
        # Most annotation slots hold padding, which the bag embedding skips.
        "residue_annotation_tokens": annotations  # (b, l, m)
        * (torch.rand(b, l, 16, generator=generator) < 0.2),
        "average_plddt": torch.rand(b, 1, generator=generator),  # (b, 1)
        "per_res_plddt": torch.rand(b, l, generator=generator),  # (b, l)
        "structure_coords": coordinates,
        "chain_id": chain_id,
        # ESM3's SDK marks padding with a boolean sequence id, true at every token.
        "sequence_id": tokens.ne(tokenizer.pad_token_id),  # (b, l)
    }


def official_forward(official: nn.Module, backend: str, **inputs: Tensor):
    """ESM3's forward along the attention path that FastPLMs' backend mirrors."""
    # inputs: (b, l) tracks such as sequence_tokens, function_tokens (b, l, 8), residue_annotation_tokens (b, l, 16), structure_coords (b, l, 3, 3), average_plddt (b, 1)
    return official(**inputs, output_attentions=backend == "eager")


OUTPUTS = (
    "sequence_logits",  # (b, l, c)
    "structure_logits",  # (b, l, c)
    "secondary_structure_logits",  # (b, l, c)
    "sasa_logits",  # (b, l, c)
    "function_logits",  # (b, l, k, c)
    "residue_logits",  # (b, l, c)
    "embeddings",  # (b, l, d), the last block's output before the final norm
)


# The constants, tokenizer, frames, and state.


def test_the_track_constants_match(biohub_esm: OfficialPackage) -> None:
    constants = biohub_esm["esm.utils.constants.esm3"]

    for name in TRACK_CONSTANTS:
        assert getattr(modeling_esm3, name) == getattr(constants, name), name


def test_the_sequence_tokenizer_matches_esm3(
    biohub_esm: OfficialPackage, official: nn.Module
) -> None:
    official_tokenizer = biohub_esm["esm.tokenization"].EsmSequenceTokenizer()
    fastplms_tokenizer = EsmSequenceTokenizer()
    sequences = [*SEQUENCES, "ACDEFGHIKLMNPQRSTVWYXBUZO", "MK<mask>TA<unk>Y|GS"]

    expected = official_tokenizer(sequences, padding=True, return_tensors="pt")
    actual = fastplms_tokenizer(sequences, padding=True, return_tensors="pt")

    assert fastplms_tokenizer.get_vocab() == official_tokenizer.get_vocab()
    assert_identical(actual["input_ids"], expected["input_ids"])  # (b, l)
    assert_identical(actual["attention_mask"], expected["attention_mask"])  # (b, l)


def test_the_backbone_frames_match(biohub_esm: OfficialPackage, official: nn.Module) -> None:
    coordinates = track_inputs(official, SEQUENCES)["structure_coords"]  # (b, l, 3, 3)

    expected_frames, expected_mask = biohub_esm[
        "esm.utils.structure.affine3d"
    ].build_affine3d_from_coordinates(coordinates)
    actual_frames, actual_mask = build_affine3d_from_coordinates(coordinates)

    assert_identical(actual_frames.tensor, expected_frames.tensor, "frames")  # (b, l, 12)
    assert_identical(actual_mask, expected_mask, "frame mask")  # (b, l)


def test_the_declared_transform_maps_every_official_tensor(official: nn.Module) -> None:
    fastplms = build_fastplms(official, "sdpa")
    official_state = official.state_dict()

    # FastPLMs holds ESM3's module tree under `esm3`.
    assert fastplms.state_dict().keys() == {f"esm3.{name}" for name in official_state}
    for name, tensor in official_state.items():
        assert_identical(fastplms.state_dict()[f"esm3.{name}"], tensor, name)


# The network, component by component.


def test_the_input_encoder_matches(networks) -> None:
    official, fastplms, _ = networks
    inputs = track_inputs(official, SEQUENCES)
    encoder_inputs = [
        inputs[name]
        for name in (
            "sequence_tokens",
            "structure_tokens",
            "average_plddt",
            "per_res_plddt",
            "ss8_tokens",
            "sasa_tokens",
            "function_tokens",
            "residue_annotation_tokens",
        )
    ]

    with torch.no_grad():
        expected = official.encoder(*encoder_inputs)  # (b, l, d)
        actual = fastplms.esm3.encoder(*encoder_inputs)  # (b, l, d)

    assert_identical(actual, expected)


def test_the_rotary_embeddings_match_across_lengths(networks) -> None:
    official, fastplms, _ = networks
    official_rotary = official.transformer.blocks[0].attn.rotary
    fastplms_rotary = fastplms.esm3.transformer.blocks[0].attn.rotary
    generator = torch.Generator().manual_seed(SEED)

    for l in (12, 5, 12, 30, 7):
        Q = torch.randn(3, l, HEADS, HIDDEN // HEADS, generator=generator)  # (b, l, h, d_h)
        K = torch.randn(3, l, HEADS, HIDDEN // HEADS, generator=generator)  # (b, l, h, d_h)
        expected_q, expected_k = official_rotary(Q.clone(), K.clone())  # (b, l, h, d_h) each
        actual_q, actual_k = fastplms_rotary(Q.clone(), K.clone())  # (b, l, h, d_h) each
        assert_identical(actual_q, expected_q, f"query, l={l}")
        assert_identical(actual_k, expected_k, f"key, l={l}")


@pytest.mark.parametrize("layer_index", range(LAYERS))
def test_each_block_matches_part_by_part(
    biohub_esm: OfficialPackage, networks, layer_index: int
) -> None:
    """Attention and its weights, geometric attention, the feed-forward block, the whole block."""
    official, fastplms, backend = networks
    official_block = official.transformer.blocks[layer_index]
    fastplms_block = fastplms.esm3.transformer.blocks[layer_index]
    inputs = track_inputs(official, SEQUENCES)
    sequence_id, chain_id = inputs["sequence_id"], inputs["chain_id"]  # (b, l) each
    b, l = sequence_id.shape
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(b, l, HIDDEN, generator=generator)  # (b, l, d)
    official_frames, official_frame_mask = biohub_esm[
        "esm.utils.structure.affine3d"
    ].build_affine3d_from_coordinates(inputs["structure_coords"])
    fastplms_frames, fastplms_frame_mask = build_affine3d_from_coordinates(  # (b, l); (b, l)
        inputs["structure_coords"]
    )
    mask, block_mask, fastplms_frame_mask, semantics, effective_backend = (
        fastplms.esm3.transformer._prepare_attention_masks(
            sequence_id=sequence_id,
            attention_mask=None,
            affine_mask=fastplms_frame_mask,
            batch_size=b,
            seq_len=l,
            device=hidden_states.device,
            attention_backend=fastplms.esm3.transformer.attention_backend,
            output_attentions=False,
        )
    )
    returns_weights = backend == "eager"

    with torch.no_grad():
        expected_attention, expected_weights = official_block.attn(  # (b, l, d); (b, h, l, l)
            hidden_states, sequence_id, output_attentions=returns_weights
        )
        actual_attention, actual_weights = fastplms_block.attn(  # (b, l, d); (b, h, l, l)
            hidden_states,
            mask,
            block_mask,
            semantics,
            output_attentions=returns_weights,
            effective_backend=effective_backend,
        )
        expected_feed_forward = official_block.ffn(hidden_states)  # (b, l, d)
        actual_feed_forward = fastplms_block.ffn(hidden_states)  # (b, l, d)
        expected_block = official_block(  # (b, l, d)
            hidden_states,
            sequence_id,
            official_frames,
            official_frame_mask,
            chain_id,
            output_attentions=returns_weights,
        )[0]
        actual_block = fastplms_block(  # (b, l, d)
            hidden_states,
            sequence_id,
            mask,
            block_mask,
            semantics,
            fastplms_frames,
            fastplms_frame_mask,
            chain_id,
            effective_backend=effective_backend,
        )[0]
        if official_block.use_geom_attn:
            expected_geometric = official_block.geom_attn(  # (b, l, d)
                hidden_states, official_frames, official_frame_mask, sequence_id, chain_id
            )
            actual_geometric = fastplms_block.geom_attn(  # (b, l, d)
                hidden_states, fastplms_frames, fastplms_frame_mask, sequence_id, chain_id
            )
            assert_identical(actual_geometric, expected_geometric, "geometric attention")
            # Residues with coordinates carry frames, so the comparison is not of zeros.
            assert expected_geometric[official_frame_mask].abs().min() > 0

    assert fastplms_block.use_geom_attn == official_block.use_geom_attn == (layer_index == 0)
    assert_identical(actual_attention, expected_attention, "attention")
    if returns_weights:
        assert_identical(actual_weights, expected_weights, "attention weights")
    assert_identical(actual_feed_forward, expected_feed_forward, "feed-forward block")
    assert_identical(actual_block, expected_block, "block")


def test_the_final_norm_and_output_heads_match(networks) -> None:
    official, fastplms, _ = networks
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(3, 9, HIDDEN, generator=generator)  # (b, l, d)

    with torch.no_grad():
        assert_identical(
            fastplms.esm3.transformer.norm(hidden_states),  # (b, l, d)
            official.transformer.norm(hidden_states),  # (b, l, d)
            "final norm",
        )
        expected = official.output_heads(hidden_states, hidden_states)
        actual = fastplms.esm3.output_heads(hidden_states, hidden_states)
    for name in OUTPUTS:
        assert_identical(getattr(actual, name), getattr(expected, name), name)


def select_tracks(inputs: dict[str, Tensor], tracks: str) -> dict[str, Tensor]:
    # inputs: (b, l) tracks as track_inputs builds them, plus function_tokens (b, l, 8), structure_coords (b, l, 3, 3), average_plddt (b, 1)
    if tracks == "every_track":
        return inputs  # (b, l) every track, as given
    if tracks == "sequence_only":
        return {name: inputs[name] for name in ("sequence_tokens", "sequence_id")}  # (b, l) sequence_tokens and sequence_id
    # Without a sequence id ESM3 lets every position attend to padding, and so does FastPLMs.
    return {name: value for name, value in inputs.items() if name != "sequence_id"}  # (b, l) every track except sequence_id


@pytest.mark.parametrize("tracks", ["every_track", "sequence_only", "no_sequence_id"])
@pytest.mark.parametrize("sequences", [(SEQUENCES[0],), SEQUENCES], ids=["alone", "padded"])
def test_every_track_output_matches(networks, sequences: tuple[str, ...], tracks: str) -> None:
    official, fastplms, backend = networks
    inputs = select_tracks(track_inputs(official, sequences), tracks)

    with torch.no_grad():
        expected = official_forward(official, backend, **inputs)
        actual = fastplms(**inputs, output_hidden_states=True)

    for name in OUTPUTS:
        assert_identical(getattr(actual, name), getattr(expected, name), name)
    assert_identical(actual.last_hidden_state, expected.embeddings, "last hidden state")
    assert len(actual.hidden_states) == LAYERS
    assert_identical(actual.hidden_states[-1], expected.embeddings, "last block output")


def test_the_attention_maps_match(networks) -> None:
    """Returning attention maps runs FastPLMs' eager attention whatever the configured backend."""
    official, fastplms, _ = networks
    inputs = track_inputs(official, SEQUENCES)

    with torch.no_grad():
        expected = official(**inputs, output_attentions=True)
        actual = fastplms(**inputs, output_attentions=True)

    assert len(actual.attentions) == len(expected.attentions) == LAYERS
    for index, (actual_map, expected_map) in enumerate(
        zip(actual.attentions, expected.attentions, strict=True)
    ):
        assert_identical(actual_map, expected_map, f"attention map {index}")  # (b, h, l, l)
    assert_identical(actual.sequence_logits, expected.sequence_logits, "sequence logits")


def test_a_tokenizer_mask_matches_esm3_sequence_ids_at_residues(networks) -> None:
    """FastPLMs' tokenizer-style call, ids and a padding mask, gives ESM3's residue outputs.

    ESM3 masks padding by sequence id, which lets padding attend to padding; FastPLMs' attention
    mask hides padded keys from every query. Only the padding rows differ.
    """
    official, fastplms, backend = networks
    tokens = tokenize(official, SEQUENCES)  # (b, l)
    tokens_present = tokens.ne(official.tokenizers.sequence.pad_token_id)  # (b, l)

    with torch.no_grad():
        expected = official_forward(
            official, backend, sequence_tokens=tokens, sequence_id=tokens_present
        )
        actual = fastplms(input_ids=tokens, attention_mask=tokens_present.long())

    for name in OUTPUTS:
        assert_identical(
            getattr(actual, name)[tokens_present],  # (r, ...), r tokens present
            getattr(expected, name)[tokens_present],  # (r, ...)
            name,
        )


def test_the_comparisons_see_a_padding_leak(networks) -> None:
    """Letting padded keys through moves residue outputs far outside even float32 rounding."""
    official, fastplms, backend = networks
    inputs = track_inputs(official, SEQUENCES)
    tokens_present = inputs["sequence_id"]  # (b, l)
    leaking = {name: value for name, value in inputs.items() if name != "sequence_id"}

    with torch.no_grad():
        expected = official_forward(official, backend, **inputs).sequence_logits  # (b, l, c)
        leaked = fastplms(**leaking).sequence_logits  # (b, l, c)

    with pytest.raises(AssertionError, match="exceeds"):
        assert_equal_to_rounding(
            leaked[tokens_present], expected[tokens_present], "residue logits, padding visible"
        )
