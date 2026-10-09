"""FastPLMs' ANKH computes what the official ANKH network computes, module by module, on CPU.

ANKH's official package, pinned in `models.toml` as `ankh`, defines no network of its own: its
loaders return transformers' `T5EncoderModel` and `T5ForConditionalGeneration`, which
`test_the_official_package_loads_transformers_t5` checks in the pinned tree. The package cannot
be imported under transformers 5, which removed the TensorFlow T5 classes its first import names,
so these tests build the official network from the installed transformers' T5 classes, which the
ANKH reference container pins at 4.25.1.

FastPLMs' encoder reimplements T5's. Its sequence-to-sequence class is transformers' own T5 class
and is compared too, because it carries FastPLMs' configuration and state. The network is small
and random with ANKH's configuration: gated GELU, 64 relative-position buckets over a distance of
128, RMS norms, and unscaled attention. FastPLMs receives the official state through the declared
transform, `ankh_t5_to_fastplms_v1`, which keeps every T5 name.

Every comparison is exact (`rtol=0, atol=0`): each FastPLMs backend is compared with the
transformers attention implementation of the same name, and the two run the same float32
operations in the same order.

The tokenizer tests at the end read the official base checkpoint's tokenizer files from the
Hugging Face cache and carry the `checkpoint` marker. Its weights, 2.9 GB in one file with the
decoder, are not loaded here.

Shape symbols: `b` batch, `l` sequence length, `d` hidden width, `h` attention heads, `c` the
vocabulary.
"""

from __future__ import annotations

import pytest
import torch

from pathlib import Path
from tests.parity.support.pinned_oracles import (
    assert_equal_to_rounding,
    assert_identical,
    cached_checkpoint,
    randomize_parameters,
    require_pinned_tree,
)
from tokenizers import Tokenizer
from torch import Tensor
from transformers import AutoTokenizer, T5Config, T5EncoderModel, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention

from fastplms.models.ankh.modeling_ankh import (
    AnkhSelfAttention,
    FastAnkhConfig,
    FastAnkhForConditionalGeneration,
    FastAnkhModel,
    tokenize_ankh_sequences,
)
from fastplms.registry import get_model_spec
from tools.conversion import apply_state_transform


REPRESENTATIVE = get_model_spec("ankh_base")
STATE_TRANSFORM = REPRESENTATIVE.family.state_transform
# ANKH's configuration at a small width.
NETWORK_SHAPE = {
    "vocab_size": 144,  # c
    "d_model": 32,  # d
    "d_kv": 8,
    "d_ff": 48,
    "num_heads": 4,  # h
    "num_layers": 3,
    "num_decoder_layers": 2,
    "relative_attention_num_buckets": 64,
    "relative_attention_max_distance": 128,
    "feed_forward_proj": "gated-gelu",
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "tie_word_embeddings": False,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}
SEED = 0
WEIGHT_SCALE = 0.3
BACKENDS = ("eager", "sdpa")
PAD = 0
# The ANKH tokenizer's ids for these sequences, each closed by </s> (1);
# `test_the_tokenizer_gives_the_ids_these_tests_use` checks them.
SEQUENCES = ("MKTAYIAKQRQISFVKSHF", "GSUZBOX", "MA")
SEQUENCE_IDS = (
    (19, 14, 11, 3, 18, 12, 3, 14, 16, 8, 16, 12, 7, 15, 6, 14, 7, 20, 15, 1),
    (5, 7, 26, 27, 24, 25, 23, 1),
    (19, 3, 1),
)
UNPADDED_IDS = ((19, 14, 11, 3, 18, 12, 3, 1), (5, 7, 26, 27, 24, 25, 23, 1))


def batch(rows: tuple[tuple[int, ...], ...]) -> tuple[Tensor, Tensor]:
    """Token ids right-padded with ANKH's pad id, and the tokenizer's attention mask."""
    l = max(len(row) for row in rows)
    input_ids = torch.tensor([[*row, *[PAD] * (l - len(row))] for row in rows])  # (b, l)
    attention_mask = torch.tensor(  # (b, l)
        [[1] * len(row) + [0] * (l - len(row)) for row in rows]
    )
    return input_ids, attention_mask  # (b, l), (b, l)


BATCHES = [
    *(pytest.param((row,), id=f"alone{index}") for index, row in enumerate(SEQUENCE_IDS)),
    pytest.param(SEQUENCE_IDS, id="padded"),
    pytest.param(UNPADDED_IDS, id="unpadded"),
]


def build_official() -> T5ForConditionalGeneration:
    """The official sequence-to-sequence network, with random weights and eager attention."""
    config = T5Config(**NETWORK_SHAPE, attn_implementation="eager")
    official = T5ForConditionalGeneration(config).eval()
    randomize_parameters(official, SEED, WEIGHT_SCALE)
    return official


def official_encoder(official: T5ForConditionalGeneration, backend: str) -> T5EncoderModel:
    """The official encoder network holding the sequence-to-sequence network's encoder state."""
    encoder = T5EncoderModel(T5Config(**NETWORK_SHAPE, attn_implementation=backend)).eval()
    encoder.load_state_dict(
        {
            name: tensor
            for name, tensor in official.state_dict().items()
            if not name.startswith(("decoder.", "lm_head."))
        },
        strict=True,
    )
    return encoder


def build_fastplms(official: T5ForConditionalGeneration, backend: str) -> FastAnkhModel:
    """FastPLMs' ANKH encoder holding the official state, mapped by the declared transform."""
    fastplms = FastAnkhModel(FastAnkhConfig(**NETWORK_SHAPE, attn_backend=backend)).eval()
    state = apply_state_transform(STATE_TRANSFORM, official.state_dict())
    # The encoder view takes the encoder's tensors; its loader ignores the decoder and head.
    fastplms.load_state_dict(
        {name: tensor for name, tensor in state.items() if name in fastplms.state_dict()},
        strict=True,
    )
    return fastplms


@pytest.fixture(scope="module")
def official() -> T5ForConditionalGeneration:
    return build_official()


@pytest.fixture(scope="module", params=BACKENDS)
def networks(
    request, official: T5ForConditionalGeneration
) -> tuple[T5EncoderModel, FastAnkhModel, str]:
    backend = request.param
    return official_encoder(official, backend), build_fastplms(official, backend), backend


# The official package and the state.


def test_the_official_package_loads_transformers_t5() -> None:
    models = require_pinned_tree("ankh") / "src" / "ankh" / "models"
    loaders = (models / "ankh_transformers.py").read_text(encoding="utf-8")

    assert "T5EncoderModel.from_pretrained(" in loaders
    assert "T5ForConditionalGeneration.from_pretrained(" in loaders


def test_the_declared_transform_keeps_every_official_tensor(
    official: T5ForConditionalGeneration,
) -> None:
    official_state = official.state_dict()

    state = apply_state_transform(STATE_TRANSFORM, official_state)

    assert state.keys() == official_state.keys()
    for name, tensor in official_state.items():
        assert_identical(state[name], tensor, name)


# The encoder, component by component.


def test_the_relative_position_buckets_match() -> None:
    relative_position = torch.arange(-300, 301)[None, :] - torch.arange(0, 3)[:, None]  # (3, 601)
    buckets = {
        "num_buckets": NETWORK_SHAPE["relative_attention_num_buckets"],
        "max_distance": NETWORK_SHAPE["relative_attention_max_distance"],
    }

    assert_identical(
        AnkhSelfAttention._relative_position_bucket(relative_position, **buckets),  # (3, 601)
        # The encoder's attention is bidirectional.
        T5Attention._relative_position_bucket(relative_position, bidirectional=True, **buckets),
    )


def test_the_position_bias_matches(networks) -> None:
    official, fastplms, _ = networks
    official_attention = official.encoder.block[0].layer[0].SelfAttention
    fastplms_attention = fastplms.encoder.block[0].layer[0].SelfAttention

    for l in (5, 17, 300):
        assert_identical(
            fastplms_attention.compute_bias(l, l, torch.device("cpu")),  # (1, h, l, l)
            official_attention.compute_bias(l, l),  # (1, h, l, l)
            f"l={l}",
        )


def test_the_norms_and_feed_forward_blocks_match(networks) -> None:
    official, fastplms, _ = networks
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(3, 9, NETWORK_SHAPE["d_model"], generator=generator)  # (b, l, d)

    with torch.no_grad():
        for index, (official_block, fastplms_block) in enumerate(
            zip(official.encoder.block, fastplms.encoder.block, strict=True)
        ):
            for part in (0, 1):
                assert_identical(
                    fastplms_block.layer[part].layer_norm(hidden_states),  # (b, l, d)
                    official_block.layer[part].layer_norm(hidden_states),  # (b, l, d)
                    f"block {index} norm {part}",
                )
            assert_identical(
                fastplms_block.layer[1].DenseReluDense(hidden_states),  # (b, l, d)
                official_block.layer[1].DenseReluDense(hidden_states),  # (b, l, d)
                f"block {index} feed-forward",
            )
            assert_identical(
                fastplms_block.layer[1](hidden_states),  # (b, l, d)
                official_block.layer[1](hidden_states),  # (b, l, d)
                f"block {index} feed-forward layer",
            )
        assert_identical(
            fastplms.encoder.final_layer_norm(hidden_states),  # (b, l, d)
            official.encoder.final_layer_norm(hidden_states),  # (b, l, d)
            "final norm",
        )


@pytest.mark.parametrize("rows", BATCHES)
def test_every_hidden_state_matches(networks, rows: tuple[tuple[int, ...], ...]) -> None:
    official, fastplms, _ = networks
    input_ids, attention_mask = batch(rows)  # (b, l); (b, l)

    with torch.no_grad():
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        expected = official(**inputs, output_hidden_states=True)
        actual = fastplms(**inputs, output_hidden_states=True)

    # Both report each block's input and then the final norm's output.
    states = NETWORK_SHAPE["num_layers"] + 1
    assert len(actual.hidden_states) == len(expected.hidden_states) == states
    for index, (actual_state, expected_state) in enumerate(
        zip(actual.hidden_states, expected.hidden_states, strict=True)
    ):
        assert_identical(actual_state, expected_state, f"hidden state {index}")  # (b, l, d)
    assert_identical(actual.last_hidden_state, expected.last_hidden_state, "last hidden state")


@pytest.mark.parametrize("rows", BATCHES)
def test_the_attention_maps_match(
    official: T5ForConditionalGeneration, rows: tuple[tuple[int, ...], ...]
) -> None:
    """Transformers returns attention maps from eager attention, and so does FastPLMs."""
    input_ids, attention_mask = batch(rows)  # (b, l); (b, l)
    expected_network = official_encoder(official, "eager")
    fastplms = build_fastplms(official, "sdpa")

    with torch.no_grad():
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        expected = expected_network(**inputs, output_attentions=True)
        actual = fastplms(**inputs, output_attentions=True)

    assert len(actual.attentions) == len(expected.attentions) == NETWORK_SHAPE["num_layers"]
    for index, (actual_map, expected_map) in enumerate(
        zip(actual.attentions, expected.attentions, strict=True)
    ):
        assert_identical(actual_map, expected_map, f"attention map {index}")  # (b, h, l, l)


def test_the_comparisons_see_a_padding_leak(networks) -> None:
    """Letting padded keys through moves residue states far outside even float32 rounding."""
    official, fastplms, _ = networks
    input_ids, attention_mask = batch(SEQUENCE_IDS)  # (b, l); (b, l)
    residues = attention_mask.bool()  # (b, l)

    with torch.no_grad():
        expected = official(input_ids=input_ids, attention_mask=attention_mask)
        leaked = fastplms(input_ids=input_ids, attention_mask=torch.ones_like(attention_mask))

    with pytest.raises(AssertionError, match="exceeds"):
        assert_equal_to_rounding(
            leaked.last_hidden_state[residues],  # (r, d), r residues
            expected.last_hidden_state[residues],  # (r, d)
            "residue states, padding visible",
        )


def test_the_sequence_to_sequence_network_matches(official: T5ForConditionalGeneration) -> None:
    fastplms = FastAnkhForConditionalGeneration(FastAnkhConfig(**NETWORK_SHAPE)).eval()
    state = apply_state_transform(STATE_TRANSFORM, official.state_dict())
    fastplms.load_state_dict(state, strict=True)
    input_ids, attention_mask = batch(SEQUENCE_IDS)  # (b, l); (b, l)
    decoder_input_ids = torch.tensor([[0, 5, 7, 9], [0, 3, 4, 1], [0, 19, 3, 1]])  # (b, l_decoder)

    with torch.no_grad():
        expected = official(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
        )
        actual = fastplms(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
        )

    assert_identical(  # (b, l, d)
        actual.encoder_last_hidden_state, expected.encoder_last_hidden_state, "encoder"
    )
    assert_identical(actual.logits, expected.logits, "logits")  # (b, l_decoder, c)


# The official tokenizer.


def official_tokenizer_files() -> Path:
    return cached_checkpoint(
        REPRESENTATIVE.official.repo_id,
        REPRESENTATIVE.official.revision,
        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
    )


@pytest.mark.checkpoint
def test_the_tokenizer_matches_the_checkpoints_own_pipeline() -> None:
    """FastPLMs' ANKH tokenization is the checkpoint's tokenizer.json pipeline, run directly.

    The file has no pre-tokenizer, so each residue is one token, raw or split into words.
    """
    snapshot = official_tokenizer_files()
    pipeline = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    sequences = [*SEQUENCES, "ACDEFGHIKLMNPQRSTVWYXBUZO"]

    encoded = tokenize_ankh_sequences(AutoTokenizer.from_pretrained(snapshot), sequences)

    for sequence, ids in zip(sequences, encoded["input_ids"], strict=True):
        assert ids == pipeline.encode(sequence).ids, sequence
        assert ids == pipeline.encode(list(sequence), is_pretokenized=True).ids, sequence


@pytest.mark.checkpoint
def test_the_tokenizer_gives_the_ids_these_tests_use() -> None:
    tokenizer = AutoTokenizer.from_pretrained(official_tokenizer_files())

    encoded = tokenize_ankh_sequences(tokenizer, list(SEQUENCES))

    assert tuple(tuple(ids) for ids in encoded["input_ids"]) == SEQUENCE_IDS


@pytest.mark.checkpoint
def test_transformers_5_splits_the_documented_usage_around_unknown_tokens() -> None:
    """Why FastPLMs configures ANKH's tokenizer rather than calling it as ANKH documents.

    ANKH's README tokenizes residues split into words. Transformers 5's T5 tokenizer replaces the
    file's absent pre-tokenizer with a metaspace one that prefixes every word with a word marker,
    which ANKH's vocabulary lacks, so each residue arrives after an unknown token. This fails once
    transformers keeps the file's pipeline.
    """
    snapshot = official_tokenizer_files()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    pipeline = Tokenizer.from_file(str(snapshot / "tokenizer.json"))

    ids = tokenizer([list(SEQUENCES[2])], is_split_into_words=True)["input_ids"][0]

    unknown = pipeline.token_to_id("<unk>")
    assert ids == [unknown, SEQUENCE_IDS[2][0], unknown, SEQUENCE_IDS[2][1], SEQUENCE_IDS[2][2]]
