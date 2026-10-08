"""FastPLMs' ESM++ computes what Biohub's native ESMC computes, module by module, on CPU.

ESM++ is FastPLMs' ESMC. The official side is Biohub's `esm` package at the commit `models.toml`
pins as `biohub-esm`, imported unchanged from its pinned tree by
`tests.parity.support.pinned_oracles`; without that tree these tests skip. ESMC builds a small
network and fills it with random weights, and FastPLMs receives its state through the transform
the manifest declares for the family, `esmc_to_fastplms_v1`, which keeps ESMC's module names.

Every comparison is exact (`rtol=0, atol=0`): both sides lay activations out as (b, l, d) and run
the same float32 operations in the same order. ESMC's forward runs SDPA with a boolean mask unless
it is asked for attention weights, and then it computes the attention products itself. FastPLMs'
SDPA backend makes the same SDPA call, and its eager backend computes the products as ESMC does
when it returns weights, so each backend is compared with the ESMC path it mirrors.

The two report hidden states by different conventions. ESMC returns each block's output, and the
final norm's output as `embeddings`. FastPLMs, as Transformers models do, returns the input to
each block and then the final norm's output. FastPLMs' state i + 1 is therefore ESMC's state i for
every block but the last, whose output ESMC also reports before the final norm; the block
comparisons cover that one.

The loading tests at the end read three files from the Hugging Face cache, never the network: the
official 300M checkpoint, Biohub's native file of the same network, and FastPLMs' published
conversion. They carry the `checkpoint` marker.

Shape symbols: `b` batch, `l` sequence length, `d` hidden width, `h` attention heads, `d_h`
per-head width, `c` the vocabulary, `n` layers.
"""

from __future__ import annotations

import json
import pytest
import torch

from pathlib import Path
from safetensors.torch import load_file
from tests.parity.support.pinned_oracles import (
    OfficialPackage,
    assert_equal_to_rounding,
    assert_identical,
    cached_checkpoint,
    official_package,
    randomize_parameters,
)
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    EsmSequenceTokenizer,
)
from fastplms.registry import CheckpointSource, get_model_spec
from tools.conversion import apply_state_transform


REPRESENTATIVE = get_model_spec("esmc_small")
STATE_TRANSFORM = REPRESENTATIVE.family.state_transform
LAYERS = 3  # n
HIDDEN = 64  # d
# h, so d_h = 16, for which ESMC's scale d_h ** -0.5 and FastPLMs' 1 / sqrt(d_h) are one float.
HEADS = 4
SEED = 0
WEIGHT_SCALE = 0.3
# ESMC normalizes queries and keys before their product, so the gain of those two norms sets how
# far attention is from uniform. At this gain a residue's strongest key takes about half of its
# attention, so a key the mask should hide would show.
QUERY_KEY_NORM_GAIN = 8.0
BACKENDS = ("eager", "sdpa")
# Right-padded to unequal lengths, with masked residues, the rarer residue letters, and a chain
# break.
SEQUENCES = ("MKTAYIAK<mask>RQISFVKSHF", "GSUZ<mask>BOX|MKV", "MA<mask>")
SEQUENCES_WITHOUT_PADDING = ("MKTAYIAKQR", "LVSG<mask>AAGEW")
OFFICIAL_MODULES = ("esm.models.esmc", "esm.tokenization")


@pytest.fixture(scope="module")
def biohub_esm() -> OfficialPackage:
    return official_package("biohub-esm", "esm", OFFICIAL_MODULES)


def build_official(biohub_esm: OfficialPackage) -> nn.Module:
    """ESMC at a small width, with random weights."""
    official = biohub_esm["esm.models.esmc"].ESMC(
        d_model=HIDDEN,
        n_heads=HEADS,
        n_layers=LAYERS,
        tokenizer=biohub_esm["esm.tokenization"].EsmSequenceTokenizer(),
        use_flash_attn=False,
    ).eval()
    randomize_parameters(official, SEED, WEIGHT_SCALE)
    with torch.no_grad():
        for block in official.transformer.blocks:
            block.attn.q_ln.weight.mul_(QUERY_KEY_NORM_GAIN)
            block.attn.k_ln.weight.mul_(QUERY_KEY_NORM_GAIN)
    return official


def fastplms_config(official: nn.Module, backend: str) -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=official.embed.num_embeddings,
        hidden_size=official.embed.embedding_dim,
        num_attention_heads=official.transformer.blocks[0].attn.n_heads,
        num_hidden_layers=len(official.transformer.blocks),
        pad_token_id=official.tokenizer.pad_token_id,
        mask_token_id=official.tokenizer.mask_token_id,
        attn_backend=backend,
    )


def build_fastplms(official: nn.Module, backend: str) -> ESMplusplusForMaskedLM:
    """FastPLMs' ESM++ holding the official state, mapped by the declared transform."""
    fastplms = ESMplusplusForMaskedLM(fastplms_config(official, backend)).eval()
    state = apply_state_transform(
        STATE_TRANSFORM, official.state_dict(), expected_keys=fastplms.state_dict()
    )
    fastplms.load_state_dict(state, strict=True)
    return fastplms


@pytest.fixture(scope="module")
def official(biohub_esm: OfficialPackage) -> nn.Module:
    return build_official(biohub_esm)


@pytest.fixture(scope="module", params=BACKENDS)
def networks(request, official: nn.Module) -> tuple[nn.Module, ESMplusplusForMaskedLM, str]:
    return official, build_fastplms(official, request.param), request.param


def tokenize(official: nn.Module, sequences: tuple[str, ...]) -> Tensor:
    """Token ids as ESMC lays them out: cls, residues, eos, then padding."""
    return official._tokenize(list(sequences))  # (b, l)


def official_forward(official: nn.Module, backend: str, **inputs: Tensor):
    """ESMC's forward along the attention path that FastPLMs' backend mirrors."""
    # inputs: (b, l) tracks, sequence_tokens and sequence_id
    return official(**inputs, output_attentions=backend == "eager")


def assert_outputs_match(actual, expected, n: int) -> None:
    """FastPLMs' hidden states, final state, and logits against ESMC's, whose layout differs.

    ESMC's state i is block i's output, which is FastPLMs' state i + 1 for every block but the
    last; ESMC's `embeddings` is the final norm's output, FastPLMs' state n.
    """
    for index in range(1, n):
        assert_identical(
            actual.hidden_states[index],  # (b, l, d)
            expected.hidden_states[index - 1],  # (b, l, d)
            f"hidden state {index}",
        )
    assert_identical(actual.hidden_states[n], expected.embeddings, f"hidden state {n}")
    assert_identical(actual.last_hidden_state, expected.embeddings, "last hidden state")
    assert_identical(actual.logits, expected.sequence_logits, "logits")  # (b, l, c)


def fastplms_masks(
    fastplms: ESMplusplusForMaskedLM, attention_mask: Tensor
) -> dict[str, Tensor | None]:
    """The masks FastPLMs' transformer stack builds once and hands every block."""
    # attention_mask: (b, l)
    b, l = attention_mask.shape
    mask_2d, mask_4d, _ = fastplms.transformer._prepare_attention_masks(  # (b, l); (b, 1, l, l)
        attention_mask=attention_mask,
        sequence_id=None,
        batch_size=b,
        seq_len=l,
        device=attention_mask.device,
        dtype=torch.float32,
    )
    return {"attention_mask_2d": mask_2d, "attention_mask_4d": mask_4d}  # (...) attention_mask_2d (b, l), attention_mask_4d (b, 1, l, l)


# Inputs: every sequence alone, then the padded and unpadded batches.
BATCHES = [
    *(pytest.param((sequence,), id=f"alone{index}") for index, sequence in enumerate(SEQUENCES)),
    pytest.param(SEQUENCES, id="padded"),
    pytest.param(SEQUENCES_WITHOUT_PADDING, id="unpadded"),
]


def test_the_batches_pad_where_their_names_say(official: nn.Module) -> None:
    pad = official.tokenizer.pad_token_id

    assert tokenize(official, SEQUENCES).eq(pad).any()
    assert not tokenize(official, SEQUENCES_WITHOUT_PADDING).eq(pad).any()


# The tokenizer and the state.


def test_the_tokenizer_matches_esmc(biohub_esm: OfficialPackage, official: nn.Module) -> None:
    official_tokenizer = biohub_esm["esm.tokenization"].EsmSequenceTokenizer()
    fastplms_tokenizer = EsmSequenceTokenizer()
    sequences = (*SEQUENCES, "ACDEFGHIKLMNPQRSTVWYXBUZO", "MK<mask>TA<unk>Y|GS")

    encoded = fastplms_tokenizer(list(sequences), padding=True, return_tensors="pt")
    expected = tokenize(official, sequences)  # (b, l)

    assert fastplms_tokenizer.get_vocab() == official_tokenizer.get_vocab()
    for token in ("cls", "pad", "mask", "eos", "unk", "chain_break"):
        name = f"{token}_token_id"
        assert getattr(fastplms_tokenizer, name) == getattr(official_tokenizer, name), name
    assert_identical(encoded["input_ids"], expected)  # (b, l)
    assert_identical(encoded["attention_mask"], expected.ne(official_tokenizer.pad_token_id).long())


def test_the_declared_transform_maps_every_official_tensor(official: nn.Module) -> None:
    fastplms = ESMplusplusForMaskedLM(fastplms_config(official, "sdpa"))
    official_state = official.state_dict()

    state = apply_state_transform(
        STATE_TRANSFORM, official_state, expected_keys=fastplms.state_dict()
    )
    missing, unexpected = fastplms.load_state_dict(state, strict=False)

    assert (missing, unexpected) == ([], [])
    # ESM++ keeps ESMC's module tree, so every tensor keeps its official name.
    assert state.keys() == official_state.keys() == fastplms.state_dict().keys()
    for name, tensor in official_state.items():
        assert_identical(state[name], tensor, name)


def test_both_build_the_same_rotary_frequencies_before_any_copy(
    biohub_esm: OfficialPackage,
) -> None:
    official = build_official(biohub_esm)
    fastplms = ESMplusplusForMaskedLM(fastplms_config(official, "sdpa"))

    for official_block, fastplms_block in zip(
        official.transformer.blocks, fastplms.transformer.blocks, strict=True
    ):
        assert_identical(
            fastplms_block.attn.rotary.inv_freq,  # (d_h / 2,)
            official_block.attn.rotary.inv_freq,  # (d_h / 2,)
        )


# The network, component by component.


def test_the_rotary_embeddings_match_across_lengths(networks) -> None:
    official, fastplms, _ = networks
    official_rotary = official.transformer.blocks[0].attn.rotary
    fastplms_rotary = fastplms.transformer.blocks[0].attn.rotary
    generator = torch.Generator().manual_seed(SEED)

    # Shorter, longer, and repeated lengths exercise both implementations' table caches.
    for l in (12, 5, 12, 30, 7):
        Q = torch.randn(3, l, HEADS, HIDDEN // HEADS, generator=generator)  # (b, l, h, d_h)
        K = torch.randn(3, l, HEADS, HIDDEN // HEADS, generator=generator)  # (b, l, h, d_h)
        expected_q, expected_k = official_rotary(Q.clone(), K.clone())  # (b, l, h, d_h) each
        actual_q, actual_k = fastplms_rotary(Q.clone(), K.clone())  # (b, l, h, d_h) each
        assert_identical(actual_q, expected_q, f"query, l={l}")
        assert_identical(actual_k, expected_k, f"key, l={l}")


@pytest.mark.parametrize("layer_index", range(LAYERS))
@pytest.mark.parametrize("sequences", [(SEQUENCES[0],), SEQUENCES], ids=["alone", "padded"])
def test_each_block_matches_part_by_part(
    networks, sequences: tuple[str, ...], layer_index: int
) -> None:
    """Attention, its weights, the feed-forward block, and the whole residual block."""
    official, fastplms, backend = networks
    official_block = official.transformer.blocks[layer_index]
    fastplms_block = fastplms.transformer.blocks[layer_index]
    tokens = tokenize(official, sequences)  # (b, l)
    b, l = tokens.shape
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(b, l, HIDDEN, generator=generator)  # (b, l, d)
    # Given no sequence id, ESMC's forward passes its blocks this boolean one, true at tokens.
    sequence_id = tokens.ne(official.tokenizer.pad_token_id)  # (b, l)
    masks = fastplms_masks(fastplms, sequence_id)
    returns_weights = backend == "eager"

    with torch.no_grad():
        expected_attention, expected_weights = official_block.attn(  # (b, l, d); (b, h, l, l)
            hidden_states, sequence_id, output_attentions=returns_weights
        )
        actual_attention, actual_weights, _ = fastplms_block.attn(  # (b, l, d); (b, h, l, l)
            hidden_states, **masks, output_attentions=returns_weights
        )
        # ESMC's block takes structure frames and chain ids for its geometric attention, which
        # ESMC's blocks do not have.
        expected_block = official_block(  # (b, l, d)
            hidden_states, sequence_id, None, None, None, output_attentions=returns_weights
        )[0]
        actual_block = fastplms_block(hidden_states, **masks)[0]  # (b, l, d)
        expected_feed_forward = official_block.ffn(hidden_states)  # (b, l, d)
        actual_feed_forward = fastplms_block.ffn(hidden_states)  # (b, l, d)

    assert_identical(actual_attention, expected_attention, "attention")
    if returns_weights:
        assert_identical(actual_weights, expected_weights, "attention weights")
    assert_identical(actual_feed_forward, expected_feed_forward, "feed-forward block")
    assert_identical(actual_block, expected_block, "block")


def test_the_final_norm_and_sequence_head_match(networks) -> None:
    official, fastplms, _ = networks
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(3, 9, HIDDEN, generator=generator)  # (b, l, d)

    with torch.no_grad():
        assert_identical(
            fastplms.transformer.norm(hidden_states),  # (b, l, d)
            official.transformer.norm(hidden_states),  # (b, l, d)
            "final norm",
        )
        assert_identical(
            fastplms.sequence_head(hidden_states),  # (b, l, c)
            official.sequence_head(hidden_states),  # (b, l, c)
            "sequence head",
        )


@pytest.mark.parametrize("sequences", BATCHES)
def test_every_hidden_state_and_the_logits_match(networks, sequences: tuple[str, ...]) -> None:
    official, fastplms, backend = networks
    tokens = tokenize(official, sequences)  # (b, l)
    # A tokenizer's mask: one at every token, zero at padding.
    attention_mask = tokens.ne(official.tokenizer.pad_token_id).long()  # (b, l)

    with torch.no_grad():
        expected = official_forward(official, backend, sequence_tokens=tokens)
        actual = fastplms(
            input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True
        )
        embedded = official.embed(tokens)  # (b, l, d)

    assert expected.hidden_states.shape[0] == LAYERS  # (n, b, l, d)
    assert len(actual.hidden_states) == LAYERS + 1
    assert_identical(actual.hidden_states[0], embedded, "hidden state 0, the embeddings")
    assert_outputs_match(actual, expected, LAYERS)


def test_chain_aware_sequence_ids_match(networks) -> None:
    """Integer sequence ids pack chains into one row, and -1 marks padding."""
    official, fastplms, backend = networks
    tokens = tokenize(official, ("MKTAYIAKQRQISF", "GSUZBOX"))  # (b, l) = (2, 16)
    # Two chains share the first row; the second row holds one chain, then padding.
    sequence_id = torch.tensor([[0] * 7 + [1] * 9, [0] * 9 + [-1] * 7])  # (b, l)

    with torch.no_grad():
        expected = official_forward(
            official, backend, sequence_tokens=tokens, sequence_id=sequence_id
        )
        actual = fastplms(input_ids=tokens, sequence_id=sequence_id, output_hidden_states=True)

    assert_outputs_match(actual, expected, LAYERS)


@pytest.mark.parametrize("sequences", BATCHES)
def test_the_attention_maps_match(networks, sequences: tuple[str, ...]) -> None:
    """Returning attention maps runs FastPLMs' eager attention whatever the configured backend."""
    official, fastplms, _ = networks
    tokens = tokenize(official, sequences)  # (b, l)

    with torch.no_grad():
        expected = official(sequence_tokens=tokens, output_attentions=True)
        # Without a mask, FastPLMs takes padding from the pad token, as ESMC does.
        actual = fastplms(input_ids=tokens, output_attentions=True)

    assert len(actual.attentions) == len(expected.attentions) == LAYERS
    for index, (actual_map, expected_map) in enumerate(
        zip(actual.attentions, expected.attentions, strict=True)
    ):
        assert_identical(actual_map, expected_map, f"attention map {index}")  # (b, h, l, l)
    assert_identical(actual.logits, expected.sequence_logits, "logits")  # (b, l, c)


def test_the_comparisons_see_a_padding_leak(networks) -> None:
    """Letting padded keys through moves residue outputs far outside even float32 rounding."""
    official, fastplms, backend = networks
    tokens = tokenize(official, SEQUENCES)  # (b, l)
    residues = tokens.ne(official.tokenizer.pad_token_id)  # (b, l)

    with torch.no_grad():
        expected = official_forward(official, backend, sequence_tokens=tokens).sequence_logits
        leaked = fastplms(  # (b, l, c)
            input_ids=tokens, attention_mask=torch.ones_like(tokens)
        ).logits

    with pytest.raises(AssertionError, match="exceeds"):
        assert_equal_to_rounding(
            leaked[residues],  # (r, c), r tokens present
            expected[residues],  # (r, c)
            "residue logits, padding visible",
        )


# Loading the official checkpoint.


# Three short natural proteins: ubiquitin, a zinc finger, and insulin's B chain.
NATURAL_SEQUENCES = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "YKCGLCERSFVEKSALSRHQRVH",
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
)
NATURAL_BATCHES = [
    *(
        pytest.param((sequence,), id=f"alone{index}")
        for index, sequence in enumerate(NATURAL_SEQUENCES)
    ),
    pytest.param(NATURAL_SEQUENCES, id="padded"),
]
# Biohub's native file of ESMC-300M, the network the pinned `esm.pretrained.ESMC_300M_202412`
# builds, under ESMC's own module names. That loader cannot read it: it hands the snapshot's root
# to huggingface_hub's `load_torch_model`, which finds no weights there, because the file sits in
# a subdirectory. These tests load the file as the loader means to.
NATIVE_REPOSITORY = "biohub/esmc-300m-2024-12"
NATIVE_REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
NATIVE_WEIGHTS = "data/weights/esmc_300m_2024_12_v0.pth"


def cached_snapshot(source: CheckpointSource) -> Path:
    filenames = tuple(item.path for item in source.files)
    return cached_checkpoint(source.repo_id, source.revision, filenames)


def official_checkpoint_state() -> dict[str, Tensor]:
    """The official Hub checkpoint, whose names come from Transformer Engine's fused modules."""
    return load_file(cached_snapshot(REPRESENTATIVE.official) / "model.safetensors")  # (...) one tensor per official parameter name, checkpoint-defined shapes


def native_checkpoint_state() -> dict[str, Tensor]:
    snapshot = cached_checkpoint(NATIVE_REPOSITORY, NATIVE_REVISION, (NATIVE_WEIGHTS,))
    return torch.load(snapshot / NATIVE_WEIGHTS, map_location="cpu", weights_only=True)  # (...) one tensor per native parameter name, checkpoint-defined shapes


@pytest.fixture(scope="module")
def native_official(biohub_esm: OfficialPackage) -> nn.Module:
    """ESMC at the official checkpoint's shape, holding its state through the declared transform."""
    config = json.loads((cached_snapshot(REPRESENTATIVE.official) / "config.json").read_text())
    official = biohub_esm["esm.models.esmc"].ESMC(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        tokenizer=biohub_esm["esm.tokenization"].EsmSequenceTokenizer(),
        use_flash_attn=False,
    ).eval()
    state = apply_state_transform(
        STATE_TRANSFORM, official_checkpoint_state(), expected_keys=official.state_dict()
    )
    official.load_state_dict(state, strict=True)
    return official


@pytest.fixture(scope="module", params=BACKENDS)
def checkpoint_networks(
    request, native_official: nn.Module
) -> tuple[nn.Module, ESMplusplusForMaskedLM, str]:
    return native_official, build_fastplms(native_official, request.param), request.param


@pytest.mark.checkpoint
def test_the_official_checkpoint_loads_key_for_key(checkpoint_networks) -> None:
    _, fastplms, _ = checkpoint_networks
    official_state = official_checkpoint_state()

    state = apply_state_transform(
        STATE_TRANSFORM, official_state, expected_keys=fastplms.state_dict()
    )
    missing, unexpected = fastplms.load_state_dict(state, strict=False)

    assert (missing, unexpected) == ([], [])
    # The transform drops only Transformer Engine's per-module extra states, which are empty here.
    dropped = {name for name in official_state if name.endswith("._extra_state")}
    assert len(state) == len(official_state) - len(dropped)
    assert all(official_state[name].numel() == 0 for name in dropped)
    for name, tensor in fastplms.state_dict().items():
        assert_identical(tensor, state[name], name)


@pytest.mark.checkpoint
def test_the_declared_transform_recovers_biohubs_native_file() -> None:
    """The official checkpoint, renamed by the transform, is Biohub's native state exactly."""
    native_state = native_checkpoint_state()

    state = apply_state_transform(STATE_TRANSFORM, official_checkpoint_state())

    assert state.keys() == native_state.keys()
    for name, tensor in native_state.items():
        assert_identical(state[name], tensor, name)


@pytest.mark.checkpoint
@pytest.mark.parametrize("sequences", NATURAL_BATCHES)
def test_the_official_checkpoint_matches_esmc(
    checkpoint_networks, sequences: tuple[str, ...]
) -> None:
    official, fastplms, backend = checkpoint_networks
    tokens = tokenize(official, sequences)  # (b, l)
    n = len(official.transformer.blocks)

    with torch.no_grad():
        expected = official_forward(official, backend, sequence_tokens=tokens)
        actual = fastplms(input_ids=tokens, output_hidden_states=True)

    assert_outputs_match(actual, expected, n)


@pytest.mark.checkpoint
def test_the_published_checkpoint_holds_the_official_tensors() -> None:
    """FastPLMs' published ESM++ small is the official checkpoint under the declared transform."""
    published = load_file(cached_snapshot(REPRESENTATIVE.fast) / "model.safetensors")

    state = apply_state_transform(STATE_TRANSFORM, official_checkpoint_state())

    assert published.keys() == state.keys()
    for name, tensor in state.items():
        assert_identical(published[name], tensor, name)


@pytest.mark.checkpoint
def test_the_published_checkpoint_loads_completely_and_matches_esmc(
    native_official: nn.Module,
) -> None:
    fastplms, loading = ESMplusplusForMaskedLM.from_pretrained(
        cached_snapshot(REPRESENTATIVE.fast), attn_implementation="sdpa", output_loading_info=True
    )
    fastplms.eval()
    tokens = tokenize(native_official, NATURAL_SEQUENCES)  # (b, l)

    with torch.no_grad():
        expected = native_official(sequence_tokens=tokens)
        actual = fastplms(input_ids=tokens)

    assert {name: sorted(keys) for name, keys in loading.items()} == {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    assert_identical(  # (b, l, d)
        actual.last_hidden_state, expected.embeddings, "last hidden state"
    )
    assert_identical(actual.logits, expected.sequence_logits, "logits")  # (b, l, c)


@pytest.mark.checkpoint
@pytest.mark.parametrize("source", ["official", "fast"])
def test_the_hub_tokenizers_match_esmc(native_official: nn.Module, source: str) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        cached_snapshot(getattr(REPRESENTATIVE, source))
    )
    sequences = (*NATURAL_SEQUENCES, "ACDEFGHIKLMNPQRSTVWYXBUZO", "MK<mask>TA<unk>Y|GS")

    encoded = tokenizer(list(sequences), padding=True, return_tensors="pt")
    expected = tokenize(native_official, sequences)  # (b, l)

    assert tokenizer.get_vocab() == native_official.tokenizer.get_vocab()
    assert_identical(encoded["input_ids"], expected)  # (b, l)
    padding = native_official.tokenizer.pad_token_id
    assert_identical(encoded["attention_mask"], expected.ne(padding).long())  # (b, l)
