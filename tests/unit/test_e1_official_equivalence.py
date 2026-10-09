"""FastPLMs' E1 computes what Profluent's E1 computes, module by module, on CPU.

The official side is Profluent's `E1` package at the commit `models.toml` pins, imported unchanged
from its pinned tree by `tests.parity.support.pinned_oracles`; without that tree these tests skip.
The official network is built small and filled with random weights, and FastPLMs' network loads
its state under the same names: the declared transform, `e1_to_fastplms_v1`, only stores floating
tensors in BF16, and these tests run in float32.

Per-token pieces run the same float32 operations on the same rows on both sides and are compared
exactly (`rtol=0, atol=0`): batch preparation, tokenization, embeddings, rotary tables, RMS norms,
feed-forward blocks, attention masks, the language-model head, and the loss. Attention itself
differs in kernel. Without a CUDA device the official code packs each row's sequences and runs
PyTorch's FlexAttention, which on CPU evaluates the unfused scores with its own reductions;
FastPLMs' default runs SDPA over the padded batch with dense masks. Those comparisons are held to
float32 rounding (`assert_equal_to_rounding`), and a test below shows that bound rejects a mask
that lets one sequence of a multi-sequence input see the next.

E1's within-sequence layers attend inside each sequence of a row, and its global layers, every
third layer here as in the checkpoints, attend block-causally across the row's sequences. A row
of comma-separated sequences, context first and query last, exercises both.

The loading tests at the end read the official 150M checkpoint from the Hugging Face cache and
carry the `checkpoint` marker.

Shape symbols: `b` batch, `l` sequence length, `d` hidden width, `h` attention heads, `h_kv`
key-value heads, `d_h` per-head width, `c` the vocabulary, `n` layers.
"""

from __future__ import annotations

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

from fastplms.attention import AttentionBackend
from fastplms.models.e1.modeling_e1 import (
    DataPrepConfig,
    E1BatchPreparer,
    E1Config,
    E1ForMaskedLM,
    RMSNorm,
    RotaryPositionalEmbedding,
    build_block_causal_mask_4d,
    build_within_seq_mask_4d,
    get_tokenizer,
)
from fastplms.registry import get_model_spec
from tools.conversion import apply_state_transform


REPRESENTATIVE = get_model_spec("e1_150m")
STATE_TRANSFORM = REPRESENTATIVE.family.state_transform
SEED = 0
WEIGHT_SCALE = 0.3
# Two small networks with the checkpoints' layer pattern: two within-sequence layers, then one
# global layer. The first has the checkpoints' gated feed-forward and query-key clipping and adds
# grouped key-value heads; the second has plain multi-head attention and an ungated block.
NETWORK_SHAPES = {
    "gated_grouped": {
        "hidden_size": 32,  # d
        "intermediate_size": 48,
        "gated_mlp": True,
        "num_hidden_layers": 3,  # n
        "num_attention_heads": 4,  # h, so d_h = 8
        "num_key_value_heads": 2,  # h_kv
        "global_attention_every_n_layers": 3,
        "clip_qkv": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta_global": 500000.0,
        "max_num_positions_global": 65536,
    },
    "plain_multihead": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "gated_mlp": False,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "global_attention_every_n_layers": 3,
        "rms_norm_eps": 1e-5,
    },
}
# Unequal lengths; masked residues ("?"); a row of context and query sequences; rare letters.
SEQUENCES = ("MKTAYIAKQR?QISFVKSHF", "GSUZBOX,ACD?EFW,MKV?L", "MA")
OFFICIAL_MODULES = (
    "E1.batch_preparer",
    "E1.modeling",
    "E1.model.attention",
    "E1.model.ffn",
    "E1.model.flex_attention",
    "E1.tokenizer",
)


@pytest.fixture(scope="module")
def official_e1() -> OfficialPackage:
    # Without CUDA the official code runs FlexAttention through Dynamo's eager backend, as the
    # suite's hermetic environment hides the GPU. With a GPU visible it compiles FlexAttention
    # with Inductor, which on CPU tensors needs a C++ toolchain this test does not assume.
    if torch.cuda.is_available():
        pytest.skip("the official E1 compiles FlexAttention with Inductor when CUDA is visible")
    return official_package("e1", "E1", OFFICIAL_MODULES, source_directory="src")


def build_official(official_e1: OfficialPackage, shape: dict[str, object]) -> nn.Module:
    config = official_e1["E1.modeling"].E1Config(**shape, torch_dtype="float32")
    official = official_e1["E1.modeling"].E1ForMaskedLM(config).eval()
    randomize_parameters(official, SEED, WEIGHT_SCALE)
    return official


def build_fastplms(official: nn.Module, shape: dict[str, object]) -> E1ForMaskedLM:
    fastplms = E1ForMaskedLM(E1Config(**shape, dtype="float32", attn_backend="sdpa")).eval()
    fastplms.load_state_dict(official.state_dict(), strict=True)
    return fastplms


@pytest.fixture(scope="module", params=sorted(NETWORK_SHAPES))
def networks(request, official_e1: OfficialPackage) -> tuple[nn.Module, E1ForMaskedLM]:
    shape = NETWORK_SHAPES[request.param]
    official = build_official(official_e1, shape)
    return official, build_fastplms(official, shape)


def official_batch(official_e1: OfficialPackage, sequences: tuple[str, ...]) -> dict[str, Tensor]:
    """The official preparer's padded batch: (b, l) ids, positions, sequence ids, and labels."""
    preparer = official_e1["E1.batch_preparer"].E1BatchPreparer(device=torch.device("cpu"))
    batch = preparer.get_batch_kwargs(list(sequences))
    return {  # (...) input_ids, within_seq_position_ids, global_position_ids, sequence_ids, labels: each (b, l)
        name: batch[name]
        for name in (
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
            "labels",
        )
    }


def model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    # batch: (b, l) one tensor per name, including labels
    return {name: value for name, value in batch.items() if name != "labels"}  # (b, l) the same tensors without labels


# Preparing sequences.


@pytest.mark.parametrize("remove_x_tokens", [False, True])
@pytest.mark.parametrize("preserve_context_labels", [False, True])
def test_batch_preparation_matches(
    official_e1: OfficialPackage, remove_x_tokens: bool, preserve_context_labels: bool
) -> None:
    official_module = official_e1["E1.batch_preparer"]
    official = official_module.E1BatchPreparer(
        data_prep_config=official_module.DataPrepConfig(remove_X_tokens=remove_x_tokens),
        preserve_context_labels=preserve_context_labels,
        device=torch.device("cpu"),
    )
    fastplms = E1BatchPreparer(
        data_prep_config=DataPrepConfig(remove_X_tokens=remove_x_tokens),
        tokenizer=get_tokenizer(),
        preserve_context_labels=preserve_context_labels,
    )
    sequences = [*SEQUENCES, "XAXC,DX?"]

    expected = official.get_batch_kwargs(sequences)
    actual = fastplms.get_batch_kwargs(sequences)

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, Tensor):
            assert_identical(actual[name], value, name)  # (b, l)
        else:
            assert actual[name] == value, name


def test_the_tokenizers_hold_the_same_vocabulary_and_encodings(
    official_e1: OfficialPackage,
) -> None:
    official = official_e1["E1.tokenizer"].get_tokenizer()
    fastplms = get_tokenizer()
    text = "<bos>1ACDEFGHIKLMNPQRSTVWYXBUZOJ?2<eos><pad>"

    assert fastplms.get_vocab() == official.get_vocab()
    assert fastplms.encode(text).ids == official.encode(text).ids
    assert fastplms.padding["pad_id"] == official.padding["pad_id"] == 0


# The state maps one to one.


def test_the_official_state_loads_key_for_key(networks) -> None:
    official, fastplms = networks
    official_state = official.state_dict()

    assert fastplms.state_dict().keys() == official_state.keys()
    for name, tensor in fastplms.state_dict().items():
        assert_identical(tensor, official_state[name], name)


def test_the_declared_transform_only_stores_floats_in_bf16(networks) -> None:
    official, fastplms = networks
    official_state = official.state_dict()

    state = apply_state_transform(
        STATE_TRANSFORM, official_state, expected_keys=fastplms.state_dict()
    )

    for name, tensor in official_state.items():
        assert_identical(state[name], tensor.to(torch.bfloat16), name)


# The network, component by component.


@pytest.mark.parametrize(("base", "max_positions"), [(10000.0, 8192), (500000.0, 65536)])
def test_the_rotary_embeddings_match(
    official_e1: OfficialPackage, base: float, max_positions: int
) -> None:
    official = official_e1["E1.model.attention"].RotaryPositionalEmbedding(
        8, max_position_embeddings=max_positions, base=base
    )
    fastplms = RotaryPositionalEmbedding(8, max_position_embeddings=max_positions, base=base)
    batch = official_batch(official_e1, SEQUENCES)
    generator = torch.Generator().manual_seed(SEED)

    residues = batch["sequence_ids"].ne(-1)  # (b, l)
    for positions in (batch["within_seq_position_ids"], batch["global_position_ids"]):  # (b, l)
        b, l = positions.shape
        Q = torch.randn(b, l, 4, 8, generator=generator)  # (b, l, h, d_h)
        K = torch.randn(b, l, 2, 8, generator=generator)  # (b, l, h_kv, d_h)
        expected_q, expected_k = official(Q, K, positions)
        actual_q, actual_k = fastplms(Q, K, positions)
        # Padding carries position -1, which reads each table's last row: the official table is
        # built to the configured maximum and FastPLMs' to the batch's longest position, so the
        # two rotate padding differently. No padding key is ever attended, which the whole-model
        # comparisons below confirm at every position.
        assert_identical(actual_q[residues], expected_q[residues], "rotated queries")  # (r, h, d_h)
        assert_identical(actual_k[residues], expected_k[residues], "rotated keys")  # (r, h_kv, d_h)


def test_the_norms_feed_forward_blocks_and_head_match(networks) -> None:
    official, fastplms = networks
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(3, 11, 32, generator=generator)  # (b, l, d)

    with torch.no_grad():
        for index, (official_layer, fastplms_layer) in enumerate(
            zip(official.model.layers, fastplms.model.layers, strict=True)
        ):
            official_block = official_layer.norm_attn_norm
            fastplms_block = fastplms_layer.norm_attn_norm
            for norm in ("input_layernorm", "post_attention_layernorm"):
                assert isinstance(getattr(fastplms_block, norm), RMSNorm)
                assert_identical(
                    getattr(fastplms_block, norm)(hidden_states),  # (b, l, d)
                    getattr(official_block, norm)(hidden_states),  # (b, l, d)
                    f"layer {index} {norm}",
                )
            assert_identical(
                fastplms_layer.ffn(hidden_states),  # (b, l, d)
                official_layer.ffn(hidden_states),  # (b, l, d)
                f"layer {index} feed-forward",
            )
        assert_identical(
            fastplms.model.norm(hidden_states),  # (b, l, d)
            official.model.norm(hidden_states),  # (b, l, d)
            "final norm",
        )
        assert_identical(
            fastplms.mlm_head(hidden_states),  # (b, l, c)
            official.mlm_head(hidden_states),  # (b, l, c)
            "language-model head",
        )


def test_the_attention_masks_match_the_official_block_masks(official_e1: OfficialPackage) -> None:
    """FastPLMs' dense masks allow exactly the pairs the official FlexAttention masks allow."""
    sequence_ids = official_batch(official_e1, SEQUENCES)["sequence_ids"]  # (b, l)
    b, l = sequence_ids.shape
    flex_attention = official_e1["E1.model.flex_attention"]
    block_mask = flex_attention.create_block_causal_mask_optimized(sequence_ids)
    batch_index = torch.arange(b)[:, None, None]  # (b, 1, 1)
    query_index = torch.arange(l)[None, :, None]  # (1, l, 1)
    key_index = torch.arange(l)[None, None, :]  # (1, 1, l)
    head_index = torch.zeros((), dtype=torch.long)
    official_block_causal = block_mask.mask_mod(  # (b, l, l)
        batch_index, head_index, query_index, key_index
    )
    # The official within-sequence path unpads each row and attends inside each sequence.
    valid = sequence_ids.ne(-1)  # (b, l)
    same_sequence = sequence_ids[:, :, None].eq(sequence_ids[:, None, :])  # (b, l, l)
    official_within = same_sequence & valid[:, :, None] & valid[:, None, :]  # (b, l, l)

    assert_identical(build_block_causal_mask_4d(sequence_ids), official_block_causal[:, None])
    assert_identical(build_within_seq_mask_4d(sequence_ids), official_within[:, None])


def official_attention_args(
    official_e1: OfficialPackage, sequence_ids: Tensor
) -> dict[str, object]:
    """The attention arguments the official encoder builds before its first layer."""
    # sequence_ids: (b, l)
    flex_attention = official_e1["E1.model.flex_attention"]
    block_mask = flex_attention.create_block_causal_mask_optimized(sequence_ids)
    return {"flex_attention_args": {"block_mask": block_mask}}


@pytest.mark.parametrize("layer_index", [0, 2], ids=["within_sequence", "global"])
def test_each_layer_matches_part_by_part(
    official_e1: OfficialPackage, networks, layer_index: int
) -> None:
    """The attention module and the whole decoder layer, for each kind of attention layer."""
    official, fastplms = networks
    batch = model_inputs(official_batch(official_e1, SEQUENCES))
    b, l = batch["input_ids"].shape
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(b, l, 32, generator=generator)  # (b, l, d)
    positions = {
        name: batch[name]  # (b, l)
        for name in ("within_seq_position_ids", "global_position_ids", "sequence_ids")
    }
    official_args = official_attention_args(official_e1, batch["sequence_ids"])
    fastplms_args = fastplms.model._build_forward_attention_args(
        batch["sequence_ids"], None, AttentionBackend.SDPA
    )
    official_layer = official.model.layers[layer_index]
    fastplms_layer = fastplms.model.layers[layer_index]
    valid = batch["sequence_ids"].ne(-1)  # (b, l)

    with torch.no_grad():
        expected_attention = official_layer.norm_attn_norm.self_attn(
            hidden_states, **positions, attention_args=official_args
        )[0]  # (b, l, d)
        actual_attention = fastplms_layer.norm_attn_norm.self_attn(
            hidden_states, **positions, attention_args=fastplms_args
        )[0]  # (b, l, d)
        expected_layer = official_layer(hidden_states, **positions, attention_args=official_args)[0]
        actual_layer = fastplms_layer(hidden_states, **positions, attention_args=fastplms_args)[0]

    assert_equal_to_rounding(actual_attention, expected_attention, "attention")
    assert_equal_to_rounding(actual_layer, expected_layer, "decoder layer")  # (b, l, d)
    # Residue positions carry the biology; padding positions still agree to rounding above.
    assert_equal_to_rounding(actual_layer[valid], expected_layer[valid], "decoder layer residues")


def test_every_hidden_state_the_logits_and_the_loss_match(
    official_e1: OfficialPackage, networks
) -> None:
    official, fastplms = networks
    batch = official_batch(official_e1, SEQUENCES)

    with torch.no_grad():
        expected = official(**batch, output_hidden_states=True)
        actual = fastplms(**batch, output_hidden_states=True)

    n = official.config.num_hidden_layers
    assert len(actual.hidden_states) == len(expected.hidden_states) == n + 1
    # The embeddings, token plus sequence-index, are exact; every later state went through
    # attention.
    assert_identical(actual.hidden_states[0], expected.hidden_states[0], "embeddings")  # (b, l, d)
    for index in range(1, n + 1):
        assert_equal_to_rounding(
            actual.hidden_states[index],  # (b, l, d)
            expected.hidden_states[index],  # (b, l, d)
            f"hidden state {index}",
        )
    assert_equal_to_rounding(actual.last_hidden_state, expected.embeddings, "last hidden state")
    assert_equal_to_rounding(actual.logits, expected.logits, "logits")  # (b, l, c)
    assert_equal_to_rounding(actual.loss, expected.loss, "loss")  # ()


def test_the_rounding_bound_rejects_context_leaking_into_its_sequences(
    official_e1: OfficialPackage, networks
) -> None:
    """Merging a row's sequences into one lets every token see its neighbors' residues."""
    official, fastplms = networks
    batch = model_inputs(official_batch(official_e1, SEQUENCES))
    merged = dict(batch)
    merged["sequence_ids"] = batch["sequence_ids"].clamp(max=0)  # (b, l), padding stays -1

    with torch.no_grad():
        expected = official(**batch).logits  # (b, l, c)
        leaked = fastplms(**merged).logits  # (b, l, c)

    with pytest.raises(AssertionError, match="exceeds"):
        assert_equal_to_rounding(leaked, expected, "logits with merged sequences")


# Loading the official checkpoint.


# Two short natural proteins, then the second as the query of a one-homolog context.
NATURAL_SEQUENCES = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKA,FVNQHLCGSHLVEALYLV?GERGFFYTPKT",
)


def official_checkpoint() -> Path:
    return cached_checkpoint(
        REPRESENTATIVE.official.repo_id,
        REPRESENTATIVE.official.revision,
        tuple(item.path for item in REPRESENTATIVE.official.files),
    )


def load_official(official_e1: OfficialPackage, snapshot: Path) -> nn.Module:
    """The official network built on the CPU, holding the checkpoint's state in float32.

    The official `from_pretrained` is not usable here. Transformers 5 builds a model on the meta
    device before it loads weights, and the official rotary module computes its tables in its
    constructor as non-persistent buffers, which no checkpoint holds, so they arrive holding
    whatever memory they were given. `test_the_official_loader_leaves_its_rotary_tables_unset`
    records that. The official constructor on a real device computes the tables as E1 intends.
    """
    modeling = official_e1["E1.modeling"]
    official = modeling.E1ForMaskedLM(modeling.E1Config.from_pretrained(snapshot)).eval()
    state = load_file(snapshot / "model.safetensors")
    official.load_state_dict({name: tensor.float() for name, tensor in state.items()}, strict=True)
    return official


@pytest.mark.checkpoint
def test_the_official_loader_leaves_its_rotary_tables_unset(official_e1: OfficialPackage) -> None:
    """Why the loading tests build the official network directly: its loader breaks here.

    This fails once the official loader computes the tables, and `load_official` can then use it.
    """
    loaded = official_e1["E1.modeling"].E1ForMaskedLM.from_pretrained(
        official_checkpoint(), dtype=torch.float32
    )
    rotary = loaded.model.layers[0].norm_attn_norm.self_attn.rotary_emb
    rebuilt = type(rotary)(rotary.dim, rotary.max_position_embeddings, rotary.base)

    assert rebuilt.cos_cached[0].eq(1).all()  # cos 0 = 1 at position 0
    assert not torch.equal(rotary.cos_cached, rebuilt.cos_cached)
    assert not torch.equal(rotary.inv_freq, rebuilt.inv_freq)


@pytest.mark.checkpoint
def test_the_official_checkpoint_loads_completely_and_matches_e1(
    official_e1: OfficialPackage,
) -> None:
    snapshot = official_checkpoint()
    official = load_official(official_e1, snapshot)
    fastplms, loading = E1ForMaskedLM.from_pretrained(
        snapshot, dtype=torch.float32, attn_backend="sdpa", output_loading_info=True
    )
    fastplms.eval()
    batch = official_batch(official_e1, NATURAL_SEQUENCES)

    with torch.no_grad():
        expected = official(**batch)
        actual = fastplms(**batch)

    assert {name: sorted(keys) for name, keys in loading.items()} == {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    official_state = official.state_dict()
    for name, tensor in fastplms.state_dict().items():
        assert_identical(tensor, official_state[name], name)
    assert_equal_to_rounding(actual.logits, expected.logits, "logits")  # (b, l, c)
    assert_equal_to_rounding(actual.last_hidden_state, expected.embeddings, "last hidden state")
    assert_equal_to_rounding(actual.loss, expected.loss, "loss")


@pytest.mark.checkpoint
def test_the_declared_transform_maps_the_official_checkpoint(official_e1: OfficialPackage) -> None:
    official_state = load_file(official_checkpoint() / "model.safetensors")
    shape = E1Config.from_pretrained(official_checkpoint())
    with torch.device("meta"):
        fastplms = E1ForMaskedLM(shape)

    state = apply_state_transform(
        STATE_TRANSFORM, official_state, expected_keys=fastplms.state_dict()
    )

    assert state.keys() == official_state.keys()
    for name, tensor in official_state.items():
        assert tensor.dtype == torch.bfloat16, name
        assert_identical(state[name], tensor, name)
