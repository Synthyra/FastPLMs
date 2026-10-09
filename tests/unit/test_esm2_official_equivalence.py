"""FastPLMs' ESM2 computes what Meta's fair-esm computes, module by module, on CPU.

The official side is fair-esm at the commit `models.toml` pins, imported unchanged from its pinned
tree by `tests.parity.support.pinned_oracles`; without that tree these tests skip. fair-esm builds
a small network and fills it with random weights, and FastPLMs receives its state through the
transform the manifest declares for ESM2, `esm2_hf_to_fastplms_v1`.

Where both sides run the same float32 operations on the same rows in the same order, the
comparison is exact (`rtol=0, atol=0`). That covers the state, rotary tables, embeddings, masks,
activation, final norm, and head, and, with eager attention and one sequence, every layer, hidden
state, attention map, contact map, and logit. fair-esm lays activations out as (l, b, d) and
FastPLMs as (b, l, d). For one sequence those are the same memory. For several, each projection
hands the CPU GEMM its rows in another order, and the GEMM tiles, and so rounds, a few rows
differently. FastPLMs' default SDPA also fuses the attention products into one kernel that sums
in another order. Those comparisons are held to float32 rounding (`assert_equal_to_rounding`),
and `test_the_rounding_bound_rejects_a_padding_leak` shows that bound fails a real fault.

fair-esm marks padding with a key-padding mask, where True hides a key; FastPLMs marks the keys
each query may see, where True shows one. Either gives a hidden key's score minus infinity.

The loading tests at the end read the official 8M checkpoint from local caches, never the network,
and carry the `checkpoint` marker.

Shape symbols: `b` batch, `l` sequence length, `d` hidden width, `h` attention heads, `d_h`
per-head width, `c` the vocabulary, `n` layers.
"""

from __future__ import annotations

import os
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

from fastplms.attention import AttentionBackend, get_attention_mask
from fastplms.digests import file_sha256
from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmTokenizer,
)
from fastplms.registry import get_model_spec
from tools.conversion import apply_state_transform


REPRESENTATIVE = get_model_spec("esm2_8m")
STATE_TRANSFORM = REPRESENTATIVE.family.state_transform
LAYERS = 2  # n
HIDDEN = 32  # d
HEADS = 4  # h, so d_h = 8
SEED = 0
# Wide enough that attention is far from uniform, so a key the mask should hide would show.
WEIGHT_SCALE = 0.3
BACKENDS = ("eager", "sdpa")
# Right-padded to unequal lengths, with masked residues and the rarer residue letters.
SEQUENCES = ("MKTAYIAK<mask>RQISFVKSHF", "GSUZ<mask>BOX", "MA<mask>")
SEQUENCES_WITHOUT_PADDING = ("MKTAYIAKQR", "LVSG<mask>AAGEW")


@pytest.fixture(scope="module")
def fair_esm() -> OfficialPackage:
    return official_package(
        "fair-esm", "esm", ("esm", "esm.model.esm2", "esm.modules", "esm.pretrained")
    )


def build_official(fair_esm: OfficialPackage, token_dropout: bool) -> nn.Module:
    """fair-esm's ESM2 at a small width, with random weights."""
    alphabet = fair_esm["esm"].data.Alphabet.from_architecture("ESM-1b")
    official = fair_esm["esm.model.esm2"].ESM2(
        num_layers=LAYERS,
        embed_dim=HIDDEN,
        attention_heads=HEADS,
        alphabet=alphabet,
        token_dropout=token_dropout,
    ).eval()
    randomize_parameters(official, SEED, WEIGHT_SCALE)
    return official


def fastplms_config(official: nn.Module, backend: str) -> FastEsmConfig:
    return FastEsmConfig(
        vocab_size=official.alphabet_size,
        hidden_size=official.embed_dim,
        num_hidden_layers=official.num_layers,
        num_attention_heads=official.attention_heads,
        intermediate_size=4 * official.embed_dim,
        pad_token_id=official.padding_idx,
        mask_token_id=official.mask_idx,
        # fair-esm's layer norms are torch's LayerNorm at its default epsilon.
        layer_norm_eps=1e-5,
        token_dropout=official.token_dropout,
        emb_layer_norm_before=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        attn_backend=backend,
    )


def build_fastplms(official: nn.Module, backend: str) -> FastEsmForMaskedLM:
    """FastPLMs' ESM2 holding the official state, mapped by the declared transform."""
    fastplms = FastEsmForMaskedLM(fastplms_config(official, backend)).eval()
    state = apply_state_transform(
        STATE_TRANSFORM, official.state_dict(), expected_keys=fastplms.state_dict()
    )
    fastplms.load_state_dict(state, strict=True)
    return fastplms


@pytest.fixture(scope="module", params=[True, False], ids=["token_dropout", "no_token_dropout"])
def official(request, fair_esm: OfficialPackage) -> nn.Module:
    return build_official(fair_esm, token_dropout=request.param)


@pytest.fixture(scope="module", params=BACKENDS)
def networks(request, official: nn.Module) -> tuple[nn.Module, FastEsmForMaskedLM, str]:
    return official, build_fastplms(official, request.param), request.param


def tokenize(official: nn.Module, sequences: tuple[str, ...]) -> Tensor:
    """Token ids as fair-esm's batch converter lays them out: cls, residues, eos, then padding."""
    converter = official.alphabet.get_batch_converter()
    _, _, tokens = converter([(str(index), sequence) for index, sequence in enumerate(sequences)])
    return tokens  # (b, l)


def assert_equivalent(actual: Tensor, expected: Tensor, exact: bool, what: str) -> None:
    # actual, expected: (...) equal shapes of any rank
    if exact:
        assert_identical(actual, expected, what)
    else:
        assert_equal_to_rounding(actual, expected, what)


def is_exact(backend: str, batch_size: int) -> bool:
    """Eager attention on one sequence runs the official arithmetic row for row."""
    return backend == "eager" and batch_size == 1


def fastplms_masks(attention_mask: Tensor, backend: str) -> dict[str, Tensor | None]:
    """The masks FastPLMs' encoder builds once and hands every layer."""
    # attention_mask: (b, l)
    b, l = attention_mask.shape
    mask_2d, mask_4d, _ = get_attention_mask(  # (b, l); (b, 1, 1, l)
        effective_backend=AttentionBackend(backend),
        batch_size=b,
        seq_len=l,
        device=attention_mask.device,
        attention_mask=attention_mask,
        dtype=torch.float32,
    )
    return {"attention_mask_2d": mask_2d, "attention_mask_4d": mask_4d}  # (...) attention_mask_2d (b, l), attention_mask_4d (b, 1, 1, l)


# Inputs: every sequence alone, then the padded and unpadded batches.
BATCHES = [
    *(pytest.param((sequence,), id=f"alone{index}") for index, sequence in enumerate(SEQUENCES)),
    pytest.param(SEQUENCES, id="padded"),
    pytest.param(SEQUENCES_WITHOUT_PADDING, id="unpadded"),
]


def test_the_batches_pad_where_their_names_say(fair_esm: OfficialPackage) -> None:
    official = build_official(fair_esm, token_dropout=True)

    assert tokenize(official, SEQUENCES).eq(official.padding_idx).any()
    assert not tokenize(official, SEQUENCES_WITHOUT_PADDING).eq(official.padding_idx).any()


# The state maps one to one.


def test_the_declared_transform_maps_every_official_tensor(official: nn.Module) -> None:
    fastplms = FastEsmForMaskedLM(fastplms_config(official, "sdpa"))
    official_state = official.state_dict()

    state = apply_state_transform(
        STATE_TRANSFORM, official_state, expected_keys=fastplms.state_dict()
    )
    missing, unexpected = fastplms.load_state_dict(state, strict=False)

    assert (missing, unexpected) == ([], [])
    assert state.keys() == fastplms.state_dict().keys()
    # One FastPLMs tensor per official name: fair-esm stores its tied output embedding under
    # two names, and FastPLMs keeps the two as independent tensors.
    assert len(official_state) == len(state)
    for official_name, name in (
        ("embed_tokens.weight", "esm.embeddings.word_embeddings.weight"),
        ("lm_head.weight", "lm_head.decoder.weight"),
        ("layers.1.self_attn.q_proj.weight", "esm.encoder.layer.1.attention.self.query.weight"),
        ("layers.0.fc2.bias", "esm.encoder.layer.0.output.dense.bias"),
        ("contact_head.regression.weight", "esm.contact_head.regression.weight"),
    ):
        assert_identical(state[name], official_state[official_name], name)


def test_fastplms_unties_the_output_embedding_the_official_network_ties(
    official: nn.Module,
) -> None:
    fastplms = build_fastplms(official, "sdpa")

    assert official.lm_head.weight is official.embed_tokens.weight
    assert fastplms.lm_head.decoder.weight is not fastplms.esm.embeddings.word_embeddings.weight
    assert_identical(fastplms.lm_head.decoder.weight, official.lm_head.weight)


def test_both_build_the_same_rotary_frequencies_before_any_copy(fair_esm: OfficialPackage) -> None:
    official = build_official(fair_esm, token_dropout=True)
    fastplms = FastEsmForMaskedLM(fastplms_config(official, "sdpa"))

    for official_layer, fastplms_layer in zip(
        official.layers, fastplms.esm.encoder.layer, strict=True
    ):
        assert_identical(
            fastplms_layer.attention.self.rotary_embeddings.inv_freq,  # (d_h / 2,)
            official_layer.self_attn.rot_emb.inv_freq,  # (d_h / 2,)
        )


# The network, component by component.


def test_the_rotary_embeddings_match_across_lengths(networks) -> None:
    official, fastplms, _ = networks
    official_rotary = official.layers[0].self_attn.rot_emb
    fastplms_rotary = fastplms.esm.encoder.layer[0].attention.self.rotary_embeddings
    generator = torch.Generator().manual_seed(SEED)

    # Shorter, longer, and repeated lengths exercise both implementations' table caches.
    for l in (12, 5, 12, 30, 7):
        Q = torch.randn(3, HEADS, l, HIDDEN // HEADS, generator=generator)  # (b, h, l, d_h)
        K = torch.randn(3, HEADS, l, HIDDEN // HEADS, generator=generator)  # (b, h, l, d_h)
        # fair-esm rotates heads flattened into the batch: (b * h, l, d_h).
        expected_q, expected_k = official_rotary(Q.flatten(0, 1), K.flatten(0, 1))
        actual_q, actual_k = fastplms_rotary(Q, K)  # (b, h, l, d_h); (b, h, l, d_h)
        assert_identical(actual_q.flatten(0, 1), expected_q, f"query, l={l}")
        assert_identical(actual_k.flatten(0, 1), expected_k, f"key, l={l}")


@pytest.mark.parametrize("sequences", BATCHES)
def test_the_embeddings_match(networks, sequences: tuple[str, ...]) -> None:
    """Token dropout's rescaling and padding's zeroing are elementwise, so exact in any batch."""
    official, fastplms, _ = networks
    tokens = tokenize(official, sequences)  # (b, l)

    with torch.no_grad():
        expected = official(tokens, repr_layers=[0])["representations"][0]  # (b, l, d)
        actual = fastplms.esm.embeddings(  # (b, l, d)
            input_ids=tokens, attention_mask=tokens.ne(official.padding_idx)
        )

    assert_identical(actual, expected)


def test_the_padding_masks_hide_the_same_keys(networks) -> None:
    official, _, backend = networks
    tokens = tokenize(official, SEQUENCES)  # (b, l)
    key_padding = tokens.eq(official.padding_idx)  # (b, l), True hides a key

    visible = fastplms_masks(tokens.ne(official.padding_idx), backend)["attention_mask_4d"]

    assert visible.shape == (len(SEQUENCES), 1, 1, tokens.shape[1])  # (b, 1, 1, l)
    assert torch.equal(visible, ~key_padding[:, None, None, :])


@pytest.mark.parametrize("layer_index", range(LAYERS))
@pytest.mark.parametrize("sequences", [(SEQUENCES[0],), SEQUENCES], ids=["alone", "padded"])
def test_each_layer_matches_part_by_part(
    fair_esm: OfficialPackage, networks, sequences: tuple[str, ...], layer_index: int
) -> None:
    """Self-attention, the feed-forward activation, and the whole pre-norm layer."""
    official, fastplms, backend = networks
    official_layer = official.layers[layer_index]
    fastplms_layer = fastplms.esm.encoder.layer[layer_index]
    tokens = tokenize(official, sequences)  # (b, l)
    b, l = tokens.shape
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(b, l, HIDDEN, generator=generator)  # (b, l, d)
    key_padding = tokens.eq(official.padding_idx)  # (b, l)
    masks = fastplms_masks(~key_padding, backend)
    # fair-esm's layout, laid out contiguously as the layer norm before its attention returns it.
    official_input = hidden_states.transpose(0, 1).contiguous()  # (l, b, d)
    # fair-esm drops an all-false key-padding mask before its layers see it.
    official_padding = key_padding if key_padding.any() else None

    with torch.no_grad():
        pairs = {
            "self-attention": (
                official_layer.self_attn(  # (l, b, d)
                    query=official_input,
                    key=official_input,
                    value=official_input,
                    key_padding_mask=official_padding,
                    need_weights=False,
                )[0].transpose(0, 1),
                fastplms_layer.attention.output.dense(  # (b, l, d)
                    fastplms_layer.attention.self(hidden_states, **masks)[0]
                ),
            ),
            "layer": (
                official_layer(  # (l, b, d), then (b, l, d)
                    official_input, self_attn_padding_mask=official_padding
                )[0].transpose(0, 1),
                fastplms_layer(hidden_states, **masks)[0],  # (b, l, d)
            ),
        }
        # The activation runs on one layout on both sides, so it is exact in any batch.
        assert_identical(
            fastplms_layer.intermediate(hidden_states),  # (b, l, 4 * d)
            fair_esm["esm.modules"].gelu(official_layer.fc1(hidden_states)),  # (b, l, 4 * d)
            "feed-forward activation",
        )

    for part, (expected, actual) in pairs.items():
        assert expected.shape == (b, l, HIDDEN), part
        assert_equivalent(actual, expected, is_exact(backend, b), part)


def test_the_final_norm_and_language_model_head_match(networks) -> None:
    official, fastplms, _ = networks
    generator = torch.Generator().manual_seed(SEED)
    hidden_states = torch.randn(3, 9, HIDDEN, generator=generator)  # (b, l, d)

    with torch.no_grad():
        assert_identical(
            fastplms.esm.encoder.emb_layer_norm_after(hidden_states),  # (b, l, d)
            official.emb_layer_norm_after(hidden_states),  # (b, l, d)
            "final layer norm",
        )
        assert_identical(
            fastplms.lm_head(hidden_states),  # (b, l, c)
            official.lm_head(hidden_states),  # (b, l, c)
            "language-model head",
        )


@pytest.mark.parametrize("sequences", BATCHES)
def test_every_hidden_state_and_the_logits_match(networks, sequences: tuple[str, ...]) -> None:
    official, fastplms, backend = networks
    tokens = tokenize(official, sequences)  # (b, l)
    exact = is_exact(backend, tokens.shape[0])

    with torch.no_grad():
        expected = official(tokens, repr_layers=list(range(LAYERS + 1)))
        actual = fastplms(
            input_ids=tokens,
            attention_mask=tokens.ne(official.padding_idx),
            output_hidden_states=True,
        )

    # fair-esm's representation i and FastPLMs' hidden state i are the input to layer i, and the
    # last of each is the final layer norm's output.
    assert len(actual.hidden_states) == LAYERS + 1
    for index in range(LAYERS + 1):
        assert_equivalent(
            actual.hidden_states[index],  # (b, l, d)
            expected["representations"][index],  # (b, l, d)
            exact,
            f"hidden state {index}",
        )
    assert_equivalent(actual.logits, expected["logits"], exact, "logits")  # (b, l, c)
    assert_equivalent(actual.last_hidden_state, expected["representations"][LAYERS], exact, "last")


@pytest.mark.parametrize("sequences", BATCHES)
def test_the_attention_maps_and_contacts_match(networks, sequences: tuple[str, ...]) -> None:
    """Returning attention maps runs eager attention whatever the configured backend."""
    official, fastplms, _ = networks
    tokens = tokenize(official, sequences)  # (b, l)
    attention_mask = tokens.ne(official.padding_idx)  # (b, l)
    exact = tokens.shape[0] == 1

    with torch.no_grad():
        expected = official(tokens, return_contacts=True)
        attentions = fastplms(
            input_ids=tokens, attention_mask=attention_mask, output_attentions=True
        ).attentions
        contacts = fastplms.predict_contacts(tokens, attention_mask=attention_mask.long())

    # fair-esm zeroes every attention entry whose query or key is padding.
    visible = (attention_mask[:, :, None] & attention_mask[:, None, :]).float()  # (b, l, l)
    stacked = torch.stack(attentions, dim=1) * visible[:, None, None]  # (b, n, h, l, l)
    assert_equivalent(stacked, expected["attentions"], exact, "attention maps")
    assert_equivalent(contacts, expected["contacts"], exact, "contacts")  # (b, l - 2, l - 2)


def test_the_rounding_bound_rejects_a_padding_leak(networks) -> None:
    """Letting padded keys through moves the outputs far outside the rounding bound."""
    official, fastplms, _ = networks
    tokens = tokenize(official, SEQUENCES)  # (b, l)

    with torch.no_grad():
        expected = official(tokens)["logits"]  # (b, l, c)
        leaked = fastplms(  # (b, l, c)
            input_ids=tokens, attention_mask=torch.ones_like(tokens)
        ).logits

    with pytest.raises(AssertionError, match="exceeds"):
        assert_equal_to_rounding(leaked, expected, "logits with padded keys visible")


# Loading the official checkpoint.


def native_oracle_asset(role: str) -> Path:
    """A hash-pinned fair-esm file from the local cache the ESM2 reference adapter fills."""
    (asset,) = (asset for asset in REPRESENTATIVE.oracle_assets if asset.role == role)
    torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    root = Path(os.environ.get("FASTPLMS_ORACLE_CACHE", str(torch_home / "fair-esm")))
    path = root / asset.path
    if not path.is_file():
        pytest.skip(f"the fair-esm {role} file {asset.path} is not cached under {root}")
    assert file_sha256(path) == asset.sha256, f"{path} differs from the pinned {asset.sha256}"
    return path


def load_native_official(fair_esm: OfficialPackage) -> nn.Module:
    """The pinned 8M network through fair-esm's own loader and Meta's hash-pinned files."""
    weights = torch.load(native_oracle_asset("weights"), map_location="cpu", weights_only=False)
    regression = torch.load(
        native_oracle_asset("contact_regression"), map_location="cpu", weights_only=False
    )
    model_name = REPRESENTATIVE.official.repo_id.rsplit("/", 1)[-1]
    load = fair_esm["esm.pretrained"].load_model_and_alphabet_core
    model, _ = load(model_name, weights, regression)
    return model.eval()


def official_hub_checkpoint() -> Path:
    return cached_checkpoint(
        REPRESENTATIVE.official.repo_id,
        REPRESENTATIVE.official.revision,
        (
            "config.json",
            "model.safetensors",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    )


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


@pytest.fixture(scope="module")
def native_official(fair_esm: OfficialPackage) -> nn.Module:
    return load_native_official(fair_esm)


@pytest.mark.checkpoint
def test_the_official_checkpoint_loads_key_for_key(native_official: nn.Module) -> None:
    fastplms = FastEsmForMaskedLM(fastplms_config(native_official, "sdpa"))

    state = apply_state_transform(
        STATE_TRANSFORM, native_official.state_dict(), expected_keys=fastplms.state_dict()
    )
    missing, unexpected = fastplms.load_state_dict(state, strict=False)

    assert (missing, unexpected) == ([], [])
    for name, tensor in fastplms.state_dict().items():
        assert_identical(tensor, state[name], name)


@pytest.mark.checkpoint
@pytest.mark.parametrize("sequences", NATURAL_BATCHES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_official_checkpoint_matches_fair_esm(
    native_official: nn.Module, backend: str, sequences: tuple[str, ...]
) -> None:
    fastplms = build_fastplms(native_official, backend)
    tokens = tokenize(native_official, sequences)  # (b, l)
    attention_mask = tokens.ne(native_official.padding_idx)  # (b, l)
    exact = is_exact(backend, tokens.shape[0])

    with torch.no_grad():
        expected = native_official(
            tokens, repr_layers=[native_official.num_layers], return_contacts=True
        )
        actual = fastplms(input_ids=tokens, attention_mask=attention_mask)
        contacts = fastplms.predict_contacts(tokens, attention_mask=attention_mask.long())

    assert_equivalent(actual.logits, expected["logits"], exact, "logits")  # (b, l, c)
    assert_equivalent(
        actual.last_hidden_state,  # (b, l, d)
        expected["representations"][native_official.num_layers],  # (b, l, d)
        exact,
        "last hidden state",
    )
    # Contacts come from eager attention maps whatever the configured backend.
    assert_equivalent(contacts, expected["contacts"], tokens.shape[0] == 1, "contacts")


@pytest.mark.checkpoint
def test_the_official_hub_checkpoint_holds_the_native_tensors(native_official: nn.Module) -> None:
    """Meta's Hub upload and its native file hold the same values under two schemas.

    The Hub state carries two tables rotary ESM2 never reads, the absolute position embeddings
    and their index buffer, and stores the tied output embedding once.
    """
    hub_state = load_file(official_hub_checkpoint() / "model.safetensors")
    native_state = apply_state_transform(STATE_TRANSFORM, native_official.state_dict())

    unused = {"esm.embeddings.position_embeddings.weight", "esm.embeddings.position_ids"}
    assert unused <= hub_state.keys()
    assert hub_state.keys() - unused == native_state.keys() - {"lm_head.decoder.weight"}
    for name in native_state.keys() - {"lm_head.decoder.weight"}:
        assert_identical(hub_state[name], native_state[name], name)
    assert_identical(
        native_state["lm_head.decoder.weight"], hub_state["esm.embeddings.word_embeddings.weight"]
    )


@pytest.mark.checkpoint
def test_the_official_tokenizer_matches_the_fair_esm_alphabet(fair_esm: OfficialPackage) -> None:
    tokenizer = FastEsmTokenizer.from_pretrained(official_hub_checkpoint())
    alphabet = fair_esm["esm"].data.Alphabet.from_architecture("ESM-1b")
    sequences = [*NATURAL_SEQUENCES, "ACDEFGHIKLMNPQRSTVWYXBUZO", "MK<mask>TA<unk>Y"]

    _, _, expected = alphabet.get_batch_converter()(list(enumerate(sequences)))
    actual = tokenizer(sequences, return_tensors="pt", padding=True)

    assert tokenizer.get_vocab() == alphabet.tok_to_idx
    assert_identical(actual["input_ids"], expected)  # (b, l)
    assert_identical(actual["attention_mask"], expected.ne(alphabet.padding_idx).long())
