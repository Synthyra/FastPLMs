"""Hermetic CPU contracts for ANKH's encoder and complete T5 views."""

from __future__ import annotations

import json
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import ClassVar
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AutoTokenizer, GenerationConfig

import fastplms.models.ankh.modeling_ankh as ankh_module
from fastplms.models.ankh.modeling_ankh import (
    FAST_ANKH_ENCODER,
    FastAnkhConfig,
    FastAnkhForConditionalGeneration,
    FastAnkhForMaskedLMExtension,
    FastAnkhForSequenceClassification,
    FastAnkhForTokenClassification,
    FastAnkhModel,
    configure_ankh_tokenizer,
    normalize_ankh_decoder_prompt,
    normalize_ankh_sequence,
    tokenize_ankh_decoder_prompts,
    tokenize_ankh_sequences,
)


def _config(**overrides: object) -> FastAnkhConfig:
    values = {
        "vocab_size": 16,
        "d_model": 8,
        "d_kv": 4,
        "d_ff": 16,
        "num_heads": 2,
        "num_layers": 2,
        "num_decoder_layers": 2,
        "dropout_rate": 0.0,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
        "attn_backend": "eager",
        "use_cache": False,
    }
    values.update(overrides)
    return FastAnkhConfig(**values)


class _TinyTokenizer:
    all_special_ids = (0, 1, 7)
    name_or_path = "tiny-ankh"
    vocab_size = 16
    special_tokens_map: ClassVar[dict[str, str]] = {
        "pad_token": "<pad>",
        "eos_token": "</s>",
    }
    model_max_length = 64
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, sequences, **_kwargs):
        alphabet = {"A": 2, "C": 3, "D": 4, "E": 5, "F": 6, "S": 7}
        rows = [[alphabet.get(character, 8) for character in value] + [1] for value in sequences]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor(  # (b, l)
            [row + [0] * (width - len(row)) for row in rows]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0).to(dtype=torch.long),
        }

    def get_vocab(self):
        return {"<pad>": 0, "</s>": 1, "A": 2, "C": 3, "D": 4, "E": 5, "F": 6}

    def get_added_vocab(self):
        return {"<extra_id_0>": 7}


class _RecordingTokenizer(_TinyTokenizer):
    is_fast = True

    def __init__(self) -> None:
        self.backend_tokenizer = SimpleNamespace(pre_tokenizer=None)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, sequences, **kwargs):
        values = [sequences] if isinstance(sequences, str) else list(sequences)
        self.calls.append((values, dict(kwargs)))
        return super().__call__(values, **kwargs)


def test_ankh_tokenization_normalizes_raw_sequences_and_tight_sentinel_prompts() -> None:
    tokenizer = _RecordingTokenizer()

    source = tokenize_ankh_sequences(
        tokenizer,
        ["M S T N P K", "AC\nDE"],
        return_tensors="pt",
        padding=True,
    )
    prompt = tokenize_ankh_decoder_prompts(
        tokenizer,
        ["M <extra_id_0>", "A\t<extra_id_0>"],
        return_tensors="pt",
        add_special_tokens=False,
    )

    assert tokenizer.calls[0][0] == ["MSTNPK", "ACDE"]
    assert tokenizer.calls[1][0] == ["M<extra_id_0>", "A<extra_id_0>"]
    assert source["input_ids"].ndim == 2
    assert prompt["input_ids"].ndim == 2
    assert type(tokenizer.backend_tokenizer.pre_tokenizer).__name__ == "Metaspace"
    assert normalize_ankh_sequence(" M S T \n") == "MST"
    assert normalize_ankh_decoder_prompt(" M <extra_id_0> ") == "M<extra_id_0>"


def test_ankh_tokenization_rejects_empty_inputs_and_real_slow_tokenizers() -> None:
    with pytest.raises(ValueError, match="protein sequences must not be empty"):
        normalize_ankh_sequence(" \n\t")
    with pytest.raises(ValueError, match="decoder prompts must not be empty"):
        normalize_ankh_decoder_prompt("  ")
    with pytest.raises(TypeError, match="requires a fast tokenizer"):
        configure_ankh_tokenizer(SimpleNamespace(is_fast=False))


def test_offline_auto_tokenizer_flags_and_seq2seq_generation_config(
    tmp_path: Path,
) -> None:
    """Transformers 5.13 must resolve both tokenizer flags from local artifact bytes."""

    vocabulary = [
        ("<pad>", 0.0),
        ("</s>", 0.0),
        ("<unk>", 0.0),
        ("A", -1.0),
        ("C", -1.0),
        ("D", -1.0),
        ("X", -1.0),
        ("<extra_id_0>", 0.0),
    ]
    backend = Tokenizer(models.Unigram(vocabulary, unk_id=2))
    replacement = "\N{LOWER ONE EIGHTH BLOCK}"
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=replacement,
        prepend_scheme="never",
        split=False,
    )
    backend.decoder = decoders.Metaspace(
        replacement=replacement,
        prepend_scheme="never",
        split=False,
    )
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[("</s>", 1)],
    )
    backend.add_special_tokens(["<pad>", "</s>", "<unk>", "<extra_id_0>"])
    backend.save(str(tmp_path / "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "extra_ids": 1,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<extra_id_0>"],
    }
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<extra_id_0>"],
    }
    generation_config = {
        "decoder_start_token_id": 0,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    for name, payload in (
        ("tokenizer_config.json", tokenizer_config),
        ("special_tokens_map.json", special_tokens),
        ("generation_config.json", generation_config),
    ):
        (tmp_path / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )

    encoded = []
    for use_fast in (True, False):
        tokenizer = AutoTokenizer.from_pretrained(
            tmp_path,
            use_fast=use_fast,
            local_files_only=True,
        )
        assert tokenizer.is_fast
        configure_ankh_tokenizer(tokenizer)
        encoded.append(
            tokenizer(
                ["ACD", "AX"],
                padding=True,
                return_tensors="pt",
            )
        )

    assert torch.equal(encoded[0]["input_ids"], encoded[1]["input_ids"])
    assert torch.equal(encoded[0]["attention_mask"], encoded[1]["attention_mask"])
    loaded_generation = GenerationConfig.from_pretrained(
        tmp_path,
        local_files_only=True,
    )
    assert loaded_generation.decoder_start_token_id == 0
    assert loaded_generation.pad_token_id == 0
    assert loaded_generation.eos_token_id == 1


def test_ankh_explicit_and_model_owned_tokenizers_share_the_raw_sequence_contract() -> None:
    explicit = _RecordingTokenizer()
    owned = _RecordingTokenizer()
    model = FastAnkhModel(_config())
    model.tokenizer = owned

    explicit_ids = model._tokenize_sequence_batch(
        ["A C D", "EF"],
        tokenizer=explicit,
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    owned_ids = model._tokenize_sequence_batch(
        ["A C D", "EF"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]

    assert explicit.calls[-1][0] == ["ACD", "EF"]
    assert owned.calls[-1][0] == ["ACD", "EF"]
    torch.testing.assert_close(explicit_ids, owned_ids)

    result = model.embed_dataset(
        ["A C D"],
        tokenizer=explicit,
        full_embeddings=True,
    )
    assert explicit.calls[-1][0] == ["ACD"]
    assert result[0].load_tensor().shape == (3, model.config.d_model)


def test_ankh_ttt_uses_raw_residue_tokenization() -> None:
    tokenizer = _RecordingTokenizer()
    model = FastAnkhForMaskedLMExtension(_config())
    model.tokenizer = tokenizer

    input_ids = model._ttt_tokenize(seq=["A C D", "EF"])

    assert tokenizer.calls[-1][0] == ["ACD", "EF"]
    assert input_ids.ndim == 2


def test_ankh_decoder_embedding_inputs_use_tight_sentinel_tokenization() -> None:
    tokenizer = _RecordingTokenizer()
    model = FastAnkhForConditionalGeneration(_config()).eval()

    decoder_input_ids, decoder_attention_mask, resolved = model._prepare_decoder_embedding_inputs(
        batch_size=2,
        decoder_inputs=["M <extra_id_0>", "A\t<extra_id_0>"],
        decoder_input_ids=None,
        decoder_attention_mask=None,
        tokenizer=tokenizer,
    )

    assert resolved is tokenizer
    assert tokenizer.calls[-1][0] == ["M<extra_id_0>", "A<extra_id_0>"]
    assert decoder_input_ids.shape == decoder_attention_mask.shape


@pytest.mark.parametrize("dropout_rate", (True, False, None, "0.1"))
def test_ankh_config_rejects_non_real_or_boolean_dropout(dropout_rate: object) -> None:
    with pytest.raises(TypeError, match="dropout_rate must be a real number"):
        _config(dropout_rate=dropout_rate)


@pytest.mark.parametrize("dropout_rate", (-0.01, 1.0, float("inf"), float("nan")))
def test_ankh_config_rejects_dropout_outside_half_open_probability_range(
    dropout_rate: float,
) -> None:
    with pytest.raises(ValueError, match=r"dropout_rate must be in \[0, 1\)"):
        _config(dropout_rate=dropout_rate)


def test_ankh_custom_encoder_invokes_every_t5_stack_dropout_site() -> None:
    model = FastAnkhModel(_config(num_layers=1, dropout_rate=0.4)).train()
    layer = model.encoder.block[0]
    dropout_sites = {
        "stack_input_and_final": model.encoder.dropout,
        "attention_residual": layer.layer[0].dropout,
        "ff_internal": layer.layer[1].DenseReluDense.dropout,
        "ff_residual": layer.layer[1].dropout,
    }
    observed_calls = {name: 0 for name in dropout_sites}
    handles = []
    for name, module in dropout_sites.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output, site=name: observed_calls.__setitem__(
                    site, observed_calls[site] + 1
                )
            )
        )

    try:
        model(
            input_ids=torch.tensor([[2, 3, 4, 1]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert observed_calls == {
        "stack_input_and_final": 2,
        "attention_residual": 1,
        "ff_internal": 1,
        "ff_residual": 1,
    }
    assert all(module.p == pytest.approx(0.4) for module in dropout_sites.values())


def test_ankh_dropout_is_eval_exact_and_seeded_in_training() -> None:
    input_ids = torch.tensor([[2, 3, 4, 1], [5, 6, 1, 0]])  # (b=2, l=4)
    attention_mask = input_ids.ne(0)  # (b, l)

    torch.manual_seed(31)
    zero_dropout = FastAnkhModel(_config(dropout_rate=0.0)).eval()
    torch.manual_seed(31)
    configured_dropout = FastAnkhModel(_config(dropout_rate=0.5)).eval()

    zero_output = zero_dropout(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    configured_output = configured_dropout(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    torch.testing.assert_close(configured_output, zero_output, rtol=0.0, atol=0.0)

    configured_dropout.train()
    torch.manual_seed(47)
    first = configured_dropout(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    torch.manual_seed(47)
    repeated = configured_dropout(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    torch.manual_seed(53)
    different_seed = configured_dropout(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state

    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
    assert not torch.equal(first, configured_output)
    assert not torch.equal(first, different_seed)


@pytest.mark.parametrize(
    "model_class",
    (
        FAST_ANKH_ENCODER,
        FastAnkhModel,
        FastAnkhForMaskedLMExtension,
        FastAnkhForSequenceClassification,
        FastAnkhForTokenClassification,
    ),
)
@pytest.mark.parametrize("argument", ("use_cache", "decoder_input_ids", "misspelled_option"))
def test_ankh_encoder_views_reject_every_unexpected_forward_argument(
    model_class: (
        type[FastAnkhModel]
        | type[FastAnkhForMaskedLMExtension]
        | type[FastAnkhForSequenceClassification]
        | type[FastAnkhForTokenClassification]
    ),
    argument: str,
) -> None:
    model = model_class(_config(num_labels=3)).eval()

    with pytest.raises(TypeError, match=argument):
        model(input_ids=torch.tensor([[2, 3, 1]]), **{argument: True})


def test_seq2seq_embedding_selects_encoder_and_explicit_decoder_layers() -> None:
    torch.manual_seed(7)
    model = FastAnkhForConditionalGeneration(_config()).eval()
    model.tokenizer = _TinyTokenizer()
    input_ids = torch.tensor([[2, 3, 4, 1], [5, 6, 1, 0]])  # (b=2, l_enc=4)
    attention_mask = input_ids.ne(0)  # (b, l_enc)
    decoder_input_ids = torch.tensor([[2, 7, 1, 0], [3, 1, 0, 0]])  # (b, l_dec=4)
    decoder_attention_mask = decoder_input_ids.ne(0)  # (b, l_dec)

    encoder_states = model._embed(
        input_ids,
        attention_mask,
        hidden_state_source="encoder",
        store_all_hidden_states=True,
    )
    decoder_states = model._embed(
        input_ids,
        attention_mask,
        hidden_state_source="decoder",
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        store_all_hidden_states=True,
    )

    assert encoder_states.shape == (2, 3, 4, 8)
    assert decoder_states.shape == (2, 3, 4, 8)
    for layer_index in range(encoder_states.shape[1]):
        assert torch.equal(
            model._embed(
                input_ids,
                attention_mask,
                hidden_state_source="encoder",
                hidden_state_index=layer_index,
            ),
            encoder_states[:, layer_index],
        )
    for layer_index in range(decoder_states.shape[1]):
        assert torch.equal(
            model._embed(
                input_ids,
                attention_mask,
                hidden_state_source="decoder",
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                hidden_state_index=layer_index,
            ),
            decoder_states[:, layer_index],
        )
    assert torch.equal(
        model._embed(input_ids, attention_mask),
        encoder_states[:, -1],
    )


def test_decoder_default_attention_mask_preserves_t5_start_and_masks_padding() -> None:
    model = FastAnkhForConditionalGeneration(_config()).eval()
    decoder_input_ids = torch.tensor([[0, 5, 1, 0], [0, 6, 7, 1]])  # (b=2, l_dec=4)

    prepared_ids, prepared_mask, _ = model._prepare_decoder_embedding_inputs(
        batch_size=2,
        decoder_inputs=None,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=None,
        tokenizer=_TinyTokenizer(),
    )

    assert torch.equal(prepared_ids, decoder_input_ids)
    assert torch.equal(
        prepared_mask,
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
    )


def test_seq2seq_view_forces_eager_without_changing_encoder_backend_contract() -> None:
    seq2seq_config = _config(attn_backend=None)
    seq2seq = FastAnkhForConditionalGeneration(seq2seq_config)
    encoder = FastAnkhModel(_config(attn_backend=None))

    assert seq2seq.config.attn_backend == "eager"
    assert seq2seq.config._attn_implementation == "eager"
    assert encoder.attn_backend == "sdpa"


def test_decoder_embeddings_require_explicit_aligned_inputs() -> None:
    model = FastAnkhForConditionalGeneration(_config()).eval()
    model.tokenizer = _TinyTokenizer()
    input_ids = torch.tensor([[2, 3, 1], [4, 1, 0]])  # (b=2, l=3)

    with pytest.raises(ValueError, match="requires exactly one"):
        model._embed(input_ids, hidden_state_source="decoder")
    with pytest.raises(ValueError, match="exactly one"):
        model._embed(
            input_ids,
            hidden_state_source="decoder",
            decoder_inputs=["AC", "D"],
            decoder_input_ids=input_ids,
        )
    with pytest.raises(ValueError, match="align one-to-one"):
        model._embed(
            input_ids,
            hidden_state_source="decoder",
            decoder_inputs=["AC"],
        )
    with pytest.raises(ValueError, match="only valid"):
        model._embed(
            input_ids,
            hidden_state_source="encoder",
            decoder_input_ids=input_ids,
        )


def test_decoder_embedding_batch_masks_start_eos_padding_and_sentinels() -> None:
    model = FastAnkhForConditionalGeneration(_config()).eval()
    tokenizer = _TinyTokenizer()
    decoder_input_ids = torch.tensor([[2, 7, 1, 0], [3, 1, 0, 0]])  # (b=2, l_dec=4)
    decoder_attention_mask = decoder_input_ids.ne(0)  # (b, l_dec)

    batch = model._embedding_batch(
        ["ACD", "EF"],
        tokenizer=tokenizer,
        hidden_state_source="decoder",
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert batch.X.shape == (2, 4, 8)
    assert torch.equal(
        batch.residue_mask,
        torch.tensor([[True, False, False, False], [True, False, False, False]]),
    )

    string_batch = model._embedding_batch(
        ["ACD", "EF"],
        tokenizer=tokenizer,
        hidden_state_source="decoder",
        decoder_inputs=["AS", "C"],
    )
    assert string_batch.X.shape == (2, 3, 8)
    assert torch.equal(
        string_batch.residue_mask,
        torch.tensor([[True, False, False], [True, False, False]]),
    )


def test_decoder_embed_dataset_slices_aligned_inputs_and_records_provenance() -> None:
    model = FastAnkhForConditionalGeneration(_config()).eval()
    tokenizer = _TinyTokenizer()
    decoder_input_ids = torch.tensor([[2, 7, 1, 0], [3, 1, 0, 0]])  # (b=2, l_dec=4)
    decoder_attention_mask = decoder_input_ids.ne(0)  # (b, l_dec)

    result = model.embed_dataset(
        ["ACD", "EF"],
        tokenizer=tokenizer,
        batch_size=1,
        full_embeddings=True,
        hidden_state_source="decoder",
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert [tuple(record.load_tensor().shape) for record in result] == [(1, 8), (1, 8)]
    assert result.metadata["hidden_state_source"] == "decoder"
    assert result.metadata["hidden_state_index"] == -1
    assert result.metadata["store_all_hidden_states"] is False
    assert result.metadata["decoder_alignment"] == "input-position"
    assert len(result.metadata["decoder_input_fingerprint"]) == 64
    assert len(result.metadata["decoder_attention_mask_fingerprint"]) == 64
    assert result.metadata["model_embedding"]["hidden_state_stack"] == "decoder"
    assert (
        result.metadata["model_embedding"]["decoder_residue_mask"]
        == "attention-mask-minus-tokenizer-specials"
    )


def test_encoder_only_view_rejects_decoder_hidden_states() -> None:
    model = FastAnkhModel(_config()).eval()
    with pytest.raises(ValueError, match="AutoModelForSeq2SeqLM"):
        model._embed(
            torch.tensor([[2, 3, 1]]),
            hidden_state_source="decoder",
            decoder_input_ids=torch.tensor([[2, 1]]),
        )


def test_sdpa_output_attentions_fallback_keeps_padding_mask_and_backend() -> None:
    model = FastAnkhModel(_config(attn_backend="sdpa")).eval()
    input_ids = torch.tensor([[2, 3, 1, 0], [4, 1, 0, 0]])  # (b=2, l=4)
    attention_mask = input_ids.ne(0)  # (b, l)

    with pytest.warns(
        RuntimeWarning,
        match="requested 'sdpa'.*using 'eager'.*call only",
    ) as fallback_warnings:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    assert len(fallback_warnings) == 1
    assert model.attn_backend == "sdpa"
    assert output.attentions is not None
    for layer_attention in output.attentions:
        assert torch.count_nonzero(layer_attention[0, :, :, 3]) == 0
        assert torch.count_nonzero(layer_attention[1, :, :, 2:]) == 0


@pytest.mark.parametrize(
    "model_class",
    (
        FastAnkhModel,
        FastAnkhForMaskedLMExtension,
        FastAnkhForSequenceClassification,
        FastAnkhForTokenClassification,
    ),
)
def test_encoder_auto_classes_honor_tuple_and_dict_outputs(
    model_class: (
        type[FastAnkhModel]
        | type[FastAnkhForMaskedLMExtension]
        | type[FastAnkhForSequenceClassification]
        | type[FastAnkhForTokenClassification]
    ),
) -> None:
    model = model_class(_config(num_labels=3)).eval()
    input_ids = torch.tensor([[2, 3, 1], [4, 1, 0]])  # (b=2, l=3)
    attention_mask = input_ids.ne(0)  # (b, l)

    dictionary_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=False,
    )

    assert isinstance(tuple_output, tuple)
    expected = (
        dictionary_output.last_hidden_state
        if model_class is FastAnkhModel
        else dictionary_output.logits
    )
    assert torch.equal(tuple_output[0], expected)
    assert tuple_output[1] is not None


@pytest.mark.parametrize(
    "model_class",
    (
        FastAnkhModel,
        FastAnkhForMaskedLMExtension,
        FastAnkhForSequenceClassification,
        FastAnkhForTokenClassification,
    ),
)
def test_encoder_auto_classes_resize_shared_input_embeddings(
    model_class: (
        type[FastAnkhModel]
        | type[FastAnkhForMaskedLMExtension]
        | type[FastAnkhForSequenceClassification]
        | type[FastAnkhForTokenClassification]
    ),
) -> None:
    model = model_class(_config(num_labels=3))
    assert model.shared is model.get_input_embeddings()
    assert model.encoder.embed_tokens is model.shared

    resized = model.resize_token_embeddings(19)

    assert resized is model.get_input_embeddings()
    assert model.config.vocab_size == 19
    assert model.shared is model.get_input_embeddings()
    assert model.encoder.embed_tokens is model.shared
    if model_class is FastAnkhForMaskedLMExtension:
        assert model.get_output_embeddings().out_features == 19


@pytest.mark.parametrize(
    ("model_class", "labels"),
    (
        (
            FastAnkhForMaskedLMExtension,
            torch.tensor([[2, 3, 1], [4, 1, -100]]),
        ),
        (
            FastAnkhForSequenceClassification,
            torch.tensor([1, 2]),
        ),
        (
            FastAnkhForTokenClassification,
            torch.tensor([[1, 2, 0], [2, 0, -100]]),
        ),
    ),
)
def test_encoder_task_heads_produce_finite_loss_and_gradients(
    model_class: (
        type[FastAnkhForMaskedLMExtension]
        | type[FastAnkhForSequenceClassification]
        | type[FastAnkhForTokenClassification]
    ),
    labels: torch.Tensor,
) -> None:
    model = model_class(_config(num_labels=3)).train()
    input_ids = torch.tensor([[2, 3, 1], [4, 1, 0]])  # (b=2, l=3)
    output = model(
        input_ids=input_ids,
        attention_mask=input_ids.ne(0),
        labels=labels,
        return_dict=True,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.shared.weight.grad is not None
    assert torch.isfinite(model.shared.weight.grad).all()


def test_seq2seq_head_produces_finite_loss_and_gradients() -> None:
    model = FastAnkhForConditionalGeneration(_config()).train()
    input_ids = torch.tensor([[2, 3, 1], [4, 1, 0]])  # (b=2, l=3)
    output = model(
        input_ids=input_ids,
        attention_mask=input_ids.ne(0),
        labels=torch.tensor([[2, 1, -100], [3, 1, -100]]),
        return_dict=True,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.shared.weight.grad is not None
    assert torch.isfinite(model.shared.weight.grad).all()


def test_complete_t5_checkpoint_loads_clean_encoder_and_seq2seq_views(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    full = FastAnkhForConditionalGeneration(_config()).eval()
    full.save_pretrained(tmp_path, safe_serialization=True)

    encoder, encoder_info = FastAnkhModel.from_pretrained(
        tmp_path,
        local_files_only=True,
        output_loading_info=True,
    )
    reloaded, seq2seq_info = FastAnkhForConditionalGeneration.from_pretrained(
        tmp_path,
        local_files_only=True,
        output_loading_info=True,
    )

    assert not hasattr(encoder, "decoder")
    assert not encoder_info["missing_keys"]
    assert not encoder_info["unexpected_keys"]
    assert not seq2seq_info["missing_keys"]
    assert not seq2seq_info["unexpected_keys"]
    expected_state = full.state_dict()
    observed_state = reloaded.state_dict()
    assert set(observed_state) == set(expected_state)
    assert all(torch.equal(observed_state[key], value) for key, value in expected_state.items())


def test_tokenizer_load_context_is_per_instance_and_offline_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    class Tokenizer(_TinyTokenizer):
        backend_tokenizer = SimpleNamespace(pre_tokenizer=None)

    def load_tokenizer(source: str, **kwargs):
        requests.append((source, kwargs))
        return Tokenizer()

    monkeypatch.setattr(
        ankh_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(load_tokenizer),
    )
    first = FastAnkhModel(_config())
    second = FastAnkhModel(_config())
    first.config._name_or_path = "ankh-first"
    first.config._commit_hash = "a" * 40
    second.config._name_or_path = "ankh-second"
    second.config._commit_hash = "b" * 40
    first_token = object()
    second_token = object()
    first.__dict__["_fastplms_tokenizer_load_context"] = {
        "cache_dir": "cache-one",
        "local_files_only": True,
        "token": first_token,
    }
    second.__dict__["_fastplms_tokenizer_load_context"] = {
        "cache_dir": "cache-two",
        "local_files_only": False,
        "token": second_token,
    }

    assert first.tokenizer is first.tokenizer
    assert second.tokenizer is second.tokenizer
    assert requests[0][0] == "ankh-first"
    assert requests[0][1] == {
        "cache_dir": "cache-one",
        "local_files_only": True,
        "token": first_token,
        "revision": "a" * 40,
    }
    assert requests[1][0] == "ankh-second"
    assert requests[1][1] == {
        "cache_dir": "cache-two",
        "local_files_only": False,
        "token": second_token,
        "revision": "b" * 40,
    }


def test_tokenizer_load_context_is_isolated_during_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent models must never exchange revision, cache, or token context."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    rendezvous = Barrier(2)
    request_lock = Lock()
    requests: dict[str, dict[str, object]] = {}

    class Tokenizer(_TinyTokenizer):
        def __init__(self, source: str) -> None:
            self.source = source
            self.backend_tokenizer = SimpleNamespace(pre_tokenizer=None)

    def load_tokenizer(source: str, **kwargs):
        rendezvous.wait(timeout=3.0)
        with request_lock:
            requests[source] = dict(kwargs)
        return Tokenizer(source)

    monkeypatch.setattr(
        ankh_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(load_tokenizer),
    )
    first = FastAnkhModel(_config(num_layers=1, num_decoder_layers=1))
    second = FastAnkhModel(_config(num_layers=1, num_decoder_layers=1))
    first.config._name_or_path = "ankh-first"
    first.config._commit_hash = "a" * 40
    second.config._name_or_path = "ankh-second"
    second.config._commit_hash = "b" * 40
    first_token = object()
    second_token = object()
    first.__dict__["_fastplms_tokenizer_load_context"] = {
        "cache_dir": "cache-one",
        "local_files_only": True,
        "token": first_token,
    }
    second.__dict__["_fastplms_tokenizer_load_context"] = {
        "cache_dir": "cache-two",
        "local_files_only": True,
        "token": second_token,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(getattr, first, "tokenizer")
        second_future = executor.submit(getattr, second, "tokenizer")
        first_tokenizer = first_future.result(timeout=5.0)
        second_tokenizer = second_future.result(timeout=5.0)

    assert first_tokenizer.source == "ankh-first"
    assert second_tokenizer.source == "ankh-second"
    assert requests == {
        "ankh-first": {
            "cache_dir": "cache-one",
            "local_files_only": True,
            "token": first_token,
            "revision": "a" * 40,
        },
        "ankh-second": {
            "cache_dir": "cache-two",
            "local_files_only": True,
            "token": second_token,
            "revision": "b" * 40,
        },
    }
    assert first.tokenizer is first_tokenizer
    assert second.tokenizer is second_tokenizer
