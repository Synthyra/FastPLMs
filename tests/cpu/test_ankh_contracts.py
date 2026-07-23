"""Mandatory ANKH encoder/decoder and Hugging Face contracts.

The detailed assertions live beside the unit tests so they are also useful to
developers running a focused file.  This module is the positive, offline CPU
allowlist used by the required status check.
"""

import pytest
import torch
from transformers.utils import ModelOutput

from tests.unit import test_ankh_cpu_contract as contracts


def _assert_nested_output_close(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_output_close(actual_value, expected_value)
        return
    assert actual == expected


def _encoder_labels(model_class, input_ids, attention_mask):
    if model_class is contracts.FastAnkhForMaskedLMExtension:
        return input_ids.masked_fill(~attention_mask, -100)
    if model_class is contracts.FastAnkhForSequenceClassification:
        return torch.tensor([1, 2])
    if model_class is contracts.FastAnkhForTokenClassification:
        return input_ids.remainder(3).masked_fill(~attention_mask, -100)
    return None


test_complete_t5_checkpoint_loads_clean_encoder_and_seq2seq_views = (
    contracts.test_complete_t5_checkpoint_loads_clean_encoder_and_seq2seq_views
)
test_decoder_default_attention_mask_preserves_t5_start_and_masks_padding = (
    contracts.test_decoder_default_attention_mask_preserves_t5_start_and_masks_padding
)
test_decoder_embedding_batch_masks_start_eos_padding_and_sentinels = (
    contracts.test_decoder_embedding_batch_masks_start_eos_padding_and_sentinels
)
test_decoder_embed_dataset_slices_aligned_inputs_and_records_provenance = (
    contracts.test_decoder_embed_dataset_slices_aligned_inputs_and_records_provenance
)
test_decoder_embeddings_require_explicit_aligned_inputs = (
    contracts.test_decoder_embeddings_require_explicit_aligned_inputs
)
test_encoder_auto_classes_honor_tuple_and_dict_outputs = (
    contracts.test_encoder_auto_classes_honor_tuple_and_dict_outputs
)
test_encoder_auto_classes_resize_shared_input_embeddings = (
    contracts.test_encoder_auto_classes_resize_shared_input_embeddings
)
test_encoder_only_view_rejects_decoder_hidden_states = (
    contracts.test_encoder_only_view_rejects_decoder_hidden_states
)
test_encoder_task_heads_produce_finite_loss_and_gradients = (
    contracts.test_encoder_task_heads_produce_finite_loss_and_gradients
)
test_ankh_decoder_embedding_inputs_use_tight_sentinel_tokenization = (
    contracts.test_ankh_decoder_embedding_inputs_use_tight_sentinel_tokenization
)
test_ankh_explicit_and_model_owned_tokenizers_share_the_raw_sequence_contract = (
    contracts.test_ankh_explicit_and_model_owned_tokenizers_share_the_raw_sequence_contract
)
test_ankh_tokenization_normalizes_raw_sequences_and_tight_sentinel_prompts = (
    contracts.test_ankh_tokenization_normalizes_raw_sequences_and_tight_sentinel_prompts
)
test_ankh_tokenization_rejects_empty_inputs_and_real_slow_tokenizers = (
    contracts.test_ankh_tokenization_rejects_empty_inputs_and_real_slow_tokenizers
)
test_ankh_ttt_uses_raw_residue_tokenization = contracts.test_ankh_ttt_uses_raw_residue_tokenization
test_sdpa_output_attentions_fallback_keeps_padding_mask_and_backend = (
    contracts.test_sdpa_output_attentions_fallback_keeps_padding_mask_and_backend
)
test_seq2seq_embedding_selects_encoder_and_explicit_decoder_layers = (
    contracts.test_seq2seq_embedding_selects_encoder_and_explicit_decoder_layers
)
test_seq2seq_head_produces_finite_loss_and_gradients = (
    contracts.test_seq2seq_head_produces_finite_loss_and_gradients
)
test_seq2seq_view_forces_eager_without_changing_encoder_backend_contract = (
    contracts.test_seq2seq_view_forces_eager_without_changing_encoder_backend_contract
)
test_tokenizer_load_context_is_per_instance_and_offline_scoped = (
    contracts.test_tokenizer_load_context_is_per_instance_and_offline_scoped
)
test_tokenizer_load_context_is_isolated_during_concurrent_first_access = (
    contracts.test_tokenizer_load_context_is_isolated_during_concurrent_first_access
)


@pytest.mark.parametrize(
    "model_class",
    (
        contracts.FastAnkhModel,
        contracts.FastAnkhForMaskedLMExtension,
        contracts.FastAnkhForSequenceClassification,
        contracts.FastAnkhForTokenClassification,
    ),
)
def test_ankh_encoder_views_backward_and_save_reload(model_class, tmp_path) -> None:
    model = model_class(contracts._config(num_labels=3)).eval()
    input_ids = torch.tensor([[2, 3, 1], [4, 1, 0]])
    attention_mask = input_ids.ne(0)
    labels = _encoder_labels(model_class, input_ids, attention_mask)
    model_arguments = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_attentions": True,
        "output_hidden_states": True,
    }
    if labels is not None:
        model_arguments["labels"] = labels
    output = model(
        **model_arguments,
        return_dict=True,
    )
    tuple_output = model(
        **model_arguments,
        return_dict=False,
    )

    assert isinstance(output, ModelOutput)
    assert output.hidden_states is not None
    assert output.attentions is not None
    _assert_nested_output_close(tuple_output, output.to_tuple())
    if labels is not None:
        torch.testing.assert_close(tuple_output[0], output.loss)
        torch.testing.assert_close(tuple_output[1], output.logits)
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unexpected_cpu_contract=True,
        )

    tensor = output.last_hidden_state if model_class is contracts.FastAnkhModel else output.logits
    loss = output.loss if labels is not None else tensor.square().mean()
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.shared.weight.grad is not None

    save_dir = tmp_path / model_class.__name__
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_dir, local_files_only=True).eval()
    with torch.inference_mode():
        observed = reloaded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    observed_tensor = (
        observed.last_hidden_state if model_class is contracts.FastAnkhModel else observed.logits
    )
    torch.testing.assert_close(observed_tensor, tensor.detach(), rtol=0.0, atol=0.0)


def test_ankh_seq2seq_view_honors_tuple_output_and_resize() -> None:
    model = contracts.FastAnkhForConditionalGeneration(contracts._config()).eval()
    input_ids = torch.tensor([[2, 3, 1, 0], [4, 1, 0, 0]])
    attention_mask = input_ids.ne(0)
    decoder_input_ids = torch.tensor([[0, 5, 1, 0], [0, 6, 7, 1]])
    decoder_attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    structured = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=False,
    )

    assert isinstance(structured, ModelOutput)
    assert isinstance(tuple_output, tuple)
    assert structured.decoder_hidden_states is not None
    assert structured.encoder_hidden_states is not None
    assert structured.decoder_attentions is not None
    assert structured.cross_attentions is not None
    assert structured.encoder_attentions is not None
    causal_future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    for decoder_attention in structured.decoder_attentions:
        assert torch.count_nonzero(decoder_attention.masked_select(causal_future)) == 0
        assert torch.count_nonzero(decoder_attention[0, :, :, 3]) == 0
    for cross_attention in structured.cross_attentions:
        assert torch.count_nonzero(cross_attention[0, :, :, 3]) == 0
        assert torch.count_nonzero(cross_attention[1, :, :, 2:]) == 0
    _assert_nested_output_close(tuple_output, structured.to_tuple())
    torch.testing.assert_close(tuple_output[0], structured.logits)

    labels = torch.tensor([[5, 1, -100, -100], [6, 7, 1, -100]])
    loss_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    loss_tuple = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=False,
    )
    _assert_nested_output_close(loss_tuple, loss_output.to_tuple())
    torch.testing.assert_close(loss_tuple[0], loss_output.loss)
    torch.testing.assert_close(loss_tuple[1], loss_output.logits)
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            unexpected_cpu_contract=True,
        )

    model.resize_token_embeddings(19)
    assert model.get_input_embeddings().num_embeddings == 19
    assert model.get_output_embeddings().out_features == 19
    assert model.encoder.embed_tokens is model.shared
    assert model.decoder.embed_tokens is model.shared
