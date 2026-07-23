"""Mandatory E1 cache, output, resize, and backend contracts."""

import pytest
import torch

from fastplms.embeddings import EmbeddingResult, load_sqlite_result
from tests.unit import test_e1_cache_contract as contracts

test_e1_cache_hit_does_not_slice_target_outputs_twice = (
    contracts.test_e1_cache_hit_does_not_slice_target_outputs_twice
)
test_e1_cache_miss_slices_every_sequence_aligned_output_alias = (
    contracts.test_e1_cache_miss_slices_every_sequence_aligned_output_alias
)
test_e1_cached_flex_dispatch_keeps_or_discards_context_by_layer = (
    contracts.test_e1_cached_flex_dispatch_keeps_or_discards_context_by_layer
)
test_e1_cached_sdpa_preserves_layer_attention_semantics = (
    contracts.test_e1_cached_sdpa_preserves_layer_attention_semantics
)
test_e1_from_pretrained_tokenizer_context_is_thread_local = (
    contracts.test_e1_from_pretrained_tokenizer_context_is_thread_local
)
test_e1_lazy_tokenizer_uses_resolved_weight_commit_per_instance = (
    contracts.test_e1_lazy_tokenizer_uses_resolved_weight_commit_per_instance
)
test_e1_legacy_backend_setter_rejects_unadvertised_backends = (
    contracts.test_e1_legacy_backend_setter_rejects_unadvertised_backends
)
test_e1_loss_bearing_head_tuples_start_with_loss_then_logits = (
    contracts.test_e1_loss_bearing_head_tuples_start_with_loss_then_logits
)
test_e1_masked_lm_resizes_input_and_output_embeddings_together = (
    contracts.test_e1_masked_lm_resizes_input_and_output_embeddings_together
)
test_e1_public_models_honor_config_output_flags_and_return_dict = (
    contracts.test_e1_public_models_honor_config_output_flags_and_return_dict
)
test_e1_public_forwards_reject_unknown_arguments = (
    contracts.test_e1_public_forwards_reject_unknown_arguments
)
test_e1_base_model_rejects_misaligned_biological_indices = (
    contracts.test_e1_base_model_rejects_misaligned_biological_indices
)
test_e1_config_round_trip_preserves_cache_policy = (
    contracts.test_e1_config_round_trip_preserves_cache_policy
)
test_e1_encoder_embedding_filters_training_only_preparer_fields = (
    contracts.test_e1_encoder_embedding_filters_training_only_preparer_fields
)


@pytest.mark.parametrize("pooling", ("mean", "cls"))
def test_e1_msa_embeddings_use_ordered_shared_sqlite_persistence(tmp_path, pooling) -> None:
    model = contracts.E1ForMaskedLM(contracts._tiny_e1_config()).eval()
    output = tmp_path / f"e1-msa-{pooling}.sqlite"

    embeddings = model.embed_dataset_with_msa(
        ["ACDEFG", "ACDEFG"],
        msa_lookup={},
        batch_size=1,
        max_len=16,
        pooling=pooling,
        progress=False,
        embed_dtype=torch.float32,
        output=output,
        format="sqlite",
    )

    assert isinstance(embeddings, EmbeddingResult)
    assert [(record.id, record.sequence) for record in embeddings] == [
        ("0", "ACDEFG"),
        ("1", "ACDEFG"),
    ]
    assert all(record.load_tensor().shape == (model.config.hidden_size,) for record in embeddings)
    assert embeddings.metadata["descriptor_index"] == "sqlite-records"
    assert embeddings.metadata["family_adapter"]["kind"] == "e1-msa-v1"
    reopened = load_sqlite_result(output)
    assert [record.sequence for record in reopened] == ["ACDEFG", "ACDEFG"]

    if pooling == "cls":
        with pytest.raises(ValueError, match=r"does not support pooling operations.*cls"):
            model.embed_dataset(
                ["ACDEFG"],
                batch_size=1,
                pooling="cls",
            )


def test_e1_mlm_loss_excludes_hf_ignore_and_padding_labels_from_normalization() -> None:
    model = contracts.E1ForMaskedLM(contracts._tiny_e1_config()).eval()
    batch = contracts._tiny_e1_batch()
    labels = torch.tensor(
        [[-100, 5, model.model.padding_idx, -100]],
        dtype=torch.long,
    )

    output = model(**batch, labels=labels, return_dict=True)
    assert output.loss is not None
    valid = labels.ne(-100) & labels.ne(model.model.padding_idx)
    expected = torch.nn.functional.cross_entropy(output.logits[valid], labels[valid])

    torch.testing.assert_close(output.loss, expected)
    actual_gradient = torch.autograd.grad(output.loss, output.logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, output.logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
