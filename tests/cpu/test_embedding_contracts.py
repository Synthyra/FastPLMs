"""Mandatory real-family, ordered, streaming, and persistent embedding contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from fastplms.embeddings import embed_dataset, load_sqlite_result
from fastplms.models.ankh.modeling_ankh import FastAnkhModel
from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMModel
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2Model
from fastplms.models.e1.modeling_e1 import E1Model
from fastplms.models.esm2.modeling_fastesm import FastEsmModel
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)
from tests.cpu.test_sequence_autoclass_contracts import (
    _dplm_config_values,
    _esm2_config,
)
from tests.unit import test_embeddings_api as contracts
from tests.unit.test_ankh_cpu_contract import _config as _ankh_config
from tests.unit.test_e1_cache_contract import _tiny_e1_config

test_all_hidden_state_embeddings_trim_token_axis_and_round_trip = (
    contracts.test_all_hidden_state_embeddings_trim_token_axis_and_round_trip
)
test_all_poolers_and_output_slices = contracts.test_all_poolers_and_output_slices
test_bounded_length_bucketing_restores_input_order = (
    contracts.test_bounded_length_bucketing_restores_input_order
)
test_decoder_companions_are_fingerprinted_and_bucket_aligned = (
    contracts.test_decoder_companions_are_fingerprinted_and_bucket_aligned
)
test_decoder_embeddings_require_an_explicit_model_capability = (
    contracts.test_decoder_embeddings_require_an_explicit_model_capability
)
test_failed_safetensors_overwrite_preserves_previous_valid_generation = (
    contracts.test_failed_safetensors_overwrite_preserves_previous_valid_generation
)
test_fasta_parser_streams_without_path_read_text = (
    contracts.test_fasta_parser_streams_without_path_read_text
)
test_fasta_preserves_headers_order_and_duplicates = (
    contracts.test_fasta_preserves_headers_order_and_duplicates
)
test_full_embeddings_contain_biological_residues_only = (
    contracts.test_full_embeddings_contain_biological_residues_only
)
test_interrupted_embedding_overwrite_preserves_previous_generation = (
    contracts.test_interrupted_embedding_overwrite_preserves_previous_generation
)
test_invalid_storage_and_pooling_fail_before_input_consumption = (
    contracts.test_invalid_storage_and_pooling_fail_before_input_consumption
)
test_legacy_sqlite_converter_accepts_compact_blobs_without_pickle = (
    contracts.test_legacy_sqlite_converter_accepts_compact_blobs_without_pickle
)
test_large_streaming_inputs_use_bounded_disk_windows = (
    contracts.test_large_streaming_inputs_use_bounded_disk_windows
)
test_mapping_inputs_embed_values_with_mapping_keys_as_ids = (
    contracts.test_mapping_inputs_embed_values_with_mapping_keys_as_ids
)
test_model_state_fingerprint_rehashes_data_and_storage_alias_mutations = (
    contracts.test_model_state_fingerprint_rehashes_data_and_storage_alias_mutations
)
test_runtime_versions_are_part_of_resume_identity = (
    contracts.test_runtime_versions_are_part_of_resume_identity
)
test_tokenizer_content_changes_run_fingerprint = (
    contracts.test_tokenizer_content_changes_run_fingerprint
)
test_native_sequence_tokenizer_loader_context_is_bound_without_secret_values = (
    contracts.test_native_sequence_tokenizer_loader_context_is_bound_without_secret_values
)
test_local_artifact_identity_fills_embedding_provenance = (
    contracts.test_local_artifact_identity_fills_embedding_provenance
)
test_max_length_counts_biological_residues_not_special_tokens = (
    contracts.test_max_length_counts_biological_residues_not_special_tokens
)
test_truncate_false_rejects_overlength_inputs_before_custom_adapter_inference = (
    contracts.test_truncate_false_rejects_overlength_inputs_before_custom_adapter_inference
)
test_truncate_false_rejects_overlength_inputs_before_raw_adapter_inference = (
    contracts.test_truncate_false_rejects_overlength_inputs_before_raw_adapter_inference
)
test_persistent_resume_metadata_records_true_commit_granularity = (
    contracts.test_persistent_resume_metadata_records_true_commit_granularity
)
test_result_preserves_order_and_duplicates = contracts.test_result_preserves_order_and_duplicates
test_resume_recovers_from_authoritative_manifest_when_index_is_missing = (
    contracts.test_resume_recovers_from_authoritative_manifest_when_index_is_missing
)
test_resume_requires_matching_fingerprint = contracts.test_resume_requires_matching_fingerprint
test_open_safetensors_reader_survives_successful_overwrite = (
    contracts.test_open_safetensors_reader_survives_successful_overwrite
)
test_safetensors_manifest_rejects_shard_path_traversal = (
    contracts.test_safetensors_manifest_rejects_shard_path_traversal
)
test_safetensors_round_trip_is_lazy = contracts.test_safetensors_round_trip_is_lazy
test_safetensors_descriptor_shards_have_a_bounded_record_count = (
    contracts.test_safetensors_descriptor_shards_have_a_bounded_record_count
)
test_safetensors_tensor_corruption_is_detected_on_materialization = (
    contracts.test_safetensors_tensor_corruption_is_detected_on_materialization
)
test_safetensors_streaming_resumes_an_ordered_prefix = (
    contracts.test_safetensors_streaming_resumes_an_ordered_prefix
)
test_safetensors_generation_gc_is_dry_run_and_explicitly_exclusive = (
    contracts.test_safetensors_generation_gc_is_dry_run_and_explicitly_exclusive
)
test_sqlite_filtered_retrieval_preserves_selector_order_and_duplicates = (
    contracts.test_sqlite_filtered_retrieval_preserves_selector_order_and_duplicates
)
test_sqlite_successful_overwrite_becomes_default_and_retains_prior_run = (
    contracts.test_sqlite_successful_overwrite_becomes_default_and_retains_prior_run
)
test_interrupted_sqlite_overwrite_retains_prior_run_and_resumable_prefix = (
    contracts.test_interrupted_sqlite_overwrite_retains_prior_run_and_resumable_prefix
)
test_sqlite_first_batch_publication_is_atomic_and_hidden_run_resumes = (
    contracts.test_sqlite_first_batch_publication_is_atomic_and_hidden_run_resumes
)
test_sqlite_same_run_replacement_is_deferred_until_first_batch_commit = (
    contracts.test_sqlite_same_run_replacement_is_deferred_until_first_batch_commit
)
test_sqlite_prepublication_schema_remains_readable_and_migrates = (
    contracts.test_sqlite_prepublication_schema_remains_readable_and_migrates
)
test_sqlite_loading_and_lazy_tensor_reads_use_read_only_connections = (
    contracts.test_sqlite_loading_and_lazy_tensor_reads_use_read_only_connections
)
test_sqlite_round_trip_is_lazy_and_bf16_lossless = (
    contracts.test_sqlite_round_trip_is_lazy_and_bf16_lossless
)
test_sqlite_tensor_corruption_is_detected_on_materialization = (
    contracts.test_sqlite_tensor_corruption_is_detected_on_materialization
)
test_sqlite_streaming_resumes_an_ordered_prefix = (
    contracts.test_sqlite_streaming_resumes_an_ordered_prefix
)


class _TinyProteinTokenizer:
    """Deterministic tokenizer for real tiny tokenizer-mode model families."""

    pad_token_id = 1
    aa_cls_token = "<cls_aa>"
    aa_eos_token = "<eos_aa>"
    all_special_ids = (0, 1, 2, 5, 32)
    name_or_path = "offline/tiny-protein-tokenizer"
    model_max_length = 64
    padding_side = "right"
    truncation_side = "right"
    special_tokens_map: ClassVar[dict[str, str]] = {
        "cls_token": "<cls>",
        "eos_token": "<eos>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
    }
    _residue_ids: ClassVar[dict[str, int]] = {
        residue: token_id
        for residue, token_id in zip(
            "ACDEFGHIKLMNPQRSTVWYX",
            (
                3,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ),
            strict=True,
        )
    }

    def get_vocab(self) -> dict[str, int]:
        return {
            "<cls>": 0,
            "<pad>": 1,
            "<eos>": 2,
            "<mask>": 5,
            "<mask_aa>": 32,
            **self._residue_ids,
        }

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

    def __call__(
        self,
        sequences: str | list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[int] | list[list[int]]]:
        scalar = isinstance(sequences, str)
        sequence_rows = [sequences] if scalar else list(sequences)
        rows: list[list[int]] = []
        for sequence in sequence_rows:
            if sequence.startswith(self.aa_cls_token):
                if not sequence.endswith(self.aa_eos_token):
                    raise ValueError("DPLM2 sequence is missing its amino-acid EOS token")
                sequence = sequence[len(self.aa_cls_token) : -len(self.aa_eos_token)]
            row = [0, *(self._residue_ids[residue] for residue in sequence), 2]
            if kwargs.get("truncation") and kwargs.get("max_length") is not None:
                row = row[: int(kwargs["max_length"])]
            rows.append(row)

        width = max(map(len, rows))
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": padded[0] if scalar else padded}
        input_ids = torch.tensor(padded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }


def _real_family_model(family: str):
    tokenizer = _TinyProteinTokenizer()
    if family == "esm2":
        model = FastEsmModel(_esm2_config())
    elif family == "esm_plusplus":
        model = ESMplusplusModel(
            ESMplusplusConfig(
                vocab_size=16,
                hidden_size=8,
                num_attention_heads=2,
                num_hidden_layers=1,
                dropout=0.0,
                pad_token_id=1,
                mask_token_id=5,
                attn_backend="eager",
            )
        )
    elif family == "dplm":
        model = DPLMModel(DPLMConfig(**_dplm_config_values(33)))
    elif family == "dplm2":
        from tests.cpu.test_sequence_autoclass_contracts import _dplm2_config_values

        model = DPLM2Model(DPLM2Config(**_dplm2_config_values()))
    elif family == "ankh":
        model = FastAnkhModel(_ankh_config(num_layers=1, num_decoder_layers=1))
    elif family == "esm3":
        return FastESM3Model(
            FastESM3Config(
                hidden_size=8,
                num_attention_heads=2,
                num_vector_heads=2,
                num_hidden_layers=1,
                attn_backend="eager",
            )
        ).eval()
    elif family == "e1":
        return E1Model(_tiny_e1_config()).eval()
    else:
        raise AssertionError(f"Unknown embedding family: {family}")
    model.tokenizer = tokenizer
    return model.eval()


@pytest.mark.parametrize(
    ("family", "persist"),
    (
        ("esm2", True),
        ("esm_plusplus", False),
        ("dplm", False),
        ("dplm2", False),
        ("ankh", False),
        ("esm3", False),
        ("e1", False),
    ),
)
def test_every_real_sequence_family_uses_ordered_biological_embedding_path(
    family: str,
    persist: bool,
    tmp_path: Path,
) -> None:
    model = _real_family_model(family)
    sequences = ["ACD", "G", "ACD"]
    output = tmp_path / "real-family.sqlite" if persist else None

    result = model.embed_dataset(
        sequences,
        batch_size=3,
        full_embeddings=True,
        output=output,
        format="sqlite",
    )

    assert [record.id for record in result] == ["0", "1", "2"]
    assert [record.sequence for record in result] == sequences
    assert [record.load_tensor().shape[0] for record in result] == [3, 1, 3]
    assert all(torch.isfinite(record.load_tensor()).all() for record in result)
    torch.testing.assert_close(result[0].load_tensor(), result[2].load_tensor())
    assert result.metadata["residue_mask_policy"] == "biological-residues-only"

    if output is not None:
        reopened = load_sqlite_result(output)
        assert [record.sequence for record in reopened] == sequences
        for source, restored in zip(result, reopened, strict=True):
            torch.testing.assert_close(
                restored.load_tensor(),
                source.load_tensor(),
                rtol=0.0,
                atol=0.0,
            )


def test_generator_inputs_are_consumed_once_and_keep_stable_order() -> None:
    consumed: list[str] = []

    def sequences():
        for sequence in ("A", "CCCC", "GG"):
            consumed.append(sequence)
            yield sequence

    result = embed_dataset(
        contracts.SyntheticEmbeddingModel(),
        sequences(),
        batch_size=2,
        batch_window_size=3,
    )

    assert consumed == ["A", "CCCC", "GG"]
    assert [record.sequence for record in result] == consumed


test_strict_embedding_controls_fail_before_consuming_inputs = (
    contracts.test_strict_embedding_controls_fail_before_consuming_inputs
)
test_decoder_input_ids_require_nonempty_2d_int32_or_int64 = (
    contracts.test_decoder_input_ids_require_nonempty_2d_int32_or_int64
)
test_decoder_attention_masks_are_exact_finite_binary_shapes = (
    contracts.test_decoder_attention_masks_are_exact_finite_binary_shapes
)
test_embedding_batch_adapter_outputs_are_validated = (
    contracts.test_embedding_batch_adapter_outputs_are_validated
)
test_embedding_value_types_fail_closed = contracts.test_embedding_value_types_fail_closed
test_pagerank_controls_require_finite_valid_values = (
    contracts.test_pagerank_controls_require_finite_valid_values
)
test_tensor_sha256_uses_bounded_chunks_and_preserves_legacy_digest = (
    contracts.test_tensor_sha256_uses_bounded_chunks_and_preserves_legacy_digest
)
test_sqlite_lazy_references_are_absolute_after_cwd_change = (
    contracts.test_sqlite_lazy_references_are_absolute_after_cwd_change
)
