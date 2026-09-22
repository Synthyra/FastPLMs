"""Mandatory sequence- and token-classification head contracts.

The CPU AutoClass evidence matrix cites these tests as runtime evidence for each
advertised classification head, so they must be collected by the CPU gate.
"""

from __future__ import annotations

from tests.unit import test_esm3_classification as esm3_heads
from tests.unit import test_esmfold2_classification as esmfold2_heads
from tests.unit import test_esmfold_classification as esmfold_heads


test_esm3_sequence_classifier_uses_final_residue_embeddings = (
    esm3_heads.test_esm3_sequence_classifier_uses_final_residue_embeddings
)
test_esm3_sequence_classifier_is_right_padding_invariant = (
    esm3_heads.test_esm3_sequence_classifier_is_right_padding_invariant
)
test_esm3_sequence_classifier_problem_type_losses = (
    esm3_heads.test_esm3_sequence_classifier_problem_type_losses
)
test_esm3_token_classifier_masks_special_padding_and_ignored_labels = (
    esm3_heads.test_esm3_token_classifier_masks_special_padding_and_ignored_labels
)
test_esm3_token_classifier_regression_and_multilabel_losses = (
    esm3_heads.test_esm3_token_classifier_regression_and_multilabel_losses
)
test_esm3_classifiers_support_tuple_and_dictionary_outputs = (
    esm3_heads.test_esm3_classifiers_support_tuple_and_dictionary_outputs
)
test_esm3_classifier_save_reload_round_trip = esm3_heads.test_esm3_classifier_save_reload_round_trip
test_esm3_sequence_classifier_preserves_multimodal_inputs = (
    esm3_heads.test_esm3_sequence_classifier_preserves_multimodal_inputs
)

test_prepare_classifier_inputs_is_residue_only_and_rejects_complexes = (
    esmfold_heads.test_prepare_classifier_inputs_is_residue_only_and_rejects_complexes
)
test_classifier_train_scope_is_exact = esmfold_heads.test_classifier_train_scope_is_exact
test_sequence_classifier_bypasses_folding_trunk = (
    esmfold_heads.test_sequence_classifier_bypasses_folding_trunk
)
test_token_classifier_masks_ignored_regression_and_multilabel_targets = (
    esmfold_heads.test_token_classifier_masks_ignored_regression_and_multilabel_targets
)
test_classifier_rejects_all_padding_rows = esmfold_heads.test_classifier_rejects_all_padding_rows

test_esmfold2_classifier_wrappers_bypass_structure_trunk = (
    esmfold2_heads.test_esmfold2_classifier_wrappers_bypass_structure_trunk
)
test_esmfold2_classifier_training_scopes_are_exact = (
    esmfold2_heads.test_esmfold2_classifier_training_scopes_are_exact
)
test_esmfold2_classifier_inputs_reject_non_protein_inputs = (
    esmfold2_heads.test_esmfold2_classifier_inputs_reject_non_protein_inputs
)
test_esmfold2_classifier_config_round_trips = (
    esmfold2_heads.test_esmfold2_classifier_config_round_trips
)
test_esmfold2_classifier_config_rejects_unknown_scope = (
    esmfold2_heads.test_esmfold2_classifier_config_rejects_unknown_scope
)
