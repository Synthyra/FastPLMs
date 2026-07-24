"""Mandatory tiny state-conversion contracts for complete ANKH publication."""

from tests.release import test_conversion_tools as conversion_contracts


test_ankh_transform_requires_and_preserves_complete_t5_state = (
    conversion_contracts.test_ankh_transform_requires_and_preserves_complete_t5_state
)
test_ankh_transform_rejects_encoder_only_publication_state = (
    conversion_contracts.test_ankh_transform_rejects_encoder_only_publication_state
)
