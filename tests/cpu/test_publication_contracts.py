"""CPU allowlist aliases for the focused publication contracts."""

from tests.release import test_publish_files_only as publication_contracts


test_compile_model_files_builds_current_runtime_bundle = (
    publication_contracts.test_compile_model_files_builds_current_runtime_bundle
)
test_selection_defaults_to_every_model_and_rejects_bad_ids = (
    publication_contracts.test_selection_defaults_to_every_model_and_rejects_bad_ids
)
test_files_only_compiles_and_uploads_without_an_artifact = (
    publication_contracts.test_files_only_compiles_and_uploads_without_an_artifact
)
test_default_mode_adds_prepared_artifact_files_and_weights = (
    publication_contracts.test_default_mode_adds_prepared_artifact_files_and_weights
)
test_default_mode_explains_how_to_build_a_missing_artifact = (
    publication_contracts.test_default_mode_explains_how_to_build_a_missing_artifact
)
test_dry_run_compiles_but_does_not_commit = (
    publication_contracts.test_dry_run_compiles_but_does_not_commit
)
test_cli_accepts_default_and_files_only_modes = (
    publication_contracts.test_cli_accepts_default_and_files_only_modes
)
