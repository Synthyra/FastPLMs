"""Declared container commands and evidence stages for remote validation."""

from __future__ import annotations

from .contracts import _BIOHUB_BUILD_TARGET, _PORTABLE_RELEASE_BACKENDS, Suite


def _compose_run(service: str, *command: str) -> tuple[str, ...]:
    return (
        "sudo",
        "docker",
        "compose",
        "-f",
        "docker/compose.yaml",
        "run",
        "--rm",
        service,
        *command,
    )


_BUILD_ARTIFACTS = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.artifacts.build_all",
    "--output-root",
    "dist/hub",
    "--source-root",
    "/workspace",
)
_BUILD_BENCHMARK_ARTIFACTS = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.artifacts.build_all",
    "--benchmark-suite",
    "--output-root",
    "dist/hub",
    "--source-root",
    "/workspace",
)
_PREPARE_REFERENCES = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.remote.prepare_references",
    "--output-root",
    "artifacts/reference",
)
_SEQUENCE_REFERENCE_CONTAINERS = (
    "reference-esm2",
    "reference-biohub-esm",
    "reference-e1",
    "reference-dplm",
    "reference-ankh",
)
_RUN_NATIVE_REFERENCES = tuple(
    _compose_run(
        container,
        "python",
        "-m",
        "tests.parity.support.native_reference",
        "--request-dir",
        f"/exchange/requests/{container}",
        "--output-dir",
        "/exchange/results",
    )
    for container in _SEQUENCE_REFERENCE_CONTAINERS
)
_RUN_CHECK_ARTIFACTS = _compose_run(
    "artifact",
    "python",
    "-m",
    "pytest",
    "tests/release/test_published_automodel.py",
    "tests/release/test_manifest_readiness.py",
    "-m",
    "artifact",
    "-k",
    "not test_local_artifact_locked_flash_backend",
    "--junitxml=artifacts/junit/check-artifact.xml",
)
_RUN_CHECK_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/integration/test_official_goldens.py",
    "tests/structure/test_structure_official_goldens.py",
    "-m",
    "gpu and not large",
    "--junitxml=artifacts/junit/check-goldens.xml",
)
_RUN_PYTHON_MATRIX = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.remote.python_matrix",
    "--output",
    "artifacts/python-matrix.json",
    "--junit-output",
    "artifacts/junit/python-matrix.xml",
)
_RUN_NIGHTLY_FP8 = _compose_run(
    "fp8",
    "python",
    "-m",
    "pytest",
    "tests/structure/test_esmfold2_fp8_compliance.py",
    "--junitxml=artifacts/junit/nightly-fp8.xml",
)
_RUN_NIGHTLY_SEQUENCE_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/integration/test_official_goldens.py",
    "--junitxml=artifacts/junit/nightly-sequence-goldens.xml",
)
_RUN_NIGHTLY_STRUCTURE_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/structure/test_structure_official_goldens.py",
    "--junitxml=artifacts/junit/nightly-structure-goldens.xml",
)
_RUN_NIGHTLY_BENCHMARK = _compose_run(
    "benchmark",
    "--artifact-root",
    "dist/hub",
    "--backends",
    *_PORTABLE_RELEASE_BACKENDS,
    "--output",
    "artifacts/benchmarks/nightly-h100.json",
    "--junit-output",
    "artifacts/junit/nightly-benchmark.xml",
)
_RUN_RELEASE_BENCHMARK = _compose_run(
    "benchmark",
    "--artifact-root",
    "dist/hub",
    "--backends",
    *_PORTABLE_RELEASE_BACKENDS,
    "--output",
    "artifacts/benchmarks/release-h100.json",
    "--junit-output",
    "artifacts/junit/release-benchmark.xml",
)
_PREPARE_BOLTZ2_BUNDLE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_BOLTZ2_REFERENCE = _compose_run(
    "reference-boltz2",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "produce-reference",
    "--exchange-root",
    "/exchange",
)
_RUN_BOLTZ2_CANDIDATE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "produce-candidate",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_PREPARE_ESMFOLD_BUNDLE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.esmfold_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_ESMFOLD_REFERENCES = tuple(
    _compose_run(
        "reference-esmfold",
        "python",
        "-m",
        "tests.structure.support.esmfold_bundle",
        "produce-reference",
        "--exchange-root",
        "/exchange",
        "--precision",
        precision,
    )
    for precision in ("fp32", "bf16")
)
_RUN_ESMFOLD_CANDIDATES = tuple(
    _compose_run(
        "structure",
        "python",
        "-m",
        "tests.structure.support.esmfold_bundle",
        "produce-candidate",
        "--exchange-root",
        "/workspace/artifacts/reference",
        "--precision",
        precision,
    )
    for precision in ("fp32", "bf16")
)
_PREPARE_ESMFOLD2_BUNDLES = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.esmfold2_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_ESMFOLD2_REFERENCE = _compose_run(
    "reference-esmfold2",
    "python",
    "-m",
    "tests.structure.support.esmfold2_bundle",
    "produce-reference",
    "--exchange-root",
    "/exchange",
    "--all",
)
_RUN_ESMFOLD2_CANDIDATES = tuple(
    _compose_run(
        "fp8" if precision == "fp8" else "structure",
        "python",
        "-m",
        "tests.structure.support.esmfold2_bundle",
        "produce-candidate",
        "--exchange-root",
        "/workspace/artifacts/reference",
        "--all",
        "--precision",
        precision,
    )
    for precision in ("bf16", "fp8")
)
_RUN_STRUCTURE_REFERENCES = (
    _PREPARE_BOLTZ2_BUNDLE,
    _RUN_BOLTZ2_REFERENCE,
    _RUN_BOLTZ2_CANDIDATE,
    _PREPARE_ESMFOLD_BUNDLE,
    *_RUN_ESMFOLD_REFERENCES,
    *_RUN_ESMFOLD_CANDIDATES,
    _PREPARE_ESMFOLD2_BUNDLES,
    _RUN_ESMFOLD2_REFERENCE,
    *_RUN_ESMFOLD2_CANDIDATES,
)

_RUN_RELEASE_STRUCTURE_REFERENCES = (
    _PREPARE_ESMFOLD_BUNDLE,
    *_RUN_ESMFOLD_REFERENCES,
    *_RUN_ESMFOLD_CANDIDATES,
    _PREPARE_ESMFOLD2_BUNDLES,
    _RUN_ESMFOLD2_REFERENCE,
    _RUN_ESMFOLD2_CANDIDATES[0],
)

# These source-parity modules are self-contained in the candidate environment.
# Direct model parity plus ANKH and E1 parity remain in the isolated native
# reference workflow because their official dependencies conflict with it.
_RELEASE_LOCAL_PARITY_TESTS = (
    "tests/parity/test_esmfold2_common_parity.py",
    "tests/parity/test_esmfold2_protein_data_parity.py",
    "tests/parity/test_esmfold2_reimplemented_source_parity.py",
    "tests/parity/test_esmfold2_residue_config_parity.py",
    "tests/parity/test_esmfold2_source_slice3_parity.py",
    "tests/parity/test_esmfold2_source_slice4_parity.py",
)


SUITES = {
    "check": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "tests/integration",
            "tests/release",
            "-m",
            "not gpu and not slow and not structure and not artifact",
            "--junitxml=artifacts/junit/check.xml",
        ),
        pre_commands=(
            _RUN_CHECK_GOLDENS,
        ),
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "gpu-golden-smoke": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            (
                "tests/release/test_validation_stack.py::"
                "test_release_cuda_gpu_is_available_without_running_a_model"
            ),
            "tests/integration/test_official_goldens.py",
            "tests/structure/test_structure_official_goldens.py",
            "-m",
            "gpu and not large",
            "--junitxml=artifacts/junit/gpu-golden-smoke.xml",
        ),
    ),
    "unit": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "--junitxml=artifacts/junit/unit.xml",
        ),
    ),
    "integration": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/integration",
            "--junitxml=artifacts/junit/integration.xml",
        ),
    ),
    "compliance": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            _BIOHUB_BUILD_TARGET,
            "reference-esm2",
            "reference-biohub-esm",
            "reference-e1",
            "reference-dplm",
            "reference-ankh",
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "fp8",
            "python",
            "-m",
            "pytest",
            "tests/parity/test_native_results.py",
            (
                "tests/release/test_validation_stack.py::"
                "test_release_cuda_gpu_is_available_without_running_a_model"
            ),
            (
                "tests/release/test_validation_stack.py::"
                "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
            ),
            "tests/structure/test_esmfold_folding_compliance.py",
            "tests/structure/test_esmfold2_folding_compliance.py",
            "tests/structure/test_esmfold2_fp8_compliance.py",
            "--junitxml=artifacts/junit/compliance.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REFERENCES,
            *_RUN_RELEASE_STRUCTURE_REFERENCES,
        ),
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "structure": Suite(
        (
            "candidate-structure",
            "candidate-fp8",
            _BIOHUB_BUILD_TARGET,
            "reference-boltz2",
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/structure",
            "tests/parity/test_boltz_source_refactor.py",
            "--ignore=tests/structure/test_structure_models.py",
            "-m",
            "structure",
            "--junitxml=artifacts/junit/structure.xml",
        ),
        pre_commands=_RUN_STRUCTURE_REFERENCES,
    ),
    "feature": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/integration/test_binder_design.py",
            "tests/integration/test_dplm_generation.py",
            "tests/integration/test_e1_rag.py",
            "tests/integration/test_esm3.py",
            "tests/integration/test_ttt.py",
            "tests/release/test_conversion_tools.py",
            "--junitxml=artifacts/junit/feature.xml",
        ),
    ),
    "artifact": Suite(
        ("candidate", "candidate-artifact"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "artifact",
            "python",
            "-m",
            "pytest",
            "tests/release",
            "-m",
            "artifact",
            "-k",
            "not test_local_artifact_locked_flash_backend",
            "--junitxml=artifacts/junit/artifact.xml",
        ),
        pre_commands=(_BUILD_ARTIFACTS,),
    ),
    "benchmark": Suite(
        ("candidate", "candidate-fp8"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "benchmark",
            "--artifact-root",
            "dist/hub",
            "--backends",
            *_PORTABLE_RELEASE_BACKENDS,
            "--output",
            "artifacts/benchmarks/h100-current.json",
            "--baseline",
            "benchmarks/baselines/h100.json",
            "--junit-output",
            "artifacts/junit/benchmark.xml",
        ),
        pre_commands=(_BUILD_BENCHMARK_ARTIFACTS,),
        required_paths=("benchmarks/baselines/h100.json",),
        pre_command_timeout_seconds=14_400,
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "benchmark-capture": Suite(
        ("candidate", "candidate-fp8"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "benchmark",
            "--artifact-root",
            "dist/hub",
            "--backends",
            *_PORTABLE_RELEASE_BACKENDS,
            "--output",
            "artifacts/benchmarks/h100-baseline-candidate.json",
            "--junit-output",
            "artifacts/junit/benchmark-capture.xml",
        ),
        pre_commands=(_BUILD_BENCHMARK_ARTIFACTS,),
        pre_command_timeout_seconds=14_400,
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "nightly": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            "candidate-artifact",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/integration/test_backend_consistency.py",
            "tests/integration/test_binder_design.py",
            "tests/integration/test_dplm_generation.py",
            "tests/integration/test_e1_rag.py",
            "tests/integration/test_esm3.py",
            "tests/integration/test_ttt.py",
            "tests/unit/test_fine_tuning_example.py",
            "--junitxml=artifacts/junit/nightly-features.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _RUN_CHECK_ARTIFACTS,
            _RUN_NIGHTLY_SEQUENCE_GOLDENS,
            _RUN_NIGHTLY_STRUCTURE_GOLDENS,
            _RUN_NIGHTLY_FP8,
            _RUN_NIGHTLY_BENCHMARK,
        ),
        pre_command_timeout_seconds=14_400,
        command_timeout_seconds=21_600,
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "release": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            "candidate-artifact",
            _BIOHUB_BUILD_TARGET,
            *_SEQUENCE_REFERENCE_CONTAINERS,
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "tests/integration",
            "tests/release",
            "tests/parity/test_native_results.py",
            *_RELEASE_LOCAL_PARITY_TESTS,
            "tests/structure",
            "tests/parity/test_boltz_source_refactor.py",
            "--ignore=tests/structure/test_structure_models.py",
            "--ignore=tests/structure/test_esmfold2_fp8_compliance.py",
            "--ignore=tests/integration/test_flash_attention_backends.py",
            (
                "--deselect=tests/release/test_validation_stack.py::"
                "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
            ),
            (
                "--deselect=tests/structure/test_boltz2_folding_compliance.py::"
                "test_boltz2_live_folding_matches_pinned_official"
            ),
            "-m",
            "not artifact",
            "--junitxml=artifacts/junit/release.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _RUN_CHECK_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REFERENCES,
            *_RUN_RELEASE_STRUCTURE_REFERENCES,
            _RUN_PYTHON_MATRIX,
            _RUN_RELEASE_BENCHMARK,
        ),
        pre_command_timeout_seconds=21_600,
        attention_backends=_PORTABLE_RELEASE_BACKENDS,
    ),
    "python-matrix": Suite(
        ("candidate",),
        _RUN_PYTHON_MATRIX,
    ),
}
