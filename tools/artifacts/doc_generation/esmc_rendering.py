"""Render validated ESMC backend measurements and release disclosures."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping

from tools.artifacts.doc_generation.card_metadata import (
    _table_row,
)
from tools.artifacts.doc_generation.esmc_evidence import (
    ESMC_MEASURED_BACKENDS,
    ESMC_PANEL_KINDS,
    ESMC_REFERENCE_SOURCE_NAMES,
    ESMC_UNAVAILABLE_BACKENDS,
    EsmcReportError,
    EsmcReportSet,
    _esmc_require_finite,
    _esmc_require_gpu_capability,
    _esmc_require_list,
    _esmc_require_mapping,
    _esmc_require_object,
)

ESMC_RELEASE_DOCUMENTATION = """\
Detailed backend measurements, release guardrails, and the GH200 package
compatibility exception are maintained in the
[attention backend guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/attention_backends.md)
and
[release evidence manifest](https://github.com/Synthyra/FastPLMs/blob/main/docs/generated/capability_evidence.md).
"""


def _esmc_kernel_label(kernel: object) -> str:
    kernel = _esmc_require_object(kernel, context="Rendered ESMC kernel identity")
    if kernel["provider"] == "torch":
        return f"Torch {kernel['torch_version']}"
    return f"{kernel['repository']} v{kernel['version']} ({kernel['expected_variant']})"


def _esmc_reference_source_table(value: object) -> list[str]:
    sources = _esmc_require_mapping(
        value,
        set(ESMC_REFERENCE_SOURCE_NAMES),
        context="Rendered ESMC reference sources",
    )
    lines = [
        "Every report carries both official reference source attestations:",
        "",
        _table_row(
            "Source",
            "Schema",
            "Package",
            "Revision",
            "Import file",
            "Tree SHA-256",
            "Attestation SHA-256",
            "Files",
        ),
        _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
    ]
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        source = _esmc_require_object(
            sources[source_name],
            context=f"Rendered ESMC reference source {source_name}",
        )
        lines.append(
            _table_row(
                f"`{source_name}`",
                f"`{source['schema_version']}`",
                f"`{source['import_name']} {source['package_version']}`",
                f"`{source['source_revision']}`",
                f"`{source['import_file']}` under `{source['import_root']}`",
                f"`{source['tree_sha256']}`",
                f"`{source['attestation_sha256']}`",
                f"`{source['file_count']}`",
            )
        )
    return lines


def _esmc_number(value: object) -> str:
    numeric = _esmc_require_finite(value, context="Rendered ESMC metric")
    if numeric == 0:
        return "0"
    return f"{numeric:.6g}"


def _esmc_range(values: Iterable[object]) -> str:
    numbers = [
        _esmc_require_finite(value, context="Rendered ESMC range metric") for value in values
    ]
    return f"{_esmc_number(min(numbers))} to {_esmc_number(max(numbers))}"


def _esmc_distribution(values: Iterable[object]) -> str:
    numbers = [
        _esmc_require_finite(value, context="Rendered ESMC distribution metric") for value in values
    ]
    return (
        f"{_esmc_number(min(numbers))} / "
        f"{_esmc_number(statistics.median(numbers))} / {_esmc_number(max(numbers))}"
    )


def _esmc_pip_check_disclosure(
    evidence: EsmcReportSet | None,
    *,
    heading: str,
) -> str:
    if evidence is None:
        status = "The frozen oracle lock permits"
        exception: Mapping[str, object] = {
            "accepted_diagnostic": (
                "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
            ),
            "distribution": "nvidia-cusparselt-cu13",
            "version": "0.8.1",
            "wheel_filename": ("nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"),
            "wheel_sha256": ("4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"),
            "filename_platform_tag": "py3-none-manylinux2014_aarch64",
            "wheel_metadata_platform_tag": "py3-none-manylinux2014_sbsa",
            "target_hardware": "NVIDIA GH200 480GB",
            "target_operating_system": "linux",
            "target_architecture": "aarch64",
            "resolution": "validated-vendor-metadata-exception-no-wheel-rewrite",
        }
    else:
        status = "The validated oracle environment recorded"
        pip_check = _esmc_require_mapping(
            evidence.reference_environment.get("pip_check"),
            {
                "status",
                "returncode",
                "diagnostics",
                "accepted_platform_exceptions",
            },
            context="Rendered ESMC pip-check evidence",
        )
        diagnostics = _esmc_require_list(
            pip_check["diagnostics"], context="Rendered ESMC pip-check diagnostics"
        )
        exceptions = _esmc_require_list(
            pip_check["accepted_platform_exceptions"],
            context="Rendered ESMC pip-check platform exceptions",
        )
        if (
            pip_check["status"] != "accepted-platform-exception"
            or pip_check["returncode"] != 1
            or len(diagnostics) != 1
            or len(exceptions) != 1
        ):
            raise EsmcReportError("Rendered ESMC pip-check exception identity is invalid")
        exception = _esmc_require_object(
            exceptions[0], context="Rendered ESMC pip-check platform exception"
        )
        if diagnostics[0] != exception.get("accepted_diagnostic"):
            raise EsmcReportError("Rendered ESMC pip-check diagnostic is not attested")
    return f"""\
{heading} Locked oracle package compatibility exception

{status} exactly one nonzero `pip check` diagnostic:
`{exception["accepted_diagnostic"]}`. It applies only to
`{exception["distribution"]}=={exception["version"]}` on
`{exception["target_hardware"]}` / `{exception["target_operating_system"]}` /
`{exception["target_architecture"]}`. The vendor filename tag is
`{exception["filename_platform_tag"]}`, while the wheel metadata declares
`{exception["wheel_metadata_platform_tag"]}`. The exact wheel is
`{exception["wheel_filename"]}` with SHA-256 `{exception["wheel_sha256"]}`.
FastPLMs accepts this vendor metadata mismatch only after the lock, installed
inventory, wheel bytes, metadata tag, and target identity all match. The wheel
is not rewritten (`{exception["resolution"]}`). Any additional diagnostic or
identity drift fails closed.
"""


def _esmc_diagnostic_table(
    backends: Iterable[tuple[str, str]],
    *,
    model_id: str,
    evidence: EsmcReportSet | None,
) -> str:
    backend_rows = tuple(backends)
    if evidence is None:
        lines = [
            _table_row("Backend", "Support", "Measurement status"),
            _table_row("---", "---", "---"),
            _table_row(
                "`sdpa`",
                "Recommended fidelity path",
                "Pending release measurement",
            ),
        ]
        for backend, support in backend_rows:
            if backend in ESMC_UNAVAILABLE_BACKENDS:
                status = "Unavailable on current GH200/aarch64 lock"
            else:
                status = "Pending release measurement"
            lines.append(
                _table_row(
                    f"`{backend}`",
                    support,
                    status,
                )
            )
        return "\n".join(lines)

    display_backends = ("sdpa", *(backend for backend, _ in backend_rows))
    model_reports = tuple(
        evidence.get(model_id, backend, panel)
        for backend in display_backends
        for panel in ESMC_PANEL_KINDS
    )
    if len(model_reports) != len(display_backends) * len(ESMC_PANEL_KINDS):
        raise EsmcReportError(f"ESMC evidence for {model_id!r} is incomplete")
    measured_reports = tuple(
        report for report in model_reports if report["record_status"] == "measured"
    )
    unavailable_reports = tuple(
        report for report in model_reports if report["record_status"] == "unavailable"
    )
    expected_measured = len(set(display_backends).intersection(ESMC_MEASURED_BACKENDS)) * len(
        ESMC_PANEL_KINDS
    )
    expected_unavailable = len(set(display_backends).intersection(ESMC_UNAVAILABLE_BACKENDS)) * len(
        ESMC_PANEL_KINDS
    )
    if (
        len(measured_reports) != expected_measured
        or len(unavailable_reports) != expected_unavailable
    ):
        raise EsmcReportError(
            f"ESMC evidence for {model_id!r} must contain {expected_measured} measurements "
            f"and {expected_unavailable} structured unavailable records"
        )
    gpu = _esmc_require_object(
        evidence.candidate_environment["gpu"],
        context="Rendered ESMC candidate GPU identity",
    )
    capability = _esmc_require_gpu_capability(
        gpu["capability"],
        context="Rendered ESMC candidate GPU capability",
    )
    reference = _esmc_require_object(
        model_reports[0]["reference"], context="Rendered ESMC reference identity"
    )
    lines = [
        "The following values come from the complete validated schema-v3 release set.",
        f"All reports used `{gpu['name']}` (SM{capability[0]}{capability[1]}, "
        f"{gpu['total_memory_bytes']} bytes), BF16, runtime "
        f"`{evidence.runtime_identity.runtime_revision}`, source tree "
        f"`{evidence.runtime_identity.source_tree_sha256}`, and runtime bundle "
        f"`{evidence.runtime_identity.runtime_bundle_sha256}`. Results are evidence for this",
        "exact accelerator identity and are not cross-device equivalence claims.",
        "",
    ]
    lines.extend(_esmc_reference_source_table(reference["reference_sources"]))
    lines.extend(
        [
            "",
            "### Measurement identity",
            "",
            _table_row(
                "Panel",
                "Configured/effective",
                "dtype",
                "Kernel",
                "Release gate",
                "Catastrophe gate",
                "Band warnings",
                "Report SHA-256",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        ]
    )
    for report in model_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        release_gate = _esmc_require_object(
            report["release_gate"], context="Rendered ESMC release gate"
        )
        violations = _esmc_require_list(
            report["published_band_violations"],
            context="Rendered ESMC band warnings",
        )
        lines.append(
            _table_row(
                f"`{panel['kind']}` (`{str(panel['definition_sha256'])[:12]}`)",
                (
                    f"`{report['configured_backend']}` / `{report['effective_backend']}`"
                    if report["effective_backend"] is not None
                    else f"`{report['configured_backend']}` / not dispatched"
                ),
                f"`{report['dtype']}`",
                _esmc_kernel_label(report["kernel"]),
                f"`{release_gate['mode']}` / `{release_gate['status']}`",
                f"`{report['catastrophic_gate']}`",
                str(len(violations)),
                f"`{report['report_sha256']}`",
            )
        )
    if unavailable_reports:
        lines.extend(
            (
                "",
                "### Locked-platform unavailable backends",
                "",
                "These records are availability evidence, not numerical measurements. The",
                "backend remains supported, but dispatch fails closed when its locked kernel",
                "is unavailable on the exact report-bound release environment named below.",
                "",
                _table_row(
                    "Backend",
                    "Panel",
                    "Platform",
                    "Dispatch contract",
                    "Historical evidence",
                    "Reason",
                    "Report SHA-256",
                ),
                _table_row("---", "---", "---", "---", "---", "---", "---"),
            )
        )
        for report in unavailable_reports:
            panel = _esmc_require_object(
                report["panel"], context="Rendered unavailable ESMC panel identity"
            )
            unavailable = _esmc_require_object(
                report["unavailability"], context="Rendered ESMC unavailability identity"
            )
            lines.append(
                _table_row(
                    f"`{report['configured_backend']}`",
                    f"`{panel['kind']}`",
                    f"`{unavailable['platform']}` / `{unavailable['accelerator']}`",
                    f"`{unavailable['dispatch_contract']}`",
                    f"`{unavailable['historical_evidence']}`",
                    str(unavailable["reason"]),
                    f"`{report['report_sha256']}`",
                )
            )
    lines.extend(
        (
            "",
            "### Panel aggregates",
            "",
            "Tensor cells are the minimum-to-maximum range across every hidden-state layer,",
            "last hidden state, and logits entry in `panel_tensor_metrics`. Top-1 and JSD are",
            "the panel-level `panel_logits_metrics` aggregates. These are measured values,",
            "not release thresholds.",
            "",
            _table_row(
                "Backend",
                "Panel",
                "Relative L2",
                "Q99.9",
                "Residue cosine P01",
                "Pooled cosine min",
                "Top-1",
                "JSD",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_metrics = _esmc_require_list(
            report["panel_tensor_metrics"], context="Rendered ESMC panel metrics"
        )
        metrics = [
            _esmc_require_object(metric, context="Rendered ESMC panel tensor metric")
            for metric in raw_metrics
        ]
        logits_metrics = _esmc_require_object(
            report["panel_logits_metrics"], context="Rendered ESMC logits metrics"
        )
        lines.append(
            _table_row(
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                _esmc_range(metric["relative_l2"] for metric in metrics),
                _esmc_range(metric["relative_q999"] for metric in metrics),
                _esmc_range(metric["residue_cosine_p01"] for metric in metrics),
                _esmc_range(metric["pooled_cosine_min"] for metric in metrics),
                _esmc_number(logits_metrics["confident_top1_agreement"]),
                _esmc_number(logits_metrics["mean_jsd"]),
            )
        )
    lines.extend(
        (
            "",
            "### Per-case distributions",
            "",
            "Tensor cells are minimum / median / maximum across every case, output, and",
            "hidden-state layer in `cases[].tensor_metrics`. Top-1 and JSD use the same",
            "minimum / median / maximum summary over `cases[].logits_metrics`.",
            "",
            _table_row(
                "Backend",
                "Panel",
                "Relative L2",
                "Q99.9",
                "Residue cosine P01",
                "Pooled cosine min",
                "Top-1",
                "JSD",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_cases = _esmc_require_list(report["cases"], context="Rendered ESMC cases")
        cases = [_esmc_require_object(case, context="Rendered ESMC case") for case in raw_cases]
        tensor_metrics: list[Mapping[str, object]] = []
        case_logits: list[Mapping[str, object]] = []
        for case in cases:
            raw_case_metrics = _esmc_require_list(
                case["tensor_metrics"], context="Rendered ESMC case tensor metrics"
            )
            tensor_metrics.extend(
                _esmc_require_object(metric, context="Rendered ESMC case tensor metric")
                for metric in raw_case_metrics
            )
            case_logits.append(
                _esmc_require_object(
                    case["logits_metrics"], context="Rendered ESMC case logits metrics"
                )
            )
        lines.append(
            _table_row(
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                _esmc_distribution(metric["relative_l2"] for metric in tensor_metrics),
                _esmc_distribution(metric["relative_q999"] for metric in tensor_metrics),
                _esmc_distribution(metric["residue_cosine_p01"] for metric in tensor_metrics),
                _esmc_distribution(metric["pooled_cosine_min"] for metric in tensor_metrics),
                _esmc_distribution(metric["confident_top1_agreement"] for metric in case_logits),
                _esmc_distribution(metric["mean_jsd"] for metric in case_logits),
            )
        )
    return "\n".join(lines)


def _render_esmc_capability_evidence(evidence: EsmcReportSet | None) -> list[str]:
    lines = [
        "## Frozen ESMC release evidence",
        "",
    ]
    if evidence is None:
        lines.extend(
            (
                "**Status: pending.** Default documentation generation never discovers or",
                "trusts reports implicitly. Release rendering requires an explicitly selected,",
                "complete schema-v3 set of exactly 30 records on one exact GH200/aarch64",
                "target: 18 measured eager, SDPA, and Flex records plus 12 structured",
                "FlashAttention 2/3 unavailable records across three checkpoints and two",
                "immutable sequence panels.",
                "The set must also carry the final candidate/reference image identities,",
                "dependency lock, installed inventory, and official source attestations.",
                "A partial, stale, malformed, self-digest-invalid, or cross-device set fails",
                "closed and cannot replace this status.",
                "",
            )
        )
        lines.extend(_esmc_pip_check_disclosure(evidence, heading="###").rstrip().splitlines())
        lines.append("")
        return lines

    gpu = _esmc_require_object(
        evidence.candidate_environment["gpu"],
        context="Rendered ESMC candidate GPU identity",
    )
    capability = _esmc_require_gpu_capability(
        gpu["capability"],
        context="Rendered ESMC candidate GPU capability",
    )
    reference = _esmc_require_object(
        evidence.reports[0]["reference"], context="Rendered ESMC reference identity"
    )
    lines.extend(
        (
            f"**Status: validated complete set ({len(evidence.reports)}/30 records).**",
            "The set contains 18 measured eager, SDPA, and Flex records and 12",
            "structured FlashAttention 2/3 locked-platform unavailable records.",
            "",
            f"Exact device: `{gpu['name']}`; capability: `SM{capability[0]}"
            f"{capability[1]}`; memory: `{gpu['total_memory_bytes']}` bytes; "
            "dtype: `bfloat16`.",
            f"Runtime revision: `{evidence.runtime_identity.runtime_revision}`; source-tree "
            f"SHA-256: `{evidence.runtime_identity.source_tree_sha256}`; runtime-bundle "
            f"SHA-256: `{evidence.runtime_identity.runtime_bundle_sha256}`.",
            "",
        )
    )
    lines.extend(_esmc_reference_source_table(reference["reference_sources"]))
    lines.extend(("",))
    lines.extend(_esmc_pip_check_disclosure(evidence, heading="###").rstrip().splitlines())
    lines.extend(
        (
            "",
            "Results are not transferred to another accelerator identity. Each model card",
            "defines and publishes the corresponding per-case minimum/median/maximum",
            "distributions.",
            "",
            _table_row(
                "Checkpoint",
                "Backend",
                "Panel",
                "Relative L2 range",
                "Q99.9 range",
                "Residue cosine range",
                "Pooled cosine range",
                "Top-1",
                "JSD",
                "Band warnings",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    measured_reports = tuple(
        report for report in evidence.reports if report["record_status"] == "measured"
    )
    unavailable_reports = tuple(
        report for report in evidence.reports if report["record_status"] == "unavailable"
    )
    if len(measured_reports) != 18 or len(unavailable_reports) != 12:
        raise EsmcReportError(
            "Rendered ESMC set must contain 18 measurements and 12 unavailable records"
        )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_metrics = _esmc_require_list(
            report["panel_tensor_metrics"], context="Rendered ESMC panel metrics"
        )
        metrics = [
            _esmc_require_object(metric, context="Rendered ESMC panel tensor metric")
            for metric in raw_metrics
        ]
        logits = _esmc_require_object(
            report["panel_logits_metrics"], context="Rendered ESMC logits metrics"
        )
        violations = _esmc_require_list(
            report["published_band_violations"],
            context="Rendered ESMC band warnings",
        )
        lines.append(
            _table_row(
                f"`{report['model_id']}`",
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}` (`{str(panel['definition_sha256'])[:12]}`)",
                _esmc_range(metric["relative_l2"] for metric in metrics),
                _esmc_range(metric["relative_q999"] for metric in metrics),
                _esmc_range(metric["residue_cosine_p01"] for metric in metrics),
                _esmc_range(metric["pooled_cosine_min"] for metric in metrics),
                _esmc_number(logits["confident_top1_agreement"]),
                _esmc_number(logits["mean_jsd"]),
                str(len(violations)),
            )
        )
    lines.extend(
        (
            "",
            "### Current locked-platform Flash availability",
            "",
            _table_row(
                "Checkpoint",
                "Backend",
                "Panel",
                "Status",
                "Dispatch contract",
                "Historical evidence",
                "Reason",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in unavailable_reports:
        panel = _esmc_require_object(
            report["panel"], context="Rendered unavailable ESMC panel identity"
        )
        unavailable = _esmc_require_object(
            report["unavailability"], context="Rendered ESMC unavailability identity"
        )
        lines.append(
            _table_row(
                f"`{report['model_id']}`",
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                (f"`unavailable` on `{unavailable['platform']}` / `{unavailable['accelerator']}`"),
                f"`{unavailable['dispatch_contract']}`",
                f"`{unavailable['historical_evidence']}`",
                str(unavailable["reason"]),
            )
        )
    lines.append("")
    return lines
