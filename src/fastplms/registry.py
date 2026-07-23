"""Typed access to the FastPLMs model and provenance manifest.

The registry is intentionally independent of Torch and Transformers. Tooling can
therefore inspect supported checkpoints, licenses, and reference sources without
initializing a model runtime or downloading any files.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlparse

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HUB_LICENSE_NAME_RE = re.compile(r"[^a-z0-9.]+")
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_PATH_NAMES = frozenset(
    {"AUX", "CON", "NUL", "PRN"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_REPOSITORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REFERENCE_CONTAINER_RE = re.compile(r"^reference-[a-z0-9]+(?:-[a-z0-9]+)*$")
_REFERENCE_ADAPTER_RE = re.compile(
    r"^tests\.parity\.support\.reference_adapters\.[a-z_][a-z0-9_]*$"
)
_DOCUMENTATION_FRAGMENT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ALLOWED_ATTENTION = frozenset(
    {"eager", "sdpa", "flex_attention", "flash_attention_2", "flash_attention_3"}
)
_ALLOWED_DTYPES = frozenset({"float32", "bfloat16"})
_ALLOWED_PRECISIONS = frozenset({"default", "auto", "fp32", "bf16", "fp8"})
_ALLOWED_BF16_EXECUTIONS = frozenset({"static_parameters", "fp32_parameters_autocast"})
HUB_LICENSE_IDENTIFIERS = frozenset({"mit", "apache-2.0", "cc-by-nc-sa-4.0", "other"})
_ALLOWED_TOKENIZER_MODES = frozenset({"tokenizer", "sequence", "structure"})
_ALLOWED_SIZE_CATEGORIES = frozenset({"small", "medium", "large", "xlarge", "structure"})
RuntimeExtra = Literal["core", "structure"]
TestTier = Literal["check", "compliance", "structure", "feature", "artifact", "benchmark"]
VramTier = Literal["sequence", "large-sequence", "structure", "structure-6b"]
GenerationContract = Literal["not_applicable", "required", "official_unavailable"]
RuntimeAssetTrustKind = Literal["hash_pinned_pickle"]
Bf16Execution = Literal["static_parameters", "fp32_parameters_autocast"]
DtypeName = Literal["float32", "bfloat16"]
_ALLOWED_EXTRAS = frozenset({"core", "structure"})
_ALLOWED_TEST_TIERS = frozenset(
    {"check", "compliance", "structure", "feature", "artifact", "benchmark"}
)
_ALLOWED_VRAM_TIERS = frozenset({"sequence", "large-sequence", "structure", "structure-6b"})
_ALLOWED_GENERATION_CONTRACTS = frozenset({"not_applicable", "required", "official_unavailable"})
_ALLOWED_RUNTIME_ASSET_TRUST_KINDS = frozenset({"hash_pinned_pickle"})
_ALLOWED_RUNTIME_ASSET_OFFLINE_BEHAVIORS = frozenset({"requires_cached_verified_file"})
_ALLOWED_AUTO_CLASSES = frozenset(
    {
        "AutoConfig",
        "AutoModel",
        "AutoModelForMaskedLM",
        "AutoModelForProteinFolding",
        "AutoModelForSequenceClassification",
        "AutoModelForSeq2SeqLM",
        "AutoModelForTokenClassification",
    }
)
_WEIGHT_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".safetensors")
_ALLOWED_ORACLE_ASSET_ROLES = frozenset({"weights", "contact_regression"})
_FAIR_ESM_ASSET_HOST = "dl.fbaipublicfiles.com"
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "legal_files",
        "attention_kernels",
        "upstreams",
        "families",
        "models",
        "runtime_assets",
    }
)
_UPSTREAM_FIELDS = frozenset(
    {
        "id",
        "path",
        "url",
        "revision",
        "license",
        "license_files",
        "license_digests",
        "distribution_files",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "architecture",
        "upstreams",
        "tokenizer_mode",
        "public_input",
        "extra",
        "reference_container",
        "reference_adapter",
        "attention",
        "dtypes",
        "bf16_execution",
        "precisions",
        "experimental_precisions",
        "vram_tier",
        "checkpoint_license",
        "hub_license",
        "hub_license_name",
        "hub_license_link",
        "state_transform",
        "conversion_provenance",
        "representative",
        "documentation",
        "test_tiers",
        "runtime_paths",
        "requires_complete_weight_publication",
        "weights_publication_allowed",
        "auto_map",
        "tokenizer_class",
        "backbone_model",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "id",
        "family",
        "size_category",
        "generation_contract",
        "fast_repo",
        "fast_revision",
        "fast_files",
        "fast_unresolved_files",
        "official_repo",
        "official_revision",
        "official_files",
        "official_unresolved_files",
        "oracle_assets",
        "official_golden",
        "artifact_source",
        "canonical_state_sha256",
        "tokenizer_source",
        "auto_map",
        "notes",
        "msa_conditioning",
    }
)
_RUNTIME_ASSET_FIELDS = frozenset(
    {
        "id",
        "repository",
        "revision",
        "path",
        "sha256",
        "size",
        "consumer_family",
        "trust_kind",
        "license",
        "offline_behavior",
    }
)


class RegistryError(ValueError):
    """Raised when the model manifest is incomplete or internally inconsistent."""


def _portable_relative_path(value: str, context: str) -> PurePosixPath:
    """Return one normalized cross-platform relative path or fail closed."""

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    unsafe_windows_part = any(
        part.rstrip(" .") != part
        or part.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_PATH_NAMES
        or any(
            ord(character) < 32 or character in _WINDOWS_INVALID_PATH_CHARACTERS
            for character in part
        )
        for part in posix.parts
    )
    if (
        not value
        or not posix.parts
        or posix == PurePosixPath(".")
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or "." in posix.parts
        or ".." in posix.parts
        or value != posix.as_posix()
        or any(
            part.lower() in {".git", ".cache", "__pycache__"}
            for part in posix.parts
        )
        or unsafe_windows_part
    ):
        raise RegistryError(f"{context} is not portable: {value!r}")
    return posix


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Expected content identity for one pinned file."""

    path: str
    algorithm: str
    digest: str

    @classmethod
    def parse(cls, value: str) -> FileDigest:
        try:
            path, encoded_digest = value.split("=", maxsplit=1)
            algorithm, digest = encoded_digest.split(":", maxsplit=1)
        except ValueError as error:
            raise RegistryError("File digests must use '<path>=<algorithm>:<digest>'.") from error

        _portable_relative_path(path, "Checkpoint file path")

        expected_length = {"git-sha1": 40, "sha256": 64}.get(algorithm)
        if expected_length is None:
            raise RegistryError(f"Unsupported file digest algorithm: {algorithm!r}")
        if len(digest) != expected_length or _HEX_RE.fullmatch(digest) is None:
            raise RegistryError(f"Invalid {algorithm} digest for {path!r}: {digest!r}")
        return cls(path=path, algorithm=algorithm, digest=digest)

    @property
    def encoded(self) -> str:
        return f"{self.algorithm}:{self.digest}"


@dataclass(frozen=True, slots=True)
class CheckpointSource:
    """One immutable Hugging Face repository snapshot."""

    repo_id: str
    revision: str
    files: tuple[FileDigest, ...]
    unresolved_files: tuple[str, ...] = ()

    @property
    def file_map(self) -> Mapping[str, FileDigest]:
        return MappingProxyType({item.path: item for item in self.files})


@dataclass(frozen=True, slots=True)
class OracleAsset:
    """Hash-pinned external file required by a native parity oracle."""

    role: str
    path: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class RuntimeAsset:
    """Immutable runtime data with an explicit deserialization trust boundary."""

    id: str
    repository: str
    revision: str
    path: str
    sha256: str
    size: int
    consumer_family: str
    trust_kind: RuntimeAssetTrustKind
    license_expression: str
    offline_behavior: str


@dataclass(frozen=True, slots=True)
class OfficialGolden:
    """Hash-pinned official output bundle required by the check tier."""

    metadata: FileDigest
    tensors: FileDigest


@dataclass(frozen=True, slots=True)
class UpstreamSource:
    """Pinned official implementation used as a parity oracle."""

    id: str
    path: str
    url: str
    revision: str
    license_expression: str
    license_files: tuple[str, ...]
    license_digests: tuple[FileDigest, ...] = ()
    distribution_files: tuple[FileDigest, ...] = ()


@dataclass(frozen=True, slots=True)
class AttentionKernelSpec:
    """Immutable Hugging Face kernel used by one attention backend."""

    implementation: str
    repository: str
    revision: str
    version: int
    expected_variant: str
    dtypes: tuple[DtypeName, ...]


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """Shared runtime and compliance contract for one architecture family."""

    id: str
    architecture: str
    upstreams: tuple[str, ...]
    tokenizer_mode: str
    public_input: str
    extra: RuntimeExtra
    reference_container: str
    reference_adapter: str
    attention: tuple[str, ...]
    dtypes: tuple[DtypeName, ...]
    bf16_execution: Bf16Execution
    precisions: tuple[str, ...]
    vram_tier: VramTier
    checkpoint_license: str
    hub_license: str
    state_transform: str
    representative: str
    documentation: str
    test_tiers: tuple[TestTier, ...]
    runtime_paths: tuple[str, ...]
    auto_map_items: tuple[tuple[str, str], ...]
    requires_complete_weight_publication: bool = False
    weights_publication_allowed: bool = False
    experimental_precisions: tuple[str, ...] = ()
    tokenizer_class: str | None = None
    hub_license_name: str | None = None
    hub_license_link: str | None = None
    conversion_provenance: str = ""
    backbone_model: str | None = None

    @property
    def auto_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.auto_map_items))

    @property
    def hub_license_metadata(self) -> Mapping[str, str]:
        """Return valid Hugging Face model-card license fields."""

        metadata = {"license": self.hub_license}
        if self.hub_license_name is not None:
            # Hugging Face validates custom license names as lowercase slugs,
            # while the manifest retains the reader-facing display name used
            # in generated prose.
            metadata["license_name"] = _HUB_LICENSE_NAME_RE.sub(
                "-",
                self.hub_license_name.lower(),
            ).strip("-.")
        if self.hub_license_link is not None:
            metadata["license_link"] = self.hub_license_link
        return MappingProxyType(metadata)

    @property
    def stable_precisions(self) -> tuple[str, ...]:
        """Return precision policies covered by the release contract."""

        experimental = set(self.experimental_precisions)
        return tuple(precision for precision in self.precisions if precision not in experimental)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Complete immutable source and runtime contract for one checkpoint."""

    id: str
    family: ModelFamily
    fast: CheckpointSource
    official: CheckpointSource
    size_category: str
    generation_contract: GenerationContract = "not_applicable"
    oracle_assets: tuple[OracleAsset, ...] = ()
    official_golden: OfficialGolden | None = None
    artifact_source: str = "fast"
    canonical_state_sha256: str | None = None
    tokenizer_source_id: str | None = None
    auto_map_items: tuple[tuple[str, str], ...] = ()
    notes: str = ""
    msa_conditioning: bool | None = None

    @property
    def is_deep_reference(self) -> bool:
        return self.id == self.family.representative

    @property
    def auto_map(self) -> Mapping[str, str]:
        if self.auto_map_items:
            return MappingProxyType(dict(self.auto_map_items))
        return self.family.auto_map

    @property
    def artifact_checkpoint(self) -> CheckpointSource:
        """Return the checkpoint selected for local artifact construction."""

        return self.fast if self.artifact_source == "fast" else self.official

    @property
    def oracle_asset_map(self) -> Mapping[str, OracleAsset]:
        """Return native oracle assets keyed by their declared role."""

        return MappingProxyType({asset.role: asset for asset in self.oracle_assets})


class ModelRegistry(Mapping[str, ModelSpec]):
    """Validated mapping of model IDs to typed model specifications."""

    def __init__(
        self,
        *,
        schema_version: int,
        upstreams: Mapping[str, UpstreamSource],
        families: Mapping[str, ModelFamily],
        models: Mapping[str, ModelSpec],
        runtime_assets: Mapping[str, RuntimeAsset] = MappingProxyType({}),
        attention_kernels: Mapping[str, AttentionKernelSpec] = MappingProxyType({}),
        legal_files: tuple[FileDigest, ...] = (),
    ) -> None:
        self.schema_version = schema_version
        self.upstreams = MappingProxyType(dict(upstreams))
        self.attention_kernels = MappingProxyType(dict(attention_kernels))
        self.families = MappingProxyType(dict(families))
        self._models = MappingProxyType(dict(models))
        self.runtime_assets = MappingProxyType(dict(runtime_assets))
        self.legal_files = legal_files

    def __getitem__(self, key: str) -> ModelSpec:
        return self._models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def by_family(self, family_id: str) -> tuple[ModelSpec, ...]:
        if family_id not in self.families:
            raise KeyError(family_id)
        return tuple(model for model in self._models.values() if model.family.id == family_id)

    def supported_attention_dtypes(
        self,
        family_id: str,
        implementation: str,
    ) -> tuple[DtypeName, ...]:
        """Return manifest-supported dtypes for one family/backend pair."""

        family = self.families[family_id]
        if implementation not in family.attention:
            raise KeyError(
                f"Family {family_id!r} does not advertise attention backend "
                f"{implementation!r}."
            )
        kernel = self.attention_kernels.get(implementation)
        if kernel is None:
            return family.dtypes
        return tuple(dtype for dtype in family.dtypes if dtype in kernel.dtypes)

    def require_resolved(self, model_id: str | None = None) -> None:
        """Fail release validation when required file identities remain unresolved."""

        selected = self._models.values() if model_id is None else (self._models[model_id],)
        unresolved: list[str] = []
        for model in selected:
            for label, checkpoint in (("fast", model.fast), ("official", model.official)):
                for path in checkpoint.unresolved_files:
                    unresolved.append(f"{model.id}.{label}:{path}")
        if unresolved:
            detail = ", ".join(unresolved)
            raise RegistryError(f"Release provenance is unresolved: {detail}")


def _reject_unknown_fields(
    table: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(table).difference(allowed))
    if unknown:
        raise RegistryError(f"{context} contains unknown fields: {unknown}.")


def _require_str(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{context}.{key} must be a non-empty string.")
    return value


def _require_enum(
    table: Mapping[str, Any],
    key: str,
    context: str,
    allowed: frozenset[str],
) -> str:
    value = _require_str(table, key, context)
    if value not in allowed:
        raise RegistryError(
            f"{context}.{key} must be one of {sorted(allowed)}; received {value!r}."
        )
    return value


def _parse_reference_container(table: Mapping[str, Any], context: str) -> str:
    value = _require_str(table, "reference_container", context)
    if _REFERENCE_CONTAINER_RE.fullmatch(value) is None:
        raise RegistryError(
            f"{context}.reference_container must be a portable 'reference-<name>' target."
        )
    return value


def _parse_reference_adapter(table: Mapping[str, Any], context: str) -> str:
    value = _require_str(table, "reference_adapter", context)
    if _REFERENCE_ADAPTER_RE.fullmatch(value) is None:
        raise RegistryError(
            f"{context}.reference_adapter must name one module under "
            "tests.parity.support.reference_adapters."
        )
    return value


def _parse_documentation_path(table: Mapping[str, Any], context: str) -> str:
    value = _require_str(table, "documentation", context)
    if value.count("#") > 1 or "\\" in value:
        raise RegistryError(f"{context}.documentation must be a portable documentation path.")
    raw_path, separator, fragment = value.partition("#")
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] != "docs"
        or path.suffix != ".md"
        or path.as_posix() != raw_path
    ):
        raise RegistryError(
            f"{context}.documentation must reference a normalized Markdown file under docs/."
        )
    if separator and _DOCUMENTATION_FRAGMENT_RE.fullmatch(fragment) is None:
        raise RegistryError(f"{context}.documentation has an invalid heading fragment.")
    return value


def _require_str_list(table: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise RegistryError(f"{context}.{key} must be a non-empty string array.")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RegistryError(f"{context}.{key} contains duplicate values.")
    return result


def _optional_str_list(table: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RegistryError(f"{context}.{key} must be a string array.")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RegistryError(f"{context}.{key} contains duplicate values.")
    return result


def _optional_str(table: Mapping[str, Any], key: str, context: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise RegistryError(f"{context}.{key} must be a non-empty single-line string.")
    return value


def _parse_hub_license(
    table: Mapping[str, Any],
    *,
    checkpoint_license: str,
    context: str,
) -> tuple[str, str | None, str | None]:
    expected_fields = {"hub_license", "hub_license_name", "hub_license_link"}
    unknown_fields = sorted(
        key for key in table if key.startswith("hub_") and key not in expected_fields
    )
    if unknown_fields:
        raise RegistryError(f"{context} contains unsupported Hub license fields: {unknown_fields}.")
    identifier = _require_str(table, "hub_license", context)
    if identifier not in HUB_LICENSE_IDENTIFIERS:
        raise RegistryError(
            f"{context}.hub_license must be a supported Hugging Face license identifier."
        )
    expected_identifier: str | None = None
    for prefix, candidate in (
        ("MIT", "mit"),
        ("Apache-2.0", "apache-2.0"),
        ("CC-BY-NC-SA-4.0", "cc-by-nc-sa-4.0"),
        ("Profluent-E1-Agreement", "other"),
        ("Unresolved", "other"),
    ):
        if checkpoint_license.startswith(prefix):
            expected_identifier = candidate
            break
    if expected_identifier is None:
        raise RegistryError(
            f"{context}.checkpoint_license has no declared Hugging Face identifier mapping."
        )
    if identifier != expected_identifier:
        raise RegistryError(
            f"{context}.hub_license must be {expected_identifier!r} for "
            f"checkpoint terms {checkpoint_license!r}."
        )

    name = _optional_str(table, "hub_license_name", context)
    link = _optional_str(table, "hub_license_link", context)
    if identifier != "other":
        if name is not None or link is not None:
            raise RegistryError(
                f"{context} may define hub_license_name and hub_license_link only "
                "when hub_license='other'."
            )
        return identifier, None, None
    if name is None or link is None:
        raise RegistryError(
            f"{context} must define hub_license_name and hub_license_link when hub_license='other'."
        )
    parsed_link = urlparse(link)
    if (
        parsed_link.scheme != "https"
        or not parsed_link.netloc
        or not parsed_link.path
        or parsed_link.username is not None
        or parsed_link.password is not None
    ):
        raise RegistryError(f"{context}.hub_license_link must be an absolute HTTPS URL.")
    return identifier, name, link


def _require_digest_list(
    table: Mapping[str, Any], key: str, context: str
) -> tuple[FileDigest, ...]:
    encoded = _require_str_list(table, key, context)
    result = tuple(FileDigest.parse(value) for value in encoded)
    paths = [item.path for item in result]
    if len(paths) != len(set(paths)):
        raise RegistryError(f"{context}.{key} contains duplicate paths.")
    return result


def _validate_revision(revision: str, context: str) -> None:
    if len(revision) != 40 or _HEX_RE.fullmatch(revision) is None:
        raise RegistryError(f"{context} must be an immutable 40-character commit revision.")


def _parse_checkpoint(table: Mapping[str, Any], prefix: str, context: str) -> CheckpointSource:
    repo_id = _require_str(table, f"{prefix}_repo", context)
    if _REPOSITORY_ID_RE.fullmatch(repo_id) is None:
        raise RegistryError(f"{context}.{prefix}_repo must be a Hugging Face repository ID.")
    revision = _require_str(table, f"{prefix}_revision", context)
    _validate_revision(revision, f"{context}.{prefix}_revision")
    encoded_files = _require_str_list(table, f"{prefix}_files", context)
    files = tuple(FileDigest.parse(value) for value in encoded_files)
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise RegistryError(f"{context}.{prefix}_files contains duplicate paths.")
    if not any(item.path.endswith(_WEIGHT_SUFFIXES) for item in files):
        raise RegistryError(f"{context}.{prefix}_files does not identify a weight file.")
    unresolved_files = _optional_str_list(table, f"{prefix}_unresolved_files", context)
    for unresolved_path in unresolved_files:
        _portable_relative_path(unresolved_path, "Unresolved checkpoint path")
        if unresolved_path in paths:
            raise RegistryError(
                f"{context}.{prefix} marks {unresolved_path!r} both resolved and unresolved."
            )
    return CheckpointSource(
        repo_id=repo_id,
        revision=revision,
        files=files,
        unresolved_files=unresolved_files,
    )


def _parse_oracle_assets(table: Mapping[str, Any], context: str) -> tuple[OracleAsset, ...]:
    raw = table.get("oracle_assets", [])
    if not isinstance(raw, list):
        raise RegistryError(f"{context}.oracle_assets must be an array of tables.")
    result: list[OracleAsset] = []
    for index, value in enumerate(raw):
        asset_context = f"{context}.oracle_assets[{index}]"
        if not isinstance(value, dict):
            raise RegistryError(f"{asset_context} must be a table.")
        expected_fields = {"role", "path", "url", "sha256", "size"}
        if set(value) != expected_fields:
            raise RegistryError(f"{asset_context} must contain exactly {sorted(expected_fields)}.")
        role = _require_str(value, "role", asset_context)
        if role not in _ALLOWED_ORACLE_ASSET_ROLES:
            raise RegistryError(f"Unsupported oracle asset role: {role!r}.")
        path = _require_str(value, "path", asset_context)
        try:
            normalized_path = _portable_relative_path(path, "Oracle asset path")
        except RegistryError as error:
            raise RegistryError(f"Invalid oracle asset path: {path!r}.") from error
        if normalized_path.suffix != ".pt":
            raise RegistryError(f"Invalid oracle asset path: {path!r}.")
        url = _require_str(value, "url", asset_context)
        parsed_url = urlparse(url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname != _FAIR_ESM_ASSET_HOST
            or parsed_url.path != f"/fair-esm/{path}"
            or parsed_url.params
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise RegistryError(f"Invalid fair-esm oracle asset URL: {url!r}.")
        sha256 = _require_str(value, "sha256", asset_context)
        if len(sha256) != 64 or _HEX_RE.fullmatch(sha256) is None:
            raise RegistryError(f"Invalid oracle asset SHA-256 for {path!r}.")
        size = value.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RegistryError(f"{asset_context}.size must be a positive byte count.")
        result.append(
            OracleAsset(
                role=role,
                path=path,
                url=url,
                sha256=sha256,
                size=size,
            )
        )
    roles = [asset.role for asset in result]
    paths = [asset.path for asset in result]
    urls = [asset.url for asset in result]
    if (
        len(roles) != len(set(roles))
        or len(paths) != len(set(paths))
        or len(urls) != len(set(urls))
    ):
        raise RegistryError(f"{context}.oracle_assets contains duplicate identities.")
    return tuple(result)


def _parse_official_golden(
    table: Mapping[str, Any],
    model_id: str,
    context: str,
) -> OfficialGolden | None:
    raw = table.get("official_golden")
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"metadata", "tensors"}:
        raise RegistryError(
            f"{context}.official_golden must contain exactly 'metadata' and 'tensors'."
        )
    parsed: dict[str, FileDigest] = {}
    for role in ("metadata", "tensors"):
        value = raw[role]
        if not isinstance(value, str):
            raise RegistryError(f"{context}.official_golden.{role} must be a file digest.")
        digest = FileDigest.parse(value)
        if digest.algorithm != "sha256":
            raise RegistryError(
                f"{context}.official_golden.{role} must use an immutable SHA-256 digest."
            )
        expected = f"tests/goldens/{model_id}.{'json' if role == 'metadata' else 'safetensors'}"
        if digest.path != expected:
            raise RegistryError(f"{context}.official_golden.{role} must use path {expected!r}.")
        parsed[role] = digest
    return OfficialGolden(metadata=parsed["metadata"], tensors=parsed["tensors"])


def _parse_attention_kernels(raw: object) -> dict[str, AttentionKernelSpec]:
    if not isinstance(raw, list) or not raw:
        raise RegistryError("The manifest must contain [[attention_kernels]] entries.")
    result: dict[str, AttentionKernelSpec] = {}
    expected_variants = {
        "flash_attention_2": "flash_attn2",
        "flash_attention_3": "flash_attn3",
    }
    for index, value in enumerate(raw):
        context = f"attention_kernels[{index}]"
        if not isinstance(value, dict):
            raise RegistryError(f"{context} must be a table.")
        expected_fields = frozenset(
            {
                "implementation",
                "repository",
                "revision",
                "version",
                "expected_variant",
                "dtypes",
            }
        )
        _reject_unknown_fields(value, expected_fields, context)
        implementation = _require_str(value, "implementation", context)
        if implementation not in expected_variants:
            raise RegistryError(f"Unsupported attention kernel {implementation!r}.")
        if implementation in result:
            raise RegistryError(f"Duplicate attention kernel {implementation!r}.")
        repository = _require_str(value, "repository", context)
        if _REPOSITORY_ID_RE.fullmatch(repository) is None:
            raise RegistryError(f"Invalid attention-kernel repository {repository!r}.")
        revision = _require_str(value, "revision", context)
        _validate_revision(revision, f"{context}.revision")
        kernel_version = value.get("version")
        if (
            isinstance(kernel_version, bool)
            or not isinstance(kernel_version, int)
            or kernel_version <= 0
        ):
            raise RegistryError(f"{context}.version must be a positive integer.")
        expected_variant = _require_str(value, "expected_variant", context)
        if expected_variant != expected_variants[implementation]:
            raise RegistryError(
                f"{context}.expected_variant must be {expected_variants[implementation]!r}."
            )
        dtypes = _require_str_list(value, "dtypes", context)
        if not set(dtypes).issubset(_ALLOWED_DTYPES):
            raise RegistryError(f"{context}.dtypes contains unsupported dtypes.")
        result[implementation] = AttentionKernelSpec(
            implementation=implementation,
            repository=repository,
            revision=revision,
            version=kernel_version,
            expected_variant=expected_variant,
            dtypes=cast(tuple[DtypeName, ...], dtypes),
        )
    if set(result) != set(expected_variants):
        raise RegistryError("The manifest must pin both FlashAttention kernel versions.")
    return result


def _parse_upstreams(raw: object) -> dict[str, UpstreamSource]:
    if not isinstance(raw, list) or not raw:
        raise RegistryError("The manifest must contain at least one [[upstreams]] entry.")
    result: dict[str, UpstreamSource] = {}
    paths: set[str] = set()
    for index, value in enumerate(raw):
        context = f"upstreams[{index}]"
        if not isinstance(value, dict):
            raise RegistryError(f"{context} must be a table.")
        _reject_unknown_fields(value, _UPSTREAM_FIELDS, context)
        source_id = _require_str(value, "id", context)
        if _IDENTIFIER_RE.fullmatch(source_id) is None:
            raise RegistryError(f"Invalid upstream ID: {source_id!r}")
        if source_id in result:
            raise RegistryError(f"Duplicate upstream ID: {source_id!r}")
        revision = _require_str(value, "revision", context)
        _validate_revision(revision, f"{context}.revision")
        path = _require_str(value, "path", context)
        try:
            normalized_path = _portable_relative_path(path, f"{context}.path")
        except RegistryError as error:
            raise RegistryError(
                f"{context}.path must be a normalized directory directly under "
                "'vendor/upstream/'."
            ) from error
        if (
            normalized_path.parts[:2] != ("vendor", "upstream")
            or len(normalized_path.parts) != 3
        ):
            raise RegistryError(
                f"{context}.path must be a normalized directory directly under "
                "'vendor/upstream/'."
            )
        if path in paths:
            raise RegistryError(f"Duplicate upstream path: {path!r}")
        paths.add(path)
        url = _require_str(value, "url", context)
        if not url.startswith("https://github.com/") or not url.endswith(".git"):
            raise RegistryError(f"{context}.url must be an HTTPS GitHub clone URL.")
        license_files = _require_str_list(value, "license_files", context)
        license_digests = _require_digest_list(value, "license_digests", context)
        if tuple(item.path for item in license_digests) != license_files:
            raise RegistryError(
                f"{context}.license_digests must cover license_files in the same order."
            )
        distribution_files = _require_digest_list(value, "distribution_files", context)
        distribution_map = {item.path: item for item in distribution_files}
        for canonical in license_digests:
            distributed = distribution_map.get(canonical.path)
            if distributed is None or distributed.encoded != canonical.encoded:
                raise RegistryError(
                    f"{context}.distribution_files must include an exact copy of "
                    f"{canonical.path!r}."
                )
        if source_id == "e1":
            required_e1 = {
                "LICENSE",
                "ATTRIBUTION",
                "NOTICE",
                "Apache-2.0.txt",
                "BSD-3-Clause.txt",
                "MODIFICATIONS.md",
            }
            missing_e1 = sorted(required_e1.difference(distribution_map))
            if missing_e1:
                raise RegistryError(f"{context} is missing E1 legal files: {missing_e1}")
        result[source_id] = UpstreamSource(
            id=source_id,
            path=path,
            url=url,
            revision=revision,
            license_expression=_require_str(value, "license", context),
            license_files=license_files,
            license_digests=license_digests,
            distribution_files=distribution_files,
        )
    return result


def _parse_families(
    raw: object,
    upstreams: Mapping[str, UpstreamSource],
) -> dict[str, ModelFamily]:
    if not isinstance(raw, dict) or not raw:
        raise RegistryError("The manifest must contain [families.<id>] tables.")
    result: dict[str, ModelFamily] = {}
    for family_id, value in raw.items():
        context = f"families.{family_id}"
        if _IDENTIFIER_RE.fullmatch(family_id) is None or not isinstance(value, dict):
            raise RegistryError(f"Invalid family table: {family_id!r}")
        checkpoint_license = _require_str(value, "checkpoint_license", context)
        hub_license, hub_license_name, hub_license_link = _parse_hub_license(
            value,
            checkpoint_license=checkpoint_license,
            context=context,
        )
        _reject_unknown_fields(value, _FAMILY_FIELDS, context)
        source_ids = _require_str_list(value, "upstreams", context)
        unknown_sources = sorted(set(source_ids).difference(upstreams))
        if unknown_sources:
            raise RegistryError(f"{context} references unknown upstreams: {unknown_sources}")
        tokenizer_mode = _require_str(value, "tokenizer_mode", context)
        if tokenizer_mode not in _ALLOWED_TOKENIZER_MODES:
            raise RegistryError(f"Unsupported tokenizer mode in {context}: {tokenizer_mode!r}")
        public_input = _require_str(value, "public_input", context)
        attention = _require_str_list(value, "attention", context)
        if not set(attention).issubset(_ALLOWED_ATTENTION):
            raise RegistryError(f"Unsupported attention implementation in {context}.")
        dtypes = _require_str_list(value, "dtypes", context)
        if not set(dtypes).issubset(_ALLOWED_DTYPES):
            raise RegistryError(f"Unsupported dtype in {context}.")
        bf16_execution = cast(
            Bf16Execution,
            _require_enum(
                value,
                "bf16_execution",
                context,
                _ALLOWED_BF16_EXECUTIONS,
            ),
        )
        precisions = _require_str_list(value, "precisions", context)
        if not set(precisions).issubset(_ALLOWED_PRECISIONS):
            raise RegistryError(f"Unsupported precision policy in {context}.")
        experimental_precisions = _optional_str_list(
            value,
            "experimental_precisions",
            context,
        )
        unknown_experimental_precisions = sorted(
            set(experimental_precisions).difference(precisions)
        )
        if unknown_experimental_precisions:
            raise RegistryError(
                f"{context}.experimental_precisions must be a subset of precisions; "
                f"unknown values: {unknown_experimental_precisions}."
            )
        extra = cast(RuntimeExtra, _require_enum(value, "extra", context, _ALLOWED_EXTRAS))
        vram_tier = cast(
            VramTier,
            _require_enum(value, "vram_tier", context, _ALLOWED_VRAM_TIERS),
        )
        test_tiers_raw = _require_str_list(value, "test_tiers", context)
        unknown_test_tiers = sorted(set(test_tiers_raw).difference(_ALLOWED_TEST_TIERS))
        if unknown_test_tiers:
            raise RegistryError(
                f"{context}.test_tiers contains unsupported tiers: {unknown_test_tiers}."
            )
        test_tiers = cast(tuple[TestTier, ...], test_tiers_raw)
        reference_container = _parse_reference_container(value, context)
        reference_adapter = _parse_reference_adapter(value, context)
        documentation = _parse_documentation_path(value, context)
        runtime_paths = _require_str_list(value, "runtime_paths", context)
        if len(runtime_paths) != len(set(runtime_paths)):
            raise RegistryError(f"{context}.runtime_paths must not contain duplicates.")
        for runtime_path in runtime_paths:
            try:
                _portable_relative_path(runtime_path, f"{context}.runtime_paths entry")
            except RegistryError as error:
                raise RegistryError(
                    f"Unsafe runtime path in {context}: {runtime_path!r}"
                ) from error
            if runtime_path.startswith("vendor/"):
                raise RegistryError(f"Unsafe runtime path in {context}: {runtime_path!r}")
        requires_complete_weight_publication = value.get(
            "requires_complete_weight_publication",
            False,
        )
        if not isinstance(requires_complete_weight_publication, bool):
            raise RegistryError(
                f"{context}.requires_complete_weight_publication must be a boolean."
            )
        if "weights_publication_allowed" not in value:
            raise RegistryError(
                f"{context}.weights_publication_allowed must be declared explicitly."
            )
        weights_publication_allowed = value["weights_publication_allowed"]
        if not isinstance(weights_publication_allowed, bool):
            raise RegistryError(f"{context}.weights_publication_allowed must be a boolean.")
        raw_auto_map = value.get("auto_map")
        if not isinstance(raw_auto_map, dict) or not raw_auto_map:
            raise RegistryError(f"{context}.auto_map must be a non-empty table.")
        auto_map: list[tuple[str, str]] = []
        for auto_class, class_path in raw_auto_map.items():
            if auto_class not in _ALLOWED_AUTO_CLASSES or not isinstance(class_path, str):
                raise RegistryError(f"Invalid AutoClass mapping in {context}: {auto_class!r}")
            if not class_path.startswith("fastplms.") or class_path.count(".") < 2:
                raise RegistryError(f"Invalid Python class path in {context}: {class_path!r}")
            auto_map.append((auto_class, class_path))
        tokenizer_class = value.get("tokenizer_class")
        if tokenizer_class is not None:
            if tokenizer_mode != "tokenizer":
                raise RegistryError(
                    f"{context}.tokenizer_class requires tokenizer_mode='tokenizer'."
                )
            if (
                not isinstance(tokenizer_class, str)
                or not tokenizer_class.startswith("fastplms.")
                or tokenizer_class.count(".") < 2
            ):
                raise RegistryError(
                    f"Invalid tokenizer class path in {context}: {tokenizer_class!r}"
                )
        backbone_model = value.get("backbone_model")
        if backbone_model is not None and (
            not isinstance(backbone_model, str)
            or _IDENTIFIER_RE.fullmatch(backbone_model) is None
        ):
            raise RegistryError(
                f"{context}.backbone_model must be a valid manifest model ID."
            )
        state_transform = _require_str(value, "state_transform", context)
        conversion_provenance = _require_str(value, "conversion_provenance", context)
        required_sections = ("Input:", "Transformation:", "Output:", "Validation:", "Limitation:")
        missing_sections = [
            section for section in required_sections if section not in conversion_provenance
        ]
        if missing_sections or state_transform not in conversion_provenance:
            raise RegistryError(
                f"{context}.conversion_provenance must identify {state_transform!r} and "
                f"contain mechanism-first sections; missing {missing_sections}."
            )
        result[family_id] = ModelFamily(
            id=family_id,
            architecture=_require_str(value, "architecture", context),
            upstreams=source_ids,
            tokenizer_mode=tokenizer_mode,
            public_input=public_input,
            extra=extra,
            reference_container=reference_container,
            reference_adapter=reference_adapter,
            attention=attention,
            dtypes=cast(tuple[DtypeName, ...], dtypes),
            bf16_execution=bf16_execution,
            precisions=precisions,
            experimental_precisions=experimental_precisions,
            vram_tier=vram_tier,
            checkpoint_license=checkpoint_license,
            hub_license=hub_license,
            state_transform=state_transform,
            representative=_require_str(value, "representative", context),
            documentation=documentation,
            test_tiers=test_tiers,
            runtime_paths=runtime_paths,
            auto_map_items=tuple(auto_map),
            requires_complete_weight_publication=requires_complete_weight_publication,
            weights_publication_allowed=weights_publication_allowed,
            tokenizer_class=tokenizer_class,
            hub_license_name=hub_license_name,
            hub_license_link=hub_license_link,
            conversion_provenance=conversion_provenance,
            backbone_model=backbone_model,
        )
    return result


def _parse_runtime_assets(
    raw: object,
    families: Mapping[str, ModelFamily],
) -> dict[str, RuntimeAsset]:
    if not isinstance(raw, list) or not raw:
        raise RegistryError("The manifest must contain at least one [[runtime_assets]] entry.")
    result: dict[str, RuntimeAsset] = {}
    identities: set[tuple[str, str, str]] = set()
    for index, value in enumerate(raw):
        context = f"runtime_assets[{index}]"
        if not isinstance(value, dict):
            raise RegistryError(f"{context} must be a table.")
        _reject_unknown_fields(value, _RUNTIME_ASSET_FIELDS, context)
        asset_id = _require_str(value, "id", context)
        if _IDENTIFIER_RE.fullmatch(asset_id) is None:
            raise RegistryError(f"Invalid runtime asset ID: {asset_id!r}")
        if asset_id in result:
            raise RegistryError(f"Duplicate runtime asset ID: {asset_id!r}")
        repository = _require_str(value, "repository", context)
        if _REPOSITORY_ID_RE.fullmatch(repository) is None:
            raise RegistryError(f"{context}.repository must be a Hugging Face repository ID.")
        revision = _require_str(value, "revision", context)
        _validate_revision(revision, f"{context}.revision")
        path = _require_str(value, "path", context)
        try:
            normalized_path = _portable_relative_path(path, "Runtime asset path")
        except RegistryError as error:
            raise RegistryError(f"Runtime asset path is not portable: {path!r}") from error
        sha256 = _require_str(value, "sha256", context)
        if len(sha256) != 64 or _HEX_RE.fullmatch(sha256) is None:
            raise RegistryError(f"Invalid runtime asset SHA-256 for {path!r}.")
        size = value.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RegistryError(f"{context}.size must be a positive byte count.")
        consumer_family = _require_str(value, "consumer_family", context)
        if consumer_family not in families:
            raise RegistryError(
                f"{context}.consumer_family references unknown family {consumer_family!r}."
            )
        trust_kind = cast(
            RuntimeAssetTrustKind,
            _require_enum(
                value,
                "trust_kind",
                context,
                _ALLOWED_RUNTIME_ASSET_TRUST_KINDS,
            ),
        )
        license_expression = _require_str(value, "license", context)
        offline_behavior = _require_str(value, "offline_behavior", context)
        if offline_behavior not in _ALLOWED_RUNTIME_ASSET_OFFLINE_BEHAVIORS:
            raise RegistryError(
                f"{context}.offline_behavior is unsupported: {offline_behavior!r}."
            )
        if trust_kind == "hash_pinned_pickle" and normalized_path.suffix != ".pkl":
            raise RegistryError(
                f"{context}.path must end in '.pkl' for trust_kind='hash_pinned_pickle'."
            )
        identity = (repository, revision, path)
        if identity in identities:
            raise RegistryError(f"Duplicate runtime asset identity: {identity!r}")
        identities.add(identity)
        result[asset_id] = RuntimeAsset(
            id=asset_id,
            repository=repository,
            revision=revision,
            path=path,
            sha256=sha256,
            size=size,
            consumer_family=consumer_family,
            trust_kind=trust_kind,
            license_expression=license_expression,
            offline_behavior=offline_behavior,
        )
    return result


def _parse_models(
    raw: object,
    families: Mapping[str, ModelFamily],
) -> dict[str, ModelSpec]:
    if not isinstance(raw, list) or not raw:
        raise RegistryError("The manifest must contain at least one [[models]] entry.")
    result: dict[str, ModelSpec] = {}
    fast_repositories: set[str] = set()
    for index, value in enumerate(raw):
        context = f"models[{index}]"
        if not isinstance(value, dict):
            raise RegistryError(f"{context} must be a table.")
        _reject_unknown_fields(value, _MODEL_FIELDS, context)
        model_id = _require_str(value, "id", context)
        if _IDENTIFIER_RE.fullmatch(model_id) is None:
            raise RegistryError(f"Invalid model ID: {model_id!r}")
        if model_id in result:
            raise RegistryError(f"Duplicate model ID: {model_id!r}")
        family_id = _require_str(value, "family", context)
        if family_id not in families:
            raise RegistryError(f"{context} references unknown family {family_id!r}.")
        fast = _parse_checkpoint(value, "fast", context)
        official = _parse_checkpoint(value, "official", context)
        if fast.repo_id in fast_repositories:
            raise RegistryError(f"Duplicate FastPLMs repository ID: {fast.repo_id!r}")
        fast_repositories.add(fast.repo_id)
        family = families[family_id]
        oracle_assets = _parse_oracle_assets(value, context)
        official_golden = _parse_official_golden(value, model_id, context)
        size_category = _require_str(value, "size_category", context)
        if size_category not in _ALLOWED_SIZE_CATEGORIES:
            raise RegistryError(f"Unsupported size category in {context}: {size_category!r}")
        generation_contract = cast(
            GenerationContract,
            _require_enum(
                value,
                "generation_contract",
                context,
                _ALLOWED_GENERATION_CONTRACTS,
            ),
        )
        if family.tokenizer_mode == "structure" and size_category != "structure":
            raise RegistryError(
                f"Structure checkpoint {model_id!r} must use size_category='structure'."
            )
        artifact_source = value.get("artifact_source", "fast")
        if artifact_source not in {"fast", "official"}:
            raise RegistryError(f"{context}.artifact_source must be 'fast' or 'official'.")
        canonical_state_sha256 = value.get("canonical_state_sha256")
        if artifact_source == "official":
            if (
                not isinstance(canonical_state_sha256, str)
                or len(canonical_state_sha256) != 64
                or _HEX_RE.fullmatch(canonical_state_sha256) is None
            ):
                raise RegistryError(
                    f"{context}.canonical_state_sha256 must be a SHA-256 commitment "
                    "for an official-source artifact."
                )
        elif canonical_state_sha256 is not None:
            raise RegistryError(
                f"{context}.canonical_state_sha256 is restricted to official-source artifacts."
            )
        if family.tokenizer_mode == "tokenizer" and not any(
            "tokenizer" in item.path or "vocab" in item.path for item in fast.files
        ):
            raise RegistryError(f"{context} does not pin a tokenizer asset.")
        tokenizer_source_id = value.get("tokenizer_source")
        if tokenizer_source_id is not None and (
            family.tokenizer_mode != "tokenizer"
            or not isinstance(tokenizer_source_id, str)
            or _IDENTIFIER_RE.fullmatch(tokenizer_source_id) is None
        ):
            raise RegistryError(f"{context}.tokenizer_source is invalid.")
        notes = value.get("notes", "")
        if not isinstance(notes, str):
            raise RegistryError(f"{context}.notes must be a string.")
        msa_conditioning = value.get("msa_conditioning")
        if family_id == "esmfold2":
            if not isinstance(msa_conditioning, bool):
                raise RegistryError(
                    f"{context}.msa_conditioning must be an explicit boolean for "
                    "ESMFold2 checkpoints."
                )
        elif "msa_conditioning" in value:
            raise RegistryError(
                f"{context}.msa_conditioning is only valid for ESMFold2 checkpoints."
            )
        raw_auto_map = value.get("auto_map")
        auto_map: list[tuple[str, str]] = []
        if raw_auto_map is not None:
            if not isinstance(raw_auto_map, dict) or not raw_auto_map:
                raise RegistryError(f"{context}.auto_map must be a non-empty table.")
            for auto_class, class_path in raw_auto_map.items():
                if auto_class not in _ALLOWED_AUTO_CLASSES or not isinstance(class_path, str):
                    raise RegistryError(f"Invalid AutoClass mapping in {context}: {auto_class!r}")
                if not class_path.startswith("fastplms.") or class_path.count(".") < 2:
                    raise RegistryError(f"Invalid Python class path in {context}: {class_path!r}")
                auto_map.append((auto_class, class_path))
        result[model_id] = ModelSpec(
            id=model_id,
            family=family,
            fast=fast,
            official=official,
            size_category=size_category,
            generation_contract=generation_contract,
            oracle_assets=oracle_assets,
            official_golden=official_golden,
            artifact_source=artifact_source,
            canonical_state_sha256=canonical_state_sha256,
            tokenizer_source_id=tokenizer_source_id,
            auto_map_items=tuple(auto_map),
            notes=notes,
            msa_conditioning=msa_conditioning,
        )
    return result


def _validate_registry(
    upstreams: Mapping[str, UpstreamSource],
    attention_kernels: Mapping[str, AttentionKernelSpec],
    families: Mapping[str, ModelFamily],
    models: Mapping[str, ModelSpec],
) -> None:
    for spec in models.values():
        if spec.tokenizer_source_id is None:
            continue
        source = models.get(spec.tokenizer_source_id)
        if source is None:
            raise RegistryError(
                f"Model {spec.id!r} references unknown tokenizer source "
                f"{spec.tokenizer_source_id!r}."
            )
        if not any(
            PurePosixPath(item.path).name
            in {
                "added_tokens.json",
                "merges.txt",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "vocab.txt",
            }
            for item in source.official.files
        ):
            raise RegistryError(
                f"Tokenizer source {source.id!r} has no official tokenizer assets."
            )
    expected_esmfold2 = {
        "esmfold2": ("Synthyra/ESMFold2", "biohub/ESMFold2"),
        "esmfold2_fast": ("Synthyra/ESMFold2-Fast", "biohub/ESMFold2-Fast"),
        "esmfold2_experimental_cutoff2025": (
            "Synthyra/ESMFold2-Experimental-Cutoff2025",
            "biohub/ESMFold2-Experimental-Cutoff2025",
        ),
        "esmfold2_experimental_fast_cutoff2025": (
            "Synthyra/ESMFold2-Experimental-Fast-Cutoff2025",
            "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
        ),
    }
    actual_esmfold2 = {
        model.id: (model.fast.repo_id, model.official.repo_id)
        for model in models.values()
        if model.family.id == "esmfold2"
    }
    if actual_esmfold2 != expected_esmfold2:
        raise RegistryError(
            "ESMFold2 support must contain exactly the four approved model IDs and "
            "official/Synthyra repositories."
        )

    golden_paths: list[str] = []
    for model in models.values():
        if model.official_golden is not None:
            golden_paths.extend(
                (
                    model.official_golden.metadata.path,
                    model.official_golden.tensors.path,
                )
            )
    if len(golden_paths) != len(set(golden_paths)):
        raise RegistryError("Official golden paths must be unique across model declarations.")
    unused_upstreams = sorted(
        set(upstreams).difference(
            source for family in families.values() for source in family.upstreams
        )
    )
    if unused_upstreams:
        raise RegistryError(
            f"Upstream sources are not connected to a model family: {unused_upstreams}"
        )
    advertised_flash = {
        implementation
        for family in families.values()
        for implementation in family.attention
        if implementation.startswith("flash_attention_")
    }
    missing_kernels = sorted(advertised_flash.difference(attention_kernels))
    if missing_kernels:
        raise RegistryError(
            f"Advertised FlashAttention backends lack kernel specs: {missing_kernels}."
        )
    for family in families.values():
        for implementation in family.attention:
            kernel = attention_kernels.get(implementation)
            if kernel is not None and not set(family.dtypes).intersection(kernel.dtypes):
                raise RegistryError(
                    f"Family {family.id!r} and attention kernel {implementation!r} "
                    "have no supported dtype in common."
                )
        family_models = [model for model in models.values() if model.family.id == family.id]
        if not family_models:
            raise RegistryError(f"Family {family.id!r} has no checkpoints.")
        representative = models.get(family.representative)
        if representative is None or representative.family.id != family.id:
            raise RegistryError(
                f"Family {family.id!r} has invalid representative {family.representative!r}."
            )
        if family.backbone_model is not None and family.backbone_model not in models:
            raise RegistryError(
                f"Family {family.id!r} references unknown backbone model "
                f"{family.backbone_model!r}."
            )


def _load_manifest_bytes(raw_bytes: bytes) -> ModelRegistry:
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"Unable to parse model manifest: {error}") from error
    _reject_unknown_fields(data, _ROOT_FIELDS, "manifest")
    if data.get("schema_version") != 1:
        raise RegistryError("Unsupported model manifest schema_version; expected 1.")
    legal_files = _require_digest_list(data, "legal_files", "manifest")
    required_legal_paths = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    if {item.path for item in legal_files} != required_legal_paths:
        raise RegistryError("manifest.legal_files must contain LICENSE and THIRD_PARTY_NOTICES.md.")
    attention_kernels = _parse_attention_kernels(data.get("attention_kernels"))
    upstreams = _parse_upstreams(data.get("upstreams"))
    families = _parse_families(data.get("families"), upstreams)
    runtime_assets = _parse_runtime_assets(data.get("runtime_assets"), families)
    models = _parse_models(data.get("models"), families)
    _validate_registry(upstreams, attention_kernels, families, models)
    return ModelRegistry(
        schema_version=1,
        upstreams=upstreams,
        attention_kernels=attention_kernels,
        families=families,
        models=models,
        runtime_assets=runtime_assets,
        legal_files=legal_files,
    )


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    """Load and validate a model manifest without importing model code."""

    if path is None:
        manifest = resources.files("fastplms").joinpath("models.toml")
        return _load_manifest_bytes(manifest.read_bytes())
    return _load_manifest_bytes(Path(path).read_bytes())


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the validated package registry, cached after its first read."""

    return load_model_registry()


def get_model_spec(model_id: str) -> ModelSpec:
    """Return one model specification by its stable manifest ID."""

    try:
        return get_model_registry()[model_id]
    except KeyError as error:
        supported = ", ".join(get_model_registry())
        raise KeyError(
            f"Unknown FastPLMs model ID {model_id!r}. Supported IDs: {supported}"
        ) from error


__all__ = [
    "HUB_LICENSE_IDENTIFIERS",
    "CheckpointSource",
    "FileDigest",
    "GenerationContract",
    "ModelFamily",
    "ModelRegistry",
    "ModelSpec",
    "OracleAsset",
    "RegistryError",
    "RuntimeAsset",
    "RuntimeAssetTrustKind",
    "RuntimeExtra",
    "TestTier",
    "UpstreamSource",
    "VramTier",
    "get_model_registry",
    "get_model_spec",
    "load_model_registry",
]
