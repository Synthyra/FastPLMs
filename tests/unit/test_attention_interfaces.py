"""Transformers registry and Flex cache contracts."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AttentionInterface

_TRANSFORMERS_FLASH_HANDLERS = {
    name: AttentionInterface()[name] for name in ("flash_attention_2", "flash_attention_3")
}

import fastplms.attention.interfaces as attention_interfaces  # noqa: E402
import fastplms.models.ankh.modeling_ankh as ankh_module  # noqa: E402
from fastplms.attention import (  # noqa: E402
    FASTPLMS_ATTENTION_FUNCTIONS,
    FASTPLMS_ATTENTION_MASKS,
    FastPLMsAttentionMixin,
    _core,
    _kernel_lock,
    validate_transformers_attention_interfaces,
)
from fastplms.embeddings.runner import _attention_kernel_metadata  # noqa: E402
from fastplms.models.ankh.modeling_ankh import (  # noqa: E402
    AnkhSelfAttention,
    FastAnkhConfig,
    FastAnkhModel,
)
from fastplms.models.dplm.modeling_dplm import (  # noqa: E402
    DPLMConfig,
    DPLMModel,
)
from fastplms.models.dplm2.modeling_dplm2 import (  # noqa: E402
    DPLM2Config,
    DPLM2Model,
)
from fastplms.registry import get_model_registry  # noqa: E402

FUNCTION_BACKENDS = (
    "sdpa",
    "flex_attention",
    "flash_attention_2",
    "flash_attention_3",
)
MASK_BACKENDS = ("eager", *FUNCTION_BACKENDS)


class _RejectingTransformersBase:
    def _check_and_adjust_attn_implementation(self, *args, **kwargs) -> str:
        raise AssertionError("Transformers' source-Flash resolver must not run.")


class _SupportedFlashMixin(FastPLMsAttentionMixin, _RejectingTransformersBase):
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
        "flash_attention_2",
        "flash_attention_3",
    )


@pytest.mark.parametrize(
    (
        "implementation",
        "repository",
        "revision",
        "kernel",
        "variant",
    ),
    (
        (
            "flash_attention_2",
            "kernels-community/flash-attn2",
            "db6b51744f0cd7061386442c09df890fc6d9f47e",
            SimpleNamespace(fwd=object(), varlen_fwd=object()),
            "flash_attn2",
        ),
        (
            "flash_attention_3",
            "kernels-community/flash-attn3",
            "43f0bd269777115d94ff826e0d113ce9c1c9087b",
            SimpleNamespace(flash_attn_func=object(), flash_attn_varlen_func=object()),
            "flash_attn3",
        ),
    ),
)
def test_flash_backend_loads_only_its_hugging_face_kernel(
    monkeypatch,
    implementation: str,
    repository: str,
    revision: str,
    kernel: object,
    variant: str,
) -> None:
    requested: list[tuple[str, str]] = []

    def locked_kernel(repo_id: str, revision: str) -> object:
        requested.append((repo_id, revision))
        return kernel

    monkeypatch.setattr(_core, "load_locked_kernel", locked_kernel)
    _core._FLASH_KERNELS.clear()

    assert _core._ensure_flash_kernels_loaded(implementation) == (kernel, variant)
    assert _core._ensure_flash_kernels_loaded(implementation) == (kernel, variant)
    assert requested == [(repository, revision)]


def test_flash_kernel_variant_mismatch_fails_closed(monkeypatch) -> None:
    requested: list[tuple[str, str]] = []
    flash_attention_3_kernel = SimpleNamespace(
        flash_attn_func=object(),
        flash_attn_varlen_func=object(),
    )

    def locked_kernel(repo_id: str, revision: str) -> object:
        requested.append((repo_id, revision))
        return flash_attention_3_kernel

    monkeypatch.setattr(_core, "load_locked_kernel", locked_kernel)
    _core._FLASH_KERNELS.clear()

    with pytest.raises(
        RuntimeError,
        match=(
            "kernels-community/flash-attn2@"
            "db6b51744f0cd7061386442c09df890fc6d9f47e exposed 'flash_attn3'; "
            "expected 'flash_attn2'"
        ),
    ):
        _core._ensure_flash_kernels_loaded("flash_attention_2")
    assert requested == [
        (
            "kernels-community/flash-attn2",
            "db6b51744f0cd7061386442c09df890fc6d9f47e",
        )
    ]


def test_flash_kernel_load_error_retains_pinned_identity_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("incompatible precompiled kernel")

    def locked_kernel(repo_id: str, revision: str) -> object:
        assert repo_id == "kernels-community/flash-attn3"
        assert revision == "43f0bd269777115d94ff826e0d113ce9c1c9087b"
        raise failure

    monkeypatch.setattr(_core, "load_locked_kernel", locked_kernel)
    _core._FLASH_KERNELS.clear()

    with pytest.raises(
        RuntimeError,
        match=(
            "Unable to load the manifest-pinned kernel "
            "kernels-community/flash-attn3@"
            "43f0bd269777115d94ff826e0d113ce9c1c9087b"
        ),
    ) as captured:
        _core._ensure_flash_kernels_loaded("flash_attention_3")
    assert captured.value.__cause__ is failure


def test_locked_kernel_is_hash_validated_before_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    lock_path = tmp_path / "kernels.lock"
    lock_path.write_text(
        json.dumps(
            [
                {
                    "repo_id": "kernels-community/flash-attn2",
                    "sha": revision,
                    "variants": {
                        "torch213-cxx11-cu130-x86_64-linux": {
                            "hash": f"sha256-{'a' * 64}",
                            "hash_type": "git_lfs_concat",
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    kernel = object()
    validated_path = tmp_path / "validated-variant"

    class KernelLock:
        @classmethod
        def from_json(cls, entry: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(sha=entry["sha"], variants=entry["variants"])

    def install_kernel(
        repository: str,
        *,
        revision: str,
        variant_locks: dict[str, object],
    ) -> Path:
        assert repository == "kernels-community/flash-attn2"
        assert revision == "d" * 40
        assert set(variant_locks) == {"torch213-cxx11-cu130-x86_64-linux"}
        events.append("validate")
        return validated_path

    def get_local_kernel(path: Path) -> object:
        assert path == validated_path
        events.append("import")
        return kernel

    monkeypatch.setattr(_kernel_lock, "_kernel_lock_path", lambda: lock_path)
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        SimpleNamespace(
            get_local_kernel=get_local_kernel,
            install_kernel=install_kernel,
        ),
    )
    monkeypatch.setitem(sys.modules, "kernels.lockfile", SimpleNamespace(KernelLock=KernelLock))

    assert (
        _kernel_lock.load_locked_kernel("kernels-community/flash-attn2", revision)
        is kernel
    )
    assert events == ["validate", "import"]


def test_locked_kernel_offline_resolves_sparse_snapshot_without_hub_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    variant_path = snapshot / "build" / "torch213-cxx11-cu130-x86_64-linux"
    variant_path.mkdir(parents=True)
    variant = SimpleNamespace(variant_str=variant_path.name)
    expected_hash = f"sha256-{'a' * 64}"
    events: list[str] = []
    kernel = object()

    def validate_kernel(*, repo_path: Path, variant: str, hash: str) -> None:
        assert repo_path == snapshot
        assert variant == variant_path.name
        assert hash == expected_hash
        events.append("validate")

    def get_local_kernel(path: Path) -> object:
        assert path == variant_path
        events.append("import")
        return kernel

    monkeypatch.setattr(_kernel_lock, "_offline_snapshot_path", lambda *_: snapshot)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_local_kernel=get_local_kernel))
    monkeypatch.setitem(
        sys.modules,
        "kernels.utils",
        SimpleNamespace(validate_kernel=validate_kernel),
    )
    monkeypatch.setitem(
        sys.modules,
        "kernels.variants",
        SimpleNamespace(
            get_variants_local=lambda path: [variant],
            resolve_variants=lambda variants: (variants, []),
        ),
    )

    assert (
        _kernel_lock._load_offline_locked_kernel(
            "kernels-community/flash-attn2",
            "d" * 40,
            {variant_path.name: SimpleNamespace(hash=expected_hash)},
        )
        is kernel
    )
    assert events == ["validate", "import"]


def test_locked_kernel_offline_rejects_unlocked_cached_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "build" / "unexpected-variant").mkdir(parents=True)
    monkeypatch.setattr(_kernel_lock, "_offline_snapshot_path", lambda *_: snapshot)

    with pytest.raises(RuntimeError, match="contains unlocked variants"):
        _kernel_lock._load_offline_locked_kernel(
            "kernels-community/flash-attn2",
            "d" * 40,
            {"expected-variant": SimpleNamespace(hash=f"sha256-{'a' * 64}")},
        )


def test_transformers_flash_hook_defers_binary_loading_until_execution(monkeypatch) -> None:
    dependency_checks: list[None] = []
    monkeypatch.setattr(
        attention_interfaces,
        "require_kernels_package",
        lambda: dependency_checks.append(None),
    )
    model = object.__new__(_SupportedFlashMixin)

    for implementation in ("flash_attention_2", "flash_attention_3"):
        assert (
            model._check_and_adjust_attn_implementation(
                implementation,
                is_init_check=True,
            )
            == implementation
        )

    assert dependency_checks == [None, None]


@pytest.mark.parametrize(
    "implementation",
    (
        "kernels-community/flash-attn2",
        "another-owner/custom-kernel",
        "paged|flash_attention_2",
        "flash_attention_4",
    ),
)
def test_transformers_flash_hook_rejects_external_and_unadvertised_kernels(
    implementation: str,
) -> None:
    model = object.__new__(_SupportedFlashMixin)
    with pytest.raises(ValueError, match="does not support"):
        model._check_and_adjust_attn_implementation(implementation)

    with pytest.raises(ValueError, match="does not load external"):
        model._check_and_adjust_attn_implementation(
            "flash_attention_2",
            allow_all_kernels=True,
        )


def test_transformers_flash_hook_rejects_an_unadvertised_family() -> None:
    model = object.__new__(FastPLMsAttentionMixin)
    with pytest.raises(ValueError, match="does not support"):
        model._check_and_adjust_attn_implementation("flash_attention_2")


def test_public_attention_setter_matches_transformers_513_kernel_policy(monkeypatch) -> None:
    signature = inspect.signature(FastPLMsAttentionMixin.set_attn_implementation)
    assert tuple(signature.parameters) == (
        "self",
        "attn_implementation",
        "allow_all_kernels",
    )
    assert signature.parameters["allow_all_kernels"].default is False

    model = object.__new__(_SupportedFlashMixin)
    with pytest.raises(ValueError, match="does not load external"):
        model.set_attn_implementation(
            "flash_attention_2",
            allow_all_kernels=True,
        )

    def unavailable_kernel_runtime() -> None:
        raise RuntimeError("precompiled kernel runtime is unavailable")

    monkeypatch.setattr(
        attention_interfaces,
        "require_kernels_package",
        unavailable_kernel_runtime,
    )
    with pytest.raises(RuntimeError, match="kernel runtime is unavailable"):
        model.set_attn_implementation("flash_attention_2")


def test_flash_kernel_identity_is_typed_and_manifest_owned() -> None:
    registry = get_model_registry()
    assert {
        implementation: (
            spec.repository,
            spec.revision,
            spec.version,
            spec.expected_variant,
            spec.dtypes,
        )
        for implementation, spec in registry.attention_kernels.items()
    } == {
        "flash_attention_2": (
            "kernels-community/flash-attn2",
            "db6b51744f0cd7061386442c09df890fc6d9f47e",
            2,
            "flash_attn2",
            ("bfloat16",),
        ),
        "flash_attention_3": (
            "kernels-community/flash-attn3",
            "43f0bd269777115d94ff826e0d113ce9c1c9087b",
            1,
            "flash_attn3",
            ("bfloat16",),
        ),
    }
    source = Path(_core.__file__).read_text(encoding="utf-8")
    assert "kernels-community/flash-attn2" not in source
    assert "kernels-community/flash-attn3" not in source
    for family in registry.families.values():
        if any(name.startswith("flash_attention_") for name in family.attention):
            assert "registry.py" in family.runtime_paths

    assert _attention_kernel_metadata("flash_attention_3") == {
        "repository": "kernels-community/flash-attn3",
        "revision": "43f0bd269777115d94ff826e0d113ce9c1c9087b",
        "version": 1,
        "expected_variant": "flash_attn3",
        "dtypes": ["bfloat16"],
    }


def test_kernels_flash_rejects_fp32_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(_implementation: str) -> None:
        raise AssertionError("unsupported dtypes must fail before kernel loading")

    monkeypatch.setattr(_core, "_ensure_flash_kernels_loaded", unexpected_load)
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_device",
        lambda *_args: torch.device("cuda"),
    )
    X = torch.zeros(1, 4, 2, 8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"bfloat16.*received float32"):
        _core.kernels_flash_attention_func(X, X, X, implementation="flash_attention_3")


def test_kernels_flash_rejects_mixed_qkv_dtypes_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(_implementation: str) -> None:
        raise AssertionError("mismatched dtypes must fail before kernel loading")

    monkeypatch.setattr(_core, "_ensure_flash_kernels_loaded", unexpected_load)
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_device",
        lambda *_args: torch.device("cuda"),
    )
    Q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    K = torch.zeros(1, 4, 2, 8, dtype=torch.float32)
    V = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Q, K, and V to share one dtype"):
        _core.kernels_flash_attention_func(Q, K, V, implementation="flash_attention_3")


def test_kernels_flash_rejects_cpu_bf16_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(_implementation: str) -> None:
        raise AssertionError("CPU tensors must fail before kernel loading")

    monkeypatch.setattr(_core, "_ensure_flash_kernels_loaded", unexpected_load)
    X = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"requires CUDA Q, K, and V.*cpu"):
        _core.kernels_flash_attention_func(X, X, X, implementation="flash_attention_2")


def test_kernels_flash_rejects_mixed_devices_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(_implementation: str) -> None:
        raise AssertionError("mixed devices must fail before kernel loading")

    monkeypatch.setattr(_core, "_ensure_flash_kernels_loaded", unexpected_load)
    Q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    K = torch.empty(1, 4, 2, 8, dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match=r"on one device.*cpu, meta, cpu"):
        _core.kernels_flash_attention_func(Q, K, Q, implementation="flash_attention_3")


def test_causal_masked_flash_uses_varlen_and_zeroes_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_device",
        lambda *_args: torch.device("cpu"),
    )
    monkeypatch.setattr(
        _core,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (object(), "flash_attn3"),
    )
    observed: dict[str, object] = {}

    def varlen_forward(**kwargs):
        observed.update(kwargs)
        return kwargs["query_states"] + 1

    monkeypatch.setattr(_core, "_kernels_flash_varlen_forward", varlen_forward)
    monkeypatch.setattr(
        _core,
        "_kernels_flash_forward",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a masked causal call must not use dense FlashAttention")
        ),
    )
    X = torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

    output = _core.kernels_flash_attention_func(
        X,
        X,
        X,
        attention_mask_2d=mask,
        causal=True,
        implementation="flash_attention_3",
    )

    assert observed["causal"] is True
    assert observed["query_states"].shape == (5, 2, 8)
    assert torch.equal(output[mask.bool()], torch.ones(5, 2, 8, dtype=torch.bfloat16))
    assert torch.count_nonzero(output[~mask.bool()]) == 0


def test_masked_flash_validates_padding_mask_shape_before_kernel_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_device",
        lambda *_args: torch.device("cpu"),
    )
    monkeypatch.setattr(
        _core,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (_ for _ in ()).throw(
            AssertionError("invalid masks must fail before kernel loading")
        ),
    )
    X = torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"expected \(2, 4\), received \(2, 3\)"):
        _core.kernels_flash_attention_func(
            X,
            X,
            X,
            attention_mask_2d=torch.ones(2, 3, dtype=torch.bool),
            causal=True,
            implementation="flash_attention_2",
        )


def test_flash_extra_has_no_source_build_path() -> None:
    root = Path(__file__).resolve().parents[2]
    dependency_files = (
        root / "pyproject.toml",
        root / "uv.lock",
        root / "docker" / "Dockerfile",
    )
    for path in dependency_files:
        text = path.read_text(encoding="utf-8")
        assert 'name = "flash-attn"' not in text
        assert "flash-attn>=" not in text
        assert "no-build-isolation-package" not in text
        assert "extra-build-dependencies" not in text

    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert 'import_module("flash_attn' not in text
        assert "from flash_attn" not in text
        assert "import flash_attn" not in text


def test_transformers_513_exposes_every_advertised_handler() -> None:
    assert version("transformers") == "5.13.0"
    validate_transformers_attention_interfaces()
    for name in FUNCTION_BACKENDS:
        assert name in FASTPLMS_ATTENTION_FUNCTIONS
        assert callable(FASTPLMS_ATTENTION_FUNCTIONS[name])
    for name in MASK_BACKENDS:
        assert name in FASTPLMS_ATTENTION_MASKS
        assert callable(FASTPLMS_ATTENTION_MASKS[name])
    transformers_registry = AttentionInterface()
    for name in ("flash_attention_2", "flash_attention_3"):
        assert transformers_registry[name] is _TRANSFORMERS_FLASH_HANDLERS[name]
        assert FASTPLMS_ATTENTION_FUNCTIONS[name] is not transformers_registry[name]


@pytest.mark.parametrize(
    ("family_id", "relative_path", "class_name", "flash_backends"),
    (
        (
            "esm2",
            "src/fastplms/models/esm2/modeling_fastesm.py",
            "FastEsmPreTrainedModel",
            ("flash_attention_2", "flash_attention_3"),
        ),
        (
            "esm_plusplus",
            "src/fastplms/models/esm_plusplus/modeling_esm_plusplus.py",
            "PreTrainedESMplusplusModel",
            ("flash_attention_2", "flash_attention_3"),
        ),
        (
            "dplm",
            "src/fastplms/models/dplm/modeling_dplm.py",
            "DPLMPreTrainedModel",
            ("flash_attention_3",),
        ),
        (
            "dplm2",
            "src/fastplms/models/dplm2/modeling_dplm2.py",
            "DPLM2PreTrainedModel",
            (),
        ),
        ("e1", "src/fastplms/models/e1/modeling_e1.py", "E1PreTrainedModel", ()),
        (
            "ankh",
            "src/fastplms/models/ankh/modeling_ankh.py",
            "AnkhPreTrainedModel",
            (),
        ),
        (
            "esm3",
            "src/fastplms/models/esm3/modeling_esm3.py",
            "FastESM3PreTrainedModel",
            (),
        ),
        (
            "esmfold",
            "src/fastplms/models/esmfold/modeling_fast_esmfold.py",
            "FastEsmForProteinFolding",
            (),
        ),
        (
            "esmfold2",
            "src/fastplms/models/esmfold2/attention.py",
            "ESMFold2AttentionMixin",
            (),
        ),
    ),
)
def test_model_flash_flags_match_the_manifest(
    family_id: str,
    relative_path: str,
    class_name: str,
    flash_backends: tuple[str, ...],
) -> None:
    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    module_constants = {
        target.id: ast.literal_eval(statement.value)
        for statement in module.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and isinstance(statement.value, (ast.Constant, ast.Tuple, ast.List))
    }
    class_node = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )

    def assignment_value(value: ast.expr) -> object:
        if isinstance(value, ast.Name):
            return module_constants[value.id]
        return ast.literal_eval(value)

    assignments = {
        target.id: assignment_value(statement.value)
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "_supports_flash_attn",
            "_supports_flash_attn_2",
            "_supports_flash_attn_3",
            "_fastplms_attention_implementations",
        }
    }
    assert assignments["_supports_flash_attn_2"] is ("flash_attention_2" in flash_backends)
    assert assignments["_supports_flash_attn_3"] is ("flash_attention_3" in flash_backends)
    assert assignments.get("_supports_flash_attn", False) is bool(flash_backends)
    expected = get_model_registry().families[family_id].attention
    assert assignments["_fastplms_attention_implementations"] == expected
    assert tuple(name for name in expected if name.startswith("flash_")) == flash_backends


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size", "unsupported"),
    (
        (
            DPLMModel,
            DPLMConfig,
            33,
            ("flash_attention_2",),
        ),
        (
            DPLM2Model,
            DPLM2Config,
            64,
            ("eager", "flex_attention", "flash_attention_2", "flash_attention_3"),
        ),
    ),
)
def test_dplm_families_reject_unadvertised_attention(
    model_class: type,
    config_class: type,
    vocab_size: int,
    unsupported: tuple[str, ...],
) -> None:
    config = config_class(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        attn_backend="sdpa",
    )
    model = model_class(config)
    expected_flex_support = "flex_attention" not in unsupported
    assert model._supports_flex_attn is expected_flex_support

    for implementation in unsupported:
        with pytest.raises(ValueError, match="does not support"):
            model.set_attn_implementation(implementation)
        with pytest.raises(ValueError, match="does not support"):
            model.attn_backend = implementation


@pytest.mark.parametrize("should_raise", (False, True))
def test_ankh_sdpa_restores_reduced_math_policy(
    monkeypatch: pytest.MonkeyPatch,
    should_raise: bool,
) -> None:
    original_policy = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    attention = AnkhSelfAttention(
        FastAnkhConfig(
            vocab_size=16,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_heads=2,
            num_layers=1,
            attn_backend="sdpa",
        )
    )
    query = torch.randn(1, 2, 3, 4)

    def fake_sdpa(*args, **_kwargs):
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        if should_raise:
            raise RuntimeError("forced SDPA failure")
        return args[0]

    monkeypatch.setattr(ankh_module.F, "scaled_dot_product_attention", fake_sdpa)
    try:
        if should_raise:
            with pytest.raises(RuntimeError, match="forced SDPA failure"):
                attention._sdpa_attn(query, query, query, None)
        else:
            output = attention._sdpa_attn(query, query, query, None)
            assert output.shape == (1, 3, 8)
        assert not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(original_policy)


def test_ankh_rejects_unadvertised_flex_attention() -> None:
    model = FastAnkhModel(
        FastAnkhConfig(
            vocab_size=16,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_heads=2,
            num_layers=1,
            attn_backend="sdpa",
        )
    )
    with pytest.raises(ValueError, match="does not support 'flex_attention'"):
        model.set_attn_implementation("flex_attention")


def test_attention_mixin_leaves_unspecified_backend_to_transformers() -> None:
    observed: list[str | None] = []

    class Base:
        def __init__(self, config) -> None:
            observed.append(config._attn_implementation)
            config._attn_implementation_internal = "sdpa"

    class Model(FastPLMsAttentionMixin, Base):
        _fastplms_attention_implementations = ("eager", "sdpa")

    config = SimpleNamespace(
        _attn_implementation=None,
        attn_backend=None,
    )
    Model(config)

    assert observed == [None]
    assert config._attn_implementation_internal == "sdpa"
    assert config.attn_backend == "sdpa"


def test_compiled_flex_cache_key_covers_every_execution_dimension(monkeypatch) -> None:
    source = object()
    compiled: list[object] = []

    def fake_compile(function, *, dynamic):
        assert function is source
        assert dynamic is False
        result = object()
        compiled.append(result)
        return result

    monkeypatch.setattr(_core, "flex_attention", source)
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        torch.nn.attention.flex_attention,
        "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG",
        False,
        raising=False,
    )
    _core._compiled_flex_attention.clear()

    base = {
        "device": torch.device("cuda:0"),
        "dtype": torch.bfloat16,
        "shape": (2, 8, 64, 64),
        "sequence_lengths": (64, 31),
        "mask_semantics": "padding",
    }
    first = _core._get_flex_attention_fn(**base)
    assert _core._get_flex_attention_fn(**base) is first

    variants = (
        {**base, "device": torch.device("cuda:1")},
        {**base, "dtype": torch.float32},
        {**base, "shape": (2, 8, 128, 64)},
        {**base, "sequence_lengths": (64, 30)},
        {**base, "mask_semantics": "chain_and_padding"},
    )
    values = [_core._get_flex_attention_fn(**variant) for variant in variants]
    assert all(value is not first for value in values)
    assert len({id(value) for value in values}) == len(values)
    assert len(compiled) == 1 + len(variants)


def test_flex_block_mask_supports_disjoint_valid_spans_and_exact_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[object, object]] = []

    def fake_create_block_mask(mask_mod, *args, **kwargs):
        block_mask = object()
        created.append((mask_mod, block_mask))
        return block_mask

    monkeypatch.setattr(_core, "flex_attention", object())
    monkeypatch.setattr(_core, "create_block_mask", fake_create_block_mask)
    _core._flex_block_masks.clear()
    first_pattern = torch.tensor(((1, 1, 0, 1), (1, 0, 1, 0)), dtype=torch.bool)

    _, _, first = _core.get_attention_mask(
        _core.AttentionBackend.FLEX_ATTENTION,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        attention_mask=first_pattern,
        dtype=torch.bfloat16,
    )
    _, _, repeated = _core.get_attention_mask(
        _core.AttentionBackend.FLEX_ATTENTION,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        attention_mask=first_pattern.clone(),
        dtype=torch.bfloat16,
    )
    assert repeated is first
    assert len(created) == 1
    mask_mod = created[0][0]
    for batch_index in range(2):
        for query_index in range(4):
            for key_index in range(4):
                assert bool(mask_mod(batch_index, 0, query_index, key_index)) is bool(
                    first_pattern[batch_index, key_index]
                )

    second_pattern = torch.tensor(((1, 0, 1, 1), (0, 1, 0, 1)), dtype=torch.bool)
    _, _, second = _core.get_attention_mask(
        _core.AttentionBackend.FLEX_ATTENTION,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        attention_mask=second_pattern,
        dtype=torch.bfloat16,
    )
    assert second is not first
    assert len(created) == 2
