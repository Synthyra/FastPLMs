"""Opt-in ``attn_implementation="auto"``: a request that always ends in a named backend."""

from __future__ import annotations

import copy
import json
import warnings
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace

from fastplms.attention import AttentionResolution, _auto, interfaces
from fastplms.attention._auto import (
    AttentionExecutionContext,
    attention_execution_context,
    resolve_auto_attention,
)
from fastplms.models.ankh.modeling_ankh import AnkhPreTrainedModel
from fastplms.models.dplm.modeling_dplm import DPLMPreTrainedModel
from fastplms.models.dplm2.modeling_dplm2 import DPLM2PreTrainedModel
from fastplms.models.e1.modeling_e1 import E1PreTrainedModel
from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmModel,
    FastEsmPreTrainedModel,
)
from fastplms.models.esm3.modeling_esm3 import FastESM3PreTrainedModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus import PreTrainedESMplusplusModel
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding
from fastplms.models.esmfold2.attention import ESMFold2AttentionMixin
from fastplms.registry import RegistryError, get_model_registry, load_model_registry


ROOT = Path(__file__).resolve().parents[2]
FLASH_FIRST = ("flash_attention_2", "sdpa")
CUDA_BF16 = AttentionExecutionContext(torch.device("cuda", 0), torch.bfloat16)
_FAMILY_CLASSES = {
    "esm2": FastEsmPreTrainedModel,
    "esm_plusplus": PreTrainedESMplusplusModel,
    "esm3": FastESM3PreTrainedModel,
    "e1": E1PreTrainedModel,
    "dplm": DPLMPreTrainedModel,
    "dplm2": DPLM2PreTrainedModel,
    "ankh": AnkhPreTrainedModel,
    "esmfold": FastEsmForProteinFolding,
    "esmfold2": ESMFold2AttentionMixin,
}


def _tiny_config(attn_implementation: str | None = "auto", **overrides: object) -> FastEsmConfig:
    config = FastEsmConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        pad_token_id=1,
        mask_token_id=5,
        **overrides,
    )
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    return config


def _forward(model: torch.nn.Module) -> None:
    input_ids = torch.tensor([[0, 4, 5, 2]])  # (b=1, l=4)
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


@pytest.fixture
def flash_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FastEsmPreTrainedModel, "_fastplms_attention_auto_order", FLASH_FIRST)


@pytest.fixture
def usable_flash_machine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Pretend the locked kernels load on a capable GPU, and record what was loaded."""
    loaded: list[str] = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 9))
    monkeypatch.setattr(_auto, "require_kernels_package", lambda: None)
    monkeypatch.setattr(interfaces, "require_kernels_package", lambda: None)
    monkeypatch.setattr(_auto, "_ensure_flash_kernels_loaded", loaded.append)
    return loaded


def test_manifest_orders_match_the_family_classes() -> None:
    families = get_model_registry().families

    assert set(_FAMILY_CLASSES) == {
        name for name, family in families.items() if family.attention_auto_order
    }
    for name, family_class in _FAMILY_CLASSES.items():
        family = families[name]
        assert family_class._fastplms_attention_auto_order == family.attention_auto_order
        assert set(family.attention_auto_order) <= set(family.attention)
        if family.attention_auto_order[0].startswith("flash_attention"):
            assert family.attention_auto_evidence is not None
        if family.attention_auto_evidence is not None:
            assert (ROOT / family.attention_auto_evidence).is_file()
        # A family that advertises FlashAttention cites the measurement behind its order.
        if any(name.startswith("flash_attention") for name in family.attention):
            assert family.attention_auto_evidence is not None
    # Boltz2 keeps its single eager implementation and rejects the request.
    assert families["boltz2"].attention_auto_order == ()


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ('attention_auto_order = ["flash_attention_2"]', "must end in 'eager' or 'sdpa'"),
        (
            'attention_auto_order = ["flash_attention_2", "sdpa"]',
            "must cite attention_auto_evidence",
        ),
        ('attention_auto_order = ["sdpa", "sdpa"]', "duplicate values"),
        (
            'attention_auto_order = ["sdpa"]\nattention_auto_evidence = "benchmarks/result.json"',
            "must be a path under docs/evidence/",
        ),
    ),
)
def test_manifest_rejects_an_unsafe_automatic_order(
    replacement: str, message: str, tmp_path: Path
) -> None:
    manifest = (ROOT / "src/fastplms/models.toml").read_text(encoding="utf-8")
    # The first declaration is ESM2's, which advertises FlashAttention and cites evidence.
    esm2_declaration = "\n".join(
        (
            'attention_auto_order = ["sdpa"]',
            'attention_auto_evidence = "docs/evidence/attention/backend_latency.json"',
        )
    )
    assert manifest.index(esm2_declaration) == manifest.index("attention_auto_order")
    path = tmp_path / "models.toml"
    path.write_text(manifest.replace(esm2_declaration, replacement, 1), encoding="utf-8")

    with pytest.raises(RegistryError, match=message):
        load_model_registry(path)


def test_manifest_rejects_an_order_outside_the_advertised_backends(tmp_path: Path) -> None:
    manifest = (ROOT / "src/fastplms/models.toml").read_text(encoding="utf-8")
    # DPLM2 advertises SDPA only.
    assert manifest.count('attention = ["sdpa"]\nattention_auto_order = ["sdpa"]') == 1
    path = tmp_path / "models.toml"
    path.write_text(
        manifest.replace(
            'attention = ["sdpa"]\nattention_auto_order = ["sdpa"]',
            'attention = ["sdpa"]\nattention_auto_order = ["eager"]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="must be a subset of attention"):
        load_model_registry(path)


@pytest.mark.parametrize(
    ("context", "reason"),
    (
        (AttentionExecutionContext(torch.device("cpu"), torch.bfloat16), "requires a CUDA device"),
        (
            AttentionExecutionContext(torch.device("cuda", 0), torch.float32),
            "supports only bfloat16; the attention inputs would be float32",
        ),
    ),
)
def test_flash_is_skipped_before_any_kernel_is_loaded(
    context: AttentionExecutionContext, reason: str, usable_flash_machine: list[str]
) -> None:
    resolution = resolve_auto_attention(FLASH_FIRST, context)

    assert resolution.resolved == "sdpa"
    assert [candidate.usable for candidate in resolution.candidates] == [False, True]
    assert reason in resolution.candidates[0].reason
    assert usable_flash_machine == []


def test_flash_is_skipped_on_an_older_gpu_and_when_the_kernel_cannot_load(
    monkeypatch: pytest.MonkeyPatch, usable_flash_machine: list[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (7, 5))
    older_gpu = resolve_auto_attention(FLASH_FIRST, CUDA_BF16)
    assert older_gpu.resolved == "sdpa"
    assert "capability 8.0 or newer; this GPU has 7.5" in older_gpu.candidates[0].reason
    assert usable_flash_machine == []

    def unavailable(_implementation: str) -> None:
        raise RuntimeError("Unable to load the manifest-pinned kernel.")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (9, 0))
    monkeypatch.setattr(_auto, "_ensure_flash_kernels_loaded", unavailable)
    offline = resolve_auto_attention(FLASH_FIRST, CUDA_BF16)
    assert offline.resolved == "sdpa"
    assert offline.candidates[0].reason == "Unable to load the manifest-pinned kernel."


def test_flash_is_selected_when_every_gate_passes(usable_flash_machine: list[str]) -> None:
    resolution = resolve_auto_attention(FLASH_FIRST, CUDA_BF16)

    assert resolution == AttentionResolution(
        requested="auto",
        resolved="flash_attention_2",
        candidates=resolution.candidates,
        context=CUDA_BF16,
        deferred=False,
    )
    assert [candidate.implementation for candidate in resolution.candidates] == [
        "flash_attention_2"
    ]
    assert usable_flash_machine == ["flash_attention_2"]


def test_execution_context_prefers_explicit_values_over_the_parameters() -> None:
    module = torch.nn.Linear(2, 2)

    assert attention_execution_context(module) == AttentionExecutionContext(
        torch.device("cpu"), torch.float32
    )
    assert attention_execution_context(module, "cuda:0", torch.bfloat16) == CUDA_BF16
    with pytest.raises(RuntimeError, match="requires a model with parameters"):
        attention_execution_context(torch.nn.Identity())


def test_context_free_order_resolves_at_construction_and_never_stores_auto() -> None:
    model = FastEsmModel(_tiny_config())

    resolution = model.attention_resolution
    assert resolution is not None
    assert (resolution.requested, resolution.resolved, resolution.deferred) == (
        "auto",
        "sdpa",
        False,
    )
    assert model.config.attn_backend == "sdpa"
    assert model.config._attn_implementation == "sdpa"
    assert len(model._forward_pre_hooks) == 0
    assert model.resolve_attn_implementation() is resolution


def test_named_requests_keep_their_strict_behavior() -> None:
    model = FastEsmModel(_tiny_config("eager"))

    assert model.attention_resolution is None
    assert model.config.attn_backend == "eager"
    with pytest.raises(RuntimeError, match="was not configured with attn_implementation='auto'"):
        model.resolve_attn_implementation()
    with pytest.raises(ValueError, match="does not support 'not_a_backend'"):
        model.set_attn_implementation("not_a_backend")


def test_a_family_without_an_order_rejects_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FastEsmPreTrainedModel, "_fastplms_attention_auto_order", ())

    with pytest.raises(ValueError, match="does not support attn_implementation='auto'"):
        FastEsmModel(_tiny_config())


@pytest.mark.usefixtures("flash_first")
def test_flash_first_order_runs_provisionally_until_the_first_forward() -> None:
    model = FastEsmForMaskedLM(_tiny_config()).eval()

    pending = model.attention_resolution
    assert pending is not None
    assert (pending.resolved, pending.deferred, pending.candidates) == ("sdpa", True, ())
    assert model.config.attn_backend == "sdpa"
    # Only the outer model holds the request; the wrapped encoder was built on SDPA.
    assert model.esm.attention_resolution is None
    assert len(model._forward_pre_hooks) == 1

    _forward(model)

    settled = model.attention_resolution
    assert settled is not None
    assert (settled.resolved, settled.deferred) == ("sdpa", False)
    assert settled.context == AttentionExecutionContext(torch.device("cpu"), torch.float32)
    assert "requires a CUDA device" in settled.candidates[0].reason
    assert len(model._forward_pre_hooks) == 0


@pytest.mark.usefixtures("flash_first")
def test_explicit_resolution_applies_flash_to_every_layer(usable_flash_machine: list[str]) -> None:
    model = FastEsmForMaskedLM(_tiny_config()).eval()

    resolution = model.resolve_attn_implementation(device="cuda:0", dtype=torch.bfloat16)

    assert resolution.resolved == "flash_attention_2"
    assert usable_flash_machine == ["flash_attention_2"]
    assert model.config.attn_backend == "flash_attention_2"
    assert model.config._attn_implementation == "flash_attention_2"
    layer_backends = {layer.attention.self.attn_backend.value for layer in model.esm.encoder.layer}
    assert layer_backends == {"flash_attention_2"}
    assert model.esm.encoder.attention_backend.value == "flash_attention_2"
    assert len(model._forward_pre_hooks) == 0
    # A settled request is not judged again.
    assert model.resolve_attn_implementation() is resolution


@pytest.mark.usefixtures("flash_first")
def test_saved_configuration_keeps_the_checkpoint_backend(
    usable_flash_machine: list[str], tmp_path: Path
) -> None:
    model = FastEsmModel(_tiny_config(attn_backend="eager"))
    model.resolve_attn_implementation(device="cuda:0", dtype=torch.bfloat16)
    assert model.config.attn_backend == "flash_attention_2"

    model.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["attn_backend"] == "eager"
    assert "auto" not in json.dumps(saved)
    assert model.config.attn_backend == "flash_attention_2"


@pytest.mark.usefixtures("flash_first")
def test_a_named_request_replaces_a_pending_automatic_one() -> None:
    model = FastEsmModel(_tiny_config())
    assert len(model._forward_pre_hooks) == 1

    model.set_attn_implementation("eager")

    assert model.attention_resolution is None
    assert len(model._forward_pre_hooks) == 0
    assert model.config.attn_backend == "eager"

    model.set_attn_implementation("auto")
    pending = model.attention_resolution
    assert pending is not None and pending.deferred
    assert len(model._forward_pre_hooks) == 1
    with pytest.raises(ValueError, match="does not load external attention kernels"):
        model.set_attn_implementation("auto", allow_all_kernels=True)


@pytest.mark.usefixtures("flash_first")
def test_a_copied_model_settles_its_own_request() -> None:
    model = FastEsmModel(_tiny_config()).eval()
    clone = copy.deepcopy(model)

    _forward(clone)

    clone_resolution = clone.attention_resolution
    original_resolution = model.attention_resolution
    assert clone_resolution is not None and not clone_resolution.deferred
    assert original_resolution is not None and original_resolution.deferred
    assert len(model._forward_pre_hooks) == 1


def test_embedding_runner_settles_a_pending_request_before_fingerprinting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastplms.embeddings import runner

    observed: list[str] = []

    class PendingModel:
        attention_resolution = SimpleNamespace(deferred=True)

        def resolve_attn_implementation(self) -> None:
            observed.append("resolved")
            type(self).attention_resolution = SimpleNamespace(deferred=False)

    def stop_at_fingerprint(*_args: object, **_kwargs: object) -> None:
        observed.append("fingerprint")
        raise _StopAtFingerprintError

    monkeypatch.setattr(runner, "_run_fingerprint", stop_at_fingerprint)
    monkeypatch.setattr(runner, "_tokenizer_metadata", lambda _model, _tokenizer: {})
    monkeypatch.setattr(runner, "_embedding_context", lambda *_args, **_kwargs: ({}, None))

    with pytest.raises(_StopAtFingerprintError):
        runner.embed_dataset(PendingModel(), ["MKT"], pooling="mean", tokenizer=object())

    assert observed == ["resolved", "fingerprint"]


class _StopAtFingerprintError(Exception):
    pass


class _TransformersStyleConfig:
    """``_attn_implementation`` reads and writes one internal field, as in Transformers."""

    def __init__(self, attn_implementation: str | None) -> None:
        self._attn_implementation_internal = attn_implementation
        self.attn_backend: str | None = None

    @property
    def _attn_implementation(self) -> str | None:
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | None) -> None:
        self._attn_implementation_internal = value


class _AcceptingTransformersBase:
    """The ``PreTrainedModel`` behaviors the attention mixins rely on."""

    def __init__(self, config: _TransformersStyleConfig) -> None:
        self.config = config

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str, **_kwargs: object
    ) -> str:
        return attn_implementation

    def modules(self) -> tuple[object, ...]:
        return (self,)


def test_esmfold2_selects_its_order_for_the_outer_model_and_the_esmc_backbone() -> None:
    class Model(ESMFold2AttentionMixin, _AcceptingTransformersBase):
        pass

    config = _TransformersStyleConfig("auto")
    model = Model(config)

    resolution = model.attention_resolution
    assert resolution is not None
    assert (resolution.resolved, resolution.deferred) == ("sdpa", False)
    assert config.esmc_attn_backend == "sdpa"
    assert config.attn_backend == "sdpa"
    assert config._attn_implementation_internal == "sdpa"

    model.set_attn_implementation("eager")
    assert model.attention_resolution is None
    assert config.esmc_attn_backend == "eager"
    model.set_attn_implementation("auto")
    assert config.esmc_attn_backend == "sdpa"


def test_ankh_generation_model_accepts_the_request_as_eager() -> None:
    from fastplms.models.ankh.modeling_ankh import FastAnkhConfig, FastAnkhForConditionalGeneration

    config = FastAnkhConfig(
        vocab_size=16,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
    )
    config._attn_implementation = "auto"

    model = FastAnkhForConditionalGeneration(config)

    assert model.config.attn_backend == "eager"
    assert model.config._attn_implementation == "eager"


@pytest.mark.usefixtures("flash_first")
def test_from_pretrained_accepts_the_request_and_settles_on_the_first_forward(
    tmp_path: Path,
) -> None:
    FastEsmModel(_tiny_config("sdpa")).save_pretrained(tmp_path)

    model = FastEsmModel.from_pretrained(tmp_path, attn_implementation="auto").eval()

    pending = model.attention_resolution
    assert pending is not None and pending.deferred
    assert model.config._attn_implementation == "sdpa"
    _forward(model)
    settled = model.attention_resolution
    assert settled is not None
    assert (settled.resolved, settled.deferred) == ("sdpa", False)


@pytest.mark.usefixtures("flash_first")
def test_a_compiled_model_settles_the_request_eagerly_and_matches_eager_output() -> None:
    torch.manual_seed(0)
    model = FastEsmModel(_tiny_config()).eval()
    input_ids = torch.tensor([[0, 4, 5, 2]])  # (b=1, l=4)
    attention_mask = torch.ones_like(input_ids)  # (b, l)
    compiled = torch.compile(model, backend="eager")

    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        compiled_output = compiled(input_ids=input_ids, attention_mask=attention_mask)
        eager_output = model(input_ids=input_ids, attention_mask=attention_mask)

    settled = model.attention_resolution
    assert settled is not None and not settled.deferred
    assert len(model._forward_pre_hooks) == 0
    assert torch.equal(compiled_output.last_hidden_state, eager_output.last_hidden_state)
    # Dynamo warns when it traces the cached manifest loader, which it must not reach.
    assert not [warning for warning in caught if "lru_cache" in str(warning.message)]
