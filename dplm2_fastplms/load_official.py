"""Load official DPLM2 model from the dplm/byprot submodule for comparison."""

import dataclasses
import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import transformers


def _ensure_local_dplm_module_on_path() -> pathlib.Path:
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [script_root / "dplm" / "src", script_root / "DPLM" / "src"]

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "dplm" / "src")
        candidates.append(parent / "DPLM" / "src")

    deduplicated: list[pathlib.Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduplicated.append(resolved)

    existing: list[pathlib.Path] = []
    for candidate in deduplicated:
        if candidate.exists():
            existing.append(candidate)
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    assert len(existing) > 0, (
        "Expected local dplm/src path for byprot imports. "
        f"Checked: {', '.join(str(p) for p in deduplicated)}"
    )

    byprot_spec = None
    try:
        byprot_spec = importlib.util.find_spec("byprot")
    except ValueError as exc:
        message = str(exc)
        if "byprot.__spec__ is None" not in message:
            raise
        stale = [k for k in list(sys.modules.keys()) if k == "byprot" or k.startswith("byprot.")]
        for k in stale:
            del sys.modules[k]
        byprot_spec = importlib.util.find_spec("byprot")

    if byprot_spec is not None:
        return existing[0]

    raise FileNotFoundError(
        "byprot module import failed. Expected local dplm submodule at one of: "
        f"{', '.join(str(p) for p in deduplicated)}. "
        "Run `git submodule update --init --recursive dplm`."
    )


def _patch_lightning_fabric_fsdp() -> None:
    fsdp_module = importlib.import_module("lightning_fabric.strategies.fsdp")
    fsdp_symbols = dir(fsdp_module)
    if "_has_meta_device_parameters" in fsdp_symbols:
        return
    assert "_has_meta_device_parameters_or_buffers" in fsdp_symbols, (
        "Expected lightning_fabric.strategies.fsdp to expose "
        "_has_meta_device_parameters_or_buffers for byprot compatibility."
    )
    fsdp_module._has_meta_device_parameters = fsdp_module._has_meta_device_parameters_or_buffers


def _patch_transformers_esm_star_exports() -> None:
    esm_module = importlib.import_module("transformers.models.esm.modeling_esm")
    esm_module.__all__ = [name for name in dir(esm_module) if not name.startswith("_")]


def _install_byprot_package_shims(dplm_src_path: pathlib.Path) -> None:
    byprot_root = dplm_src_path / "byprot"
    models_root = byprot_root / "models"
    datamodules_root = byprot_root / "datamodules"
    dataset_root = datamodules_root / "dataset"
    assert byprot_root.exists(), f"Expected byprot package at {byprot_root}"
    assert models_root.exists(), f"Expected byprot.models package at {models_root}"
    assert datamodules_root.exists(), f"Expected byprot.datamodules package at {datamodules_root}"
    assert dataset_root.exists(), f"Expected byprot.datamodules.dataset package at {dataset_root}"

    if "byprot" in sys.modules:
        byprot_module = sys.modules["byprot"]
    else:
        byprot_module = types.ModuleType("byprot")
        sys.modules["byprot"] = byprot_module
    byprot_module.__path__ = [str(byprot_root)]
    byprot_module.__file__ = str(byprot_root / "__init__.py")

    if "byprot.models" in sys.modules:
        models_module = sys.modules["byprot.models"]
    else:
        models_module = types.ModuleType("byprot.models")
        sys.modules["byprot.models"] = models_module
    models_module.__path__ = [str(models_root)]
    models_module.__file__ = str(models_root / "__init__.py")
    if "MODEL_REGISTRY" not in models_module.__dict__:
        models_module.MODEL_REGISTRY = {}
    if "register_model" not in models_module.__dict__:
        def register_model(name):
            def decorator(cls):
                models_module.MODEL_REGISTRY[name] = cls
                return cls
            return decorator
        models_module.register_model = register_model

    if "byprot.datamodules" not in sys.modules:
        datamodules_module = types.ModuleType("byprot.datamodules")
        sys.modules["byprot.datamodules"] = datamodules_module
    else:
        datamodules_module = sys.modules["byprot.datamodules"]
    datamodules_module.__path__ = [str(datamodules_root)]
    datamodules_module.__file__ = str(datamodules_root / "__init__.py")

    if "byprot.datamodules.dataset" not in sys.modules:
        dataset_module = types.ModuleType("byprot.datamodules.dataset")
        sys.modules["byprot.datamodules.dataset"] = dataset_module
    else:
        dataset_module = sys.modules["byprot.datamodules.dataset"]
    dataset_module.__path__ = [str(dataset_root)]
    dataset_module.__file__ = str(dataset_root / "__init__.py")

    byprot_module.models = models_module
    byprot_module.datamodules = datamodules_module
    datamodules_module.dataset = dataset_module


def _purge_modules(prefixes: list[str]) -> None:
    module_names = list(sys.modules.keys())
    for name in module_names:
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
                break


def _import_byprot_module_with_dataclass_patch(module_name: str, purge_prefixes: list[str]):
    _purge_modules(purge_prefixes)
    original_dataclass = dataclasses.dataclass

    def _patched_dataclass(_cls=None, **kwargs):
        if _cls is None:
            def _decorator(cls):
                return _patched_dataclass(cls, **kwargs)
            return _decorator
        patched_kwargs = dict(kwargs)
        if _cls.__module__.startswith("byprot.") and "unsafe_hash" not in patched_kwargs:
            patched_kwargs["unsafe_hash"] = True
        return original_dataclass(_cls, **patched_kwargs)

    dataclasses.dataclass = _patched_dataclass
    try:
        module = importlib.import_module(module_name)
    finally:
        dataclasses.dataclass = original_dataclass
    return module


def _install_dplm2_tokenizer_hf_compat(tokenizer_cls) -> None:
    if "mask_token_id" in tokenizer_cls.__dict__:
        return

    def _resolve_token_id(self, attribute_name: str, fallback_token: str) -> int:
        token_value = getattr(self, attribute_name)
        if token_value is None:
            token_value = fallback_token
        return self._token_to_id[token_value]

    def _get_mask_token(self):
        return self.aa_mask_token if self.aa_mask_token is not None else "<mask_aa>"

    def _get_mask_token_id(self):
        return _resolve_token_id(self, "aa_mask_token", "<mask_aa>")

    def _get_cls_token(self):
        return self.aa_cls_token if self.aa_cls_token is not None else "<cls_aa>"

    def _get_cls_token_id(self):
        return _resolve_token_id(self, "aa_cls_token", "<cls_aa>")

    def _get_eos_token(self):
        return self.aa_eos_token if self.aa_eos_token is not None else "<eos_aa>"

    def _get_eos_token_id(self):
        return _resolve_token_id(self, "aa_eos_token", "<eos_aa>")

    def _get_unk_token(self):
        return self.aa_unk_token if self.aa_unk_token is not None else "<unk_aa>"

    def _get_unk_token_id(self):
        return _resolve_token_id(self, "aa_unk_token", "<unk_aa>")

    tokenizer_cls.mask_token = property(_get_mask_token)
    tokenizer_cls.mask_token_id = property(_get_mask_token_id)
    tokenizer_cls.cls_token = property(_get_cls_token)
    tokenizer_cls.cls_token_id = property(_get_cls_token_id)
    tokenizer_cls.eos_token = property(_get_eos_token)
    tokenizer_cls.eos_token_id = property(_get_eos_token_id)
    tokenizer_cls.unk_token = property(_get_unk_token)
    tokenizer_cls.unk_token_id = property(_get_unk_token_id)


class _OfficialComplianceOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor):
        self.logits = logits
        self.hidden_states = (last_hidden_state,)


class _OfficialDPLM2ComplianceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> _OfficialComplianceOutput:
        del attention_mask, output_hidden_states, output_attentions, kwargs
        outputs = self.model(input_ids=input_ids)
        assert isinstance(outputs, dict), f"Expected dict from official DPLM2, got {type(outputs)}."
        assert "logits" in outputs, "Official DPLM2 output is missing 'logits'."
        assert "last_hidden_state" in outputs, "Official DPLM2 output is missing 'last_hidden_state'."
        return _OfficialComplianceOutput(
            logits=outputs["logits"],
            last_hidden_state=outputs["last_hidden_state"],
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official DPLM2 model from the dplm/byprot submodule.

    Args:
        reference_repo_id: e.g. "airkingbd/dplm2_150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    dplm_src_path = _ensure_local_dplm_module_on_path()
    _patch_lightning_fabric_fsdp()
    _patch_transformers_esm_star_exports()
    _install_byprot_package_shims(dplm_src_path)

    dplm2_module = _import_byprot_module_with_dataclass_patch(
        module_name="byprot.models.dplm2.dplm2",
        purge_prefixes=["byprot.models.utils", "byprot.models.dplm2"],
    )
    dplm2_modeling_module = importlib.import_module(
        "byprot.models.dplm2.modules.dplm2_modeling_esm"
    )
    tokenized_protein_module = importlib.import_module(
        "byprot.datamodules.dataset.tokenized_protein"
    )
    DPLM2Tokenizer = tokenized_protein_module.DPLM2Tokenizer
    _install_dplm2_tokenizer_hf_compat(DPLM2Tokenizer)
    setattr(transformers, "DPLM2Tokenizer", DPLM2Tokenizer)
    tokenization_auto_module = importlib.import_module(
        "transformers.models.auto.tokenization_auto"
    )
    setattr(tokenization_auto_module, "DPLM2Tokenizer", DPLM2Tokenizer)
    EsmForDPLM2 = dplm2_modeling_module.EsmForDPLM2
    EsmForDPLM2.all_tied_weights_keys = {}

    dplm_modeling_module = _import_byprot_module_with_dataclass_patch(
        module_name="byprot.models.dplm.modules.dplm_modeling_esm",
        purge_prefixes=["byprot.models.utils", "byprot.models.dplm"],
    )
    EsmForDPLM = dplm_modeling_module.EsmForDPLM
    EsmForDPLM.all_tied_weights_keys = {}
    byprot_models_module = importlib.import_module("byprot.models")
    byprot_models_module.MODEL_REGISTRY["dplm_esm"] = EsmForDPLM
    byprot_models_module.MODEL_REGISTRY["dplm2_esm"] = EsmForDPLM2

    MultimodalDiffusionProteinLanguageModel = dplm2_module.MultimodalDiffusionProteinLanguageModel
    official_model = (
        MultimodalDiffusionProteinLanguageModel.from_pretrained(reference_repo_id)
        .to(device=device, dtype=dtype)
        .eval()
    )
    wrapped = _OfficialDPLM2ComplianceWrapper(official_model).eval()
    tokenizer = official_model.tokenizer
    return wrapped, tokenizer
