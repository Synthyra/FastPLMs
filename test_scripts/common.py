import contextlib
import datetime
import importlib
import importlib.util
import pathlib
import random
import shutil
import sys
import types

import numpy as np
import torch
from huggingface_hub import login
from typing import Dict, Iterable, List, Optional
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    EsmForMaskedLM,
    EsmTokenizer,
)

from test_scripts.model_registry import ModelSpec


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_int_list(values: str) -> List[int]:
    output: List[int] = []
    for chunk in values.split(","):
        value = int(chunk.strip())
        output.append(value)
    assert len(output) > 0, "Expected at least one integer value."
    return output


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def now_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_dir(output_dir: Optional[str], suite_name: str) -> pathlib.Path:
    if output_dir is None:
        root = pathlib.Path("test_scripts") / "results" / now_timestamp()
        return ensure_dir(root / suite_name)
    root = pathlib.Path(output_dir)
    return ensure_dir(root)


def generate_sequences(num_sequences: int, min_length: int, max_length: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    sequences: List[str] = []
    for _ in range(num_sequences):
        length = rng.randint(min_length, max_length)
        sequence = "M" + "".join(rng.choices(CANONICAL_AMINO_ACIDS, k=length - 1))
        sequences.append(sequence)
    return sequences


def chunk_sequences(sequences: List[str], batch_size: int) -> List[List[str]]:
    batches: List[List[str]] = []
    for start in range(0, len(sequences), batch_size):
        batches.append(sequences[start:start + batch_size])
    return batches


def login_if_needed(token: Optional[str]) -> None:
    if token is not None:
        assert len(token) > 0, "Token cannot be empty."
        login(token=token)


def load_model(
    spec: ModelSpec,
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_backend: Optional[str] = None,
    compile_model: bool = True,
    compile_backend: Optional[str] = None,
    compile_dynamic: Optional[bool] = None,
):
    if spec.family == "esm2":
        from esm2.modeling_fastesm import (
            FastEsmConfig,
            FastEsmModel,   
            FastEsmForMaskedLM,
        )
        model_config = FastEsmConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            model_config.attn_backend = attn_backend
        if task == "base":
            model = FastEsmModel.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = FastEsmForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        else:
            raise ValueError(f"Unsupported task: {task}")
    elif spec.family == "esmplusplus":
        from esm_plusplus.modeling_esm_plusplus import (
            ESMplusplusConfig,
            ESMplusplusModel,
            ESMplusplusForMaskedLM,
        )
        model_config = ESMplusplusConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            model_config.attn_backend = attn_backend
        if task == "base":
            model = ESMplusplusModel.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = ESMplusplusForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        else:
            raise ValueError(f"Unsupported task: {task}")
    elif spec.family == "e1":
        from e1_fastplms.modeling_e1 import (
            E1Config,
            E1ForMaskedLM,
            E1Model,
        )

        model_config = E1Config.from_pretrained(spec.repo_id)
        if task == "base":
            model = E1Model.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = E1ForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        else:
            raise ValueError(f"Unsupported task: {task}")
    elif spec.family == "dplm":
        from dplm_fastplms.dplm import (
            DPLMConfig,
            DPLMForMaskedLM,
            DPLMModel,
        )

        model_config = DPLMConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            model_config.attn_backend = attn_backend
        if task == "base":
            model = DPLMModel.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = DPLMForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        else:
            raise ValueError(f"Unsupported task: {task}")
    elif spec.family == "dplm2":
        from dplm_fastplms.dplm2 import (
            DPLM2Config,
            DPLM2ForMaskedLM,
            DPLM2Model,
        )

        model_config = DPLM2Config.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            model_config.attn_backend = attn_backend
        if task == "base":
            model = DPLM2Model.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = DPLM2ForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        else:
            raise ValueError(f"Unsupported task: {task}")
    else:
        model_config = None
        if attn_backend is not None:
            model_config = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=True)
            model_config.attn_backend = attn_backend
        if task == "base":
            model = AutoModel.from_pretrained(spec.repo_id, trust_remote_code=True, torch_dtype=dtype, config=model_config)
        elif task == "masked_lm":
            model = AutoModelForMaskedLM.from_pretrained(spec.repo_id, trust_remote_code=True, torch_dtype=dtype, config=model_config)
        else:
            raise ValueError(f"Unsupported task: {task}")

    model = model.to(device).eval()
    if spec.family == "esmplusplus":
        model.all_tied_weights_keys = {}
    if compile_model:
        if compile_backend is None and compile_dynamic is None:
            model = torch.compile(model)
        elif compile_backend is None:
            model = torch.compile(model, dynamic=compile_dynamic)
        elif compile_dynamic is None:
            model = torch.compile(model, backend=compile_backend)
        else:
            model = torch.compile(model, backend=compile_backend, dynamic=compile_dynamic)

    if spec.family == "e1":
        tokenizer = None
    else:
        tokenizer = model.tokenizer
    return model, tokenizer


def _ensure_local_e1_module_on_path() -> None:
    candidates: List[pathlib.Path] = []
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(script_root / "e1" / "src")
    candidates.append(script_root / "E1" / "src")

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "e1" / "src")
        candidates.append(parent / "E1" / "src")

    deduplicated_candidates: List[pathlib.Path] = []
    seen_paths = set()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        candidate_key = str(candidate_resolved)
        if candidate_key not in seen_paths:
            seen_paths.add(candidate_key)
            deduplicated_candidates.append(candidate_resolved)

    for candidate in deduplicated_candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    if importlib.util.find_spec("E1") is not None:
        return

    checked_paths = ", ".join([str(path) for path in deduplicated_candidates])
    raise FileNotFoundError(
        "e1 module import failed. Expected local submodule at one of: "
        f"{checked_paths}. "
        "Run `git submodule update --init --recursive e1` or install e1 package."
    )


def _ensure_local_dplm_module_on_path() -> None:
    candidates: List[pathlib.Path] = []
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(script_root / "dplm" / "src")
    candidates.append(script_root / "DPLM" / "src")

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "dplm" / "src")
        candidates.append(parent / "DPLM" / "src")

    deduplicated_candidates: List[pathlib.Path] = []
    seen_paths = set()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        candidate_key = str(candidate_resolved)
        if candidate_key not in seen_paths:
            seen_paths.add(candidate_key)
            deduplicated_candidates.append(candidate_resolved)

    for candidate in deduplicated_candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    if importlib.util.find_spec("byprot") is not None:
        return

    checked_paths = ", ".join([str(path) for path in deduplicated_candidates])
    raise FileNotFoundError(
        "byprot module import failed. Expected local dplm submodule at one of: "
        f"{checked_paths}. "
        "Run `git submodule update --init --recursive dplm`."
    )


def _patch_lightning_fabric_fsdp_for_byprot() -> None:
    fsdp_module = importlib.import_module("lightning_fabric.strategies.fsdp")
    fsdp_symbols = dir(fsdp_module)
    if "_has_meta_device_parameters" in fsdp_symbols:
        return
    assert "_has_meta_device_parameters_or_buffers" in fsdp_symbols, (
        "Expected lightning_fabric.strategies.fsdp to expose "
        "_has_meta_device_parameters_or_buffers for byprot compatibility."
    )
    fsdp_module._has_meta_device_parameters = fsdp_module._has_meta_device_parameters_or_buffers


def _ensure_local_e1_tokenizer_json(spec: ModelSpec) -> None:
    e1_spec = importlib.util.find_spec("E1")
    assert e1_spec is not None, "E1 module spec was not found after path setup."
    assert e1_spec.origin is not None, "E1 module origin is required to resolve tokenizer path."
    e1_package_dir = pathlib.Path(e1_spec.origin).resolve().parent
    tokenizer_path = e1_package_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return

    script_root = pathlib.Path(__file__).resolve().parents[1]
    fallback_tokenizer_path = script_root / "e1_fastplms" / "tokenizer.json"
    assert fallback_tokenizer_path.exists(), (
        "Missing E1 tokenizer.json in both locations. "
        f"Expected either {tokenizer_path} or {fallback_tokenizer_path}."
    )
    print(
        "[test_scripts.common] Missing E1 tokenizer file at "
        f"{tokenizer_path}. Copying from local fallback {fallback_tokenizer_path}."
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback_tokenizer_path, tokenizer_path)
    assert tokenizer_path.exists(), f"Failed to materialize E1 tokenizer file at {tokenizer_path}"


def load_official_e1_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    assert spec.reference_repo_id is not None, f"Missing official e1 repo id for {spec.key}."
    _ensure_local_e1_module_on_path()
    _ensure_local_e1_tokenizer_json(spec)
    if "kernels" not in sys.modules:
        kernels_spec = importlib.util.find_spec("kernels")
        if kernels_spec is None:
            kernels_module = types.ModuleType("kernels")

            def _missing_get_kernel(_kernel_name: str):
                raise ModuleNotFoundError("No module named 'kernels'")

            kernels_module.get_kernel = _missing_get_kernel
            sys.modules["kernels"] = kernels_module
    batch_preparer_module = importlib.import_module("E1.batch_preparer")
    modeling_module = importlib.import_module("E1.modeling")
    E1BatchPreparer = batch_preparer_module.E1BatchPreparer
    E1ForMaskedLM = modeling_module.E1ForMaskedLM

    model = E1ForMaskedLM.from_pretrained(spec.reference_repo_id).to(device=device, dtype=dtype).eval()
    batch_preparer = E1BatchPreparer()
    return model, batch_preparer


def load_official_esmc_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    from esm_plusplus.get_esmc_weights import ESMplusplus_300M, ESMplusplus_600M

    assert spec.reference_repo_id is not None, f"Missing official ESMC repo id for {spec.key}."
    if "300" in spec.reference_repo_id:
        model = ESMplusplus_300M(device=device)
    elif "600" in spec.reference_repo_id:
        model = ESMplusplus_600M(device=device)
    else:
        raise ValueError(f"Unsupported ESMC reference repo id: {spec.reference_repo_id}")
    model = model.to(device).to(dtype).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


def load_official_esm2_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    assert spec.reference_repo_id is not None, f"Missing official ESM2 repo id for {spec.key}."
    tokenizer = EsmTokenizer.from_pretrained(spec.reference_repo_id)
    model = EsmForMaskedLM.from_pretrained(spec.reference_repo_id).to(device=device, dtype=dtype).eval()
    return model, tokenizer


class _OfficialComplianceOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor):
        self.logits = logits
        self.hidden_states = (last_hidden_state,)


class _OfficialDPLMComplianceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
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
        del attention_mask
        del output_hidden_states
        del output_attentions
        del kwargs
        outputs = self.model(input_ids=input_ids, return_last_hidden_state=True)
        assert isinstance(outputs, tuple), f"Expected tuple output from official DPLM, got {type(outputs)}."
        assert len(outputs) == 2, f"Expected 2-tuple output from official DPLM, got {len(outputs)} values."
        logits, last_hidden_state = outputs
        return _OfficialComplianceOutput(logits=logits, last_hidden_state=last_hidden_state)


class _OfficialDPLM2ComplianceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
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
        del attention_mask
        del output_hidden_states
        del output_attentions
        del kwargs
        outputs = self.model(input_ids=input_ids)
        assert isinstance(outputs, dict), f"Expected dict output from official DPLM2, got {type(outputs)}."
        assert "logits" in outputs, "Official DPLM2 output is missing 'logits'."
        assert "last_hidden_state" in outputs, "Official DPLM2 output is missing 'last_hidden_state'."
        return _OfficialComplianceOutput(
            logits=outputs["logits"],
            last_hidden_state=outputs["last_hidden_state"],
        )


def load_official_dplm_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    assert spec.reference_repo_id is not None, f"Missing official DPLM repo id for {spec.key}."
    _ensure_local_dplm_module_on_path()
    _patch_lightning_fabric_fsdp_for_byprot()
    dplm_module = importlib.import_module("byprot.models.dplm.dplm")
    DiffusionProteinLanguageModel = dplm_module.DiffusionProteinLanguageModel

    official_model = DiffusionProteinLanguageModel.from_pretrained(spec.reference_repo_id).to(device=device, dtype=dtype).eval()
    model = _OfficialDPLMComplianceWrapper(official_model).eval()
    tokenizer = official_model.tokenizer
    return model, tokenizer


def load_official_dplm2_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    assert spec.reference_repo_id is not None, f"Missing official DPLM2 repo id for {spec.key}."
    _ensure_local_dplm_module_on_path()
    _patch_lightning_fabric_fsdp_for_byprot()
    dplm2_module = importlib.import_module("byprot.models.dplm2.dplm2")
    MultimodalDiffusionProteinLanguageModel = dplm2_module.MultimodalDiffusionProteinLanguageModel

    official_model = MultimodalDiffusionProteinLanguageModel.from_pretrained(spec.reference_repo_id).to(
        device=device,
        dtype=dtype,
    ).eval()
    model = _OfficialDPLM2ComplianceWrapper(official_model).eval()
    tokenizer = official_model.tokenizer
    return model, tokenizer


def load_official_model_for_compliance(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    if spec.family == "e1":
        return load_official_e1_model(spec=spec, device=device, dtype=dtype)
    if spec.family == "esmplusplus":
        return load_official_esmc_model(spec=spec, device=device, dtype=dtype)
    if spec.family == "esm2":
        return load_official_esm2_model(spec=spec, device=device, dtype=dtype)
    if spec.family == "dplm":
        return load_official_dplm_model(spec=spec, device=device, dtype=dtype)
    if spec.family == "dplm2":
        return load_official_dplm2_model(spec=spec, device=device, dtype=dtype)
    raise ValueError(f"Unsupported family for official loader: {spec.family}")


def compare_model_state_dicts_fp32(reference_model, candidate_model, max_report: int = 5) -> Dict[str, object]:
    reference_state = reference_model.state_dict()
    candidate_state = candidate_model.state_dict()
    reference_keys = set(reference_state.keys())
    candidate_keys = set(candidate_state.keys())
    only_in_reference = sorted(reference_keys - candidate_keys)
    only_in_candidate = sorted(candidate_keys - reference_keys)
    common_keys = sorted(reference_keys & candidate_keys)
    shape_mismatches: List[Dict[str, object]] = []
    differing_tensors: List[Dict[str, object]] = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""

    for name in common_keys:
        reference_tensor = reference_state[name].detach().cpu().to(torch.float32)
        candidate_tensor = candidate_state[name].detach().cpu().to(torch.float32)
        if reference_tensor.shape != candidate_tensor.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "reference_shape": list(reference_tensor.shape),
                    "candidate_shape": list(candidate_tensor.shape),
                }
            )
            continue
        if torch.equal(reference_tensor, candidate_tensor):
            continue
        abs_diff = torch.abs(reference_tensor - candidate_tensor)
        param_max_abs_diff = float(torch.max(abs_diff).item())
        param_mean_abs_diff = float(torch.mean(abs_diff).item())
        differing_tensors.append(
            {
                "name": name,
                "max_abs_diff": param_max_abs_diff,
                "mean_abs_diff": param_mean_abs_diff,
            }
        )
        if param_max_abs_diff > max_abs_diff:
            max_abs_diff = param_max_abs_diff
            max_abs_diff_param = name

    match = len(only_in_reference) == 0 and len(only_in_candidate) == 0 and len(shape_mismatches) == 0 and len(differing_tensors) == 0
    return {
        "match": match,
        "only_in_reference": only_in_reference[:max_report],
        "only_in_candidate": only_in_candidate[:max_report],
        "shape_mismatches": shape_mismatches[:max_report],
        "diff_param_count": len(differing_tensors),
        "diff_params_sample": differing_tensors[:max_report],
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_param": max_abs_diff_param,
    }


def prepare_official_batch_for_compliance(
    spec: ModelSpec,
    sequence_batch: List[str],
    tokenizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if spec.family == "e1":
        assert tokenizer is not None, "Official e1 batch preparer is required for compliance comparison."
        raw_batch = tokenizer.get_batch_kwargs(sequence_batch, device=device)
        batch = {
            "input_ids": raw_batch["input_ids"],
            "within_seq_position_ids": raw_batch["within_seq_position_ids"],
            "global_position_ids": raw_batch["global_position_ids"],
            "sequence_ids": raw_batch["sequence_ids"],
            "attention_mask": (raw_batch["sequence_ids"] != -1).long(),
        }
        return batch
    assert tokenizer is not None, "Official tokenizer is required for compliance comparison."
    batch = tokenizer(sequence_batch, return_tensors="pt", padding="longest")
    return batch.to(device)


def prepare_model_batch(
    spec: ModelSpec,
    model,
    tokenizer,
    sequence_batch: List[str],
    device: torch.device,
    pad_to_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if spec.family == "e1":
        batch = model.prep_tokens.get_batch_kwargs(sequence_batch, device=device)
        return batch
    assert tokenizer is not None, "Tokenizer is required for non-e1 families."
    if pad_to_length is None:
        batch = tokenizer(sequence_batch, return_tensors="pt", padding="longest")
    else:
        batch = tokenizer(
            sequence_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=pad_to_length,
            truncation=True,
        )
    return batch.to(device)


def run_forward(
    spec: ModelSpec,
    model,
    batch: Dict[str, torch.Tensor],
    output_hidden_states: bool,
    output_attentions: bool,
):
    if spec.family == "e1":
        return model(**batch, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
    return model(**batch, output_hidden_states=output_hidden_states, output_attentions=output_attentions)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))


def maybe_tokenizer_for_embedding(spec: ModelSpec, model):
    if spec.family == "e1":
        return None
    return model.tokenizer


def flatten_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for row in rows:
        output.append(row)
    return output

