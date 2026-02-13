import contextlib
import datetime
import pathlib
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForMaskedLM

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
            return torch.float16
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
    compile_model: bool = False,
):
    if spec.family == "esm2":
        from esm2.modeling_fastesm import FastEsmConfig
        from esm2.modeling_fastesm import FastEsmForMaskedLM
        from esm2.modeling_fastesm import FastEsmModel

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
        from esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig
        from esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
        from esm_plusplus.modeling_esm_plusplus import ESMplusplusModel

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
        from e1.modeling_e1 import E1Config
        from e1.modeling_e1 import E1ForMaskedLM
        from e1.modeling_e1 import E1Model

        model_config = E1Config.from_pretrained(spec.repo_id)
        if task == "base":
            model = E1Model.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
        elif task == "masked_lm":
            model = E1ForMaskedLM.from_pretrained(spec.repo_id, config=model_config, dtype=dtype)
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
        model = torch.compile(model, dynamic=True, backend="aot_eager")
    if spec.family == "e1":
        tokenizer = None
    else:
        tokenizer = model.tokenizer
    return model, tokenizer


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
    assert tokenizer is not None, "Tokenizer is required for non-E1 families."
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

