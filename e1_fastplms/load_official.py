"""Load official E1 model from the e1 submodule for comparison."""

import importlib
import importlib.util
import pathlib
import shutil
import sys
import types
import torch
import torch.nn as nn


def _ensure_local_e1_module_on_path() -> pathlib.Path:
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [script_root / "e1" / "src", script_root / "E1" / "src"]

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "e1" / "src")
        candidates.append(parent / "E1" / "src")

    deduplicated: list[pathlib.Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduplicated.append(resolved)

    for candidate in deduplicated:
        package_marker = candidate / "E1" / "__init__.py"
        if package_marker.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "Unable to locate local E1 submodule. "
        f"Checked: {', '.join(str(p) for p in deduplicated)}"
    )


def _ensure_local_e1_tokenizer_json() -> None:
    e1_spec = importlib.util.find_spec("E1")
    assert e1_spec is not None, "Unable to find E1 package after path setup."
    assert e1_spec.origin is not None, "E1 package origin is required."
    e1_package_dir = pathlib.Path(e1_spec.origin).resolve().parent
    tokenizer_path = e1_package_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return

    script_root = pathlib.Path(__file__).resolve().parents[1]
    fallback_tokenizer_path = script_root / "e1_fastplms" / "tokenizer.json"
    assert fallback_tokenizer_path.exists(), (
        f"Missing fallback tokenizer at {fallback_tokenizer_path}."
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback_tokenizer_path, tokenizer_path)
    assert tokenizer_path.exists(), f"Failed to create tokenizer at {tokenizer_path}."


def _ensure_kernels_module_stub() -> None:
    if "kernels" in sys.modules:
        return
    kernels_spec = importlib.util.find_spec("kernels")
    if kernels_spec is not None:
        return

    kernels_module = types.ModuleType("kernels")

    def _missing_get_kernel(kernel_name: str):
        raise ModuleNotFoundError(
            f"No module named 'kernels' while requesting kernel '{kernel_name}'."
        )

    kernels_module.get_kernel = _missing_get_kernel
    sys.modules["kernels"] = kernels_module


class _OfficialE1ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ):
        batch = {
            "input_ids": input_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "sequence_ids": sequence_ids,
        }
        outputs = self.model(**batch, output_hidden_states=True)
        return outputs


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official E1 model from the e1 submodule.

    Args:
        reference_repo_id: e.g. "Profluent-Bio/E1-150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (official_model, batch_preparer) where batch_preparer is an E1BatchPreparer.
    The official model is E1ForMaskedLM with standard HF forward interface.
    """
    _ensure_local_e1_module_on_path()
    _ensure_local_e1_tokenizer_json()
    _ensure_kernels_module_stub()

    batch_preparer_module = importlib.import_module("E1.batch_preparer")
    modeling_module = importlib.import_module("E1.modeling")
    E1BatchPreparer = batch_preparer_module.E1BatchPreparer
    E1ForMaskedLM = modeling_module.E1ForMaskedLM

    model = E1ForMaskedLM.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
    ).eval()
    batch_preparer = E1BatchPreparer()
    wrapped = _OfficialE1ForwardWrapper(model).eval()
    return wrapped, batch_preparer
