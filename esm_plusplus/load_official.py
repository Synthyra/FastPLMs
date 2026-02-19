"""Load official ESMC model from the esm submodule for comparison."""

import importlib.util
import pathlib
import sys
import types

import torch
import torch.nn as nn


def _ensure_local_esm_module_on_path() -> pathlib.Path:
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [script_root / "esm"]

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "esm")

    deduplicated: list[pathlib.Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduplicated.append(resolved)

    for candidate in deduplicated:
        package_marker = candidate / "esm" / "__init__.py"
        if package_marker.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "Unable to locate local esm submodule. "
        f"Checked: {', '.join(str(p) for p in deduplicated)}"
    )


def _ensure_zstd_module_stub() -> None:
    if "zstd" in sys.modules:
        return
    try:
        zstd_spec = importlib.util.find_spec("zstd")
    except ValueError:
        zstd_spec = None
    if zstd_spec is not None:
        return

    zstd_module = types.ModuleType("zstd")

    def _missing_zstd_uncompress(data: bytes) -> bytes:
        raise ModuleNotFoundError(
            "No module named 'zstd'. Install zstd if compressed tensor "
            "deserialization is required."
        )

    zstd_module.ZSTD_uncompress = _missing_zstd_uncompress
    sys.modules["zstd"] = zstd_module


class _ESMCComplianceOutput:
    """Mimics HuggingFace model output so the test suite can access .logits and .hidden_states."""
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor, hidden_states: tuple):
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _OfficialESMCForwardWrapper(nn.Module):
    """Wraps official ESMC model to produce outputs compatible with our test suite."""
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        esmc_output = self.model(input_ids)
        # ESMC returns: sequence_logits, embeddings, hidden_states (stacked [n_layers, B, L, D])
        logits = esmc_output.sequence_logits
        embeddings = esmc_output.embeddings
        raw_hiddens = esmc_output.hidden_states
        # Convert stacked tensor to tuple for compatibility with hidden_states[-1]
        if raw_hiddens is not None:
            hidden_states = tuple(raw_hiddens[i] for i in range(raw_hiddens.shape[0]))
            hidden_states = hidden_states + (embeddings,)
        else:
            hidden_states = (embeddings,)
        return _ESMCComplianceOutput(
            logits=logits,
            last_hidden_state=embeddings,
            hidden_states=hidden_states,
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official ESMC model from the esm submodule.

    Args:
        reference_repo_id: e.g. "EvolutionaryScale/esmc-300m-2024-12"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    _ensure_local_esm_module_on_path()
    _ensure_zstd_module_stub()
    from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412

    if "300" in reference_repo_id:
        official_model = ESMC_300M_202412(device=str(device), use_flash_attn=False)
    elif "600" in reference_repo_id:
        official_model = ESMC_600M_202412(device=str(device), use_flash_attn=False)
    else:
        raise ValueError(f"Unsupported ESMC reference repo id: {reference_repo_id}")

    official_model = official_model.to(device=device, dtype=dtype).eval()
    tokenizer = official_model.tokenizer
    wrapped = _OfficialESMCForwardWrapper(official_model, tokenizer).to(device=device, dtype=dtype).eval()
    return wrapped, tokenizer
