"""Fail closed when ESM++ runtime functions overlap the Biohub parity oracle."""

from __future__ import annotations

import ast
import copy
import importlib.util
import pytest
import torch
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType

from fastplms.models.esm_plusplus.modeling_esm_plusplus import PreTrainedESMplusplusModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    RotaryEmbedding as FastRotaryEmbedding,
)


ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL = ROOT / "src/fastplms/models/esm_plusplus/modeling_esm_plusplus.py"
LOCAL_SAE = ROOT / "src/fastplms/models/esm_plusplus/modeling_esm_plusplus_sae.py"
UPSTREAM_ROTARY = ROOT / "vendor/upstream/biohub-esm/esm/layers/rotary.py"
UPSTREAM_TOKENIZER = ROOT / "vendor/upstream/biohub-esm/esm/tokenization/sequence_tokenizer.py"
UPSTREAM_SAE = (
    ROOT / "vendor/upstream/biohub-transformers/src/transformers/models/esmc/modeling_esmc_sae.py"
)
MAX_FUNCTION_SIMILARITY = 0.75

# These functions implement the same public contracts, but the repository source
# must remain independently maintained. Function-level comparisons prevent
# unrelated model code from diluting a copied implementation's similarity.
# ESMplusplusSAELayer.__init__ is excluded on purpose: parameter names and shapes
# are the published shard format, so they must converge for the weights to load.
SOURCE_PAIRS = (
    (
        LOCAL_MODEL,
        "EsmSequenceTokenizer.__init__",
        UPSTREAM_TOKENIZER,
        "EsmSequenceTokenizer.__init__",
    ),
    (
        LOCAL_MODEL,
        "RotaryEmbedding.__init__",
        UPSTREAM_ROTARY,
        "RotaryEmbedding.__init__",
    ),
    (
        LOCAL_MODEL,
        "RotaryEmbedding._update_cos_sin_cache",
        UPSTREAM_ROTARY,
        "RotaryEmbedding._update_cos_sin_cache",
    ),
    (
        LOCAL_MODEL,
        "apply_rotary_emb_torch",
        UPSTREAM_ROTARY,
        "apply_rotary_emb_torch",
    ),
    (
        LOCAL_SAE,
        "ESMplusplusSAELayer.forward",
        UPSTREAM_SAE,
        "_ESMCSAELayer.forward",
    ),
    (
        LOCAL_SAE,
        "ESMplusplusSAELayer.get_sae_output",
        UPSTREAM_SAE,
        "_ESMCSAELayer.get_sae_output",
    ),
)


def _function(path: Path, qualified_name: str) -> ast.FunctionDef:
    body: list[ast.stmt] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
    selected: ast.AST | None = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part
            ),
            None,
        )
        assert selected is not None, f"{qualified_name!r} is absent from {path}"
        body = selected.body
    assert isinstance(selected, ast.FunctionDef)
    return selected


def _normalized_ast_lines(node: ast.FunctionDef) -> list[str]:
    normalized = copy.deepcopy(node)
    normalized.name = "function"
    normalized.decorator_list = []
    normalized.returns = None
    for argument in (
        *normalized.args.posonlyargs,
        *normalized.args.args,
        *normalized.args.kwonlyargs,
    ):
        argument.annotation = None
    if normalized.args.vararg is not None:
        normalized.args.vararg.annotation = None
    if normalized.args.kwarg is not None:
        normalized.args.kwarg.annotation = None
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body.pop(0)
    ast.fix_missing_locations(normalized)
    return [
        " ".join(line.strip().split())
        for line in ast.unparse(normalized).splitlines()
        if line.strip()
    ]


def _load_upstream_rotary() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_fastplms_test_biohub_rotary",
        UPSTREAM_ROTARY,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("local_path", "local_name", "upstream_path", "upstream_name"),
    SOURCE_PAIRS,
    ids=[local_name for _, local_name, _, _ in SOURCE_PAIRS],
)
def test_esmplusplus_functions_are_independently_implemented(
    local_path: Path,
    local_name: str,
    upstream_path: Path,
    upstream_name: str,
) -> None:
    assert upstream_path.is_file(), f"pinned Biohub source is missing: {upstream_path}"
    local_lines = _normalized_ast_lines(_function(local_path, local_name))
    upstream_lines = _normalized_ast_lines(_function(upstream_path, upstream_name))
    similarity = SequenceMatcher(
        None,
        local_lines,
        upstream_lines,
        autojunk=False,
    ).ratio()
    assert similarity < MAX_FUNCTION_SIMILARITY, (
        f"{local_name} has normalized AST similarity {similarity:.3f} to "
        f"{upstream_path.relative_to(ROOT)}::{upstream_name}"
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("interleaved", (False, True))
def test_reimplemented_rotary_is_exact(dtype: torch.dtype, interleaved: bool) -> None:
    upstream_class = _load_upstream_rotary().RotaryEmbedding
    local = FastRotaryEmbedding(dim=8, interleaved=interleaved).eval()
    upstream = upstream_class(dim=8, interleaved=interleaved).eval()

    generator = torch.Generator().manual_seed(13)
    # q: (2, 17, 3, 8)
    q = torch.randn((2, 17, 3, 8), generator=generator, dtype=dtype)
    # k: (2, 17, 3, 8)
    k = torch.randn((2, 17, 3, 8), generator=generator, dtype=dtype)
    local_q, local_k = local(q, k)
    upstream_q, upstream_k = upstream(q, k)

    assert torch.equal(local.inv_freq, upstream.inv_freq)
    assert torch.equal(local._cos_cached, upstream._cos_cached)
    assert torch.equal(local._sin_cached, upstream._sin_cached)
    assert torch.equal(local_q, upstream_q)
    assert torch.equal(local_k, upstream_k)
    assert local.state_dict().keys() == upstream.state_dict().keys()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_reimplemented_scaled_rotary_cache_is_exact(dtype: torch.dtype) -> None:
    upstream_class = _load_upstream_rotary().RotaryEmbedding
    local = FastRotaryEmbedding(dim=8, scale_base=512).eval()
    upstream = upstream_class(dim=8, scale_base=512).eval()

    local._update_cos_sin_cache(19, device=torch.device("cpu"), dtype=dtype)
    upstream._update_cos_sin_cache(19, device=torch.device("cpu"), dtype=dtype)
    for name in ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached"):
        assert torch.equal(getattr(local, name), getattr(upstream, name))
    assert local.state_dict().keys() == upstream.state_dict().keys()
    assert torch.equal(local.state_dict()["scale"], upstream.state_dict()["scale"])


@pytest.mark.gpu
def test_reimplemented_rotary_matches_transformers_cuda_policy() -> None:
    assert torch.cuda.is_available(), "ESM++ rotary parity requires CUDA"
    upstream_class = _load_upstream_rotary().RotaryEmbedding
    # local: (...)
    local = FastRotaryEmbedding(dim=64).eval().to("cuda")
    # upstream: (...)
    upstream = upstream_class(dim=64).eval().to("cuda")

    # The original Biohub SDK migrates CPU-computed frequencies. The pinned
    # Biohub Transformers oracle instead recomputes them on CUDA after a device
    # move. Reproduce that public AutoModel policy on the independent upstream
    # rotary implementation before comparing outputs.
    # cpu_migrated: (...)
    cpu_migrated = upstream.inv_freq.clone()
    cuda_native = upstream._compute_inv_freq(torch.device("cuda"))
    assert not torch.equal(cpu_migrated, cuda_native)
    upstream.register_buffer("inv_freq", cuda_native, persistent=False)
    upstream._seq_len_cached = 0
    upstream._cos_cached = None
    upstream._sin_cached = None

    generator = torch.Generator(device="cuda").manual_seed(29)
    # q: (3, 65, 8, 64)
    q = torch.randn((3, 65, 8, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
    # k: (3, 65, 8, 64)
    k = torch.randn((3, 65, 8, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
    local_q, local_k = local(q, k)
    upstream_q, upstream_k = upstream(q, k)

    assert torch.equal(local.inv_freq, cuda_native)
    assert torch.equal(local._cos_cached, upstream._cos_cached)
    assert torch.equal(local._sin_cached, upstream._sin_cached)
    assert torch.equal(local_q, upstream_q)
    assert torch.equal(local_k, upstream_k)


def test_esmplusplus_advertises_only_pinned_flash_kernels() -> None:
    assert PreTrainedESMplusplusModel._supports_flash_attn is True
    assert PreTrainedESMplusplusModel._supports_flash_attn_2 is True
    assert PreTrainedESMplusplusModel._supports_flash_attn_3 is True
    assert PreTrainedESMplusplusModel._fastplms_attention_implementations == (
        "eager",
        "sdpa",
        "flex_attention",
        "flash_attention_2",
        "flash_attention_3",
    )
