"""Fresh-process import contracts for the package and model modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MODEL_MODULES = (
    "fastplms.models.ankh.modeling_ankh",
    "fastplms.models.boltz.modeling_boltz2",
    "fastplms.models.dplm.modeling_dplm",
    "fastplms.models.dplm2.modeling_dplm2",
    "fastplms.models.e1.modeling_e1",
    "fastplms.models.esm2.modeling_fastesm",
    "fastplms.models.esm3.modeling_esm3",
    "fastplms.models.esm_plusplus.modeling_esm_plusplus",
    "fastplms.models.esmfold.modeling_fast_esmfold",
    "fastplms.models.esmfold2.modeling_esmfold2",
    "fastplms.models.esmfold2.modeling_esmfold2_experimental",
)


def _run_fresh(script: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


def test_top_level_package_and_models_namespace_are_lazy() -> None:
    completed = _run_fresh(
        "import sys\n"
        "import fastplms\n"
        "import fastplms.models\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'huggingface_hub' not in sys.modules\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_model_imports_do_not_download_compile_log_or_mutate_torch_globals() -> None:
    modules = repr(MODEL_MODULES)
    completed = _run_fresh(
        f"MODULES = {modules}\n"
        "import importlib\n"
        "import logging\n"
        "import socket\n"
        "import torch\n"
        "import torch._dynamo.config as dynamo_config\n"
        "import torch._inductor.config as inductor_config\n"
        "import transformers\n"
        "import huggingface_hub\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('import attempted a forbidden side effect')\n"
        "torch.compile = forbidden\n"
        "logging.basicConfig = forbidden\n"
        "socket.create_connection = forbidden\n"
        "huggingface_hub.hf_hub_download = forbidden\n"
        "huggingface_hub.snapshot_download = forbidden\n"
        "transformers.AutoTokenizer.from_pretrained = forbidden\n"
        "def snapshot():\n"
        "    return (\n"
        "        torch.get_float32_matmul_precision(),\n"
        "        torch.backends.cuda.matmul.allow_tf32,\n"
        "        torch.backends.cudnn.allow_tf32,\n"
        "        torch.backends.cudnn.benchmark,\n"
        "        torch.backends.cudnn.deterministic,\n"
        "        repr(getattr(dynamo_config, '_config', None)),\n"
        "        repr(getattr(inductor_config, '_config', None)),\n"
        "    )\n"
        "before = snapshot()\n"
        "for module in MODULES:\n"
        "    importlib.import_module(module)\n"
        "    assert snapshot() == before, module\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_production_models_do_not_embed_unpinned_checkpoint_downloaders() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "fastplms" / "models"
    for relative_path in (
        "esm_plusplus/modeling_esm_plusplus.py",
        "esm3/modeling_esm3.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "snapshot_download" not in source
        assert "from_pretrained_esm" not in source
