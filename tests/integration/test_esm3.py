import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from fastplms.models.esm3.modeling_esm3 import (
    FastESM3Config,
    FastESM3GenerationConfig,
    FastESM3Model,
)
from tests.conftest import strict_fp32_matmul


def _small_config() -> FastESM3Config:
    return FastESM3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_vector_heads=8,
        num_hidden_layers=2,
    )


def _small_model() -> FastESM3Model:
    return FastESM3Model(_small_config()).eval()


def test_esm3_sequence_only_forward() -> None:
    model = _small_model()
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode():
        output = model(**batch)

    assert output.logits is not None
    assert output.function_logits is not None
    assert output.residue_logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert output.logits.shape[-1] == 64
    assert output.structure_logits.shape[-1] == 4096
    assert output.function_logits.shape[-2:] == (8, 260)
    assert output.residue_logits.shape[-1] == 1478
    assert not torch.isnan(output.logits).any()


def test_esm3_accepts_function_tokens_argument() -> None:
    model = _small_model()
    batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)
    function_tokens = batch["input_ids"].new_zeros((*batch["input_ids"].shape, 8))

    with torch.inference_mode():
        output = model(**batch, function_tokens=function_tokens)

    assert output.logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape


def test_esm3_loads_with_automodel(tmp_path: Path) -> None:
    model = _small_model()
    model.save_pretrained(tmp_path)
    assert (tmp_path / "modeling_esm3.py").is_file()
    assert (tmp_path / "modeling_fastplms.py").is_file()
    assert (tmp_path / "fastplms_bundle.py").is_file()
    assert (tmp_path / "fastplms" / "models" / "esm3" / "modeling_esm3.py").is_file()
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["auto_map"] == {
        "AutoConfig": "modeling_fastplms.FastESM3Config",
        "AutoModel": "modeling_fastplms.FastESM3Model",
    }

    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    batch = loaded.tokenize_sequences(["MKTAYIAKQ"], device=loaded.device)

    with torch.inference_mode():
        output = loaded(**batch)

    assert output.logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape


def test_esm3_seeded_generation_is_repeatable_and_preserves_context() -> None:
    model = _small_model()
    config = FastESM3GenerationConfig(num_steps=2, temperature=1.0, seed=73)

    first = model.generate("MK__A", config)
    second = model.generate("MK__A", config)

    assert isinstance(first, str)
    assert first == second
    assert len(first) == 5
    assert first[:2] == "MK"
    assert first[-1] == "A"
    assert "_" not in first


def test_esm3_saved_model_loads_without_installed_fastplms(tmp_path: Path) -> None:
    model_path = tmp_path / "saved"
    _small_model().save_pretrained(model_path)
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        class BlockInstalledFastPLMs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastplms":
                    raise ModuleNotFoundError("installed FastPLMs is blocked")
                return None

        sys.modules.pop("fastplms", None)
        sys.meta_path.insert(0, BlockInstalledFastPLMs())

        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            sys.argv[1],
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        assert type(model).__module__ == "fastplms.models.esm3.modeling_esm3"
        package_file = Path(sys.modules["fastplms"].__file__).resolve()
        assert any(part.startswith("_fastplms_runtime_") for part in package_file.parts)
        batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)
        with torch.inference_mode():
            output = model(**batch)
        assert output.logits is not None
        assert output.logits.shape[:2] == batch["input_ids"].shape
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(model_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_esm3_embed_dataset(tmp_path: Path) -> None:
    model = _small_model()
    save_path = tmp_path / "embeddings"

    result = model.embed_dataset(
        inputs=["MKTAYIAKQ", "GGGG"],
        batch_size=2,
        max_length=16,
        pooling=("mean", "cls"),
        output=save_path,
    )

    embeddings = result.as_dict(key="sequence")
    assert set(embeddings) == {"MKTAYIAKQ", "GGGG"}
    assert embeddings["MKTAYIAKQ"].shape == (128,)
    assert (save_path / "index.json").is_file()


@pytest.mark.gpu
def test_esm3_flex_matches_sdpa() -> None:
    model = _small_model().to(torch.device("cuda"))
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode(), strict_fp32_matmul():
        model.set_attn_implementation("sdpa")
        sdpa_output = model(**batch).last_hidden_state
        model.set_attn_implementation("flex_attention")
        flex_output = model(**batch).last_hidden_state

    max_abs = (sdpa_output - flex_output).float().abs().max().item()
    mse = ((sdpa_output - flex_output).float() ** 2).mean().item()
    assert max_abs < 1e-4
    assert mse < 1e-8
