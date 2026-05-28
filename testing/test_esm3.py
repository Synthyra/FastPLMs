import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from fastplms.esm3 import modeling_esm3
from fastplms.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from testing.conftest import strict_fp32_matmul


HUB_AUTO_MAP = {
    "AutoConfig": "modeling_esm3.FastESM3Config",
    "AutoModel": "modeling_esm3.FastESM3Model",
    "AutoModelForMaskedLM": "modeling_esm3.FastESM3Model",
}


def _small_config() -> FastESM3Config:
    config = FastESM3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_vector_heads=8,
        num_hidden_layers=2,
    )
    config.architectures = ["FastESM3Model"]
    config.auto_map = HUB_AUTO_MAP
    return config


def _small_model() -> FastESM3Model:
    try:
        model = FastESM3Model(_small_config()).eval()
    except ModuleNotFoundError as exc:
        pytest.skip(f"Biohub ESM3 runtime dependency is unavailable: {exc}")
    return model


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
    shutil.copyfile(Path(modeling_esm3.__file__), tmp_path / "modeling_esm3.py")

    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    batch = loaded.tokenize_sequences(["MKTAYIAKQ"], device=loaded.device)

    with torch.inference_mode():
        output = loaded(**batch)

    assert output.logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape


def test_esm3_embed_dataset(tmp_path: Path) -> None:
    model = _small_model()
    save_path = tmp_path / "embeddings.pth"

    embeddings = model.embed_dataset(
        sequences=["MKTAYIAKQ", "GGGG"],
        batch_size=2,
        max_len=16,
        pooling_types=["mean", "cls"],
        save=True,
        save_path=str(save_path),
    )

    assert set(embeddings) == {"MKTAYIAKQ", "GGGG"}
    assert embeddings["MKTAYIAKQ"].shape == (128,)
    assert save_path.exists()


def test_esm3_flex_matches_sdpa() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Flex attention ESM3 equivalence is validated on CUDA.")
    model = _small_model().to(torch.device("cuda"))
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode(), strict_fp32_matmul():
        model.attn_backend = "sdpa"
        sdpa_output = model(**batch).last_hidden_state
        try:
            model.attn_backend = "flex"
        except AssertionError as exc:
            pytest.skip(f"Flex attention is unavailable: {exc}")
        flex_output = model(**batch).last_hidden_state

    max_abs = (sdpa_output - flex_output).float().abs().max().item()
    mse = ((sdpa_output - flex_output).float() ** 2).mean().item()
    assert max_abs < 1e-4
    assert mse < 1e-8
