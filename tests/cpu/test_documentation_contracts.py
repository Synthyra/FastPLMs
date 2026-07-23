"""Execute curated offline examples and the Python migration snippets."""

from __future__ import annotations

import ast
import json
import sqlite3
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import transformers

from examples import (
    _runtime,
    ankh_embeddings,
    artifact_loading,
    attention_switching,
    e1_rag,
    embedding_and_retrieval,
    generation,
    structure_preparation,
    task_heads,
    ttt,
)
from fastplms.embeddings import (
    EmbeddingRecord,
    EmbeddingResult,
    embed_dataset,
    load_sqlite_result,
    save_safetensors_result,
    save_sqlite_result,
)
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esmfold2 import esmfold2_types
from tests.integration import test_ttt as ttt_contracts
from tests.unit import test_ankh_cpu_contract as ankh_contracts
from tests.unit import test_e1_cache_contract as e1_contracts
from tests.unit import test_embeddings_api as embedding_contracts

_ROOT = Path(__file__).resolve().parents[2]
_CURATED_EXAMPLE_CPU_CASES: dict[str, tuple[str, ...]] = {
    "embedding_and_retrieval.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_embedding_and_retrieval_example_executes_with_ordered_sqlite",
    ),
    "attention_switching.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_attention_switching_main_executes_optimized_and_masked_fallback",
    ),
    "ankh_embeddings.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_ankh_embedding_example_executes_encoder_and_decoder_layers",
    ),
    "generation.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_dplm_branch_offline",
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_dplm2_branch_offline",
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_esm3_trace",
    ),
    "e1_rag.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_e1_rag_example_executes_local_msa_and_shared_persistence",
    ),
    "ttt.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_ttt_example_executes_seeded_adapt_save_and_reset",
    ),
    "structure_preparation.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_structure_preparation_example_executes_each_public_branch",
    ),
    "artifact_loading.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_artifact_loading_example_executes_local_only_autoconfig",
    ),
    "task_heads.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_task_head_example_executes_all_advertised_heads_offline",
    ),
    "fine_tuning.py": (
        "tests/cpu/test_peft_contracts.py::"
        "test_fine_tuning_main_wires_both_tasks_without_external_io",
        "tests/cpu/test_peft_contracts.py::"
        "test_shipped_collators_create_tokenizer_aware_sequence_and_pair_batches",
        "tests/cpu/test_peft_contracts.py::"
        "test_shipped_initializer_drives_one_peft_step_and_atomic_final_reload",
    ),
    "binder_design_fastplms.py": (
        "tests/cpu/test_structure_contracts.py::"
        "test_public_binder_workflow_pads_heterogeneous_prepared_atoms_without_truncation",
        "tests/cpu/test_structure_contracts.py::"
        "test_binder_example_main_wires_explicit_offline_cli_arguments",
        "tests/cpu/test_structure_contracts.py::"
        "test_binder_structure_loss_is_finite_and_differentiable",
    ),
}


def _patch_transformers_imports(
    monkeypatch: pytest.MonkeyPatch,
    **replacements: object,
) -> None:
    """Patch ``from transformers import ...`` through its lazy module proxy."""

    proxy = ModuleType("transformers")
    proxy.__dict__.update(transformers.__dict__)
    proxy.__dict__.update(replacements)
    proxy.__getattr__ = lambda name: getattr(transformers, name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", proxy)


def _tiny_esm3_model() -> FastESM3Model:
    return FastESM3Model(
        FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    ).eval()


class _TinyTokenizer:
    all_special_ids = (0, 1)
    vocab_size = 16
    name_or_path = "cpu-doc-tokenizer"

    def __call__(self, sequences, **_kwargs):
        rows = [[2 + (ord(character) % 10) for character in value] + [1] for value in sequences]
        width = max(map(len, rows))
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0).long(),
        }


def _python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    for section in text.split("```python")[1:]:
        blocks.append(section.split("```", maxsplit=1)[0].strip())
    return blocks


def test_embedding_and_retrieval_example_executes_with_ordered_sqlite(tmp_path) -> None:
    model = embedding_contracts.SyntheticEmbeddingModel()
    model.embed_dataset = lambda inputs, **kwargs: embed_dataset(  # type: ignore[attr-defined]
        model, inputs, **kwargs
    )
    output = tmp_path / "example.sqlite"
    result = embedding_and_retrieval.run_embeddings(
        model,
        _TinyTokenizer(),
        {"a": "ACD", "b": "GG"},
        output=output,
        output_format="sqlite",
        max_length=16,
    )

    assert [record.id for record in result] == ["a", "b"]
    selected = load_sqlite_result(output, record_ids=["b", "a", "b"])
    assert [record.id for record in selected] == ["b", "a", "b"]


@pytest.mark.parametrize(
    "retrieval_arguments",
    (
        ("--select-id", "0"),
        (
            "--output",
            "embeddings.safetensors",
            "--format",
            "safetensors",
            "--select-id",
            "0",
        ),
    ),
)
def test_embedding_retrieval_selection_requires_sqlite_output(
    tmp_path,
    retrieval_arguments: tuple[str, ...],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        SystemExit,
        match="--select-id requires both --output and --format sqlite",
    ):
        embedding_and_retrieval.main([str(artifact), "--sequence", "ACD", *retrieval_arguments])


@pytest.mark.parametrize(
    ("module", "arguments"),
    (
        (embedding_and_retrieval, ["artifact", "--sequence", "ACD"]),
        (ankh_embeddings, ["artifact"]),
        (generation, ["dplm", "artifact"]),
        (e1_rag, ["artifact", "query.a3m"]),
        (ttt, ["artifact", "adapted"]),
    ),
)
def test_sequence_examples_share_explicit_device_dtype_contract(module, arguments) -> None:
    parsed = module.build_parser().parse_args([*arguments, "--device", "cpu", "--dtype", "float32"])
    assert parsed.device == "cpu"
    assert parsed.dtype == "float32"
    device, dtype = _runtime.resolve_execution(parsed.device, parsed.dtype)
    assert device == torch.device("cpu")
    assert dtype is torch.float32


def test_attention_switching_main_executes_optimized_and_masked_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torch.nn.functional as functional

    from fastplms import attention as attention_module

    model = ankh_contracts.FastAnkhModel(ankh_contracts._config(attn_backend="sdpa")).eval()
    artifact = tmp_path / "ankh-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")
    model_loads: list[tuple[object, dict[str, object]]] = []
    tokenizer_loads: list[tuple[object, dict[str, object]]] = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            del cls
            model_loads.append((source, kwargs))
            return model

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            del cls
            tokenizer_loads.append((source, kwargs))
            return _TinyTokenizer()

    sdpa_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_sdpa = functional.scaled_dot_product_attention

    def tracked_sdpa(query, key, value, *args, **kwargs):
        sdpa_calls.append((tuple(query.shape), tuple(key.shape)))
        return original_sdpa(query, key, value, *args, **kwargs)

    cleared: list[bool] = []
    _patch_transformers_imports(
        monkeypatch,
        AutoModel=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setattr(functional, "scaled_dot_product_attention", tracked_sdpa)
    monkeypatch.setattr(
        attention_module,
        "clear_flex_attention_caches",
        lambda: cleared.append(True),
    )

    result = attention_switching.main([str(artifact), "--backend", "sdpa"])
    output = capsys.readouterr().out

    assert result == 0
    assert sdpa_calls
    assert "optimized (2," in output
    assert "fallback (2," in output
    assert "warning" in output and "sdpa" in output and "eager" in output
    assert model.attn_backend == "sdpa"
    assert model_loads == [
        (
            artifact.resolve(),
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "attn_implementation": "sdpa",
                "dtype": torch.float32,
            },
        )
    ]
    assert tokenizer_loads == [
        (
            artifact.resolve(),
            {"trust_remote_code": True, "local_files_only": True},
        )
    ]
    assert cleared == [True]


def test_attention_example_detects_and_does_not_repair_fallback_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ankh_contracts.FastAnkhModel(ankh_contracts._config(attn_backend="sdpa")).eval()
    artifact = tmp_path / "ankh-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            return model

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            return object()

    output = SimpleNamespace(last_hidden_state=torch.zeros((2, 3, 8)))
    monkeypatch.setattr(
        attention_switching,
        "run_optimized_attention_example",
        lambda *_args, **_kwargs: output,
    )

    def mutate_during_fallback(model: Any, *_args: Any, **_kwargs: Any) -> Any:
        model.attn_backend = "eager"
        return output, ["configured=sdpa effective=eager"]

    monkeypatch.setattr(
        attention_switching,
        "run_attention_example",
        mutate_during_fallback,
    )
    _patch_transformers_imports(
        monkeypatch,
        AutoModel=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )

    with pytest.raises(RuntimeError, match="eager fallback mutated"):
        attention_switching.main([str(artifact), "--backend", "sdpa"])
    assert model.attn_backend == "eager"


@pytest.mark.parametrize("backend", ("flash_attention_2", "flash_attention_3"))
def test_attention_switching_flash_contract_fails_before_model_loading(
    backend: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")

    class ForbiddenAutoModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            del cls, args, kwargs
            raise AssertionError("invalid FlashAttention execution reached model loading")

    _patch_transformers_imports(monkeypatch, AutoModel=ForbiddenAutoModel)

    with pytest.raises(SystemExit, match=rf"{backend} requires a CUDA device"):
        attention_switching.main(
            [
                str(artifact),
                "--backend",
                backend,
                "--device",
                "cpu",
                "--dtype",
                "bfloat16",
            ]
        )
    with pytest.raises(SystemExit, match=rf"{backend} requires --dtype bfloat16"):
        attention_switching.main(
            [
                str(artifact),
                "--backend",
                backend,
                "--device",
                "cuda:0",
                "--dtype",
                "float32",
            ]
        )


def test_task_head_example_executes_all_advertised_heads_offline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fastplms.models.esm2.modeling_fastesm import (
        FastEsmConfig,
        FastEsmForMaskedLM,
        FastEsmForSequenceClassification,
        FastEsmForTokenClassification,
    )

    artifact = tmp_path / "esm2-artifact"

    def config() -> FastEsmConfig:
        return FastEsmConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
            pad_token_id=1,
            mask_token_id=5,
            eos_token_id=2,
            position_embedding_type="absolute",
            attn_backend="eager",
        )

    FastEsmForMaskedLM(config()).save_pretrained(artifact, safe_serialization=True)

    class TaskTokenizer:
        all_special_ids = (0, 1, 2, 5)
        mask_token_id = 5

        def __call__(self, sequences, **_kwargs):
            available = (3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            alphabet = dict(zip("ACDEFGHIKLMN", available, strict=True))
            rows = [[0, *(alphabet[residue] for residue in sequence), 2] for sequence in sequences]
            width = max(map(len, rows))
            input_ids = torch.tensor([row + [1] * (width - len(row)) for row in rows])
            return {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(1).long(),
            }

    class FakeMaskedLM:
        @classmethod
        def from_pretrained(cls, source: Any, **kwargs: Any) -> Any:
            return FastEsmForMaskedLM.from_pretrained(source, **kwargs)

    class FakeSequenceClassification:
        @classmethod
        def from_pretrained(cls, source: Any, **kwargs: Any) -> Any:
            return FastEsmForSequenceClassification.from_pretrained(source, **kwargs)

    class FakeTokenClassification:
        @classmethod
        def from_pretrained(cls, source: Any, **kwargs: Any) -> Any:
            return FastEsmForTokenClassification.from_pretrained(source, **kwargs)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            return TaskTokenizer()

    _patch_transformers_imports(
        monkeypatch,
        AutoModelForMaskedLM=FakeMaskedLM,
        AutoModelForSequenceClassification=FakeSequenceClassification,
        AutoModelForTokenClassification=FakeTokenClassification,
        AutoTokenizer=FakeTokenizer,
    )

    result = task_heads.main(
        [
            str(artifact),
            "--sequence",
            "ACDE",
            "--sequence",
            "AC",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--attn-backend",
            "eager",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert result == 0
    assert summary["masked_lm"]["status"] == "checkpoint-provided pretrained head"
    assert summary["contacts"]["status"] == "checkpoint-provided pretrained head"
    assert summary["contacts"]["finite"] is True
    assert summary["sequence_classification"]["status"] == ("base weights + untrained task head")
    assert summary["token_classification"]["status"] == ("base weights + untrained task head")


def test_task_head_example_rejects_missing_or_nonfinite_checkpoint_heads() -> None:
    with pytest.raises(RuntimeError, match="complete checkpoint-provided"):
        task_heads._require_checkpoint_heads(
            {"missing_keys": ["lm_head.decoder.weight", "esm.contact_head.regression.weight"]},
            ("lm_head.", "esm.contact_head."),
        )
    with pytest.raises(RuntimeError, match="Contact predictions contained non-finite"):
        task_heads._require_finite_tensor(
            "Contact predictions",
            torch.tensor([float("nan")]),
        )


def test_ankh_embedding_example_executes_encoder_and_decoder_layers() -> None:
    config = ankh_contracts._config()
    tokenizer = ankh_contracts._TinyTokenizer()
    encoder = ankh_contracts.FastAnkhModel(config).eval()
    seq2seq = ankh_contracts.FastAnkhForConditionalGeneration(ankh_contracts._config()).eval()
    encoder.tokenizer = tokenizer
    seq2seq.tokenizer = tokenizer

    encoder_final, encoder_all, decoder_final = ankh_embeddings.extract_ankh_layers(
        encoder,
        seq2seq,
        tokenizer,
        ["ACD", "EF"],
        ["AS", "D"],
    )

    assert encoder_final.metadata["hidden_state_source"] == "encoder"
    assert encoder_all[0].load_tensor().shape[0] == config.num_layers + 1
    assert decoder_final.metadata["hidden_state_source"] == "decoder"
    assert decoder_final.metadata["decoder_input_fingerprint"]

    tokenizer_calls: list[tuple[str, bool]] = []
    generation_arguments: dict[str, Any] = {}

    class PromptTokenizer:
        def __call__(
            self,
            text: str,
            *,
            return_tensors: str,
            add_special_tokens: bool = True,
        ) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            tokenizer_calls.append((text, add_special_tokens))
            ids = [[2, 3, 4, 1]] if add_special_tokens else [[9, 7]]
            input_ids = torch.tensor(ids)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class PromptedSeq2Seq:
        config = SimpleNamespace(decoder_start_token_id=0)
        device = torch.device("cpu")

        def generate(self, **kwargs: Any) -> torch.Tensor:
            generation_arguments.update(kwargs)
            return kwargs["decoder_input_ids"]

    generated = ankh_embeddings.generate_ankh_task(
        PromptedSeq2Seq(),
        PromptTokenizer(),
        "ACD",
        "M<extra_id_0>",
        max_new_tokens=3,
    )

    assert tokenizer_calls == [("ACD", True), ("M<extra_id_0>", False)]
    assert torch.equal(generated, torch.tensor([[0, 9, 7]]))
    assert torch.equal(generation_arguments["input_ids"], torch.tensor([[2, 3, 4, 1]]))
    assert generation_arguments["do_sample"] is False
    assert generation_arguments["num_beams"] == 1
    assert generation_arguments["use_cache"] is True
    assert generation_arguments["max_new_tokens"] == 3


class _GenerationExampleTokenizer:
    def __init__(self) -> None:
        self.encoded_sequences: list[str] = []

    def __call__(self, sequence: str, *, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.encoded_sequences.append(sequence)
        return {
            "input_ids": torch.tensor(
                [[0, *([3] * len(sequence)), 2]],
                dtype=torch.long,
            )
        }

    def get_vocab(self) -> dict[str, int]:
        return {
            "<cls_struct>": 10,
            "<mask_struct>": 11,
            "<eos_struct>": 12,
            "<cls_aa>": 20,
            "<mask_aa>": 21,
            "<eos_aa>": 22,
        }


class _GenerationExampleModel:
    device = torch.device("cpu")

    def __init__(self, *, multimodal: bool) -> None:
        self.multimodal = multimodal
        self.calls: list[tuple[torch.Tensor, dict[str, Any]]] = []

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        self.calls.append((input_ids.clone(), dict(kwargs)))
        sampled = torch.randint(0, 100, (1, 1), device=input_ids.device)
        output = torch.cat((input_ids, sampled), dim=1)
        return {"output_tokens": output} if self.multimodal else output


def test_generation_example_executes_seeded_dplm_branch_offline() -> None:
    generation.configure_offline()
    model = _GenerationExampleModel(multimodal=False)
    tokenizer = _GenerationExampleTokenizer()
    caller_rng = torch.random.get_rng_state()

    first = generation.generate_dplm(model, tokenizer, length=3, steps=2, seed=19)
    second = generation.generate_dplm(model, tokenizer, length=3, steps=2, seed=19)

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert tokenizer.encoded_sequences == ["AAA", "AAA"]
    assert len(model.calls) == 2
    for input_ids, kwargs in model.calls:
        assert torch.equal(input_ids, torch.tensor([[0, 3, 3, 3, 2]]))
        assert kwargs == {
            "max_iter": 2,
            "sampling_strategy": "argmax",
            "disable_resample": True,
        }
    assert generation.os.environ["HF_HUB_OFFLINE"] == "1"
    assert generation.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_generation_example_executes_seeded_dplm2_branch_offline() -> None:
    generation.configure_offline()
    model = _GenerationExampleModel(multimodal=True)
    tokenizer = _GenerationExampleTokenizer()
    caller_rng = torch.random.get_rng_state()

    first = generation.generate_dplm2(model, tokenizer, length=3, steps=2, seed=23)
    second = generation.generate_dplm2(model, tokenizer, length=3, steps=2, seed=23)

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert len(model.calls) == 2
    expected = torch.tensor([[10, 11, 11, 11, 12, 20, 21, 21, 21, 22]])
    for input_ids, kwargs in model.calls:
        assert torch.equal(input_ids, expected)
        assert kwargs == {
            "max_iter": 2,
            "sampling_strategy": "argmax",
            "unmasking_strategy": "deterministic",
        }
    assert generation.os.environ["HF_HUB_OFFLINE"] == "1"
    assert generation.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_generation_example_executes_seeded_esm3_trace() -> None:
    model = _tiny_esm3_model()
    first = generation.generate_esm3(model, "MK__A", steps=2, seed=19)
    second = generation.generate_esm3(model, "MK__A", steps=2, seed=19)

    assert first == second
    assert first.startswith("MK") and first.endswith("A")


def test_generation_example_preserves_every_esm3_multimodal_track() -> None:
    model = _tiny_esm3_model()
    request = generation.build_esm3_multimodal_request(model, "M_A")
    observed: list[dict[str, torch.Tensor]] = []
    original_forward = model.forward

    def capture_forward(*args: Any, **kwargs: Any) -> Any:
        observed.append(
            {
                name: value.detach().clone()
                for name, value in kwargs.items()
                if torch.is_tensor(value)
            }
        )
        return original_forward(*args, **kwargs)

    model.forward = capture_forward
    output = generation.generate_esm3(model, request, steps=1, seed=23)

    assert torch.is_tensor(output)
    assert len(observed) == 1
    assert set(request).issubset(observed[0])
    for name, expected in request.items():
        torch.testing.assert_close(observed[0][name], expected, equal_nan=True)


def test_e1_rag_example_executes_local_msa_and_shared_persistence(tmp_path) -> None:
    sequence = "ACDEFG"
    a3m_path = tmp_path / "query.a3m"
    a3m_path.write_text(">query\nACDEFG\n>near\nACDEYG\n", encoding="utf-8")
    output = tmp_path / "e1.sqlite"
    model = e1_contracts.E1ForMaskedLM(e1_contracts._tiny_e1_config()).eval()

    result = e1_rag.embed_local_msa(
        model,
        sequence,
        a3m_path,
        output=output,
        output_format="sqlite",
        seed=7,
    )

    assert [(record.id, record.sequence) for record in result] == [
        ("0", sequence),
        ("1", sequence),
    ]
    assert [record.id for record in load_sqlite_result(output)] == ["0", "1"]


def test_ttt_example_executes_seeded_adapt_save_and_reset(tmp_path) -> None:
    model = ttt_contracts.DummyPretrainedTTTModel(ttt_contracts.DummyPretrainedTTTConfig())
    metrics = ttt.adapt_and_save(model, "ACDE", tmp_path / "adapted", seed=7)

    assert len(metrics["losses"]) == 3
    assert (tmp_path / "adapted" / "config.json").is_file()
    assert not list(tmp_path.glob(".adapted-*"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ttt.adapt_and_save(model, "ACDE", tmp_path / "adapted", seed=7)


def test_ttt_example_resets_after_a_mutating_adaptation_failure(tmp_path) -> None:
    class FailingModel:
        def __init__(self) -> None:
            self.mutated = False
            self.reset_calls = 0

        def ttt(self, **_kwargs: Any) -> None:
            self.mutated = True
            raise RuntimeError("adaptation failed after mutation")

        def ttt_reset(self) -> None:
            self.mutated = False
            self.reset_calls += 1

    model = FailingModel()
    with pytest.raises(RuntimeError, match="adaptation failed after mutation"):
        ttt.adapt_and_save(model, "ACDE", tmp_path / "failed", seed=7)

    assert model.mutated is False
    assert model.reset_calls == 1
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_ttt_example_cleans_staging_even_when_reset_fails(tmp_path) -> None:
    class ResetFailingModel:
        def ttt(self, **_kwargs: Any) -> None:
            raise RuntimeError("adaptation failure")

        def ttt_reset(self) -> None:
            raise ValueError("reset failure")

    with pytest.raises(ValueError, match="reset failure"):
        ttt.adapt_and_save(
            ResetFailingModel(),
            "ACDE",
            tmp_path / "failed-reset",
            seed=7,
        )

    assert not (tmp_path / "failed-reset").exists()
    assert not list(tmp_path.glob(".failed-reset-*"))


def test_ttt_example_refuses_source_and_existing_destinations(tmp_path) -> None:
    artifact = tmp_path / "source"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="must not be the source artifact"):
        ttt.main([str(artifact), str(artifact)])

    destination = tmp_path / "adapted"
    destination.mkdir()
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        ttt.main([str(artifact), str(destination)])


def test_structure_preparation_example_executes_each_public_branch() -> None:
    class FakeBoltz:
        def predict_structure(self, **kwargs: Any) -> Any:
            return kwargs

    class FakeESMFold:
        def fold_protein(self, sequence: str) -> Any:
            return {"sequence": sequence}

    boltz = structure_preparation.run_structure_helper(FakeBoltz(), "boltz2", "ACD", 7)
    esmfold = structure_preparation.run_structure_helper(FakeESMFold(), "esmfold", "ACD", 7)

    assert boltz["seed"] == 7
    assert esmfold == {"sequence": "ACD"}

    request = structure_preparation.build_esmfold2_conditioned_complex(esmfold2_types)
    assert len(request.sequences) == 5
    assert (
        sum(isinstance(sequence, esmfold2_types.ProteinInput) for sequence in request.sequences)
        == 2
    )
    assert any(isinstance(sequence, esmfold2_types.RNAInput) for sequence in request.sequences)
    assert any(isinstance(sequence, esmfold2_types.DNAInput) for sequence in request.sequences)
    assert any(isinstance(sequence, esmfold2_types.LigandInput) for sequence in request.sequences)
    protein_with_msa = request.sequences[0]
    assert isinstance(protein_with_msa, esmfold2_types.ProteinInput)
    assert isinstance(protein_with_msa.msa, esmfold2_types.MSA)
    modified_protein = request.sequences[1]
    assert isinstance(modified_protein, esmfold2_types.ProteinInput)
    assert modified_protein.modifications == [esmfold2_types.Modification(position=0, ccd="MSE")]
    assert request.covalent_bonds and len(request.covalent_bonds) == 1
    assert request.distogram_conditioning and len(request.distogram_conditioning) == 1

    class FakeESMFold2:
        input_types = esmfold2_types

        def prepare_structure_input(self, prepared: Any, *, seed: int) -> Any:
            assert seed == 11
            if prepared.pocket is not None:
                raise NotImplementedError("Pocket conditioning is not implemented.")
            return prepared

    rejection = structure_preparation.verify_esmfold2_pocket_rejection(
        FakeESMFold2(),
        seed=11,
    )
    assert "Pocket conditioning" in rejection


def test_artifact_loading_example_executes_local_only_autoconfig(tmp_path) -> None:
    config = transformers.BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    config.save_pretrained(tmp_path)
    loaded = artifact_loading.load_local_artifact(tmp_path, "AutoConfig")

    assert loaded.model_type == config.model_type
    assert artifact_loading.require_local_artifact(str(tmp_path)) == tmp_path.resolve()


def test_migration_python_snippets_execute_against_tiny_offline_objects(
    monkeypatch,
    tmp_path,
) -> None:
    blocks = _python_blocks(_ROOT / "docs" / "migration.md")
    assert len(blocks) == 6

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            del cls, args, kwargs
            return SimpleNamespace(set_attn_implementation=lambda _backend: None)

    class FakeSeq2Seq(FakeAutoModel):
        pass

    _patch_transformers_imports(
        monkeypatch,
        AutoModel=FakeAutoModel,
        AutoModelForSeq2SeqLM=FakeSeq2Seq,
    )
    exec(blocks[0], {"model_id": "local", "__builtins__": __builtins__})

    model = embedding_contracts.SyntheticEmbeddingModel()
    model.embed_dataset = lambda inputs, **kwargs: embed_dataset(  # type: ignore[attr-defined]
        model, inputs, **kwargs
    )
    inputs = ["ACD", "GG", "ACD"]
    cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    namespace = {"model": model, "inputs": inputs, "__builtins__": __builtins__}
    exec(blocks[1], namespace)
    result = namespace["result"]
    exec(blocks[2], {"result": result, "__builtins__": __builtins__})

    records = EmbeddingResult(
        [
            EmbeddingRecord("a", "AC", torch.tensor([1.0, 2.0])),
            EmbeddingRecord("b", "GG", torch.tensor([3.0, 4.0])),
        ],
        {"complete": True, "run_fingerprint": "migration-example-fixture"},
    )
    (tmp_path / "embeddings.sqlite").unlink()
    save_safetensors_result(records, tmp_path / "embeddings")
    save_sqlite_result(records, tmp_path / "embeddings.sqlite")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    shape = tuple(tensor.shape)
    blob = struct.pack(f"<BBi{len(shape)}i", 1, 2, len(shape), *shape) + tensor.numpy().tobytes()
    with sqlite3.connect(tmp_path / "legacy.db") as connection:
        connection.execute(
            "CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        connection.execute("INSERT INTO embeddings VALUES (?, ?)", ("AC", blob))
    torch.save({"AC": tensor}, tmp_path / "legacy.pth")
    exec(blocks[3], {"__builtins__": __builtins__})

    exec(
        blocks[4],
        {
            "ankh_id": "local",
            "ankh_revision": "a" * 40,
            "__builtins__": __builtins__,
        },
    )

    class FakeEmbeddingView:
        def embed_dataset(self, values: Any, **kwargs: Any) -> Any:
            return {"values": list(values), "kwargs": kwargs}

    exec(
        blocks[5],
        {
            "encoder": FakeEmbeddingView(),
            "seq2seq": FakeEmbeddingView(),
            "inputs": inputs,
            "__builtins__": __builtins__,
        },
    )
    monkeypatch.chdir(cwd)


def test_capability_evidence_routes_every_curated_example_to_collected_cpu_cases() -> None:
    evidence = (_ROOT / "docs" / "generated" / "capability_evidence.md").read_text(encoding="utf-8")
    for name, required_cases in _CURATED_EXAMPLE_CPU_CASES.items():
        assert name in evidence
        missing_cases: list[str] = []
        for nodeid in required_cases:
            assert nodeid in evidence, (
                f"Capability evidence for {name} omits exact CPU node {nodeid!r}"
            )
            relative_path, separator, test_name = nodeid.partition("::")
            path = (_ROOT / relative_path).resolve()
            try:
                path.relative_to((_ROOT / "tests" / "cpu").resolve())
            except ValueError:
                missing_cases.append(nodeid)
                continue
            if not separator or not path.is_file():
                missing_cases.append(nodeid)
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            declared_tests: set[str] = set()
            for node in tree.body:
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    declared_tests.add(node.name)
                elif isinstance(node, ast.Assign):
                    declared_tests.update(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    declared_tests.add(node.target.id)
            if test_name not in declared_tests:
                missing_cases.append(nodeid)
        assert not missing_cases, (
            f"Capability evidence for {name} names missing CPU cases: {missing_cases!r}"
        )
    assert evidence.count("`cpu_contract`") >= len(_CURATED_EXAMPLE_CPU_CASES)
