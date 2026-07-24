"""Public validation must remain active when Python removes ``assert`` statements."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from fastplms.models.dplm.modeling_dplm import FAST_DPLM_ENCODER
from fastplms.models.dplm2.modeling_dplm2 import (
    DPLM2ForMaskedLM,
    _has_packed_multimodal_layout,
    _normalize_dplm2_input_ids,
)
from fastplms.models.e1.attention import _unpad_input
from fastplms.models.e1.cache import DynamicCache, KVCache
from fastplms.models.e1.modeling_e1 import FAST_E1_ENCODER, E1ForMaskedLM
from fastplms.models.e1.preparation import E1BatchPreparer
from fastplms.models.e1.retrieval import (
    IdSequence,
    _make_homologue_searcher,
    compute_ppll,
    convert_to_tensor,
    get_query_from_a3m,
    read_fasta_sequences,
)
from fastplms.models.esm3.modeling_esm3 import (
    Affine3D,
    FastESM3Config,
    FastESM3PreTrainedModel,
    RotationMatrix,
)
from fastplms.models.esmfold import modeling_fast_esmfold
from fastplms.models.esmfold.modeling_fast_esmfold import (
    EsmSelfAttention as FastEsmFoldSelfAttention,
)
from fastplms.models.esmfold.modeling_fast_esmfold import (
    FastEsmFoldConfig,
)
from fastplms.models.ttt import TTTConfig


@pytest.mark.parametrize(
    ("values", "error_type"),
    (
        ({"lr": 0.0}, ValueError),
        ({"steps": 0}, ValueError),
        ({"steps": 1.5}, TypeError),
        ({"mask_ratio": 1.1}, ValueError),
        ({"optimizer": "rmsprop"}, ValueError),
        ({"lora_target_modules": "query"}, TypeError),
    ),
)
def test_ttt_config_public_validation_uses_explicit_exceptions(
    values: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        TTTConfig(**values)


def test_sequence_model_public_validations_use_explicit_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="outside the DPLM2 embedding table"):
        _normalize_dplm2_input_ids(torch.tensor([[999]]), vocab_size=64)
    with pytest.raises(ValueError, match=r"type_ids.*shape"):
        _has_packed_multimodal_layout(
            torch.zeros(3, dtype=torch.long),
            aa_type=0,
            struct_type=1,
            pad_type=2,
        )
    dplm2_stub = SimpleNamespace(
        config=SimpleNamespace(
            use_return_dict=True,
            dplm_type="multimodal",
            hidden_size=8,
        ),
        pad_id=0,
    )
    with pytest.raises(ValueError, match="attention_mask is required"):
        DPLM2ForMaskedLM.forward(
            dplm2_stub,
            inputs_embeds=torch.zeros(1, 2, 8),
        )
    with pytest.raises(ValueError, match="type_ids is required"):
        DPLM2ForMaskedLM.forward(
            dplm2_stub,
            attention_mask=torch.ones(1, 2),
            inputs_embeds=torch.zeros(1, 2, 8),
        )
    with pytest.raises(ValueError, match="Pass either seq or input_ids"):
        DPLM2ForMaskedLM._ttt_tokenize(SimpleNamespace())
    empty_replacement_stub = SimpleNamespace(
        tokenizer=SimpleNamespace(
            all_special_ids=(),
            struct_cls_token="<cls_struct>",
            _token_to_id={"<cls_struct>": 0},
        )
    )
    with pytest.raises(RuntimeError, match="replacement set is empty"):
        DPLM2ForMaskedLM._ttt_replacement_tokens(
            empty_replacement_stub,
            torch.tensor([[1]]),
        )
    with pytest.raises(ValueError, match=r"head_mask\.dim"):
        FAST_DPLM_ENCODER._convert_head_mask_to_5d(
            SimpleNamespace(dtype=torch.float32),
            torch.ones(1, 1, 1),
            num_hidden_layers=1,
        )

    DynamicCache().crop(1)
    with pytest.raises(ValueError, match="max_length must be positive"):
        DynamicCache().crop(0)
    with pytest.raises(ValueError, match="use_cache=True"):
        KVCache().after_forward(
            {"context": ["ctx"], "context_len": [1], "use_cache": False},
            SimpleNamespace(),
        )
    with pytest.raises(TypeError, match="Sequence must be a string"):
        E1BatchPreparer.validate_sequence(
            SimpleNamespace(mask_token="<mask>"),
            123,
        )
    e1_stub = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            max_num_positions_within_seq=8,
            max_num_positions_global=8,
            max_num_sequences=2,
        )
    )
    with pytest.raises(ValueError, match="input_ids must have rank 2"):
        FAST_E1_ENCODER._prepare_hidden_states(
            e1_stub,
            torch.ones(1, 2, 3, dtype=torch.long),
            None,
            None,
            None,
            None,
        )
    with pytest.raises(ValueError, match="sequence_ids must have shape"):
        FAST_E1_ENCODER._prepare_hidden_states(
            e1_stub,
            torch.ones(1, 2, dtype=torch.long),
            None,
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long),
        )

    malformed_fasta = tmp_path / "malformed.fasta"
    malformed_fasta.write_text("ACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before header"):
        read_fasta_sequences(str(malformed_fasta))
    malformed_a3m = tmp_path / "malformed.a3m"
    malformed_a3m.write_text("ACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No FASTA header"):
        get_query_from_a3m(str(malformed_a3m))
    with pytest.raises(ValueError, match="equal aligned lengths"):
        convert_to_tensor([IdSequence("a", "AC"), IdSequence("b", "ACD")])
    with pytest.raises(ValueError, match="empty token sequence"):
        compute_ppll(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="target_db is required"):
        _make_homologue_searcher("mmseqs2", None)

    with pytest.raises(ValueError, match="hidden_size must be positive"):
        FastESM3Config(hidden_size=0)
    with pytest.raises(ValueError, match="divisible"):
        FastESM3Config(hidden_size=24, num_attention_heads=5)
    with pytest.raises(ValueError, match="currently supports only"):
        FastESM3PreTrainedModel.attn_backend.fset(
            SimpleNamespace(),
            "flash_attention_2",
        )
    with pytest.raises(
        ValueError,
        match="Rotation matrices must have trailing shape",
    ):
        RotationMatrix(torch.zeros(2, 3, 2))
    rotation = RotationMatrix.identity((2,))
    with pytest.raises(
        ValueError,
        match="Affine translation and rotation batch shapes must match",
    ):
        Affine3D(torch.zeros(1, 3), rotation)

    with pytest.raises(ValueError, match="not a multiple"):
        FastEsmFoldSelfAttention(
            FastEsmFoldConfig(
                hidden_size=10,
                num_attention_heads=3,
                attn_backend="eager",
            )
        )
    fold_attention = FastEsmFoldSelfAttention(
        FastEsmFoldConfig(
            hidden_size=8,
            num_attention_heads=2,
            attn_backend="eager",
        )
    )
    monkeypatch.setattr(modeling_fast_esmfold, "flex_attention", None)
    attention_heads = torch.zeros(1, 2, 3, 4)  # (b=1, h=2, l=3, d_h=4)
    with pytest.raises(RuntimeError, match="Flex attention is not available"):
        fold_attention._flex_attn(
            attention_heads,
            attention_heads,
            attention_heads,
        )

    query_layer = torch.zeros(1, 3, 2, 4)  # (b=1, l_q=3, h=2, d_h=4)
    key_layer = torch.zeros(1, 3, 2, 4)  # (b, l_k=3, h, d_h)
    value_layer = torch.zeros(1, 3, 2, 4)  # (b, l_k, h, d_h)
    with pytest.raises(
        ValueError,
        match="Shape mismatch between query layer and query sequence ids",
    ):
        _unpad_input(
            query_layer,
            key_layer,
            value_layer,
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
        )
    with pytest.raises(
        ValueError,
        match="key_layer and value_layer must have identical shapes",
    ):
        _unpad_input(
            query_layer,
            key_layer,
            value_layer[:, :2],
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
        )


def test_ttt_and_e1_public_state_validations_use_explicit_exceptions() -> None:
    from tests.integration.test_ttt import DummyTTTModel

    model = DummyTTTModel()
    with pytest.raises(ValueError, match="Pass either seq or input_ids"):
        model._ttt_tokenize()
    with pytest.raises(RuntimeError, match="no LoRA parameters"):
        model._ttt_lora_parameters()
    with pytest.raises(RuntimeError, match="no LoRA state"):
        model._ttt_snapshot_lora_state()

    model._ttt_ensure_initialized()
    with pytest.raises(ValueError, match="Changing lora_rank"):
        model.ttt(seq="AC", ttt_config={"lora_rank": 3})
    with pytest.raises(RuntimeError, match="state/module count mismatch"):
        model._ttt_restore_lora_state([])

    invalid_target = DummyTTTModel()
    invalid_target._ttt_cfg = invalid_target.ttt_config.merged(
        {"lora_target_modules": ("missing",)}
    )
    with pytest.raises(ValueError, match="did not find any target modules"):
        invalid_target._ttt_inject_lora()

    with pytest.raises(ValueError, match="E1 token tensors"):
        E1ForMaskedLM._ttt_tokenize(SimpleNamespace())
    with pytest.raises(TypeError, match="tensor dictionary"):
        E1ForMaskedLM._ttt_predict_logits(SimpleNamespace(), torch.tensor([[1]]))

    class EmptyContexts:
        def sample_msa_contexts(self, **_kwargs: object) -> dict[str, object]:
            return {}

    with pytest.raises(ValueError, match="sampled MSA context"):
        E1ForMaskedLM.score_ppll(
            EmptyContexts(),
            sequences=["AC"],
            a3m_path="unused.a3m",
        )


def test_representative_public_validation_survives_python_optimized_mode() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        import torch
        import torch.nn as nn

        from fastplms.models.dplm.modeling_dplm import FAST_DPLM_ENCODER
        from fastplms.models.dplm2.modeling_dplm2 import (
            DPLM2ForMaskedLM,
            _has_packed_multimodal_layout,
            _normalize_dplm2_input_ids,
        )
        from fastplms.models.e1.attention import _unpad_input
        from fastplms.models.e1.cache import DynamicCache, KVCache
        from fastplms.models.e1.modeling_e1 import E1ForMaskedLM, FAST_E1_ENCODER
        from fastplms.models.e1.retrieval import (
            _make_homologue_searcher,
            compute_ppll,
        )
        from fastplms.models.esm3.modeling_esm3 import (
            Affine3D,
            FastESM3Config,
            FastESM3PreTrainedModel,
            RotaryEmbedding,
            RotationMatrix,
        )
        from fastplms.models.esmfold import modeling_fast_esmfold
        from fastplms.models.esmfold.modeling_fast_esmfold import (
            EsmSelfAttention as FastEsmFoldSelfAttention,
            FastEsmFoldConfig,
        )
        from fastplms.models.ttt import FastPLMTestTimeTrainingMixin, TTTConfig

        class DummyTTTModel(FastPLMTestTimeTrainingMixin, nn.Module):
            # Keep the optimized subprocess independent of the broad integration module.

            def __init__(self):
                nn.Module.__init__(self)
                self.config = SimpleNamespace(vocab_size=8)
                self.backbone = nn.Sequential(nn.Linear(8, 8))
                self.init_ttt({"lora_rank": 2, "lora_alpha": 1.0})

            def _ttt_get_trainable_modules(self):
                return [self.backbone]

        def must_raise(error_type, function, *args, **kwargs):
            try:
                function(*args, **kwargs)
            except error_type:
                return
            raise RuntimeError(
                f"{function.__qualname__} did not raise {error_type.__name__} under -O"
            )

        must_raise(ValueError, TTTConfig, steps=0)
        must_raise(TypeError, TTTConfig, lora_target_modules="query")
        must_raise(
            ValueError,
            _normalize_dplm2_input_ids,
            torch.tensor([[999]]),
            64,
        )
        must_raise(
            ValueError,
            _has_packed_multimodal_layout,
            torch.zeros(3, dtype=torch.long),
            0,
            1,
            2,
        )
        dplm2_stub = SimpleNamespace(
            config=SimpleNamespace(
                use_return_dict=True,
                dplm_type="multimodal",
                hidden_size=8,
            ),
            pad_id=0,
        )
        must_raise(
            ValueError,
            DPLM2ForMaskedLM.forward,
            dplm2_stub,
            inputs_embeds=torch.zeros(1, 2, 8),
        )
        must_raise(
            ValueError,
            DPLM2ForMaskedLM.forward,
            dplm2_stub,
            attention_mask=torch.ones(1, 2),
            inputs_embeds=torch.zeros(1, 2, 8),
        )
        must_raise(ValueError, DPLM2ForMaskedLM._ttt_tokenize, SimpleNamespace())
        must_raise(
            ValueError,
            FAST_DPLM_ENCODER._convert_head_mask_to_5d,
            SimpleNamespace(dtype=torch.float32),
            torch.ones(1, 1, 1),
            1,
        )
        must_raise(ValueError, DynamicCache().crop, 0)
        must_raise(
            ValueError,
            KVCache().after_forward,
            {"context": ["ctx"], "context_len": [1], "use_cache": False},
            SimpleNamespace(),
        )
        e1_stub = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=8,
                max_num_positions_within_seq=8,
                max_num_positions_global=8,
                max_num_sequences=2,
            )
        )
        must_raise(
            ValueError,
            FAST_E1_ENCODER._prepare_hidden_states,
            e1_stub,
            torch.ones(1, 2, dtype=torch.long),
            None,
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long),
        )
        must_raise(
            ValueError,
            compute_ppll,
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.long),
        )
        must_raise(ValueError, _make_homologue_searcher, "mmseqs2", None)
        must_raise(ValueError, FastESM3Config, hidden_size=0)
        must_raise(
            ValueError,
            FastESM3PreTrainedModel.attn_backend.fset,
            SimpleNamespace(),
            "flash_attention_2",
        )
        rotary = RotaryEmbedding(4)
        rotary._update_cos_sin_cache = lambda *_args, **_kwargs: None
        must_raise(
            RuntimeError,
            rotary.forward,
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 1, 4),
        )
        must_raise(ValueError, RotationMatrix, torch.zeros(2, 3, 2))
        rotation = RotationMatrix.identity((2,))
        must_raise(
            ValueError,
            Affine3D,
            torch.zeros(1, 3),
            rotation,
        )
        must_raise(
            ValueError,
            FastEsmFoldSelfAttention,
            FastEsmFoldConfig(
                hidden_size=10,
                num_attention_heads=3,
                attn_backend="eager",
            ),
        )
        fold_attention = FastEsmFoldSelfAttention(
            FastEsmFoldConfig(
                hidden_size=8,
                num_attention_heads=2,
                attn_backend="eager",
            )
        )
        modeling_fast_esmfold.flex_attention = None
        attention_heads = torch.zeros(1, 2, 3, 4)
        must_raise(
            RuntimeError,
            fold_attention._flex_attn,
            attention_heads,
            attention_heads,
            attention_heads,
        )
        must_raise(
            ValueError,
            _unpad_input,
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
        )
        must_raise(
            ValueError,
            _unpad_input,
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
        )

        model = DummyTTTModel()
        must_raise(ValueError, model._ttt_tokenize)
        must_raise(RuntimeError, model._ttt_lora_parameters)
        model._ttt_ensure_initialized()
        must_raise(ValueError, model.ttt, seq="AC", ttt_config={"lora_rank": 3})
        must_raise(ValueError, E1ForMaskedLM._ttt_tokenize, SimpleNamespace())
        must_raise(
            TypeError,
            E1ForMaskedLM._ttt_predict_logits,
            SimpleNamespace(),
            torch.tensor([[1]]),
        )
        """
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_esmfold_public_attention_validation_has_no_optimized_away_asserts() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fastplms"
        / "models"
        / "esmfold"
        / "modeling_fast_esmfold.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    attention_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "EsmSelfAttention"
    )
    public_validation_methods = {
        method.name: method
        for method in attention_class.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name in {"__init__", "_flex_attn"}
    }

    assert set(public_validation_methods) == {"__init__", "_flex_attn"}
    for method in public_validation_methods.values():
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(method))
