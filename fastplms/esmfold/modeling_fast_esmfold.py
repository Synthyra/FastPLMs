from __future__ import annotations
"""FastESMFold: Self-contained ESMFold with FastESM2 attention backends + built-in Test-Time Training.

Usage:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True).cuda()

    # Basic folding
    result = model.fold_protein("MKTLLILAVVA...")
    print(result["plddt"], result["pdb_string"][:100])

    # Folding with TTT (test-time training improves structure prediction)
    result = model.fold_protein("MKTLLILAVVA...", ttt=True)

Dependencies: torch, transformers, einops, peft (for LoRA TTT only)
No dependency on: esm (fair-esm), proteinttt, openfold
"""
import copy
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmIntermediate,
    EsmLMHead,
    EsmOutput,
    EsmSelfOutput,
    RotaryEmbedding,
)
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding


try:
    from fastplms.attention import (
        AttentionBackend, VALID_ATTENTION_BACKENDS,
        resolve_attention_backend, get_attention_mask,
        _get_flex_attention_fn,
        _ensure_flash_kernels_loaded, FLASH_KERNEL, FLASH_KERNEL_VARIANT,
        _kernels_flash_forward, _kernels_flash_varlen_forward,
        kernels_flash_attention_func,
        index_first_axis, index_put_first_axis, pad_input, _unpad_input,
        create_block_mask, flex_attention, BlockMask,
    )
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


# =============================================================================
# Output Dataclass
# =============================================================================

@dataclass
class FastEsmEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


# =============================================================================
# FastESM2 Attention Layers (multi-backend: SDPA, Flash, Flex)
# =============================================================================

class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: Optional[str] = None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, (
            f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
            f"heads ({config.num_attention_heads})"
        )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.scale = self.attention_head_size**-0.5

        self.dropout_prob = config.attention_probs_dropout_prob
        self.config = config
        self.attn_backend = resolve_attention_backend(config.attn_backend)
        self.position_embedding_type = position_embedding_type or config.position_embedding_type
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_BHLD = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_BHLD = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_BHLD = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_BHLD = query_BHLD * self.scale

        if self.position_embedding_type == "rotary":
            query_BHLD, key_BHLD = self.rotary_embeddings(query_BHLD, key_BHLD)

        attn_output, attn_weights = self._attn(
            query_BHLD, key_BHLD, value_BHLD,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        return attn_output, attn_weights

    def _attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            return self._manual_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d)

        if self.attn_backend == AttentionBackend.KERNELS_FLASH:
            return self._kernels_flash_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_2d)
        elif self.attn_backend == AttentionBackend.FLEX:
            return self._flex_attn(query_BHLD, key_BHLD, value_BHLD, flex_block_mask)
        elif self.attn_backend == AttentionBackend.SDPA:
            return self._sdpa_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d)
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

    def _manual_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-1, -2))
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        context_BHLD = torch.matmul(attn_weights, value_BHLD)
        attn_output = rearrange(context_BHLD, "b h s d -> b s (h d)")
        return attn_output, attn_weights

    def _kernels_flash_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        query_BLHD = query_BHLD.transpose(1, 2).contiguous()
        key_BLHD = key_BHLD.transpose(1, 2).contiguous()
        value_BLHD = value_BHLD.transpose(1, 2).contiguous()
        # Q is pre-scaled by self.scale in forward() -- pass softmax_scale=1.0
        # to prevent the kernel from applying its default 1/sqrt(head_dim).
        attn_output = kernels_flash_attention_func(
            query_states=query_BLHD, key_states=key_BLHD, value_states=value_BLHD,
            attention_mask_2d=attention_mask_2d, causal=False,
            softmax_scale=1.0,
        )
        return rearrange(attn_output, "b s h d -> b s (h d)"), None

    def _flex_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        flex_block_mask: Optional[BlockMask] = None,
    ) -> Tuple[torch.Tensor, None]:
        assert flex_attention is not None, "Flex attention is not available in this environment."
        fn = _get_flex_attention_fn()
        context_BHLD = fn(query_BHLD, key_BHLD, value_BHLD, block_mask=flex_block_mask, scale=1.0)
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None

    def _sdpa_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD,
            attn_mask=attention_mask_4d,
            dropout_p=self.dropout_prob if self.training else 0.0,
            scale=1.0,
        )
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None


class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, attn_weights = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        layer_output = self._feed_forward(attention_output)
        return layer_output, attn_weights

    def _feed_forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        return self.output(intermediate_output, attention_output)


class FastEsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> FastEsmEncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights = layer_module(
                hidden_states,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
            )

            if all_attentions is not None:
                all_attentions = all_attentions + (attn_weights,)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return FastEsmEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# =============================================================================
# FastESM Backbone (replaces EsmModel inside ESMFold)
# =============================================================================

class FastEsmBackbone(nn.Module):
    """FastESM2 backbone with multi-backend attention. Drop-in replacement for
    transformers.EsmModel inside EsmForProteinFolding.

    State dict keys match HuggingFace EsmModel exactly, so pretrained weights
    load without any key remapping.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = FastEsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> FastEsmEncoderOutput:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        token_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return FastEsmEncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# =============================================================================
# TTT (Test-Time Training) Configuration and Utilities
# =============================================================================

_ESM_STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")


class LoraInjectedLinear(nn.Module):
    """LoRA-augmented linear layer matching lora_diffusion's behavior.

    Replaces an existing nn.Linear with base(x) + lora_up(lora_down(x)) * scale.
    Initialization follows cloneofsimo/lora: down=Normal(0, 1/r), up=zeros.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, scale: float = 1.0):
        super().__init__()
        self.linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        assert r <= min(in_features, out_features), f"LoRA rank {r} exceeds dimensions ({in_features}, {out_features})"
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        nn.init.normal_(self.lora_down.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scale


def inject_trainable_lora(
    model: nn.Module,
    target_class_name: str,
    r: int,
    scale: float,
) -> List[nn.Parameter]:
    """Replace nn.Linear layers inside modules matching target_class_name with LoRA.

    Matches lora_diffusion's inject_trainable_lora behavior: finds all modules whose
    class name matches target_class_name, then replaces their nn.Linear children with
    LoraInjectedLinear. Returns the list of trainable LoRA parameters.
    """
    lora_params: List[nn.Parameter] = []
    for _parent_name, parent_module in model.named_modules():
        if parent_module.__class__.__name__ != target_class_name:
            continue
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            lora_linear = LoraInjectedLinear(child_module, r=r, scale=scale)
            lora_linear = lora_linear.to(
                device=child_module.weight.device,
                dtype=child_module.weight.dtype,
            )
            setattr(parent_module, child_name, lora_linear)
            lora_params.extend(lora_linear.lora_down.parameters())
            lora_params.extend(lora_linear.lora_up.parameters())
    return lora_params


@dataclass
class TTTConfig:
    lr: float = 4e-4
    ags: int = 4
    steps: int = 10
    batch_size: int = 4
    mask_ratio: float = 0.15
    crop_size: int = 1024
    bert_leave_prob: float = 0.1
    bert_replace_prob: float = 0.1
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    seed: Optional[int] = 0
    initial_state_reset: bool = True
    freeze_embeddings: bool = True
    lora_rank: int = 8
    lora_alpha: float = 32.0
    lora_target_class: str = "EsmSelfAttention"

    def verify(self) -> None:
        assert self.lr > 0.0, "TTT learning rate must be positive."
        assert self.ags > 0, "TTT ags must be positive."
        assert self.steps >= 0, "TTT steps must be non-negative."
        assert self.batch_size > 0, "TTT batch_size must be positive."
        assert 0.0 < self.mask_ratio <= 1.0, "TTT mask_ratio must be in (0, 1]."
        assert self.crop_size > 0, "TTT crop_size must be positive."
        assert 0.0 <= self.bert_leave_prob <= 1.0
        assert 0.0 <= self.bert_replace_prob <= 1.0
        assert self.bert_leave_prob + self.bert_replace_prob <= 1.0
        assert self.optimizer in {"sgd", "adamw"}
        assert self.lora_rank >= 0
        assert self.lora_alpha > 0.0


def preserve_model_state(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        was_training = self.training
        original_device = next(self.parameters()).device
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.named_parameters()
        }
        try:
            return func(self, *args, **kwargs)
        finally:
            self.train(was_training)
            self.to(original_device)
            for name, parameter in self.named_parameters():
                if name in original_requires_grad:
                    parameter.requires_grad = original_requires_grad[name]
                else:
                    parameter.requires_grad = False
    return wrapper


# =============================================================================
# FastEsmFoldConfig
# =============================================================================

class FastEsmFoldConfig(EsmConfig):
    model_type = "fast_esmfold"

    def __init__(self, attn_backend: str = "sdpa", ttt_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.attn_backend = attn_backend
        self.ttt_config = ttt_config or {
            "lr": 4e-4,
            "steps": 10,
            "lora_rank": 8,
            "lora_alpha": 32.0,
        }


# =============================================================================
# FastEsmForProteinFolding
# =============================================================================

class FastEsmForProteinFolding(EsmForProteinFolding):
    """ESMFold with FastESM2 attention backends + built-in Test-Time Training.

    Inherits all folding logic (trunk, structure module, output_to_pdb, infer)
    from transformers.EsmForProteinFolding. Replaces the ESM2 backbone with
    FastESM2 for optimized attention and adds TTT for improved structure prediction.

    Key API:
        result = model.fold_protein("MKTL...", ttt=True)
        # result = {"plddt": float, "ptm": float, "pdb_string": str}
    """
    config_class = FastEsmFoldConfig

    def __init__(self, config: FastEsmFoldConfig):
        super().__init__(config)

        # Replace standard ESM2 backbone with FastESM2 (multi-backend attention)
        # unless use_standard_backbone is set (for TTT debugging/compatibility)
        if not config.ttt_config.get("use_standard_backbone", False):
            self.esm = FastEsmBackbone(config)
            self.esm.requires_grad_(False)
            if config.esmfold_config.fp16_esm:
                self.esm.half()

        # MLM head for TTT (pretrained EsmLMHead: Dense -> GELU -> LN -> Linear)
        self.mlm_head = EsmLMHead(config)

        # TTT state (lazy initialization)
        ttt_kwargs = {k: v for k, v in config.ttt_config.items() if k != "use_standard_backbone"}
        self._ttt_cfg = TTTConfig(**ttt_kwargs)
        self._ttt_cfg.verify()
        self._ttt_initialized = False
        self._ttt_initial_state = None
        self._ttt_generator = torch.Generator()
        if self._ttt_cfg.seed is not None:
            self._ttt_generator.manual_seed(self._ttt_cfg.seed)
        self._non_special_tokens_cache = None
        self._ttt_tokenizer = None

    def _get_ttt_tokenizer(self) -> EsmTokenizer:
        if self._ttt_tokenizer is None:
            self._ttt_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        return self._ttt_tokenizer

    def _ensure_ttt_ready(self) -> None:
        """Lazy TTT initialization. Injects LoRA adapters and saves initial state.
        Must be called after weights are loaded (not in __init__)."""
        if self._ttt_initialized:
            return
        self._ttt_initialized = True

        tokenizer = self._get_ttt_tokenizer()
        vocab = tokenizer.get_vocab()
        self._non_special_tokens_cache = [vocab[c] for c in _ESM_STANDARD_AA if c in vocab]

        if self._ttt_cfg.lora_rank > 0:
            self.mlm_head.eval()
            for p in self.mlm_head.parameters():
                p.requires_grad = False
            # Seed global state before LoRA init for reproducible weight initialization
            if self._ttt_cfg.seed is not None:
                torch.manual_seed(self._ttt_cfg.seed)
            self._inject_lora()
        else:
            # Legacy path: jointly-trained random linear projection head
            H = self.config.hidden_size
            V = self.config.vocab_size
            device = next(self.esm.parameters()).device
            self._ttt_lm_proj = nn.Linear(H, V, bias=True).to(device)

        if self._ttt_cfg.initial_state_reset:
            self._ttt_initial_state = self._ttt_get_state()

    @property
    def _uses_lora(self) -> bool:
        return self._ttt_cfg.lora_rank > 0

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into ESM2 attention layers (matching lora_diffusion behavior)."""
        self._lora_params = inject_trainable_lora(
            self.esm,
            target_class_name=self._ttt_cfg.lora_target_class,
            r=self._ttt_cfg.lora_rank,
            scale=self._ttt_cfg.lora_alpha,
        )
        assert len(self._lora_params) > 0, (
            f"No LoRA params injected. Check target_class_name='{self._ttt_cfg.lora_target_class}' "
            f"matches attention modules in the backbone."
        )

    # ---- TTT State Management ----

    def _get_lora_modules(self) -> List[LoraInjectedLinear]:
        """Find all LoraInjectedLinear modules in the backbone."""
        return [m for m in self.esm.modules() if isinstance(m, LoraInjectedLinear)]

    def _ttt_get_state(self) -> Dict[str, Any]:
        if self._uses_lora:
            lora_state = []
            for m in self._get_lora_modules():
                lora_state.append({
                    "down": m.lora_down.weight.data.clone(),
                    "up": m.lora_up.weight.data.clone(),
                })
            return {"_lora_state": lora_state}
        return {
            "esm": copy.deepcopy(self.esm),
            "_ttt_lm_proj": copy.deepcopy(self._ttt_lm_proj),
        }

    def _ttt_set_state(self, state: Dict[str, Any]) -> None:
        if "_lora_state" in state:
            modules = self._get_lora_modules()
            assert len(modules) == len(state["_lora_state"])
            for m, saved in zip(modules, state["_lora_state"]):
                m.lora_down.weight.data.copy_(saved["down"])
                m.lora_up.weight.data.copy_(saved["up"])
            return
        if "esm" in state:
            self.esm = copy.deepcopy(state["esm"])
        if "_ttt_lm_proj" in state:
            self._ttt_lm_proj = copy.deepcopy(state["_ttt_lm_proj"])

    def ttt_reset(self) -> None:
        """Reset model to pre-TTT state (restore initial LoRA or backbone weights)."""
        assert self._ttt_initial_state is not None, "TTT reset requires initial_state_reset=True."
        self._ttt_set_state(self._ttt_initial_state)

    # ---- TTT Core ----

    def _ttt_tokenize(self, seq: str) -> torch.Tensor:
        tokenizer = self._get_ttt_tokenizer()
        out = tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=self._uses_lora,
            padding=False,
            truncation=False,
        )
        return out["input_ids"]

    def _ttt_mask_token(self) -> int:
        return self._get_ttt_tokenizer().mask_token_id

    def _ttt_get_non_special_tokens(self) -> List[int]:
        if self._non_special_tokens_cache is not None:
            return self._non_special_tokens_cache
        tokenizer = self._get_ttt_tokenizer()
        vocab = tokenizer.get_vocab()
        self._non_special_tokens_cache = [vocab[c] for c in _ESM_STANDARD_AA if c in vocab]
        return self._non_special_tokens_cache

    def _ttt_predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Run ESM2 backbone + LM head to get MLM logits."""
        # Temporarily unfreeze backbone for gradient flow during TTT
        output = self.esm(input_ids=batch)
        hidden = output.last_hidden_state
        if self._uses_lora:
            return self.mlm_head(hidden)
        return self._ttt_lm_proj(hidden)

    def _ttt_sample_batch(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len = x.shape
        batch_size = self._ttt_cfg.batch_size
        crop_size = min(self._ttt_cfg.crop_size, seq_len)

        x_expanded = x.expand(batch_size, -1)
        if seq_len == crop_size:
            start_indices = torch.zeros(batch_size, dtype=torch.long)
        else:
            start_indices = torch.randint(
                0, seq_len - crop_size + 1, (batch_size,),
                generator=self._ttt_generator,
            ).to(torch.long)

        batch_cropped = torch.stack([
            x_expanded[index, start : start + crop_size]
            for index, start in enumerate(start_indices)
        ])

        non_special_tokens = set(self._ttt_get_non_special_tokens())
        mask = torch.zeros((batch_size, crop_size), dtype=torch.bool)
        mask_token_id = self._ttt_mask_token()

        for row_index in range(batch_size):
            non_special_positions = [
                col for col in range(crop_size)
                if batch_cropped[row_index, col].item() in non_special_tokens
            ]
            assert len(non_special_positions) > 0, "Sequence must contain at least one non-special token."
            num_to_mask = max(1, int(round(len(non_special_positions) * self._ttt_cfg.mask_ratio)))
            sampled_indices = torch.randperm(
                len(non_special_positions), generator=self._ttt_generator,
            )[:num_to_mask]
            positions_to_mask = torch.tensor(non_special_positions, dtype=torch.long)[sampled_indices]
            mask[row_index, positions_to_mask] = True

        batch_masked = batch_cropped.clone()
        for row_index in range(batch_size):
            masked_positions = torch.nonzero(mask[row_index], as_tuple=True)[0]
            for masked_position in masked_positions:
                probability = float(torch.rand(1, generator=self._ttt_generator).item())
                if probability < 1.0 - self._ttt_cfg.bert_leave_prob - self._ttt_cfg.bert_replace_prob:
                    batch_masked[row_index, masked_position] = mask_token_id
                    continue
                if probability < 1.0 - self._ttt_cfg.bert_leave_prob:
                    replacement_candidates = self._ttt_get_non_special_tokens()
                    replacement_index = int(torch.randint(
                        0, len(replacement_candidates), (1,), generator=self._ttt_generator,
                    ).item())
                    batch_masked[row_index, masked_position] = replacement_candidates[replacement_index]

        return batch_masked, batch_cropped, mask, start_indices

    def _ttt_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert logits.ndim == 3, "Logits must be [batch, seq, vocab]."
        _, _, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        assert int(mask_flat.sum().item()) > 0, "TTT mask must select at least one token."
        loss = F.cross_entropy(
            logits_flat[mask_flat],
            targets_flat[mask_flat],
            reduction="none",
        )
        masked_tokens_per_seq = mask.sum(dim=1).tolist()
        per_sequence_losses = torch.split(loss, masked_tokens_per_seq)
        return torch.stack([sl.mean() for sl in per_sequence_losses]).mean()

    def _ttt_get_optimizer(self, parameters) -> torch.optim.Optimizer:
        if self._ttt_cfg.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self._ttt_cfg.lr,
                momentum=self._ttt_cfg.momentum,
                weight_decay=self._ttt_cfg.weight_decay,
            )
        return torch.optim.AdamW(
            parameters,
            lr=self._ttt_cfg.lr,
            weight_decay=self._ttt_cfg.weight_decay,
        )

    def _lora_ttt(self, seq: str) -> Dict[str, List[float]]:
        """LoRA TTT: only LoRA adapter weights are trained, mlm_head is frozen."""
        x = self._ttt_tokenize(seq)
        device = next(self.parameters()).device
        non_blocking = device.type == "cuda"
        losses = []

        if self._ttt_cfg.steps == 0:
            return {"losses": losses}

        for parameter in self.parameters():
            parameter.requires_grad = False
        for p in self._lora_params:
            p.requires_grad = True
        optimizer = self._ttt_get_optimizer(self._lora_params)
        optimizer.zero_grad(set_to_none=True)

        self.eval()
        for step in range(self._ttt_cfg.steps * self._ttt_cfg.ags):
            batch_masked, targets, mask, start_indices = self._ttt_sample_batch(x)
            batch_masked = batch_masked.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            self.train()
            logits = self._ttt_predict_logits(batch_masked)
            loss = self._ttt_cross_entropy_loss(logits, targets, mask)
            loss.backward()
            losses.append(float(loss.detach().cpu().item()))

            if (step + 1) % self._ttt_cfg.ags == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        self.eval()
        return {"losses": losses}

    def _legacy_ttt(self, seq: str) -> Dict[str, List[float]]:
        """Legacy TTT: full fine-tuning of ESM2 backbone with random linear projection head."""
        x = self._ttt_tokenize(seq)
        device = next(self.parameters()).device
        non_blocking = device.type == "cuda"
        losses = []

        if self._ttt_cfg.steps == 0:
            return {"losses": losses}

        # Full fine-tune: all backbone params trainable
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.esm.parameters():
            parameter.requires_grad = True
        if self._ttt_cfg.freeze_embeddings:
            for parameter in self.esm.embeddings.parameters():
                parameter.requires_grad = False
        for parameter in self._ttt_lm_proj.parameters():
            parameter.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self._ttt_get_optimizer(trainable_params)
        optimizer.zero_grad(set_to_none=True)

        self.eval()
        for step in range(self._ttt_cfg.steps * self._ttt_cfg.ags):
            batch_masked, targets, mask, start_indices = self._ttt_sample_batch(x)
            batch_masked = batch_masked.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            self.train()
            logits = self._ttt_predict_logits(batch_masked)
            loss = self._ttt_cross_entropy_loss(logits, targets, mask)
            loss.backward()
            losses.append(float(loss.detach().cpu().item()))

            if (step + 1) % self._ttt_cfg.ags == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        self.eval()
        return {"losses": losses}

    @preserve_model_state
    def ttt(self, seq: str) -> Dict[str, List[float]]:
        """Run test-time training on a single sequence using masked language modeling.

        Adapts the ESM2 backbone (via LoRA or full fine-tuning) to the input sequence
        before structure prediction. Call fold_protein(seq, ttt=True) for the full pipeline.

        Args:
            seq: Protein sequence (single-letter amino acid codes)

        Returns:
            Dict with "losses" key containing per-step MLM loss values
        """
        self._ensure_ttt_ready()
        # TTT requires fp32 for stable gradient computation. ESMFold typically
        # runs the backbone in fp16, but small LoRA updates vanish in half precision.
        esm_dtype = next(self.esm.parameters()).dtype
        if esm_dtype != torch.float32:
            self.esm.float()
            self.mlm_head.float()
        if self._uses_lora:
            result = self._lora_ttt(seq)
        else:
            result = self._legacy_ttt(seq)
        # Restore original dtype (backbone back to fp16 for inference)
        if esm_dtype != torch.float32:
            self.esm.to(esm_dtype)
            self.mlm_head.to(esm_dtype)
        return result

    # ---- High-Level API ----

    def _fold_single(self, sequence: str, return_pdb_string: bool = True) -> Dict[str, Any]:
        """Fold a sequence once and return pLDDT, ptm, and optionally PDB string."""
        with torch.no_grad():
            output = self.infer(sequence)
        plddt = output["plddt"]
        # plddt shape is (batch, L, 37) - per-atom across atom37 types.
        # Use CA atom (index 1) only, matching PDB B-factor output.
        if plddt.dim() == 3:
            mean_plddt = float(plddt[:, :, 1].mean().item())
        elif plddt.dim() == 2:
            mean_plddt = float(plddt[:, 1].mean().item())
        else:
            mean_plddt = float(plddt.mean().item())
        result = {
            "plddt": mean_plddt,
            "ptm": float(output["ptm"].item()) if "ptm" in output else None,
        }
        if return_pdb_string:
            pdb_strings = self.output_to_pdb(output)
            result["pdb_string"] = pdb_strings[0] if isinstance(pdb_strings, list) else pdb_strings
        return result

    def fold_protein(
        self,
        sequence: str,
        return_pdb_string: bool = True,
    ) -> Dict[str, Any]:
        """Fold a protein sequence with test-time training.

        Runs TTT (masked language model adaptation via LoRA) for the configured
        number of steps, folding after each optimizer step to track pLDDT. Returns
        the structure with the highest pLDDT across all steps (including baseline).

        Args:
            sequence: Protein sequence (single-letter amino acid codes)
            return_pdb_string: If True, include PDB string in output

        Returns:
            Dict with keys:
                - plddt: float, best mean pLDDT across all TTT steps
                - ptm: float, predicted TM-score from best step
                - pdb_string: str (if return_pdb_string=True), PDB from best step
                - step_plddts: list[float], pLDDT at each step [baseline, s1, ..., s10]
                - best_step: int, which step produced best structure (0=baseline)
        """
        self._ensure_ttt_ready()

        # Cast to fp32 for TTT stability
        esm_dtype = next(self.esm.parameters()).dtype
        if esm_dtype != torch.float32:
            self.esm.float()
            self.mlm_head.float()

        device = next(self.parameters()).device
        non_blocking = device.type == "cuda"

        # Step 0: baseline fold (no TTT adaptation)
        best = self._fold_single(sequence, return_pdb_string=return_pdb_string)
        step_plddts = [best["plddt"]]

        if self._ttt_cfg.steps > 0:
            # Tokenize for masked LM training
            x = self._ttt_tokenize(sequence)

            # Freeze all, unfreeze LoRA
            for p in self.parameters():
                p.requires_grad = False
            if self._uses_lora:
                for p in self._lora_params:
                    p.requires_grad = True
                optimizer = self._ttt_get_optimizer(self._lora_params)
            else:
                for p in self.esm.parameters():
                    p.requires_grad = True
                if self._ttt_cfg.freeze_embeddings:
                    for p in self.esm.embeddings.parameters():
                        p.requires_grad = False
                for p in self._ttt_lm_proj.parameters():
                    p.requires_grad = True
                trainable = [p for p in self.parameters() if p.requires_grad]
                optimizer = self._ttt_get_optimizer(trainable)
            optimizer.zero_grad(set_to_none=True)

            self.eval()
            for step in range(self._ttt_cfg.steps * self._ttt_cfg.ags):
                batch_masked, targets, mask, _start = self._ttt_sample_batch(x)
                batch_masked = batch_masked.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                mask = mask.to(device, non_blocking=non_blocking)

                self.train()
                logits = self._ttt_predict_logits(batch_masked)
                loss = self._ttt_cross_entropy_loss(logits, targets, mask)
                loss.backward()

                if (step + 1) % self._ttt_cfg.ags == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Fold after this optimizer step
                    self.eval()
                    current = self._fold_single(sequence, return_pdb_string=return_pdb_string)
                    step_plddts.append(current["plddt"])
                    if current["plddt"] > best["plddt"]:
                        best = current

            self.eval()

            # Restore requires_grad
            for p in self.parameters():
                p.requires_grad = False

        # Reset LoRA weights for next sequence
        self.ttt_reset()

        # Restore dtype
        if esm_dtype != torch.float32:
            self.esm.to(esm_dtype)
            self.mlm_head.to(esm_dtype)

        return {
            "plddt": best["plddt"],
            "ptm": best["ptm"],
            "pdb_string": best.get("pdb_string"),
            "step_plddts": step_plddts,
            "best_step": step_plddts.index(max(step_plddts)),
        }
