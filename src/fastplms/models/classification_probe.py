"""Shared transformer probes for residue and sequence prediction tasks."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

try:
    from fastplms.attention import (
        AttentionBackend,
        _get_flex_attention_fn,
        flex_attention,
        get_attention_mask,
        resolve_attention_backend,
    )
    from fastplms.embeddings.pooling import Pooler
    from fastplms.models._esm_rotary import RotaryEmbedding
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "Pooler",
        "RotaryEmbedding",
        "_get_flex_attention_fn",
        "flex_attention",
        "get_attention_mask",
        "resolve_attention_backend",
    )
    if error.name != "fastplms" or any(
        name not in globals() for name in _COMPOSITE_REQUIRED_NAMES
    ):
        raise
    # Flat Hub composites define every shared symbol above this source.


_SUPPORTED_BACKENDS = frozenset(
    {
        AttentionBackend.EAGER,
        AttentionBackend.SDPA,
        AttentionBackend.FLEX_ATTENTION,
    }
)
_SUPPORTED_PROBLEM_TYPES = frozenset(
    {
        "regression",
        "single_label_classification",
        "multi_label_classification",
    }
)
_UNSUPPORTED_POOLING = frozenset({"cls", "parti"})


def _config_value(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, None)
    return default if value is None else value


def _attention_backend(config: Any) -> AttentionBackend:
    requested = getattr(config, "_attn_implementation", None)
    if requested is None:
        requested = getattr(config, "attn_backend", "sdpa")
    backend = resolve_attention_backend(requested)
    if backend not in _SUPPORTED_BACKENDS:
        expected = ", ".join(sorted(item.value for item in _SUPPORTED_BACKENDS))
        raise ValueError(
            f"Classification probes support only {expected}; received {backend.value!r}."
        )
    return backend


def resolve_problem_type(
    config: Any,
    labels: torch.Tensor,
    *,
    num_labels: int,
) -> str:
    """Resolve and persist the standard Transformers classification problem type."""

    problem_type = getattr(config, "problem_type", None)
    if problem_type is None:
        if num_labels == 1:
            problem_type = "regression"
        elif labels.dtype in {torch.long, torch.int}:
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"
        config.problem_type = problem_type
    if problem_type not in _SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            f"Unsupported problem_type {problem_type!r}; expected one of "
            f"{sorted(_SUPPORTED_PROBLEM_TYPES)}."
        )
    return problem_type


def sequence_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    problem_type: str,
    num_labels: int,
) -> torch.Tensor:
    """Compute a Hugging Face-compatible sequence task loss."""

    labels = labels.to(logits.device)
    if problem_type == "regression":
        if num_labels == 1:
            return F.mse_loss(logits.squeeze(-1), labels.squeeze(-1).to(logits.dtype))
        return F.mse_loss(logits, labels.to(logits.dtype))
    if problem_type == "single_label_classification":
        return F.cross_entropy(logits.reshape(-1, num_labels), labels.reshape(-1).long())
    if problem_type == "multi_label_classification":
        return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
    raise ValueError(f"Unsupported problem_type {problem_type!r}.")


def _masked_elementwise_loss(
    losses: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    valid = labels.ne(-100)
    if not bool(valid.any()):
        return losses.sum() * 0
    return losses.masked_select(valid).mean()


def token_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    problem_type: str,
    num_labels: int,
) -> torch.Tensor:
    """Compute a token task loss, excluding every label element equal to ``-100``."""

    labels = labels.to(logits.device)
    if problem_type == "regression":
        targets = labels.to(logits.dtype)
        if num_labels == 1 and targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(-1)
        if targets.shape != logits.shape:
            raise ValueError(
                "Token regression labels must match logits, except that the final "
                "singleton dimension may be omitted when num_labels=1."
            )
        return _masked_elementwise_loss(F.mse_loss(logits, targets, reduction="none"), targets)
    if problem_type == "single_label_classification":
        if not bool(labels.ne(-100).any()):
            return logits.sum() * 0
        return F.cross_entropy(
            logits.reshape(-1, num_labels),
            labels.reshape(-1).long(),
            ignore_index=-100,
        )
    if problem_type == "multi_label_classification":
        if labels.shape != logits.shape:
            raise ValueError("Multilabel token labels must have the same shape as logits.")
        losses = F.binary_cross_entropy_with_logits(
            logits,
            labels.to(logits.dtype),
            reduction="none",
        )
        return _masked_elementwise_loss(losses, labels)
    raise ValueError(f"Unsupported problem_type {problem_type!r}.")


class SwiGLU(nn.Module):
    """SwiGLU activation used by the Protify-aligned feed-forward layer."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, values = inputs.chunk(2, dim=-1)
        return F.silu(gate) * values


class ProbeSelfAttention(nn.Module):
    """Four-head RoPE self-attention with explicit, fail-closed dispatch."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        backend: AttentionBackend,
        use_bias: bool,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("classifier_probe_hidden_size must be divisible by its head count.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout
        self.backend = backend
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=use_bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.rotary = RotaryEmbedding(self.head_size)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = self._reshape(query)
        key = self._reshape(key)
        value = self._reshape(value)
        query, key = self.rotary(query, key)
        if output_attentions and self.backend != AttentionBackend.EAGER:
            raise ValueError(
                f"output_attentions=True is unavailable for {self.backend.value!r}; "
                "select 'eager' explicitly."
            )
        _, attention_mask_4d, flex_block_mask = get_attention_mask(
            self.backend,
            batch_size,
            sequence_length,
            hidden_states.device,
            attention_mask,
            hidden_states.dtype,
        )
        dropout = self.dropout if self.training else 0.0
        attention_weights = None
        if self.backend == AttentionBackend.EAGER:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
            if attention_mask_4d is not None:
                scores = scores.masked_fill(~attention_mask_4d, float("-inf"))
            attention_weights = scores.softmax(dim=-1)
            context = F.dropout(attention_weights, p=dropout, training=self.training) @ value
        elif self.backend == AttentionBackend.SDPA:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask_4d,
                dropout_p=dropout,
            )
        elif self.backend == AttentionBackend.FLEX_ATTENTION:
            if flex_attention is None:
                raise RuntimeError("'flex_attention' was requested but is unavailable.")
            flex_fn = _get_flex_attention_fn(
                device=query.device,
                dtype=query.dtype,
                shape=tuple(query.shape),
                mask_semantics="padding",
            )
            if flex_fn is None:
                raise RuntimeError("'flex_attention' was requested but is unavailable.")
            context = flex_fn(
                query,
                key,
                value,
                block_mask=flex_block_mask,
                scale=1.0 / math.sqrt(self.head_size),
                kernel_options={"PRESCALE_QK": True, "BLOCK_N": 32},
            )
        else:
            raise AssertionError(f"Unhandled attention backend {self.backend.value!r}.")
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.output(context), attention_weights


class ProteinTransformerProbe(nn.Module):
    """Project residue embeddings and refine them with exactly one pre-LN block."""

    def __init__(self, config: Any, input_size: int) -> None:
        super().__init__()
        hidden_size = int(_config_value(config, "classifier_probe_hidden_size", 512))
        num_heads = int(_config_value(config, "classifier_probe_num_heads", 4))
        dropout = float(_config_value(config, "classifier_probe_dropout", 0.1))
        use_bias = bool(
            _config_value(
                config,
                "classifier_use_bias",
                _config_value(config, "use_bias", False),
            )
        )
        if hidden_size != 512 or num_heads != 4 or hidden_size // num_heads != 128:
            raise ValueError(
                "The folding classification probe requires a 512-wide projection with "
                "four 128-wide attention heads."
            )
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = ProbeSelfAttention(
            hidden_size,
            num_heads,
            dropout,
            _attention_backend(config),
            use_bias,
        )
        intermediate_size = int(math.ceil((8 / 3) * hidden_size / 256) * 256)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 2 * intermediate_size, bias=use_bias),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size, bias=use_bias),
        )
        self.residual_dropout = nn.Dropout(dropout)

    @property
    def attn_backend(self) -> str:
        return self.attention.backend.value

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, residue, channel).")
        embeddings = embeddings.to(dtype=self.input_projection.weight.dtype)
        hidden_states = self.input_projection(self.input_norm(embeddings))
        attention_output, attention_weights = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        hidden_states = hidden_states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(hidden_states))
        )
        output = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=(attention_weights,) if output_attentions else None,
        )
        return output if return_dict else output.to_tuple()


class _ClassificationProbe(nn.Module):
    def __init__(self, config: Any, input_size: int, *, sequence_task: bool) -> None:
        super().__init__()
        self.config = config
        self.num_labels = int(_config_value(config, "num_labels", 2))
        self.transformer = ProteinTransformerProbe(config, input_size)
        self.sequence_task = sequence_task
        pooling_types = _config_value(config, "classifier_pooling_types", ["mean"])
        self.pooler = Pooler(pooling_types) if sequence_task else None
        if self.pooler is not None:
            unsupported = sorted(set(self.pooler.names) & _UNSUPPORTED_POOLING)
            if unsupported:
                raise ValueError(
                    "Classification probes consume residue-only representations and do not "
                    f"support pooling operation(s) {unsupported}."
                )
        hidden_size = self.transformer.hidden_size
        classifier_input = hidden_size * (len(self.pooler.names) if self.pooler else 1)
        classifier_hidden = int(_config_value(config, "classifier_hidden_size", 4096))
        classifier_dropout = float(_config_value(config, "classifier_dropout", 0.2))
        use_bias = bool(
            _config_value(
                config,
                "classifier_use_bias",
                _config_value(config, "use_bias", False),
            )
        )
        projection_size = int(math.ceil((2 * self.num_labels) / 256) * 256)
        classifier_layers: list[nn.Module] = [
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, classifier_hidden, bias=use_bias),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, projection_size, bias=use_bias),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
        ]
        if not sequence_task:
            classifier_layers.extend(
                [
                    nn.Linear(projection_size, projection_size, bias=use_bias),
                    nn.ReLU(),
                ]
            )
        classifier_layers.append(nn.Linear(projection_size, self.num_labels, bias=use_bias))
        self.classifier = nn.Sequential(*classifier_layers)

    def _forward_transformer(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
    ) -> BaseModelOutput:
        output_attentions = (
            bool(output_attentions)
            if output_attentions is not None
            else bool(getattr(self.config, "output_attentions", False))
        )
        output_hidden_states = (
            bool(output_hidden_states)
            if output_hidden_states is not None
            else bool(getattr(self.config, "output_hidden_states", False))
        )
        return self.transformer(
            embeddings,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


class SequenceClassificationProbe(_ClassificationProbe):
    """Protify-style sequence classifier over externally supplied residue embeddings."""

    def __init__(self, config: Any, input_size: int) -> None:
        super().__init__(config, input_size, sequence_task=True)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        if attention_mask is None:
            attention_mask = torch.ones(
                embeddings.shape[:2],
                device=embeddings.device,
                dtype=torch.bool,
            )
        outputs = self._forward_transformer(
            embeddings,
            attention_mask,
            output_attentions,
            output_hidden_states,
        )
        if self.pooler is None:
            raise AssertionError("Sequence classification requires a configured pooler.")
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            problem_type = resolve_problem_type(self.config, labels, num_labels=self.num_labels)
            loss = sequence_classification_loss(
                logits,
                labels,
                problem_type=problem_type,
                num_labels=self.num_labels,
            )
        result = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        use_return_dict = (
            bool(return_dict)
            if return_dict is not None
            else bool(getattr(self.config, "use_return_dict", True))
        )
        return result if use_return_dict else result.to_tuple()


class TokenClassificationProbe(_ClassificationProbe):
    """Protify-style residue classifier or regressor over supplied embeddings."""

    def __init__(self, config: Any, input_size: int) -> None:
        super().__init__(config, input_size, sequence_task=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]:
        outputs = self._forward_transformer(
            embeddings,
            attention_mask,
            output_attentions,
            output_hidden_states,
        )
        logits = self.classifier(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            problem_type = resolve_problem_type(self.config, labels, num_labels=self.num_labels)
            loss = token_classification_loss(
                logits,
                labels,
                problem_type=problem_type,
                num_labels=self.num_labels,
            )
        result = TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        use_return_dict = (
            bool(return_dict)
            if return_dict is not None
            else bool(getattr(self.config, "use_return_dict", True))
        )
        return result if use_return_dict else result.to_tuple()


__all__ = [
    "ProbeSelfAttention",
    "ProteinTransformerProbe",
    "SequenceClassificationProbe",
    "SwiGLU",
    "TokenClassificationProbe",
    "resolve_problem_type",
    "sequence_classification_loss",
    "token_classification_loss",
]
