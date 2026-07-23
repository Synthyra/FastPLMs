"""Discrete diffusion generation shared by DPLM and DPLM2.

The implementation keeps model-specific vocabulary rules at the public entry
points and shares only the categorical sampling and confidence-based remasking
mechanism.  It has no dependency on the pinned upstream checkout.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

import torch
from tqdm.auto import tqdm


class _MaskedLanguageModel(Protocol):
    """Structural type used by the two generation entry points."""

    config: Any

    def eval(self) -> Any: ...

    def modules(self) -> Iterable[torch.nn.Module]: ...

    def __call__(self, **kwargs: Any) -> Any: ...


_DPLM2_AA_BOUNDARY = 33
_DPLM2_AA_BOS = 0
_DPLM2_PAD = 1
_DPLM2_AA_EOS = 2
_DPLM2_AA_UNK = 3
_DPLM2_AA_X = 24
_DPLM2_AA_B = 25
_DPLM2_AA_U = 26
_DPLM2_AA_Z = 27
_DPLM2_AA_O = 28
_DPLM2_AA_MASK = 32
_DPLM2_STRUCT_BOS = 33
_DPLM2_STRUCT_EOS = 34
_DPLM2_STRUCT_UNK = 35


@contextmanager
def _temporary_eval(model: _MaskedLanguageModel):
    """Run one generation forward in eval mode and restore every module flag."""
    training_states = tuple((module, module.training) for module in model.modules())
    model.eval()
    try:
        yield
    finally:
        for module, training in training_states:
            module.training = training


def _resolve_max_iter(model: _MaskedLanguageModel, max_iter: int | None) -> int:
    if max_iter is None:
        max_iter = int(getattr(model.config, "num_diffusion_timesteps", 500))
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    return max_iter


def _validate_inputs(
    input_tokens: torch.Tensor,
    partial_masks: torch.Tensor | None,
) -> torch.Tensor | None:
    if input_tokens.ndim != 2 or input_tokens.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("input_tokens must be an integer tensor with shape (b, l)")
    if input_tokens.shape[-1] == 0:
        raise ValueError("input_tokens must contain at least one token")
    if partial_masks is None:
        return None
    if partial_masks.shape != input_tokens.shape or partial_masks.dtype != torch.bool:
        raise ValueError("partial_masks must be boolean with the same shape as input_tokens")
    if partial_masks.device != input_tokens.device:
        raise ValueError("partial_masks and input_tokens must be on the same device")
    return partial_masks


def _validate_temperature(temperature: float | None, *, default: float = 1.0) -> float:
    if temperature is None:
        temperature = default
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    return temperature


def _steps(max_iter: int, *, show_progress: bool) -> Iterable[int]:
    return tqdm(range(max_iter), desc="Decoding", disable=not show_progress)


def _categorical(
    logits: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature == 0:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores
    distribution = torch.distributions.Categorical(logits=logits.div(temperature))
    tokens = distribution.sample()
    return tokens, distribution.log_prob(tokens)


def _gumbel_argmax(
    logits: torch.Tensor,
    *,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    uniform = torch.rand_like(logits)
    noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
    return _categorical(logits + noise_scale * noise, temperature=0.0)


def _top_p(logits: torch.Tensor, probability: float = 0.95) -> torch.Tensor:
    """Apply the nucleus filter used by the official DPLM samplers."""

    original_shape = logits.shape
    flattened = logits.reshape(-1, original_shape[-1])
    sorted_logits, sorted_indices = flattened.sort(dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative > probability
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits.masked_fill_(remove, -math.inf)
    return sorted_logits.gather(1, sorted_indices.argsort(dim=-1)).reshape(original_shape)


def _lowest_confidence_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    rate: float,
    stochastic_temperature: float | None = None,
) -> torch.Tensor:
    selection_scores = scores.masked_fill(~eligible, 1000.0)
    if stochastic_temperature is not None:
        uniform = torch.rand_like(selection_scores)
        noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        selection_scores = selection_scores + stochastic_temperature * rate * noise
    cutoff_index = (eligible.sum(dim=-1, keepdim=True).to(scores.dtype) * rate).long()
    cutoff_index.clamp_(min=0, max=scores.shape[-1] - 1)
    sorted_scores = selection_scores.sort(dim=-1).values
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_index)
    return (selection_scores < cutoff) & eligible


def _reparameterize(
    output_tokens: torch.Tensor,
    output_scores: torch.Tensor,
    candidate_tokens: torch.Tensor,
    candidate_scores: torch.Tensor,
    active_mask: torch.Tensor,
    eligible: torch.Tensor,
    *,
    mask_token_id: int,
    rate: float,
    stochastic_temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remask = _lowest_confidence_mask(
        candidate_scores,
        eligible,
        rate=rate,
        stochastic_temperature=stochastic_temperature,
    )
    output_tokens.masked_fill_(remask, mask_token_id)
    output_scores.masked_fill_(remask, -math.inf)
    accept = active_mask & eligible & ~remask
    output_tokens.masked_scatter_(accept, candidate_tokens[accept])
    output_scores.masked_scatter_(accept, candidate_scores[accept])
    return remask, output_tokens, output_scores


def _logits(output: object) -> torch.Tensor:
    value = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if not torch.is_tensor(value):
        raise RuntimeError("The masked-language model did not return logits")
    return value


def _suppress_token_ids(logits: torch.Tensor, token_ids: Iterable[int]) -> None:
    vocabulary_size = logits.shape[-1]
    for token_id in token_ids:
        if 0 <= token_id < vocabulary_size:
            logits[..., token_id] = -math.inf


def _dplm_special_id(
    model: _MaskedLanguageModel,
    tokenizer: object | None,
    name: str,
    default: int,
) -> int:
    value = getattr(model.config, name, None)
    if value is None:
        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None)
        value = getattr(tokenizer, name, None)
    return default if value is None else int(value)


def _dplm_resample_repeats(
    model: _MaskedLanguageModel,
    candidate_tokens: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    invalid_token_ids: tuple[int, ...],
    mask_token_id: int,
    ratio: float,
) -> None:
    selected_rows: list[int] = []
    resample_tokens: list[torch.Tensor] = []
    resample_scores: list[torch.Tensor] = []
    resample_masks: list[torch.Tensor] = []
    for row_index, row in enumerate(candidate_tokens):
        positions: dict[int, list[int]] = {}
        for position, token in enumerate(row.tolist()):
            positions.setdefault(int(token), []).append(position)
        repeated = [indices for indices in positions.values() if len(indices) > row.numel() * ratio]
        if not repeated:
            continue
        M = torch.zeros_like(row, dtype=torch.bool)
        for indices in repeated:
            M[indices] = True
        selected_rows.append(row_index)
        resample_masks.append(M)
        resample_tokens.append(row.masked_fill(M, mask_token_id))
        resample_scores.append(candidate_scores[row_index])

    if not selected_rows:
        return
    X = torch.stack(resample_tokens)
    S = torch.stack(resample_scores)
    M = torch.stack(resample_masks)
    with _temporary_eval(model), torch.no_grad():
        logits = _logits(model(input_ids=X, return_dict=True))
    if logits.dtype != S.dtype:
        logits = logits.to(S.dtype)
    _suppress_token_ids(logits, invalid_token_ids)
    logits = _top_p(logits)
    sampled_tokens, sampled_scores = _gumbel_argmax(logits, noise_scale=1.0)
    X.masked_scatter_(M, sampled_tokens[M])
    S.masked_scatter_(M, sampled_scores[M])
    candidate_tokens[selected_rows] = X
    candidate_scores[selected_rows] = S


def generate_dplm(
    model: _MaskedLanguageModel,
    input_tokens: torch.Tensor,
    *,
    tokenizer: object | None = None,
    max_iter: int | None = None,
    temperature: float | None = None,
    partial_masks: torch.Tensor | None = None,
    sampling_strategy: str = "gumbel_argmax",
    disable_resample: bool = False,
    resample_ratio: float = 0.25,
    show_progress: bool = False,
) -> torch.Tensor:
    """Generate DPLM sequences with the official iterative unmasking process.

    ``input_tokens`` is X with shape (b, l). ``partial_masks=True`` marks fixed
    positions. The return value is the generated token tensor X with shape
    (b, l), matching the official DPLM public API.
    """

    partial_masks = _validate_inputs(input_tokens, partial_masks)
    max_iter = _resolve_max_iter(model, max_iter)
    # Upstream treats ``None`` as the falsey, zero-temperature branch for
    # vanilla categorical sampling. Gumbel and argmax strategies ignore it.
    temperature = _validate_temperature(temperature, default=0.0)
    if sampling_strategy not in {"vanilla", "argmax", "gumbel_argmax"}:
        raise ValueError(f"Unsupported DPLM sampling strategy: {sampling_strategy!r}")
    if not 0 < float(resample_ratio) <= 1:
        raise ValueError("resample_ratio must be in (0, 1]")

    pad_id = _dplm_special_id(model, tokenizer, "pad_token_id", 1)
    bos_id = _dplm_special_id(model, tokenizer, "bos_token_id", 0)
    eos_id = _dplm_special_id(model, tokenizer, "eos_token_id", 2)
    mask_id = _dplm_special_id(model, tokenizer, "mask_token_id", 32)
    x_id = 24
    X = input_tokens.clone()
    mutable = X.ne(pad_id) & X.ne(bos_id) & X.ne(eos_id)
    if partial_masks is not None:
        mutable &= ~partial_masks
    X.masked_fill_(mutable, mask_id)
    S = torch.zeros_like(X, dtype=torch.float32)
    active = mutable.clone()
    invalid_ids = (mask_id, x_id, pad_id, bos_id, eos_id)
    for step in _steps(max_iter, show_progress=show_progress):
        with _temporary_eval(model), torch.no_grad():
            logits = _logits(model(input_ids=X, return_dict=True))
        if logits.dtype != S.dtype:
            logits = logits.to(S.dtype)
        _suppress_token_ids(logits, invalid_ids)
        if sampling_strategy == "vanilla":
            candidate_tokens, candidate_scores = _categorical(
                logits,
                temperature=temperature,
            )
        elif sampling_strategy == "argmax":
            candidate_scores, candidate_tokens = logits.max(dim=-1)
        else:
            candidate_tokens, candidate_scores = _gumbel_argmax(logits, noise_scale=1.0)
            if not disable_resample:
                _dplm_resample_repeats(
                    model,
                    candidate_tokens,
                    candidate_scores,
                    invalid_token_ids=invalid_ids,
                    mask_token_id=mask_id,
                    ratio=float(resample_ratio),
                )

        eligible = X.ne(pad_id) & X.ne(bos_id) & X.ne(eos_id)
        if partial_masks is not None:
            eligible &= ~partial_masks
        rate = 1.0 - (step + 1) / max_iter
        active, X, S = _reparameterize(
            X.clone(),
            S.clone(),
            candidate_tokens,
            candidate_scores,
            active,
            eligible,
            mask_token_id=mask_id,
            rate=rate,
        )
    return X


def _normalize_dplm2_special_ids(X: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
    normalized = X.clone()
    replacements = {
        vocabulary_size: _DPLM2_AA_EOS,
        vocabulary_size + 1: _DPLM2_AA_UNK,
        vocabulary_size + 2: _DPLM2_AA_BOS,
        vocabulary_size + 3: _DPLM2_AA_MASK,
    }
    for generic_id, native_id in replacements.items():
        normalized.masked_fill_(X.eq(generic_id), native_id)
    return normalized


def _dplm2_types(X: torch.Tensor) -> torch.Tensor:
    valid = X.ne(_DPLM2_PAD)
    types = ((X < _DPLM2_AA_BOUNDARY) & valid).to(torch.int64)
    types.masked_fill_(~valid, 2)
    return types


def _dplm2_mutable(X: torch.Tensor, partial_masks: torch.Tensor | None) -> torch.Tensor:
    mutable = (
        X.ne(_DPLM2_PAD)
        & X.ne(_DPLM2_AA_BOS)
        & X.ne(_DPLM2_AA_EOS)
        & X.ne(_DPLM2_STRUCT_BOS)
        & X.ne(_DPLM2_STRUCT_EOS)
    )
    if partial_masks is not None:
        mutable &= ~partial_masks
    return mutable


def _dplm2_unmasking_temperature(strategy: str) -> float | None:
    if strategy == "deterministic":
        return None
    if strategy.startswith("stochastic"):
        suffix = strategy.removeprefix("stochastic")
        value = 1.0 if not suffix else float(suffix)
        if not math.isfinite(value) or value < 0:
            raise ValueError("The stochastic unmasking temperature must be non-negative")
        return value
    raise ValueError(f"Unsupported DPLM2 unmasking strategy: {strategy!r}")


def _annealing_temperature(strategy: str, step: int, max_iter: int) -> float | None:
    if not strategy.startswith("annealing"):
        return None
    try:
        maximum, minimum = map(float, strategy.split("@", maxsplit=1)[1].split(":"))
    except (IndexError, ValueError) as error:
        raise ValueError("Annealing must use the form 'annealing@maximum:minimum'") from error
    if not all(math.isfinite(value) and value >= 0 for value in (maximum, minimum)):
        raise ValueError("Annealing temperatures must be finite and non-negative")
    rate = 1.0 - step / max_iter
    return minimum + (maximum - minimum) * rate


def generate_dplm2(
    model: _MaskedLanguageModel,
    input_tokens: torch.Tensor,
    *,
    max_iter: int | None = None,
    temperature: float = 1.0,
    partial_masks: torch.Tensor | None = None,
    unmasking_strategy: str = "stochastic1.0",
    sampling_strategy: str = "annealing@2.0:0.1",
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate packed DPLM2 sequence and structure tracks.

    ``input_tokens`` is X with shape (b, l). A packed co-generation input has
    two equal-length modality tracks. ``partial_masks=True`` marks fixed
    positions. The output mapping matches the official DPLM2 public API.
    """

    partial_masks = _validate_inputs(input_tokens, partial_masks)
    max_iter = _resolve_max_iter(model, max_iter)
    temperature = _validate_temperature(temperature)
    unmasking_temperature = _dplm2_unmasking_temperature(unmasking_strategy)
    if sampling_strategy.startswith("annealing"):
        _annealing_temperature(sampling_strategy, 0, max_iter)
    elif sampling_strategy not in {"argmax", "gumbel_argmax"}:
        raise ValueError(f"Unsupported DPLM2 sampling strategy: {sampling_strategy!r}")
    vocabulary_size = int(model.config.vocab_size)
    if vocabulary_size <= _DPLM2_STRUCT_UNK + 1:
        raise ValueError("DPLM2 generation requires the multimodal vocabulary")
    struct_mask_id = vocabulary_size - 1

    X = _normalize_dplm2_special_ids(input_tokens, vocabulary_size)
    if X.numel() and (X.min() < 0 or X.max() >= vocabulary_size):
        raise ValueError("input_tokens contains an ID outside the DPLM2 vocabulary")
    mutable = _dplm2_mutable(X, partial_masks)
    types = _dplm2_types(X)
    X.masked_fill_(mutable & types.eq(1), _DPLM2_AA_MASK)
    X.masked_fill_(mutable & types.eq(0), struct_mask_id)
    S = torch.zeros_like(X, dtype=torch.float32)
    active = mutable.clone()
    invalid_ids = (
        _DPLM2_AA_BOS,
        _DPLM2_AA_EOS,
        _DPLM2_AA_MASK,
        _DPLM2_STRUCT_BOS,
        _DPLM2_STRUCT_EOS,
        struct_mask_id,
        _DPLM2_PAD,
        _DPLM2_AA_UNK,
        _DPLM2_STRUCT_UNK,
        _DPLM2_AA_X,
        _DPLM2_AA_B,
        _DPLM2_AA_U,
        _DPLM2_AA_Z,
        _DPLM2_AA_O,
    )
    for step in _steps(max_iter, show_progress=show_progress):
        eligible = _dplm2_mutable(X, partial_masks)
        types = _dplm2_types(X)
        with _temporary_eval(model), torch.no_grad():
            logits = _logits(model(input_ids=X, return_dict=True)).log_softmax(dim=-1)
        if logits.dtype != S.dtype:
            logits = logits.to(S.dtype)
        aa_rows, aa_columns = torch.where(types.eq(1) & eligible)
        struct_rows, struct_columns = torch.where(types.eq(0) & eligible)
        logits[aa_rows, aa_columns, _DPLM2_AA_BOUNDARY:] = -math.inf
        logits[struct_rows, struct_columns, :_DPLM2_AA_BOUNDARY] = -math.inf
        _suppress_token_ids(logits, invalid_ids)
        logits = _top_p(logits)

        if sampling_strategy == "argmax":
            candidate_scores, candidate_tokens = logits.max(dim=-1)
        elif sampling_strategy == "gumbel_argmax":
            candidate_tokens, candidate_scores = _gumbel_argmax(
                logits,
                noise_scale=temperature,
            )
            candidate_tokens.masked_scatter_(~eligible, X[~eligible])
        else:
            annealed = _annealing_temperature(sampling_strategy, step, max_iter)
            sample_temperature = temperature if annealed is None else annealed
            candidate_tokens, candidate_scores = _categorical(
                logits,
                temperature=sample_temperature,
            )

        rate = 1.0 - (step + 1) / max_iter
        new_active = torch.zeros_like(active)
        for modality, mask_id in ((1, _DPLM2_AA_MASK), (0, struct_mask_id)):
            modality_positions = types.eq(modality) & eligible
            if not bool(modality_positions.any()):
                continue
            modality_active, X, S = _reparameterize(
                X,
                S,
                candidate_tokens,
                candidate_scores,
                active,
                modality_positions,
                mask_token_id=mask_id,
                rate=rate,
                stochastic_temperature=unmasking_temperature,
            )
            new_active |= modality_active
        active = new_active
    return {"output_tokens": X}


__all__ = ["generate_dplm", "generate_dplm2"]
