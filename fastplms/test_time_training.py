from __future__ import annotations

import typing as T
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTTConfig:
    lr: float = 4e-4
    steps: int = 30
    ags: int = 16
    batch_size: int = 2
    mask_ratio: float = 0.15
    crop_size: int = 1024
    bert_leave_prob: float = 0.1
    bert_replace_prob: float = 0.1
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    seed: int | None = 0
    lora_rank: int = 8
    lora_alpha: float = 32.0
    lora_target_replace_module: str | None = None
    lora_target_modules: tuple[str, ...] | None = None
    initial_state_reset: bool = True
    automatic_best_state_reset: bool = False
    eval_each_step: bool = False
    gradient_clip: bool = False
    gradient_clip_max_norm: float = 1.0

    @classmethod
    def from_kwargs(cls, **kwargs: T.Any) -> "TTTConfig":
        valid_names = {field.name for field in fields(cls)}
        unknown_names = set(kwargs) - valid_names
        assert len(unknown_names) == 0, f"Unknown TTTConfig fields: {sorted(unknown_names)}"
        return cls(**kwargs)

    def merged(self, overrides: T.Mapping[str, T.Any] | "TTTConfig" | None) -> "TTTConfig":
        if overrides is None:
            return self
        if isinstance(overrides, TTTConfig):
            return overrides
        values = {field.name: self.__dict__[field.name] for field in fields(self)}
        for name, value in overrides.items():
            assert name in values, f"Unknown TTTConfig field: {name}"
            values[name] = value
        return TTTConfig(**values)

    def verify(self) -> None:
        assert self.lr > 0.0, "TTT learning rate must be positive."
        assert self.steps >= 1, "TTT steps must be >= 1."
        assert self.ags >= 1, "TTT gradient accumulation steps must be >= 1."
        assert self.batch_size >= 1, "TTT batch_size must be >= 1."
        assert 0.0 < self.mask_ratio <= 1.0, "TTT mask_ratio must be in (0, 1]."
        assert self.crop_size >= 1, "TTT crop_size must be >= 1."
        assert self.lora_rank >= 1, "TTT v1 is LoRA-only, so lora_rank must be >= 1."
        assert self.lora_alpha > 0.0, "TTT lora_alpha must be positive."
        assert self.optimizer in {"adamw", "sgd"}, "TTT optimizer must be 'adamw' or 'sgd'."
        assert 0.0 <= self.bert_leave_prob <= 1.0, "bert_leave_prob must be in [0, 1]."
        assert 0.0 <= self.bert_replace_prob <= 1.0, "bert_replace_prob must be in [0, 1]."
        assert self.bert_leave_prob + self.bert_replace_prob <= 1.0, (
            "bert_leave_prob + bert_replace_prob must be <= 1."
        )
        if self.gradient_clip:
            assert self.gradient_clip_max_norm > 0.0, "gradient_clip_max_norm must be positive."


class LoraInjectedLinear(nn.Module):
    def __init__(self, linear: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        weight = linear._parameters["weight"]
        assert weight.ndim == 2, "LoRA can only wrap 2D linear weights."
        self.linear = linear
        self.linear.requires_grad_(False)
        self.rank = rank
        self.scale = alpha
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        self.lora_down = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
        self.lora_down.to(device=weight.device)
        self.lora_up.to(device=weight.device)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear._parameters["weight"]

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear._parameters["bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = self.lora_up(self.lora_down(x.to(dtype=torch.float32))) * self.scale
        return base + delta.to(dtype=base.dtype)


class FastPLMTestTimeTrainingMixin:
    def init_ttt(self, ttt_config: TTTConfig | T.Mapping[str, T.Any] | None = None) -> None:
        base_config = TTTConfig()
        self._ttt_cfg = base_config.merged(ttt_config)
        self._ttt_cfg.verify()
        self._ttt_initialized = False
        self._ttt_initial_state: list[dict[str, torch.Tensor]] | None = None

    @property
    def ttt_config(self) -> TTTConfig:
        if "_ttt_cfg" not in self.__dict__:
            self.init_ttt()
        return self._ttt_cfg

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self]

    def _ttt_get_frozen_modules(self) -> list[nn.Module]:
        return []

    def _ttt_tokenize(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: T.Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del kwargs
        if input_ids is not None:
            return input_ids
        assert seq is not None, "Pass either seq or input_ids for TTT."
        tokenized = self.tokenizer(seq, return_tensors="pt", padding=True)
        return tokenized["input_ids"]

    def _ttt_mask_token(self) -> int:
        return int(self.tokenizer.mask_token_id)

    def _ttt_padding_token(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def _ttt_replacement_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.tokenizer
        special_ids = set(tokenizer.all_special_ids)
        vocab_size = int(self.config.vocab_size)
        ids = [idx for idx in range(vocab_size) if idx not in special_ids]
        assert len(ids) > 0, "TTT replacement token set is empty."
        return torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)

    def _ttt_predict_logits(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        **kwargs: T.Any,
    ) -> torch.Tensor:
        del kwargs
        if isinstance(batch, dict):
            output = self(**batch)
            return output.logits
        attention_mask = batch.ne(self._ttt_padding_token())
        output = self(input_ids=batch, attention_mask=attention_mask)
        return output.logits

    def _ttt_eval_step(
        self,
        step: int,
        loss: float,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: T.Any,
    ) -> tuple[dict[str, T.Any], float | None]:
        del step, loss, seq, input_ids, kwargs
        return {}, None

    def _ttt_is_lora_target(
        self,
        name: str,
        full_name: str,
        module: nn.Module,
        active: bool,
        target_modules: tuple[str, ...] | None,
    ) -> bool:
        if not active:
            return False
        if isinstance(module, LoraInjectedLinear):
            return False
        if (
            target_modules is not None
            and name not in target_modules
            and full_name not in target_modules
        ):
            return False
        if isinstance(module, nn.Linear):
            return True
        if "weight" not in module._parameters:
            return False
        weight = module._parameters["weight"]
        if weight is None or weight.ndim != 2:
            return False
        return "Linear" in module.__class__.__name__

    def _ttt_inject_lora(self) -> int:
        cfg = self.ttt_config
        cfg.verify()
        target_class = cfg.lora_target_replace_module
        target_modules = cfg.lora_target_modules
        wrapped = 0

        def inject(module: nn.Module, prefix: str, active: bool) -> None:
            nonlocal wrapped
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                child_active = active
                if target_class is not None:
                    child_active = active or child.__class__.__name__ == target_class
                if self._ttt_is_lora_target(name, full_name, child, child_active, target_modules):
                    setattr(
                        module,
                        name,
                        LoraInjectedLinear(child, rank=cfg.lora_rank, alpha=cfg.lora_alpha),
                    )
                    wrapped += 1
                    continue
                inject(child, full_name, child_active)

        for trainable_module in self._ttt_get_trainable_modules():
            inject(trainable_module, "", target_class is None)
        assert wrapped > 0, "TTT LoRA injection did not find any target modules."
        return wrapped

    def _ttt_lora_modules(self) -> list[LoraInjectedLinear]:
        return [module for module in self.modules() if isinstance(module, LoraInjectedLinear)]

    def _ttt_lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self._ttt_lora_modules():
            params.extend(module.lora_down.parameters())
            params.extend(module.lora_up.parameters())
        assert len(params) > 0, "TTT has no LoRA parameters."
        return params

    def _ttt_snapshot_lora_state(self) -> list[dict[str, torch.Tensor]]:
        snapshot = []
        for module in self._ttt_lora_modules():
            snapshot.append(
                {
                    "lora_down.weight": module.lora_down.weight.detach().clone(),
                    "lora_up.weight": module.lora_up.weight.detach().clone(),
                }
            )
        assert len(snapshot) > 0, "TTT has no LoRA state to snapshot."
        return snapshot

    def _ttt_restore_lora_state(self, state: list[dict[str, torch.Tensor]]) -> None:
        modules = self._ttt_lora_modules()
        assert len(modules) == len(state), "TTT LoRA state/module count mismatch."
        with torch.no_grad():
            for module, module_state in zip(modules, state):
                module.lora_down.weight.copy_(module_state["lora_down.weight"])
                module.lora_up.weight.copy_(module_state["lora_up.weight"])

    def _ttt_ensure_initialized(self) -> None:
        if "_ttt_cfg" not in self.__dict__:
            self.init_ttt()
        if self._ttt_initialized:
            return
        self._ttt_inject_lora()
        self._ttt_initial_state = self._ttt_snapshot_lora_state()
        self._ttt_initialized = True

    def ttt_reset(self) -> None:
        self._ttt_ensure_initialized()
        assert self._ttt_initial_state is not None, "TTT initial state is not available."
        self._ttt_restore_lora_state(self._ttt_initial_state)

    def _ttt_make_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.ttt_config
        params = self._ttt_lora_parameters()
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _ttt_to_device(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return {name: tensor.to(device) for name, tensor in batch.items()}
        return batch.to(device)

    def _ttt_input_ids_from_batch(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch["input_ids"]
        return batch

    def _ttt_set_input_ids(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        input_ids: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            updated = dict(batch)
            updated["input_ids"] = input_ids
            return updated
        return input_ids

    def _ttt_non_special_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_token = self._ttt_padding_token()
        mask = input_ids.ne(pad_token)
        special_ids = set(self.tokenizer.all_special_ids)
        for special_id in special_ids:
            mask = mask & input_ids.ne(int(special_id))
        return mask

    def _ttt_sample_crop(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        input_ids = self._ttt_input_ids_from_batch(batch)
        cfg = self.ttt_config
        if input_ids.shape[1] <= cfg.crop_size:
            return batch
        high = input_ids.shape[1] - cfg.crop_size + 1
        start = int(
            torch.randint(
                high,
                (1,),
                generator=generator,
                device=input_ids.device,
            ).item()
        )
        end = start + cfg.crop_size
        if isinstance(batch, dict):
            cropped = {}
            for name, tensor in batch.items():
                if tensor.ndim >= 2 and tensor.shape[1] == input_ids.shape[1]:
                    cropped[name] = tensor[:, start:end]
                else:
                    cropped[name] = tensor
            return cropped
        return input_ids[:, start:end]

    def _ttt_sample_batch(
        self,
        tokenized: torch.Tensor | dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        cfg = self.ttt_config
        batch = self._ttt_sample_crop(tokenized, generator)
        input_ids = self._ttt_input_ids_from_batch(batch)
        rows = torch.randint(
            input_ids.shape[0],
            (cfg.batch_size,),
            generator=generator,
            device=input_ids.device,
        )
        if isinstance(batch, dict):
            sampled: torch.Tensor | dict[str, torch.Tensor] = {}
            for name, tensor in batch.items():
                if tensor.ndim >= 1 and tensor.shape[0] == input_ids.shape[0]:
                    sampled[name] = tensor.index_select(0, rows)
                else:
                    sampled[name] = tensor
        else:
            sampled = input_ids.index_select(0, rows)

        sampled_ids = self._ttt_input_ids_from_batch(sampled)
        labels = sampled_ids.clone()
        non_special = self._ttt_non_special_mask(sampled_ids)
        label_mask = torch.zeros_like(non_special)
        for row_idx in range(sampled_ids.shape[0]):
            candidate_positions = torch.where(non_special[row_idx])[0]
            if candidate_positions.numel() == 0:
                continue
            num_mask = max(1, int(round(candidate_positions.numel() * cfg.mask_ratio)))
            order = torch.randperm(
                candidate_positions.numel(),
                generator=generator,
                device=sampled_ids.device,
            )
            chosen = candidate_positions[order[:num_mask]]
            label_mask[row_idx, chosen] = True
        labels = labels.masked_fill(~label_mask, -100)

        masked_ids = sampled_ids.clone()
        chosen_positions = torch.where(label_mask)
        if chosen_positions[0].numel() > 0:
            random_values = torch.rand(
                chosen_positions[0].shape,
                generator=generator,
                device=sampled_ids.device,
            )
            leave = random_values < cfg.bert_leave_prob
            replace = (random_values >= cfg.bert_leave_prob) & (
                random_values < cfg.bert_leave_prob + cfg.bert_replace_prob
            )
            mask = ~(leave | replace)
            if mask.any():
                masked_ids[
                    chosen_positions[0][mask],
                    chosen_positions[1][mask],
                ] = self._ttt_mask_token()
            if replace.any():
                replacement_tokens = self._ttt_replacement_tokens(sampled_ids)
                replacement_idx = torch.randint(
                    replacement_tokens.shape[0],
                    (int(replace.sum().item()),),
                    generator=generator,
                    device=sampled_ids.device,
                )
                masked_ids[
                    chosen_positions[0][replace],
                    chosen_positions[1][replace],
                ] = replacement_tokens[replacement_idx]

        return self._ttt_set_input_ids(sampled, masked_ids), labels

    def ttt(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        ttt_config: TTTConfig | T.Mapping[str, T.Any] | None = None,
        **kwargs: T.Any,
    ) -> dict[str, T.Any]:
        if ttt_config is not None:
            if "_ttt_initialized" in self.__dict__ and self._ttt_initialized:
                next_cfg = self.ttt_config.merged(ttt_config)
                assert next_cfg.lora_rank == self.ttt_config.lora_rank, (
                    "Changing lora_rank after TTT initialization is not supported."
                )
                assert next_cfg.lora_alpha == self.ttt_config.lora_alpha, (
                    "Changing lora_alpha after TTT initialization is not supported."
                )
                assert (
                    next_cfg.lora_target_replace_module
                    == self.ttt_config.lora_target_replace_module
                ), "Changing LoRA target class after TTT initialization is not supported."
                assert next_cfg.lora_target_modules == self.ttt_config.lora_target_modules, (
                    "Changing LoRA target modules after TTT initialization is not supported."
                )
                self._ttt_cfg = next_cfg
            else:
                self.init_ttt(ttt_config)

        self._ttt_ensure_initialized()
        cfg = self.ttt_config
        if cfg.initial_state_reset:
            self.ttt_reset()

        device = next(self.parameters()).device
        tokenized = self._ttt_tokenize(seq=seq, input_ids=input_ids, **kwargs)
        tokenized = self._ttt_to_device(tokenized, device)
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        if cfg.seed is not None:
            generator.manual_seed(cfg.seed)

        module_modes = {module: module.training for module in self.modules()}
        requires_grad = {param: param.requires_grad for param in self.parameters()}
        losses: list[float] = []
        step_metrics: list[dict[str, T.Any]] = []
        best_state: list[dict[str, torch.Tensor]] | None = None
        best_metric: float | None = None
        best_step = 0

        try:
            self.train()
            for param in self.parameters():
                param.requires_grad_(False)
            for param in self._ttt_lora_parameters():
                param.requires_grad_(True)

            optimizer = self._ttt_make_optimizer()
            optimizer.zero_grad(set_to_none=True)
            total_micro_steps = cfg.steps * cfg.ags
            for micro_step in range(total_micro_steps):
                batch, labels = self._ttt_sample_batch(tokenized, generator)
                logits = self._ttt_predict_logits(batch, **kwargs)
                labels = labels.to(device=logits.device)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                (loss / cfg.ags).backward()
                if (micro_step + 1) % cfg.ags != 0:
                    continue

                if cfg.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self._ttt_lora_parameters(),
                        cfg.gradient_clip_max_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = (micro_step + 1) // cfg.ags
                loss_value = float(loss.detach().item())
                losses.append(loss_value)
                if cfg.eval_each_step:
                    metrics, metric = self._ttt_eval_step(
                        step=step,
                        loss=loss_value,
                        seq=seq,
                        input_ids=input_ids,
                        **kwargs,
                    )
                    if len(metrics) > 0:
                        step_metrics.append(metrics)
                    if metric is not None and (
                        best_metric is None or metric > best_metric
                    ):
                        best_metric = metric
                        best_step = step
                        best_state = self._ttt_snapshot_lora_state()

            if cfg.automatic_best_state_reset and best_state is not None:
                self._ttt_restore_lora_state(best_state)
        finally:
            for param, value in requires_grad.items():
                param.requires_grad_(value)
            for module, training in module_modes.items():
                module.train(training)

        return {
            "losses": losses,
            "step_metrics": step_metrics,
            "best_step": best_step,
            "best_metric": best_metric,
        }
