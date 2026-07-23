from __future__ import annotations

import contextlib
import math
import numbers
import typing as T
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

_STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_TTT_SERIALIZATION_VERSION = 1


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

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def from_kwargs(cls, **kwargs: T.Any) -> TTTConfig:
        valid_names = {field.name for field in fields(cls)}
        unknown_names = set(kwargs) - valid_names
        if unknown_names:
            raise ValueError(f"Unknown TTTConfig fields: {sorted(unknown_names)}")
        # JSON has no tuple type. Normalize the serialized representation while
        # keeping the public constructor and runtime overrides type-strict.
        if isinstance(kwargs.get("lora_target_modules"), list):
            kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
        return cls(**kwargs)

    def merged(self, overrides: T.Mapping[str, T.Any] | TTTConfig | None) -> TTTConfig:
        if overrides is None:
            return self
        if isinstance(overrides, TTTConfig):
            return overrides
        values = {field.name: self.__dict__[field.name] for field in fields(self)}
        for name, value in overrides.items():
            if name not in values:
                raise ValueError(f"Unknown TTTConfig field: {name}")
            values[name] = value
        return TTTConfig(**values)

    def to_dict(self) -> dict[str, T.Any]:
        return asdict(self)

    def verify(self) -> None:
        numeric_fields = {
            "lr": self.lr,
            "mask_ratio": self.mask_ratio,
            "lora_alpha": self.lora_alpha,
            "bert_leave_prob": self.bert_leave_prob,
            "bert_replace_prob": self.bert_replace_prob,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"TTT {name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"TTT {name} must be finite.")

        integer_fields = {
            "steps": self.steps,
            "ags": self.ags,
            "batch_size": self.batch_size,
            "crop_size": self.crop_size,
            "lora_rank": self.lora_rank,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"TTT {name} must be an integer.")

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("TTT seed must be None or an integer.")

        boolean_fields = {
            "initial_state_reset": self.initial_state_reset,
            "automatic_best_state_reset": self.automatic_best_state_reset,
            "eval_each_step": self.eval_each_step,
            "gradient_clip": self.gradient_clip,
        }
        for name, value in boolean_fields.items():
            if type(value) is not bool:
                raise TypeError(f"TTT {name} must be a boolean.")

        if self.lr <= 0.0:
            raise ValueError("TTT learning rate must be positive.")
        if self.steps < 1:
            raise ValueError("TTT steps must be >= 1.")
        if self.ags < 1:
            raise ValueError("TTT gradient accumulation steps must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("TTT batch_size must be >= 1.")
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("TTT mask_ratio must be in (0, 1].")
        if self.crop_size < 1:
            raise ValueError("TTT crop_size must be >= 1.")
        if self.lora_rank < 1:
            raise ValueError("TTT v1 is LoRA-only, so lora_rank must be >= 1.")
        if self.lora_alpha <= 0.0:
            raise ValueError("TTT lora_alpha must be positive.")
        if not isinstance(self.optimizer, str):
            raise TypeError("TTT optimizer must be a string.")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("TTT optimizer must be 'adamw' or 'sgd'.")
        if self.momentum < 0.0:
            raise ValueError("TTT momentum must be non-negative.")
        if self.weight_decay < 0.0:
            raise ValueError("TTT weight_decay must be non-negative.")
        if not 0.0 <= self.bert_leave_prob <= 1.0:
            raise ValueError("bert_leave_prob must be in [0, 1].")
        if not 0.0 <= self.bert_replace_prob <= 1.0:
            raise ValueError("bert_replace_prob must be in [0, 1].")
        if self.bert_leave_prob + self.bert_replace_prob > 1.0:
            raise ValueError("bert_leave_prob + bert_replace_prob must be <= 1.")
        if self.gradient_clip and self.gradient_clip_max_norm <= 0.0:
            raise ValueError("gradient_clip_max_norm must be positive.")
        if self.lora_target_replace_module is not None:
            if not isinstance(self.lora_target_replace_module, str):
                raise TypeError("lora_target_replace_module must be None or a string.")
            if not self.lora_target_replace_module.strip():
                raise ValueError("lora_target_replace_module must not be empty.")
        if self.lora_target_modules is not None:
            if not isinstance(self.lora_target_modules, tuple):
                raise TypeError("lora_target_modules must be None or a tuple of strings.")
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules must not be empty.")
            if any(not isinstance(name, str) for name in self.lora_target_modules):
                raise TypeError("lora_target_modules must contain only strings.")
            if any(not name.strip() for name in self.lora_target_modules):
                raise ValueError(
                    "lora_target_modules must contain only non-empty strings."
                )
            if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
                raise ValueError("lora_target_modules must not contain duplicates.")


class LoraInjectedLinear(nn.Module):
    """ProteinTTT-compatible low-rank adapter.

    ``alpha`` is the direct adapter-output multiplier used by the pinned
    ProteinTTT ``inject_trainable_lora(..., scale=lora_alpha)`` contract.  It
    is intentionally not divided by ``rank`` as it would be in the common
    PEFT LoRA convention.
    """

    def __init__(
        self,
        linear: nn.Module,
        rank: int,
        alpha: float,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        weight = linear._parameters.get("weight")
        if not isinstance(weight, torch.Tensor):
            raise TypeError("LoRA targets must expose a tensor weight parameter.")
        if weight.ndim != 2:
            raise ValueError("LoRA can only wrap 2D linear weights.")
        self.linear = linear
        self.linear.requires_grad_(False)
        self.rank = rank
        # ProteinTTT names this setting ``lora_alpha`` but passes it directly
        # to cloneofsimo/lora's ``scale`` argument.  Preserve that numerical
        # contract for parity and for saved FastPLMs TTT configurations.
        self.scale = alpha
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        # ``nn.Linear`` initializes from the process-global CPU generator. Preserve
        # that state when TTT supplies its own generator so lazy adapter injection
        # is reproducible without perturbing the caller's RNG stream.
        with torch.random.fork_rng(devices=[], enabled=generator is not None):
            self.lora_down = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
            self.lora_up = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank, generator=generator)
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.to(device=weight.device)
        self.lora_up.to(device=weight.device)
        self.register_buffer(
            "_ttt_initial_lora_down",
            self.lora_down.weight.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "_ttt_initial_lora_up",
            self.lora_up.weight.detach().clone(),
            persistent=True,
        )

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

    def reset_lora_parameters(self) -> None:
        with torch.no_grad():
            self.lora_down.weight.copy_(self._ttt_initial_lora_down)
            self.lora_up.weight.copy_(self._ttt_initial_lora_up)


class FastPLMTestTimeTrainingMixin:
    def init_ttt(self, ttt_config: TTTConfig | T.Mapping[str, T.Any] | None = None) -> None:
        base_config = self.__dict__.get("_ttt_cfg")
        if base_config is None:
            base_config = TTTConfig()
        if not isinstance(base_config, TTTConfig):
            raise TypeError("Existing TTT configuration must be a TTTConfig instance.")
        configured = base_config.merged(ttt_config)
        serialized = getattr(getattr(self, "config", None), "fastplms_ttt", None)
        serialized_initialized = False
        if serialized is not None:
            if not isinstance(serialized, T.Mapping):
                raise ValueError("config.fastplms_ttt must be a mapping.")
            version = serialized.get("version")
            if version != _TTT_SERIALIZATION_VERSION:
                raise ValueError(
                    "Unsupported FastPLMs TTT serialization version "
                    f"{version!r}; expected {_TTT_SERIALIZATION_VERSION}."
                )
            serialized_config = serialized.get("config")
            if not isinstance(serialized_config, T.Mapping):
                raise ValueError("Serialized FastPLMs TTT state is missing its config mapping.")
            configured = TTTConfig.from_kwargs(**dict(serialized_config))
            initialized_value = serialized.get("initialized", False)
            if type(initialized_value) is not bool:
                raise ValueError("Serialized FastPLMs TTT initialized flag must be a boolean.")
            serialized_initialized = initialized_value

        self._ttt_cfg = configured
        self._ttt_cfg.verify()
        self._ttt_initialized = False
        if serialized_initialized:
            self._ttt_inject_lora()
            self._ttt_initialized = True

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
        if seq is None:
            raise ValueError("Pass either seq or input_ids for TTT.")
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
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if unknown_id is not None:
            special_ids.add(int(unknown_id))

        vocab: T.Mapping[str, T.Any] = {}
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
        elif isinstance(getattr(tokenizer, "vocab", None), T.Mapping):
            vocab = tokenizer.vocab
        elif isinstance(getattr(tokenizer, "_token_to_id", None), T.Mapping):
            vocab = tokenizer._token_to_id

        ids: list[int] = []
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        for amino_acid in _STANDARD_AMINO_ACIDS:
            token_id = convert(amino_acid) if callable(convert) else vocab.get(amino_acid)
            if (
                isinstance(token_id, int)
                and 0 <= token_id < vocab_size
                and token_id not in special_ids
                and token_id not in ids
            ):
                ids.append(token_id)
        if not ids:
            raise ValueError(
                "TTT could not resolve any canonical amino-acid token IDs from the tokenizer; "
                "refusing to sample arbitrary or reserved vocabulary entries."
            )
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
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(cfg.seed)

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
                        LoraInjectedLinear(
                            child,
                            rank=cfg.lora_rank,
                            alpha=cfg.lora_alpha,
                            generator=generator,
                        ),
                    )
                    wrapped += 1
                    continue
                inject(child, full_name, child_active)

        for trainable_module in self._ttt_get_trainable_modules():
            inject(trainable_module, "", target_class is None)
        if wrapped == 0:
            raise ValueError("TTT LoRA injection did not find any target modules.")
        return wrapped

    def _ttt_lora_modules(self) -> list[LoraInjectedLinear]:
        return [module for module in self.modules() if isinstance(module, LoraInjectedLinear)]

    def _ttt_lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self._ttt_lora_modules():
            params.extend(module.lora_down.parameters())
            params.extend(module.lora_up.parameters())
        if not params:
            raise RuntimeError("TTT has no LoRA parameters.")
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
        if not snapshot:
            raise RuntimeError("TTT has no LoRA state to snapshot.")
        return snapshot

    def _ttt_restore_lora_state(self, state: list[dict[str, torch.Tensor]]) -> None:
        modules = self._ttt_lora_modules()
        if len(modules) != len(state):
            raise RuntimeError("TTT LoRA state/module count mismatch.")
        with torch.no_grad():
            for module, module_state in zip(modules, state, strict=True):
                module.lora_down.weight.copy_(module_state["lora_down.weight"])
                module.lora_up.weight.copy_(module_state["lora_up.weight"])

    def _ttt_ensure_initialized(self) -> None:
        if "_ttt_cfg" not in self.__dict__:
            self.init_ttt()
        if self._ttt_initialized:
            return
        self._ttt_inject_lora()
        self._ttt_initialized = True

    def ttt_reset(self) -> None:
        self._ttt_ensure_initialized()
        for module in self._ttt_lora_modules():
            module.reset_lora_parameters()

    def _ttt_serialized_contract(self) -> dict[str, T.Any]:
        return {
            "version": _TTT_SERIALIZATION_VERSION,
            "initialized": bool(self._ttt_initialized),
            "config": self.ttt_config.to_dict(),
        }

    def save_pretrained(self, save_directory: T.Any, *args: T.Any, **kwargs: T.Any) -> T.Any:
        """Save initialized adapters, their reset baseline, and the TTT config.

        Adapter injection changes the module tree, so the serialized config must
        reconstruct that tree before Transformers loads the state dict. Models
        whose own state-dict hooks omit their trainable TTT modules fail closed
        instead of producing an artifact that cannot restore the adaptation.
        """

        if self._ttt_initialized:
            state_keys = set(self.state_dict())
            missing_adapter_keys = [
                name
                for name, _ in self.named_parameters()
                if ".lora_" in name and name not in state_keys
            ]
            if missing_adapter_keys:
                raise RuntimeError(
                    "This model attaches TTT adapters to transient modules that its "
                    "checkpoint excludes, so save_pretrained cannot persist the adapted "
                    "state safely. Reset the model or use a model-specific adapter export."
                )
            self.config.fastplms_ttt = self._ttt_serialized_contract()
        return super().save_pretrained(save_directory, *args, **kwargs)

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
        residue_ids = self._ttt_replacement_tokens(input_ids)
        return torch.isin(input_ids, residue_ids)

    def _ttt_validate_tokenized_batch(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        input_ids = self._ttt_input_ids_from_batch(batch)
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError(
                "TTT input_ids must have non-empty shape (batch, sequence); got "
                f"{tuple(input_ids.shape)}."
            )

        if str(getattr(self.config, "model_type", "")) == "dplm2":
            tokenizer = self.tokenizer
            token_to_id = getattr(tokenizer, "_token_to_id", {})
            struct_cls_token = getattr(tokenizer, "struct_cls_token", None)
            struct_boundary = token_to_id.get(struct_cls_token)
            if struct_boundary is None:
                raise ValueError(
                    "DPLM2 TTT could not resolve the structure-token boundary safely."
                )
            pad_token = self._ttt_padding_token()
            generic_aa_special_ids = torch.tensor(
                [int(self.config.vocab_size) + offset for offset in range(4)],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            is_structure = input_ids.ge(int(struct_boundary)) & input_ids.ne(pad_token)
            is_structure &= ~torch.isin(input_ids, generic_aa_special_ids)
            if bool(is_structure.any()):
                raise ValueError(
                    "DPLM2 TTT currently supports amino-acid-only inputs. Packed or "
                    "structure-token inputs require a modality-specific corruption objective."
                )

            if isinstance(batch, dict) and "type_ids" in batch:
                type_ids = batch["type_ids"]
                attention_mask = batch.get("attention_mask", input_ids.ne(pad_token)).bool()
                if bool(((type_ids == int(self.config.struct_type)) & attention_mask).any()):
                    raise ValueError(
                        "DPLM2 TTT currently supports amino-acid-only inputs; structure "
                        "type_ids are not accepted."
                    )

        if not bool(self._ttt_non_special_mask(input_ids).any()):
            raise ValueError(
                "TTT input contains no trainable biological residue tokens after excluding "
                "padding, boundary, mask, and reserved tokens."
            )

    def _ttt_sample_crop(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        input_ids = self._ttt_input_ids_from_batch(batch)
        cfg = self.ttt_config
        if input_ids.shape[1] <= cfg.crop_size:
            return batch
        position_has_residue = self._ttt_non_special_mask(input_ids).any(dim=0).to(torch.int64)
        prefix = F.pad(position_has_residue.cumsum(dim=0), (1, 0))
        window_counts = prefix[cfg.crop_size :] - prefix[: -cfg.crop_size]
        valid_starts = torch.where(window_counts > 0)[0]
        if valid_starts.numel() == 0:
            raise ValueError("TTT could not find a crop containing a biological residue token.")
        selected = torch.randint(
            valid_starts.numel(),
            (1,),
            generator=generator,
            device=input_ids.device,
        )
        start = int(valid_starts[selected].item())
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
        row_has_residue = self._ttt_non_special_mask(input_ids).any(dim=1)
        eligible_rows = torch.where(row_has_residue)[0]
        if eligible_rows.numel() == 0:
            raise ValueError(
                "TTT sampled batch contains no trainable biological residue tokens."
            )
        sampled_row_indices = torch.randint(
            eligible_rows.numel(),
            (cfg.batch_size,),
            generator=generator,
            device=input_ids.device,
        )
        rows = eligible_rows[sampled_row_indices]
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
            num_mask = max(1, round(candidate_positions.numel() * cfg.mask_ratio))
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

    @contextlib.contextmanager
    def _ttt_seed_scope(self, seed: int | None) -> T.Iterator[None]:
        if seed is None:
            yield
            return
        cuda_devices = sorted(
            {
                parameter.device.index
                for parameter in self.parameters()
                if parameter.device.type == "cuda" and parameter.device.index is not None
            }
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            for device_index in cuda_devices:
                with torch.cuda.device(device_index):
                    torch.cuda.manual_seed(seed)
            yield

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
                current_cfg = self.ttt_config
                if next_cfg.lora_rank != current_cfg.lora_rank:
                    raise ValueError(
                        "Changing lora_rank after TTT initialization is not supported."
                    )
                if next_cfg.lora_alpha != current_cfg.lora_alpha:
                    raise ValueError(
                        "Changing lora_alpha after TTT initialization is not supported."
                    )
                if (
                    next_cfg.lora_target_replace_module
                    != current_cfg.lora_target_replace_module
                ):
                    raise ValueError(
                        "Changing LoRA target class after TTT initialization is not supported."
                    )
                if next_cfg.lora_target_modules != current_cfg.lora_target_modules:
                    raise ValueError(
                        "Changing LoRA target modules after TTT initialization is not supported."
                    )
                self._ttt_cfg = next_cfg
            else:
                # Family constructors preconfigure the attention class that may
                # receive LoRA adapters. A first-call mapping changes only the
                # requested fields; rebuilding from TTTConfig defaults here
                # would erase that family target immediately before injection.
                self._ttt_cfg = self.ttt_config.merged(ttt_config)
                self._ttt_cfg.verify()

        cfg = self.ttt_config
        device = next(self.parameters()).device
        tokenized = self._ttt_tokenize(seq=seq, input_ids=input_ids, **kwargs)
        tokenized = self._ttt_to_device(tokenized, device)
        self._ttt_validate_tokenized_batch(tokenized)
        self._ttt_ensure_initialized()
        if cfg.initial_state_reset:
            self.ttt_reset()

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

        with self._ttt_seed_scope(cfg.seed):
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
                    if not bool(labels.ne(-100).any()):
                        raise RuntimeError(
                            "TTT produced an all-ignored label batch; refusing a NaN update."
                        )
                    logits = self._ttt_predict_logits(batch, **kwargs)
                    labels = labels.to(device=logits.device)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"TTT loss is non-finite at micro-step {micro_step + 1}."
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
                        if metric is not None and (best_metric is None or metric > best_metric):
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
