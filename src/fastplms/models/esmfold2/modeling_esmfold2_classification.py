"""Sequence and residue prediction heads for ESMFold2 checkpoints."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..classification_probe import SequenceClassificationProbe, TokenClassificationProbe
from .configuration_esmfold2 import ESMFold2Config
from .embedding import _TOKEN_TO_ID, _VALID_RESIDUES, _encode_single_chain
from .esmfold2_constants_esm3 import SEQUENCE_PAD_TOKEN
from .modeling_esmfold2 import ESMFold2Model
from .modeling_esmfold2_experimental import ESMFold2ExperimentalModel


ClassifierTrainScope = Literal["probe", "projection"]
_VALID_RESIDUE_IDS = frozenset(_TOKEN_TO_ID[residue] for residue in _VALID_RESIDUES)


class _ESMFold2ClassificationMixin:
    """Run a task probe on the checkpoint-owned ESMC sequence projection."""

    _classifier_config_type = ""
    _keys_to_ignore_on_load_unexpected: list[str] = [r"\._extra_state$"]

    def _initialize_classifier(self, classifier: nn.Module) -> None:
        self.requires_grad_(False)
        self.classifier = classifier
        self.set_classifier_train_scope(self.config.classifier_train_scope)

    def set_classifier_train_scope(self, scope: ClassifierTrainScope) -> None:
        """Select whether fine-tuning updates only the probe or its input projection too."""

        if scope not in {"probe", "projection"}:
            raise ValueError(
                "classifier_train_scope must be 'probe' or 'projection', "
                f"got {scope!r}."
            )
        self.config.classifier_train_scope = scope
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        if scope == "projection":
            self.language_model.base_z_combine.requires_grad_(True)
            self.language_model.base_z_linear.requires_grad_(True)
        if self._esmc is not None:
            self._esmc.requires_grad_(False)

    def load_esmc(self, *args: Any, **kwargs: Any) -> None:
        super().load_esmc(*args, **kwargs)
        if self._esmc is None:
            raise RuntimeError("ESMFold2 ESMC loading completed without a backbone.")
        self._esmc.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._esmc is not None:
            self._esmc.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args: Any, **kwargs: Any):
        if "config" not in kwargs:
            kwargs["config"] = ESMFold2Config.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        config = kwargs["config"]
        if not isinstance(config, ESMFold2Config):
            raise TypeError("ESMFold2 classifiers require an ESMFold2Config.")
        if config.type != cls._classifier_config_type:
            raise ValueError(
                f"{cls.__name__} requires config.type={cls._classifier_config_type!r}, "
                f"got {config.type!r}."
            )
        loaded = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.set_classifier_train_scope(model.config.classifier_train_scope)
        return loaded

    def prepare_classifier_inputs(
        self, sequence_or_sequences: str | list[str] | tuple[str, ...]
    ) -> dict[str, Tensor]:
        """Encode one or more ungapped single-chain proteins without special tokens."""

        sequences = (
            [sequence_or_sequences]
            if isinstance(sequence_or_sequences, str)
            else list(sequence_or_sequences)
        )
        if not sequences:
            raise ValueError("prepare_classifier_inputs requires at least one sequence.")
        encoded = [_encode_single_chain(sequence) for sequence in sequences]
        sequence_length = max(map(len, encoded))
        input_ids = torch.full(
            (len(encoded), sequence_length),
            SEQUENCE_PAD_TOKEN,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_index, token_ids in enumerate(encoded):
            length = len(token_ids)
            input_ids[batch_index, :length] = torch.tensor(
                token_ids, dtype=torch.long, device=self.device
            )
            attention_mask[batch_index, :length] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _classifier_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                "ESMFold2 classifier input_ids must have shape (batch, residue), "
                f"got {tuple(input_ids.shape)}."
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(SEQUENCE_PAD_TOKEN)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError(
                "ESMFold2 classifier attention_mask must match input_ids, got "
                f"{tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
            )
        residue_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not residue_mask.any(dim=1).all():
            raise ValueError("Every ESMFold2 classifier input must contain a protein residue.")
        if input_ids.masked_select(residue_mask).eq(SEQUENCE_PAD_TOKEN).any():
            raise ValueError("ESMFold2 classifier padding tokens cannot be attended residues.")
        residue_ids = input_ids.masked_select(residue_mask)
        valid_residue_ids = torch.tensor(
            sorted(_VALID_RESIDUE_IDS), dtype=input_ids.dtype, device=input_ids.device
        )
        if not torch.isin(residue_ids, valid_residue_ids).all():
            raise ValueError(
                "ESMFold2 classifiers accept residue-only single-chain protein inputs."
            )

        batch_size, sequence_length = input_ids.shape
        residue_index = torch.arange(sequence_length, device=input_ids.device).expand(
            batch_size, -1
        )
        asym_id = torch.zeros_like(input_ids)
        mol_type = torch.zeros_like(input_ids)
        with torch.no_grad():
            hidden_states = self._compute_lm_hidden_states(
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                residue_mask,
            )
        embeddings = self.project_esmc_hidden_states(hidden_states, residue_mask)
        return embeddings, residue_mask

    def _classifier_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        embeddings, residue_mask = self._classifier_embeddings(input_ids, attention_mask)
        return self.classifier(
            embeddings,
            attention_mask=residue_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class _ESMFold2SequenceClassificationMixin(_ESMFold2ClassificationMixin):
    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__(config)
        self._initialize_classifier(SequenceClassificationProbe(config, config.d_pair))

    forward = _ESMFold2ClassificationMixin._classifier_forward


class _ESMFold2TokenClassificationMixin(_ESMFold2ClassificationMixin):
    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__(config)
        self._initialize_classifier(TokenClassificationProbe(config, config.d_pair))

    forward = _ESMFold2ClassificationMixin._classifier_forward


class ESMFold2ForSequenceClassification(
    _ESMFold2SequenceClassificationMixin, ESMFold2Model
):
    """Released ESMFold2 with a sequence classification or regression probe."""

    _classifier_config_type = "release"


class ESMFold2ForTokenClassification(_ESMFold2TokenClassificationMixin, ESMFold2Model):
    """Released ESMFold2 with a residue classification or regression probe."""

    _classifier_config_type = "release"


class ESMFold2ExperimentalForSequenceClassification(
    _ESMFold2SequenceClassificationMixin, ESMFold2ExperimentalModel
):
    """Experimental ESMFold2 with a sequence classification or regression probe."""

    _classifier_config_type = "experimental"


class ESMFold2ExperimentalForTokenClassification(
    _ESMFold2TokenClassificationMixin, ESMFold2ExperimentalModel
):
    """Experimental ESMFold2 with a residue classification or regression probe."""

    _classifier_config_type = "experimental"


__all__ = [
    "ESMFold2ExperimentalForSequenceClassification",
    "ESMFold2ExperimentalForTokenClassification",
    "ESMFold2ForSequenceClassification",
    "ESMFold2ForTokenClassification",
]
