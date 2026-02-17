from typing import Any

import torch
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DynamicCache:
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.

    Args:
        key_cache (`list[torch.Tensor]`): The list of key states.
        value_cache (`list[torch.Tensor]`): The list of value states.
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            value_states (`torch.Tensor`): The new value states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            layer_idx (`int`): The index of the layer to update.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states of shape [batch_size, seq_len, num_heads, head_dim].
        """
        # Lazy initialization
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[1] if not is_empty_layer else 0
        return layer_seq_length

    def crop(self, max_length: int) -> None:
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        assert max_length > 0, "max_length must be positive"

        if self.get_seq_length() <= max_length:
            return

        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :max_length, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :max_length, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class KVCache:
    def __init__(self, cache_size: int = 4) -> None:
        self.cache_size = cache_size
        self.tensor_input_field_names = [
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
            "labels",
        ]
        self.tensor_output_field_names = ["logits", "embeddings"]
        self.cache_dict: dict[str, DynamicCache] = {}
        self.cache_queue: list[str] = []

    def reset(self) -> None:
        for k in list(self.cache_dict.keys()):
            del self.cache_dict[k]
        del self.cache_dict
        self.cache_dict = {}
        self.cache_queue = []

        torch.cuda.empty_cache()

    def before_forward(self, batch: dict[str, torch.Tensor]) -> None:
        contexts: list[str] | None = batch.get("context", None)
        if contexts is None or "context_len" not in batch:
            logger.warning_once(
                "KVCache requires the batch dict to have both `context` and `context_len` keys to trigger. Skipping."
            )
            return

        context_lens: list[int] = list(set(batch["context_len"]))
        contexts: list[str] = list(set(contexts))  # type: ignore[no-redef]
        if len(contexts) != 1 or len(context_lens) != 1:
            logger.warning(
                "SingleContextKVCache requires a single context and context length. "
                "Multiple contexts or context lengths found in a single batch. Skipping."
            )
            return

        batch_size = batch["input_ids"].shape[0]

        unique_context = contexts[0]
        unique_context_len = context_lens[0]
        batch["use_cache"] = True

        if unique_context not in self.cache_dict:
            return

        self.cache_dict[unique_context].batch_repeat_interleave(batch_size)
        past_key_values = self.cache_dict[unique_context]
        batch["past_key_values"] = past_key_values

        # Remove context from the input fields
        for field_name in self.tensor_input_field_names:
            if batch.get(field_name, None) is not None:
                batch[field_name] = batch[field_name][:, unique_context_len:]

    def after_forward(self, batch: dict[str, Any], outputs: ModelOutput) -> None:
        contexts = batch.get("context", None)
        context_lens = batch.get("context_len", [])
        if contexts is None or len(set(contexts)) != 1 or len(set(context_lens)) != 1 or context_lens[0] == 0:
            return

        assert batch["use_cache"]
        unique_context = contexts[0]
        unique_context_len = context_lens[0]

        past_key_values = getattr(outputs, "past_key_values", None)
        if not isinstance(past_key_values, DynamicCache):
            logger.warning_once("KVCache is incompatible with models that don't return a DynamicCache. Skipping.")
            return

        if "past_key_values" not in batch:
            if len(self.cache_queue) == self.cache_size:
                last_context = self.cache_queue.pop(0)
                if last_context not in self.cache_queue:
                    del self.cache_dict[last_context]
                    torch.cuda.empty_cache()

            self.cache_dict[unique_context] = past_key_values
            self.cache_queue.append(unique_context)

            # Remove context from the input fields
            for field_name in self.tensor_input_field_names:
                if field_name in batch and batch[field_name] is not None:
                    batch[field_name] = batch[field_name][:, unique_context_len:]

            # Remove context from the output fields
            for field_name in self.tensor_output_field_names:
                if field_name in outputs and outputs[field_name] is not None:
                    outputs[field_name] = outputs[field_name][:, unique_context_len:]
            if "hidden_states" in outputs and outputs["hidden_states"] is not None:
                outputs["hidden_states"] = [h[:, unique_context_len:] for h in outputs["hidden_states"]]

        self.cache_dict[unique_context].crop(unique_context_len)
        self.cache_dict[unique_context].batch_select_indices([0])
