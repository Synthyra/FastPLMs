"""Load official E1 model from the e1 package for comparison."""

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import move_model


class _OfficialE1ForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        batch = {
            "input_ids": input_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "sequence_ids": sequence_ids,
        }
        outputs = self.model(**batch, output_hidden_states=True)
        return outputs


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load the official E1 model from the e1 submodule.

    Args:
        reference_repo_id: e.g. "Profluent-Bio/E1-150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (official_model, batch_preparer) where batch_preparer is an E1BatchPreparer.
    The official model is E1ForMaskedLM with standard HF forward interface.
    """
    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM

    load_kwargs = {
        "revision": reference_revision,
        "tie_word_embeddings": False,
    }
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    # Load through the official public API on CPU, then transfer once below.
    # A Transformers ``device_map`` requires Accelerate even for one device and
    # adds no semantic value for these reference checkpoints.
    model = E1ForMaskedLM.from_pretrained(reference_repo_id, **load_kwargs).eval()
    batch_preparer = E1BatchPreparer()
    wrapped = move_model(_OfficialE1ForwardWrapper(model), device, dtype).eval()
    return wrapped, batch_preparer
