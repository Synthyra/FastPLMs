"""Load official E1 model from the e1 package for comparison."""
import torch
import torch.nn as nn


class _OfficialE1ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ):
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
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official E1 model from the e1 submodule.

    Args:
        reference_repo_id: e.g. "Profluent-Bio/E1-150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (official_model, batch_preparer) where batch_preparer is an E1BatchPreparer.
    The official model is E1ForMaskedLM with standard HF forward interface.
    """
    from E1.modeling import E1ForMaskedLM
    from E1.batch_preparer import E1BatchPreparer

    model = E1ForMaskedLM.from_pretrained(reference_repo_id).to(device=device, dtype=dtype).eval()
    batch_preparer = E1BatchPreparer()
    wrapped = _OfficialE1ForwardWrapper(model).eval()
    return wrapped, batch_preparer
