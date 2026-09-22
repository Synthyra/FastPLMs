"""ESMFold2 atom attention modes: dense by default, windowed only on request.

The windowed mode runs PyTorch's variable-length FlashAttention, so its numerical
checks need CUDA. They compare it with a dense reference that applies the same
window over unpadded atom ranks.
"""

from __future__ import annotations

import pytest
import torch

from torch import nn

from fastplms.models.esmfold2 import modeling_esmfold2_common as common
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel


# The 3D rotary table needs 14 frequency pairs, so a head must be at least 28 wide.
D_ATOM = 128  # d_atom: atom-state width.
N_HEADS = 4
HALF_WINDOW = 2
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _attention(half_window: int = HALF_WINDOW) -> common.SWA3DRoPEAttention:
    torch.manual_seed(7)
    return common.SWA3DRoPEAttention(D_ATOM, N_HEADS, half_window=half_window).eval()


def _attention_params(atom_mask: torch.Tensor) -> tuple:
    batch_size, n_atoms = atom_mask.shape
    generator = torch.Generator().manual_seed(3)
    ref_pos = torch.randn(batch_size, n_atoms, 3, generator=generator)  # (b, n_atoms, xyz)
    ref_space_uid = torch.arange(n_atoms).div(3, rounding_mode="floor").expand(batch_size, -1)
    cos, sin = common.build_3d_rope(ref_pos, ref_space_uid, head_dim=D_ATOM // N_HEADS)
    _, indices, cu_seqlens, max_seqlen, _ = common._prepare_atom_encoder_metadata(
        atom_mask, torch.zeros(batch_size, n_atoms, dtype=torch.long), 1
    )
    return cos, sin, indices, cu_seqlens, max_seqlen


def _windowed_reference(
    module: common.SWA3DRoPEAttention, x: torch.Tensor, params: tuple, atom_mask: torch.Tensor
) -> torch.Tensor:
    """Dense attention restricted to ``half_window`` real atoms on each side."""
    batch_size, n_atoms = x.shape[:2]
    qkv = module.Wqkv(x).view(batch_size, n_atoms, 3, N_HEADS, D_ATOM // N_HEADS)
    q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each (b, n_atoms, h, d_h)
    q = common.apply_rotary_emb_3d(common.qk_norm(q), params[0], params[1])
    k = common.apply_rotary_emb_3d(common.qk_norm(k), params[0], params[1])
    # The kernel receives BF16 inputs; quantize identically, then attend in FP32.
    q, k, v = (states.bfloat16().float().transpose(1, 2) for states in (q, k, v))
    rank = atom_mask.long().cumsum(-1) - 1  # (b, n_atoms), position among real atoms
    in_window = (rank[:, :, None] - rank[:, None, :]).abs() <= module.half_window
    allowed = in_window & atom_mask[:, :, None] & atom_mask[:, None, :]  # (b, n_atoms, n_atoms)
    logits = torch.matmul(q, k.transpose(-2, -1)) * module.scale  # (b, h, n_atoms, n_atoms)
    logits = logits.masked_fill(~allowed[:, None], float("-inf"))
    weights = torch.softmax(logits, dim=-1).nan_to_num(0.0)  # padded query rows have no key
    attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, n_atoms, D_ATOM)
    return module.out_proj(attended * torch.sigmoid(module.gate_proj(x)))


def test_dense_is_the_default_and_unknown_modes_raise() -> None:
    module = _attention()
    assert module._atom_attention == "dense"
    with pytest.raises(ValueError, match=r"must be one of \('dense', 'windowed'\)"):
        module.set_atom_attention("sliding")
    assert module._atom_attention == "dense"


def test_windowed_mode_names_a_missing_pytorch_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_varlen_attn", None)
    with pytest.raises(RuntimeError, match=r"requires torch\.nn\.attention\.varlen"):
        _attention().set_atom_attention("windowed")


def test_windowed_mode_rejects_cpu_tensors_instead_of_falling_back() -> None:
    if common._varlen_attn is None:
        pytest.skip("this PyTorch build has no variable-length attention")
    module = _attention()
    module.set_atom_attention("windowed")
    atom_mask = torch.ones(1, 8, dtype=torch.bool)  # (1, 8)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"), torch.no_grad():
        module(torch.randn(1, 8, D_ATOM), _attention_params(atom_mask))


@pytest.mark.parametrize("model_class", (ESMFold2Model, ESMFold2ExperimentalModel))
def test_model_setter_reaches_every_atom_attention_module(model_class: type) -> None:
    class AtomStacks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = common.SWAAtomTransformer(d_atom=D_ATOM, n_blocks=2, n_heads=N_HEADS)
            self.decoder = common.SWAAtomTransformer(d_atom=D_ATOM, n_blocks=1, n_heads=N_HEADS)

    stacks = AtomStacks()
    if common._varlen_attn is not None:
        model_class.set_atom_attention(stacks, "windowed")
        modes = [m._atom_attention for m in stacks.modules() if hasattr(m, "_atom_attention")]
        assert modes == ["windowed"] * 3
    with pytest.raises(ValueError, match="must be one of"):
        model_class.set_atom_attention(stacks, "flash")


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("padded", (False, True))
def test_windowed_attention_matches_a_dense_window_over_real_atoms(padded: bool) -> None:
    device = torch.device("cuda")
    atom_mask = torch.ones(3, 16, dtype=torch.bool)  # (b, n_atoms)
    if padded:
        atom_mask[0, 11:] = False
        atom_mask[1, 4] = False  # an interior gap shifts every later window
        atom_mask[2, 1:] = False
    module = _attention().to(device)
    params = tuple(
        value.to(device) if isinstance(value, torch.Tensor) else value
        for value in _attention_params(atom_mask)
    )
    atom_mask = atom_mask.to(device)  # (3, 16)
    x = torch.randn(3, 16, D_ATOM, generator=torch.Generator().manual_seed(1)).to(device)  # (3, 16, d_atom)

    module.set_atom_attention("windowed")
    with torch.no_grad():
        windowed = module(x, params)  # (b, n_atoms, d_atom)
        reference = _windowed_reference(module, x, params, atom_mask)
        module.set_atom_attention("dense")
        dense = module(x, params)

    torch.testing.assert_close(windowed[atom_mask], reference[atom_mask], atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(windowed[~atom_mask]) == 0
    # The window is narrower than the sample, so dense attention must differ clearly.
    assert (dense[atom_mask] - reference[atom_mask]).abs().max() > 0.05


@requires_cuda
@pytest.mark.gpu
def test_a_window_wider_than_the_sample_matches_dense_attention() -> None:
    device = torch.device("cuda")
    atom_mask = torch.ones(2, 16, dtype=torch.bool)  # (2, 16)
    module = _attention(half_window=64).to(device)
    params = tuple(
        value.to(device) if isinstance(value, torch.Tensor) else value
        for value in _attention_params(atom_mask)
    )
    x = torch.randn(2, 16, D_ATOM, generator=torch.Generator().manual_seed(2)).to(device)  # (2, 16, d_atom)

    with torch.no_grad():
        dense = module(x, params)
        module.set_atom_attention("windowed")
        windowed = module(x, params)

    torch.testing.assert_close(windowed, dense, atol=2e-2, rtol=2e-2)
