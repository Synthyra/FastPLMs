"""Decompose the diff between FastPLMs and native ESMC hidden states.

Test what LayerNorm actually absorbs by APPLYING the native norm manually
on top of fast_29 and native_29 and seeing what happens.
"""
from __future__ import annotations

import random
import torch

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm_plusplus import load_official_model as load_native_esmc


def per_position_affine_fit(fast: torch.Tensor, native: torch.Tensor):
    mean_n = native.mean(dim=-1, keepdim=True)
    mean_f = fast.mean(dim=-1, keepdim=True)
    n_c = native - mean_n
    f_c = fast - mean_f
    cov = (f_c * n_c).mean(dim=-1, keepdim=True)
    var = (n_c * n_c).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    beta = cov / var
    alpha = mean_f - beta * mean_n
    fitted = alpha + beta * native
    residual = fast - fitted
    return alpha.squeeze(-1), beta.squeeze(-1), residual


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    seq = "M" + "".join(random.choices(CANONICAL_AAS, k=63))

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESMplusplus_small", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()

    native, _ = load_native_esmc(reference_repo_id="esmc-300", device=device, dtype=torch.float32)

    enc = fast.tokenizer([seq], return_tensors="pt", padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        fout = fast(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        nout = native(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    f29 = fout.hidden_states[29].float().clone()
    n29 = nout.hidden_states[29].float().clone()
    f30 = fout.hidden_states[30].float().clone()
    n30 = nout.hidden_states[30].float().clone()

    print(f"f29 vs n29: mse={((f29-n29)**2).mean().item():.3e} maxabs={(f29-n29).abs().max().item():.3e}")
    print(f"f30 vs n30: mse={((f30-n30)**2).mean().item():.3e} maxabs={(f30-n30).abs().max().item():.3e}")

    fast_norm = fast.transformer.norm
    native_norm = native.model.transformer.norm

    print(f"\n  fast norm weight max abs: {fast_norm.weight.abs().max().item():.6f}")
    print(f"  native norm weight max abs: {native_norm.weight.abs().max().item():.6f}")
    print(f"  weights equal: {torch.equal(fast_norm.weight, native_norm.weight)}")
    print(f"  fast norm bias None: {fast_norm.bias is None}  native: {native_norm.bias is None}")

    manual_fast30 = fast_norm(f29)
    manual_native30 = native_norm(n29)
    print(f"\n  manual fast_norm(f29) vs fout.hidden_states[30]:"
          f" mse={((manual_fast30 - f30)**2).mean().item():.3e}")
    print(f"  manual native_norm(n29) vs nout.hidden_states[30]:"
          f" mse={((manual_native30 - n30)**2).mean().item():.3e}")
    print(f"  manual fast_norm(f29) vs manual native_norm(n29):"
          f" mse={((manual_fast30 - manual_native30)**2).mean().item():.3e}")

    cross_fn = fast_norm(n29)
    cross_nf = native_norm(f29)
    print(f"  cross fast_norm(n29) vs nout.hidden_states[30]:"
          f" mse={((cross_fn - n30)**2).mean().item():.3e}")
    print(f"  cross native_norm(f29) vs fout.hidden_states[30]:"
          f" mse={((cross_nf - f30)**2).mean().item():.3e}")

    print("\n  alpha/beta at a few positions (from affine fit):")
    alpha, beta, residual = per_position_affine_fit(f29, n29)
    print(f"  beta per-position: min={beta.min().item():.6f} max={beta.max().item():.6f} mean={beta.mean().item():.6f}")
    print(f"  alpha per-position abs max: {alpha.abs().max().item():.6f}")
    print(f"  residual mse: {(residual**2).mean().item():.3e} ({((residual**2).mean().item() / ((f29-n29)**2).mean().item()) * 100:.1f}% of total)")

    print("\n  check: std of diff per position vs std of native per position:")
    diff = f29 - n29
    diff_std = diff.std(dim=-1).squeeze(0)
    native_std = n29.std(dim=-1).squeeze(0)
    ratio = diff_std / native_std
    print(f"  ratio diff_std / native_std: min={ratio.min().item():.4f} max={ratio.max().item():.4f} mean={ratio.mean().item():.4f}")

    print("\n  compare f29 and n29 row-wise (per position):")
    for pos in [0, 1, 10, 32, 64]:
        if pos >= f29.shape[1]:
            continue
        fp = f29[0, pos]
        np = n29[0, pos]
        print(f"    pos {pos}: mean_f={fp.mean().item():+.4f} std_f={fp.std().item():.4f} "
              f"mean_n={np.mean().item():+.4f} std_n={np.std().item():.4f} "
              f"mean_diff={(fp - np).mean().item():+.4e} corr={torch.corrcoef(torch.stack([fp, np]))[0,1].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
