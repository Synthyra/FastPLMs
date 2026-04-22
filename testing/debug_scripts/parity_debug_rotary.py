"""Verify the rotary inv_freq difference between FastPLMs and native ESMC."""
from __future__ import annotations

import torch

from testing.official.esm_plusplus import load_official_model as load_native_esmc


def diff(label, a, b):
    a = a.float()
    b = b.float()
    print(f"  {label:55s} mse={((a-b)**2).mean().item():.3e}  maxabs={(a-b).abs().max().item():.3e}")


def main() -> int:
    device = torch.device("cuda")

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESMplusplus_small", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()
    native, _ = load_native_esmc(reference_repo_id="esmc-300", device=device, dtype=torch.float32)

    f_rot = fast.transformer.blocks[0].attn.rotary
    n_rot = native.model.transformer.blocks[0].attn.rotary

    print("Before any forward call:")
    diff("fast.inv_freq vs native.inv_freq", f_rot.inv_freq, n_rot.inv_freq)

    print("\nManual recompute on GPU vs on CPU-then-moved:")
    cpu_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device="cpu", dtype=torch.float32) / 64))
    gpu_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64))
    diff("cpu_computed.to(cuda) vs gpu_computed", cpu_inv_freq.to(device), gpu_inv_freq)
    diff("fast.inv_freq vs cpu_computed.to(cuda)", f_rot.inv_freq, cpu_inv_freq.to(device))
    diff("fast.inv_freq vs gpu_computed",           f_rot.inv_freq, gpu_inv_freq)
    diff("native.inv_freq vs cpu_computed.to(cuda)", n_rot.inv_freq, cpu_inv_freq.to(device))
    diff("native.inv_freq vs gpu_computed",         n_rot.inv_freq, gpu_inv_freq)

    print(f"\n  fast inv_freq first 3 entries: {f_rot.inv_freq[:3].tolist()}")
    print(f"  native inv_freq first 3 entries: {n_rot.inv_freq[:3].tolist()}")
    print(f"  cpu_computed first 3 entries:   {cpu_inv_freq[:3].tolist()}")
    print(f"  gpu_computed first 3 entries:   {gpu_inv_freq[:3].tolist()}")

    f_inv_before = f_rot.inv_freq.clone()
    _ = torch.randn(1, 5, 15, 64, device=device)
    f_rot._update_cos_sin_cache(5, device=device, dtype=torch.float32)
    print("\nAfter FastPLMs _update_cos_sin_cache:")
    diff("fast.inv_freq before vs after cache update", f_inv_before, f_rot.inv_freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
