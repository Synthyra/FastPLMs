"""Check whether DPLM2 fast lm_head matches native DPLM2 lm_head or embedding."""
from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, EsmForMaskedLM, AutoConfig


def main() -> int:
    device = torch.device("cuda")
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/DPLM2-150M", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()
    native = EsmForMaskedLM.from_pretrained(
        "airkingbd/dplm2_150m", dtype=torch.float32, device_map=device,
    ).eval()
    native_cfg = AutoConfig.from_pretrained("airkingbd/dplm2_150m")

    print(f"native tie_word_embeddings: {getattr(native_cfg, 'tie_word_embeddings', None)}")
    print(f"fast  tie_word_embeddings: {fast.config.tie_word_embeddings}")

    fsd = fast.state_dict()
    nsd = native.state_dict()

    f_lm = fsd["lm_head.decoder.weight"]
    n_lm = nsd["lm_head.decoder.weight"]
    n_emb = nsd["esm.embeddings.word_embeddings.weight"]
    f_emb = fsd["esm.embeddings.word_embeddings.weight"]

    print(f"fast lm_head   == native lm_head?   max|d|={(f_lm - n_lm).abs().max().item():.3e}")
    print(f"native lm_head == native word_emb?  max|d|={(n_lm - n_emb).abs().max().item():.3e}")
    print(f"fast lm_head   == native word_emb?  max|d|={(f_lm - n_emb).abs().max().item():.3e}")
    print(f"fast word_emb  == native word_emb?  max|d|={(f_emb - n_emb).abs().max().item():.3e}")
    print(f"fast word_emb  == fast lm_head?     max|d|={(f_emb - f_lm).abs().max().item():.3e}")

    # Check biases if present.
    for k in ("lm_head.decoder.bias", "lm_head.bias"):
        if k in fsd and k in nsd:
            print(f"fast[{k}] vs native[{k}] max|d|={(fsd[k] - nsd[k]).abs().max().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
