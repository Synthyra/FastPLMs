"""Figure out why native DPLM2 OOBs on tokens produced by the native tokenizer."""
from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, EsmForMaskedLM, EsmTokenizer, AutoConfig


def main() -> int:
    device = torch.device("cuda")
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/DPLM2-150M", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()
    native = EsmForMaskedLM.from_pretrained(
        "airkingbd/dplm2_150m", dtype=torch.float32, device_map=device,
    ).eval()
    native_tok = EsmTokenizer.from_pretrained("airkingbd/dplm2_150m")
    fast_tok = fast.tokenizer
    native_cfg = AutoConfig.from_pretrained("airkingbd/dplm2_150m")

    print(f"fast config.vocab_size   = {fast.config.vocab_size}")
    print(f"native config.vocab_size = {native_cfg.vocab_size}")
    print(f"fast word_emb shape: {fast.state_dict()['esm.embeddings.word_embeddings.weight'].shape}")
    print(f"native word_emb shape: {native.state_dict()['esm.embeddings.word_embeddings.weight'].shape}")

    print(f"\nfast tok vocab size: {len(fast_tok.get_vocab())}")
    print(f"native tok vocab size: {len(native_tok.get_vocab())}")
    print(f"fast tok pad_token_id: {fast_tok.pad_token_id}  cls: {fast_tok.cls_token_id}  eos: {fast_tok.eos_token_id}  mask: {fast_tok.mask_token_id}")
    print(f"native tok pad_token_id: {native_tok.pad_token_id}  cls: {native_tok.cls_token_id}  eos: {native_tok.eos_token_id}  mask: {native_tok.mask_token_id}")

    seqs = ["MALW", "MKTIIALSY"]
    fast_enc = fast_tok(seqs, return_tensors="pt", padding=True)
    native_enc = native_tok(seqs, return_tensors="pt", padding=True)
    print(f"\nfast_enc input_ids[0] = {fast_enc['input_ids'][0].tolist()}")
    print(f"native_enc input_ids[0] = {native_enc['input_ids'][0].tolist()}")
    print(f"max fast_enc id = {fast_enc['input_ids'].max().item()}")
    print(f"max native_enc id = {native_enc['input_ids'].max().item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
