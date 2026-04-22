"""Instrument each sub-operation of block 0 and block 1 to pinpoint where
FastPLMs and native ESMC diverge in fp32.

Methodology: grab intermediate tensors from both implementations via forward
hooks on corresponding submodules, then compare element-wise.
"""
from __future__ import annotations

import random
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm_plusplus import load_official_model as load_native_esmc


def diff_stats(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float()
    b = b.float()
    d = a - b
    print(f"    {label:40s} mse={((d**2).mean().item()):.3e}  maxabs={d.abs().max().item():.3e}  shape={tuple(a.shape)}")


class Capture:
    """Hook that stores the module's input and output."""
    def __init__(self):
        self.inputs = None
        self.outputs = None
    def __call__(self, module, inputs, outputs):
        self.inputs = tuple(x.detach().clone() if isinstance(x, torch.Tensor) else x for x in inputs)
        self.outputs = outputs.detach().clone() if isinstance(outputs, torch.Tensor) else outputs


def attach(mod, cap):
    return mod.register_forward_hook(cap)


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

    fast_caps = {}
    native_caps = {}
    hooks = []
    for i in (0, 1):
        fast_block = fast.transformer.blocks[i]
        native_block = native.model.transformer.blocks[i]

        for name, fmod, nmod in [
            ("attn.layernorm_qkv", fast_block.attn.layernorm_qkv, native_block.attn.layernorm_qkv),
            ("attn.q_ln",          fast_block.attn.q_ln,          native_block.attn.q_ln),
            ("attn.k_ln",          fast_block.attn.k_ln,          native_block.attn.k_ln),
            ("attn.rotary",        fast_block.attn.rotary,        native_block.attn.rotary),
            ("attn.out_proj",      fast_block.attn.out_proj,      native_block.attn.out_proj),
            ("attn",               fast_block.attn,               native_block.attn),
            ("ffn",                fast_block.ffn,                native_block.ffn),
        ]:
            fcap = Capture()
            ncap = Capture()
            fast_caps[(i, name)] = fcap
            native_caps[(i, name)] = ncap
            hooks.append(attach(fmod, fcap))
            hooks.append(attach(nmod, ncap))

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        fo = fast(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        no = native(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    for h in hooks:
        h.remove()

    for i in (0, 1):
        print(f"\n== block {i} ==")
        diff_stats(f"block {i} INPUT (x)", fast_caps[(i, "attn")].inputs[0], native_caps[(i, "attn")].inputs[0])
        diff_stats(f"  attn.layernorm_qkv OUTPUT", fast_caps[(i, "attn.layernorm_qkv")].outputs, native_caps[(i, "attn.layernorm_qkv")].outputs)
        diff_stats(f"  attn.q_ln OUTPUT",          fast_caps[(i, "attn.q_ln")].outputs,          native_caps[(i, "attn.q_ln")].outputs)
        diff_stats(f"  attn.k_ln OUTPUT",          fast_caps[(i, "attn.k_ln")].outputs,          native_caps[(i, "attn.k_ln")].outputs)

        f_rot_in_q  = fast_caps[(i, "attn.rotary")].inputs[0]
        n_rot_in_q  = native_caps[(i, "attn.rotary")].inputs[0]
        diff_stats(f"  attn.rotary INPUT q",       f_rot_in_q, n_rot_in_q)
        f_rot_out_q = fast_caps[(i, "attn.rotary")].outputs[0]
        n_rot_out_q = native_caps[(i, "attn.rotary")].outputs[0]
        diff_stats(f"  attn.rotary OUTPUT q",      f_rot_out_q, n_rot_out_q)

        diff_stats(f"  attn.out_proj INPUT",       fast_caps[(i, "attn.out_proj")].inputs[0], native_caps[(i, "attn.out_proj")].inputs[0])
        diff_stats(f"  attn.out_proj OUTPUT",      fast_caps[(i, "attn.out_proj")].outputs,   native_caps[(i, "attn.out_proj")].outputs)
        diff_stats(f"  attn OUTPUT (full)",        fast_caps[(i, "attn")].outputs[0],          native_caps[(i, "attn")].outputs)

        diff_stats(f"  ffn INPUT",                 fast_caps[(i, "ffn")].inputs[0],  native_caps[(i, "ffn")].inputs[0])
        diff_stats(f"  ffn OUTPUT",                fast_caps[(i, "ffn")].outputs,    native_caps[(i, "ffn")].outputs)
        diff_stats(f"block {i} OUTPUT",            fo.hidden_states[i],              no.hidden_states[i])

    print("\nRotary cos/sin caches at block 0 vs block 1 within FastPLMs (sanity):")
    f0 = fast.transformer.blocks[0].attn.rotary
    f1 = fast.transformer.blocks[1].attn.rotary
    n0 = native.model.transformer.blocks[0].attn.rotary
    n1 = native.model.transformer.blocks[1].attn.rotary
    diff_stats("fast b0 cos vs native b0 cos", f0._cos_cached, n0._cos_cached)
    diff_stats("fast b1 cos vs native b1 cos", f1._cos_cached, n1._cos_cached)
    diff_stats("fast b0 cos vs fast b1 cos",   f0._cos_cached, f1._cos_cached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
