"""Quick test: what does bool_tensor.masked_fill(bool_mask, float('-inf')) do?"""
import torch

am_4d = torch.tensor([[True, True, True, False, False]])[None, None]
print(f"am_4d dtype={am_4d.dtype} shape={am_4d.shape}")
print(f"am_4d: {am_4d}")

result = am_4d.masked_fill(am_4d.logical_not(), float("-inf"))
print(f"\nresult dtype={result.dtype}")
print(f"result: {result}")

# What we actually want
zero_mask = torch.zeros(am_4d.shape, dtype=torch.float32)
zero_mask = zero_mask.masked_fill(am_4d.logical_not(), float("-inf"))
print(f"\nzero_mask: {zero_mask}")

# Test with SDPA
torch.manual_seed(0)
B, H, Q, K, D = 2, 2, 4, 8, 8
q = torch.randn(B, H, Q, D, device="cuda", dtype=torch.float32)
k = torch.randn(B, H, K, D, device="cuda", dtype=torch.float32)
v = torch.randn(B, H, K, D, device="cuda", dtype=torch.float32)

# Mask: first 4 keys valid, last 4 padded for batch 0; all valid for batch 1
mask_2d = torch.tensor([[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1]], dtype=torch.bool, device="cuda")
mask_4d = mask_2d[:, None, None, :]  # (B, 1, 1, K) bool

# Approach 1: bool masked_fill (current code)
m1 = mask_4d.masked_fill(mask_4d.logical_not(), float("-inf"))
print(f"\nm1 dtype={m1.dtype} m1[0]={m1[0]}")

# Approach 2: zeros + masked_fill (proper additive mask)
m2 = torch.zeros(mask_4d.shape, dtype=torch.float32, device="cuda")
m2 = m2.masked_fill(mask_4d.logical_not(), float("-inf"))
print(f"m2 dtype={m2.dtype} m2[0]={m2[0]}")

import torch.nn.functional as F
out1 = F.scaled_dot_product_attention(q, k, v, attn_mask=m1.float() if m1.dtype == torch.bool else m1, scale=1.0)
out2 = F.scaled_dot_product_attention(q, k, v, attn_mask=m2, scale=1.0)
print(f"\nout1 vs out2 (batch=0): mse={((out1[0]-out2[0])**2).mean().item():.3e}")
print(f"out1 vs out2 (batch=1): mse={((out1[1]-out2[1])**2).mean().item():.3e}")

# Now compare with single-batch (B=1) using just batch 0's data
q_single = q[0:1]
k_single = k[0:1, :, :4, :]
v_single = v[0:1, :, :4, :]
out_single = F.scaled_dot_product_attention(q_single, k_single, v_single, scale=1.0)
print(f"\nout_single shape={out_single.shape}, out1[0:1] shape={out1[0:1].shape}")
print(f"out_single vs out1[0]: mse={((out_single[0]-out1[0])**2).mean().item():.3e} maxabs={(out_single[0]-out1[0]).abs().max().item():.3e}")
print(f"out_single vs out2[0]: mse={((out_single[0]-out2[0])**2).mean().item():.3e} maxabs={(out_single[0]-out2[0]).abs().max().item():.3e}")
