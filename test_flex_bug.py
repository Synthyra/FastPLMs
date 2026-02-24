import torch
from torch.nn.attention.flex_attention import create_block_mask

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    if device.type != "cuda":
        print("CUDA must be available for testing flex attention properly.")
        return

    batch_size = 2
    seq_len = 16
    
    attention_mask = torch.ones((batch_size, seq_len), device=device).bool()
    attention_mask[0, 8:] = False
    
    def get_flex_mask(attention_mask):
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return attention_mask[batch_idx, q_idx] & attention_mask[batch_idx, kv_idx]

        flex_block_mask = create_block_mask(
            mask_mod,
            batch_size,
            1,
            seq_len,
            seq_len,
            device=device,
        )
        return flex_block_mask

    @torch.compile
    def compiled_func(q, k, v, mask):
        block_mask = get_flex_mask(mask)
        from torch.nn.attention.flex_attention import flex_attention
        return flex_attention(q, k, v, block_mask=block_mask)

    q = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=torch.bfloat16)
    
    try:
        out = compiled_func(q, k, v, attention_mask)
        print("Original mask_mod succeeded.")
    except Exception as e:
        print("Original mask_mod failed!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
