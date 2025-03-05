import argparse
import torch
import numpy as np
import time
from transformers import T5Config, T5EncoderModel
from .t5_flex_attention import T5FlexAttention, replace_t5_attention_with_flex
from .t5_attention import T5AttentionTransformers


def test_attention_layer_equivalence(
    batch_size=2,
    seq_length=16,
    d_model=32,
    num_heads=4,
    d_kv=8,
    seed=42,
    device=None,
    measure_time=False,
    num_warmup=10,
    num_timing_runs=10,
):
    """
    Test the equivalence between T5Attention and T5FlexAttention layers.
    
    Args:
        batch_size: Batch size for test inputs
        seq_length: Sequence length for test inputs
        d_model: Model dimension
        num_heads: Number of attention heads
        d_kv: Dimension of key and value vectors
        seed: Random seed for reproducibility
        device: Device to run the test on
        measure_time: Whether to measure execution time
        num_warmup: Number of warmup runs before timing
        num_timing_runs: Number of runs to average timing over

    Returns:
        max_diff: Maximum absolute difference between outputs
        mean_diff: Mean absolute difference between outputs
        std_time: Average execution time for standard attention (if measure_time=True)
        flex_time: Average execution time for flex attention (if measure_time=True)
    """
    print(f"\n{'='*20} Testing Attention Layer Equivalence {'='*20}")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a simple T5 config
    config = T5Config(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        is_decoder=False,
        use_cache=False,
    )
    
    # Create standard T5Attention and T5FlexAttention layers
    std_attention = T5AttentionTransformers(config, has_relative_attention_bias=True).to(device)
    flex_attention = T5FlexAttention(config, has_relative_attention_bias=True, compile_flex=True).to(device)
    
    # Copy weights from standard attention to flex attention
    flex_attention.q.weight.data = std_attention.q.weight.data.clone()
    flex_attention.k.weight.data = std_attention.k.weight.data.clone()
    flex_attention.v.weight.data = std_attention.v.weight.data.clone()
    flex_attention.o.weight.data = std_attention.o.weight.data.clone()
    flex_attention.relative_attention_bias.weight.data = std_attention.relative_attention_bias.weight.data.clone()
    
    # Create random input
    hidden_states = torch.randn(batch_size, seq_length, d_model).to(device)
    
    # Set both models to eval mode
    std_attention.eval()
    flex_attention.eval()
    
    # Timing measurements
    std_time = 0
    flex_time = 0
    
    if measure_time:
        # Warmup runs
        print(f"Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = std_attention(hidden_states)[0]
                _ = flex_attention(hidden_states)[0]
        
        # Timing runs for standard attention
        print(f"Measuring standard attention over {num_timing_runs} runs...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_timing_runs):
            with torch.no_grad():
                _ = std_attention(hidden_states)[0]
            torch.cuda.synchronize() if device.type == 'cuda' else None
        std_time = (time.time() - start_time) / num_timing_runs
        
        # Timing runs for flex attention
        print(f"Measuring flex attention over {num_timing_runs} runs...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_timing_runs):
            with torch.no_grad():
                _ = flex_attention(hidden_states)[0]
            torch.cuda.synchronize() if device.type == 'cuda' else None
        flex_time = (time.time() - start_time) / num_timing_runs
        
        print(f"Standard attention average time: {std_time*1000:.4f} ms")
        print(f"Flex attention average time: {flex_time*1000:.4f} ms")
        print(f"Speedup: {std_time/flex_time:.2f}x")
    
    # Forward pass for correctness check
    with torch.no_grad():
        std_output = std_attention(hidden_states)[0]
        flex_output = flex_attention(hidden_states)[0]
    
    # Calculate differences
    abs_diff = torch.abs(std_output - flex_output)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    
    if measure_time:
        return max_diff, mean_diff, std_time, flex_time
    else:
        return max_diff, mean_diff


def test_model_equivalence(
    model_name,
    batch_size=2,
    seq_length=16,
    seed=42,
    device=None,
    measure_time=False,
    num_warmup=10,
    num_timing_runs=10,
):
    """
    Test the equivalence between a standard T5EncoderModel and one with flex attention.
    
    Args:
        model_name: Name or path of the pretrained model
        batch_size: Batch size for test inputs
        seq_length: Sequence length for test inputs
        seed: Random seed for reproducibility
        device: Device to run the test on
        measure_time: Whether to measure execution time
        num_warmup: Number of warmup runs before timing
        num_timing_runs: Number of runs to average timing over
    Returns:
        max_diff: Maximum absolute difference between outputs
        mean_diff: Mean absolute difference between outputs
        std_time: Average execution time for standard model (if measure_time=True)
        flex_time: Average execution time for flex model (if measure_time=True)
    """
    print(f"\n{'='*20} Testing Model Equivalence {'='*20}")
    print(f"Using pretrained model: {model_name}")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load the pretrained model
    std_model = T5EncoderModel.from_pretrained(model_name).to(device)
    
    # Create a copy with flex attention
    flex_model = T5EncoderModel.from_pretrained(model_name).to(device)
    flex_model = replace_t5_attention_with_flex(flex_model, compile_flex=True)
    
    # Create random input IDs
    input_ids = torch.randint(0, std_model.config.vocab_size, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Set both models to eval mode
    std_model.eval()
    flex_model.eval()
    
    # Timing measurements
    std_time = 0
    flex_time = 0
    
    if measure_time:
        # Warmup runs
        print(f"Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = std_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                _ = flex_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Timing runs for standard model
        print(f"Measuring standard model over {num_timing_runs} runs...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_timing_runs):
            with torch.no_grad():
                _ = std_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            torch.cuda.synchronize() if device.type == 'cuda' else None
        std_time = (time.time() - start_time) / num_timing_runs
        
        # Timing runs for flex model
        print(f"Measuring flex model over {num_timing_runs} runs...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_timing_runs):
            with torch.no_grad():
                _ = flex_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            torch.cuda.synchronize() if device.type == 'cuda' else None
        flex_time = (time.time() - start_time) / num_timing_runs
        
        print(f"Standard model average time: {std_time*1000:.4f} ms")
        print(f"Flex model average time: {flex_time*1000:.4f} ms")
        print(f"Speedup: {std_time/flex_time:.2f}x")
    
    # Forward pass for correctness check
    with torch.no_grad():
        std_output = std_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        flex_output = flex_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    
    # Calculate differences
    abs_diff = torch.abs(std_output - flex_output)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"Max absolute difference in last_hidden_state: {max_diff:.8f}")
    print(f"Mean absolute difference in last_hidden_state: {mean_diff:.8f}")
    
    if measure_time:
        return max_diff, mean_diff, std_time, flex_time
    else:
        return max_diff, mean_diff


if __name__ == "__main__":
    # py -m wip.t5.test_t5_flex_attention
    parser = argparse.ArgumentParser(description="Test T5 Flex Attention equivalence")
    parser.add_argument("--model_name", type=str, default="Synthyra/ANKH_base", 
                        help="Pretrained model name or path")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for test inputs (default: 2)")
    parser.add_argument("--seq_length", type=int, default=16, 
                        help="Sequence length for test inputs (default: 16)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--tolerance", type=float, default=1e-5, 
                        help="Tolerance for differences (default: 1e-5)")
    parser.add_argument("--measure_time", action="store_true",
                        help="Measure execution time")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="Number of warmup runs before timing (default: 10)")
    parser.add_argument("--num_timing_runs", type=int, default=10,
                        help="Number of runs to average timing over (default: 10)")
    parser.add_argument("--compile_flex", action="store_true",
                        help="Compile flex attention")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}")

    # Test attention layer equivalence
    if args.measure_time:
        layer_max_diff, layer_mean_diff, layer_std_time, layer_flex_time = test_attention_layer_equivalence(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
            device=device,
            measure_time=True,
            num_warmup=args.num_warmup,
            num_timing_runs=args.num_timing_runs
        )
    else:
        layer_max_diff, layer_mean_diff = test_attention_layer_equivalence(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
            device=device
        )
    
    # Test model equivalence
    if args.measure_time:
        model_max_diff, model_mean_diff, model_std_time, model_flex_time = test_model_equivalence(
            model_name=args.model_name,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
            device=device,
            measure_time=True,
            num_warmup=args.num_warmup,
            num_timing_runs=args.num_timing_runs
        )
    else:
        model_max_diff, model_mean_diff = test_model_equivalence(
            model_name=args.model_name,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
            device=device
        )
    
    # Check if differences are within tolerance
    print(f"\n{'='*20} Results {'='*20}")
    print(f"Tolerance threshold: {args.tolerance}")
    
    if layer_max_diff <= args.tolerance:
        print(f"✅ Attention layer test PASSED: Max diff {layer_max_diff:.8f} <= {args.tolerance}")
    else:
        print(f"❌ Attention layer test FAILED: Max diff {layer_max_diff:.8f} > {args.tolerance}")
    
    if model_max_diff <= args.tolerance:
        print(f"✅ Model test PASSED: Max diff {model_max_diff:.8f} <= {args.tolerance}")
    else:
        print(f"❌ Model test FAILED: Max diff {model_max_diff:.8f} > {args.tolerance}")
    
    # Print timing summary if measured
    if args.measure_time:
        print(f"\n{'='*20} Timing Summary {'='*20}")
        print(f"Attention Layer:")
        print(f"  Standard: {layer_std_time*1000:.4f} ms")
        print(f"  Flex:     {layer_flex_time*1000:.4f} ms")
        print(f"  Speedup:  {layer_std_time/layer_flex_time:.2f}x")
        
        print(f"\nFull Model:")
        print(f"  Standard: {model_std_time*1000:.4f} ms")
        print(f"  Flex:     {model_flex_time*1000:.4f} ms")
        print(f"  Speedup:  {model_std_time/model_flex_time:.2f}x") 