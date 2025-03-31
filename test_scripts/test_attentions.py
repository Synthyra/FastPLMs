import torch
import argparse
from datasets import load_dataset
from modeling_esm_plusplus import ESMplusplusModel
from tqdm import tqdm


def test_attention_outputs(model, tokenizer, seqs, batch_size=4, tolerances=[1e-3, 1e-5, 1e-7, 1e-9]):
    """
    Test if hidden states are the same with and without attention output at different tolerance levels.
    
    Args:
        model: The model to test
        tokenizer: The tokenizer to use
        seqs: List of sequences to process
        batch_size: Batch size for processing
        tolerances: List of tolerance values to test with torch.allclose
        
    Returns:
        dict: Results for each tolerance level
    """
    results = {tol: True for tol in tolerances}
    max_diff = 0.0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc='Testing attention outputs'):
            batch_seqs = seqs[i:i+batch_size]
            
            # Tokenize the batch
            tokenized = tokenizer(batch_seqs, padding=True, return_tensors='pt')
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
            
            # Get output without attention
            output_no_att = model(**tokenized).last_hidden_state.detach().cpu()
            
            # Get output with attention
            output_with_att = model(**tokenized, output_attentions=True).last_hidden_state.detach().cpu()
            
            # Calculate maximum difference
            diff = (output_no_att - output_with_att).abs().max().item()
            max_diff = max(max_diff, diff)
            print(max_diff)
            
            # Check for NaN or infinite values
            has_nan_or_inf_no_att = torch.isnan(output_no_att).any() or torch.isinf(output_no_att).any()
            has_nan_or_inf_with_att = torch.isnan(output_with_att).any() or torch.isinf(output_with_att).any()
            if has_nan_or_inf_no_att or has_nan_or_inf_with_att:
                print(f"WARNING: Found NaN or infinite values in the outputs! No att: {has_nan_or_inf_no_att}, With att: {has_nan_or_inf_with_att}")
            
            # Test different tolerance levels
            for tol in tolerances:
                if not torch.allclose(output_no_att, output_with_att, atol=tol):
                    results[tol] = False
            
    return results, max_diff


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test attention outputs in ESM++ models')
    parser.add_argument('--model', type=str, default='Synthyra/ESMplusplus_small', help='Model to test')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    args = parser.parse_args()
    
    # Load data
    seqs = load_dataset('Synthyra/NEGATOME', split='manual_stringent').filter(lambda x: len(x['SeqA']) <= 256).select(range(args.num_samples))['SeqA']
    seqs = list(set(seqs))
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESMplusplusModel.from_pretrained(args.model).to(device)
    tokenizer = model.tokenizer
    
    # Define tolerance levels to test
    tolerances = [1e-2, 1e-4, 1e-6, 1e-8]
    
    # Run tests
    print(f"Testing model: {args.model}")
    print(f"Device: {device}")
    print(f"Testing {len(seqs)} sequences with batch size {args.batch_size}")
    
    results, max_diff = test_attention_outputs(
        model, 
        tokenizer, 
        seqs, 
        batch_size=args.batch_size, 
        tolerances=tolerances
    )
    
    # Report results
    print("\nTest Results:")
    print(f"Maximum absolute difference: {max_diff:.10e}")
    print("\nTolerance tests:")
    for tol in sorted(tolerances):
        status = "PASSED" if results[tol] else "FAILED"
        print(f"  Tolerance {tol:.0e}: {status}")
    
    # Overall result
    if all(results.values()):
        print("\nAll tests PASSED! Hidden states are identical with and without attention output.")
    else:
        min_passing_tol = min([tol for tol, passed in results.items() if passed], default=None)
        if min_passing_tol:
            print(f"\nTest PASSED at tolerance {min_passing_tol:.0e} and above.")
        else:
            print("\nAll tests FAILED. Hidden states differ significantly when output_attentions is True vs False.")