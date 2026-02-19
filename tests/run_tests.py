"""
FastPLMs test suite.

Runs three tiers of tests for each model family:
  1. Weight parity (float32): Compare state_dict tensors between our model and official.
  2. Output parity (SDPA, float32): Run both models on same input, compare hidden states + logits.
  3. Flex attention sanity (bfloat16): Cast our model to bfloat16, run with flex attention,
     confirm outputs are not too different from the SDPA float32 baseline.

Usage:
    py -m tests.run_tests
    py -m tests.run_tests --families esm2 dplm
    py -m tests.run_tests --device cpu --skip-flex
"""

import argparse
import sys

import torch

from tests.common import (
    LOAD_DTYPE,
    RUNTIME_DTYPE,
    compare_state_dicts,
    generate_sequences,
    get_non_pad_mask,
    load_official_model,
    load_our_model,
    set_seed,
    tokenize_batch,
    tokenize_official_batch,
)
from tests.model_registry import REPRESENTATIVE_MODELS, ModelSpec


WEIGHT_PARITY_TOLERANCE = 0.0
OUTPUT_MSE_TOLERANCE = 1e-5
OUTPUT_MAX_ABS_TOLERANCE = 1e-3
FLEX_VS_SDPA_MAX_ABS_TOLERANCE = 0.5
FLEX_ARGMAX_MIN_ACCURACY = 0.8


def test_weight_parity(spec: ModelSpec, device: torch.device) -> bool:
    """Test 1: Load both models in float32, compare state_dict tensors exactly."""
    print(f"\n  [Weight Parity] Loading our model: {spec.repo_id}")
    our_model, _ = load_our_model(spec, device=device, dtype=LOAD_DTYPE)
    print(f"  [Weight Parity] Loading official model: {spec.reference_repo_id}")
    official_model, _ = load_official_model(spec, device=device, dtype=LOAD_DTYPE)

    our_sd = our_model.state_dict()
    if spec.family in ("esm2",):
        # ESM2: official has position_embeddings, ours uses RoPE
        official_sd = {
            k: v for k, v in official_model.state_dict().items()
            if "position_embeddings" not in k
        }
    elif spec.family in ("dplm2",):
        # DPLM2: official has contact_head weights we don't use
        excluded = {"esm.contact_head.regression.weight", "esm.contact_head.regression.bias"}
        official_sd = {k: v for k, v in official_model.state_dict().items() if k not in excluded}
    else:
        official_sd = official_model.state_dict()

    result = compare_state_dicts(official_sd, our_sd)

    if result["match"]:
        print(f"  [Weight Parity] PASS - {result['common_params']} params match exactly")
        del our_model, official_model
        torch.cuda.empty_cache()
        return True
    else:
        print(f"  [Weight Parity] FAIL")
        print(f"    Max abs diff: {result['max_abs_diff']:.2e} ({result['max_diff_param']})")
        if result["only_in_reference"]:
            print(f"    Only in official: {result['only_in_reference']}")
        if result["only_in_candidate"]:
            print(f"    Only in ours: {result['only_in_candidate']}")
        if result["diffs"]:
            for d in result["diffs"]:
                print(f"    {d}")
        del our_model, official_model
        torch.cuda.empty_cache()
        return False


def _extract_hidden_and_logits(outputs, spec: ModelSpec, is_official: bool):
    """Extract last hidden state and logits from model outputs, handling per-family differences."""
    logits = outputs.logits.detach().cpu().float()
    if outputs.hidden_states is not None:
        hidden = outputs.hidden_states[-1].detach().cpu().float()
    elif is_official and spec.family == "e1":
        hidden = outputs.embeddings.detach().cpu().float()
    else:
        hidden = outputs.last_hidden_state.detach().cpu().float()
    return hidden, logits


def test_output_parity(
    spec: ModelSpec,
    device: torch.device,
    sequences: list[str],
) -> bool:
    """Test 2: SDPA float32 output comparison between our model and official."""
    print(f"\n  [Output Parity] Loading our model (SDPA, fp32): {spec.repo_id}")
    our_model, our_tokenizer = load_our_model(spec, device=device, dtype=LOAD_DTYPE, attn_backend="sdpa")
    print(f"  [Output Parity] Loading official model: {spec.reference_repo_id}")
    official_model, official_tokenizer = load_official_model(spec, device=device, dtype=LOAD_DTYPE)

    our_batch = tokenize_batch(spec, sequences, our_model, our_tokenizer, device)
    official_batch = tokenize_official_batch(spec, sequences, official_tokenizer, device)

    with torch.no_grad():
        our_outputs = our_model(**our_batch, output_hidden_states=True)
        official_outputs = official_model(**official_batch, output_hidden_states=True)

    our_hidden, our_logits = _extract_hidden_and_logits(our_outputs, spec, is_official=False)
    official_hidden, official_logits = _extract_hidden_and_logits(official_outputs, spec, is_official=True)

    mask_2d = get_non_pad_mask(spec, our_batch)
    mask = mask_2d.unsqueeze(-1)

    hidden_diff = torch.abs(our_hidden - official_hidden)
    masked_hidden_diff = hidden_diff * mask
    hidden_mse = float((masked_hidden_diff ** 2).sum() / mask.sum() / our_hidden.shape[-1])
    hidden_max_abs = float(masked_hidden_diff.max())

    logits_diff = torch.abs(our_logits - official_logits)
    masked_logits_diff = logits_diff * mask
    logits_mse = float((masked_logits_diff ** 2).sum() / mask.sum() / our_logits.shape[-1])
    logits_max_abs = float(masked_logits_diff.max())

    our_argmax = our_logits.argmax(dim=-1)
    official_argmax = official_logits.argmax(dim=-1)
    argmax_match = float(((our_argmax == official_argmax) * mask_2d).sum() / mask_2d.sum())

    print(f"  [Output Parity] Hidden: MSE={hidden_mse:.2e}, MaxAbs={hidden_max_abs:.2e}")
    print(f"  [Output Parity] Logits: MSE={logits_mse:.2e}, MaxAbs={logits_max_abs:.2e}")
    print(f"  [Output Parity] Argmax accuracy: {argmax_match:.4f}")

    passed = (
        hidden_mse < OUTPUT_MSE_TOLERANCE
        and logits_max_abs < OUTPUT_MAX_ABS_TOLERANCE
        and argmax_match > 0.99
    )
    print(f"  [Output Parity] {'PASS' if passed else 'FAIL'}")

    del our_model, official_model
    torch.cuda.empty_cache()
    return passed


def test_flex_attention(
    spec: ModelSpec,
    device: torch.device,
    sequences: list[str],
) -> bool:
    """Test 3: Cast our model to bfloat16, run flex attention, compare to SDPA fp32 baseline."""
    print(f"\n  [Flex Attention] Loading our model (SDPA, fp32 baseline): {spec.repo_id}")
    baseline_model, tokenizer = load_our_model(spec, device=device, dtype=LOAD_DTYPE, attn_backend="sdpa")
    batch = tokenize_batch(spec, sequences, baseline_model, tokenizer, device)

    with torch.no_grad():
        baseline_outputs = baseline_model(**batch, output_hidden_states=True)
    baseline_logits = baseline_outputs.logits.detach().cpu().float()

    del baseline_model
    torch.cuda.empty_cache()

    print(f"  [Flex Attention] Loading our model (flex, bf16): {spec.repo_id}")
    flex_model, flex_tokenizer = load_our_model(spec, device=device, dtype=LOAD_DTYPE, attn_backend="flex")
    flex_model = flex_model.to(dtype=RUNTIME_DTYPE)

    flex_batch = tokenize_batch(spec, sequences, flex_model, flex_tokenizer, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=RUNTIME_DTYPE):
        flex_outputs = flex_model(**flex_batch, output_hidden_states=True)
    flex_logits = flex_outputs.logits.detach().cpu().float()

    mask_2d = get_non_pad_mask(spec, batch)
    mask_3d = mask_2d.unsqueeze(-1)
    diff = torch.abs(flex_logits - baseline_logits) * mask_3d
    max_abs_diff = float(diff.max())

    flex_argmax = flex_logits.argmax(dim=-1)
    baseline_argmax = baseline_logits.argmax(dim=-1)
    argmax_accuracy = float(((flex_argmax == baseline_argmax) * mask_2d).sum() / mask_2d.sum())

    print(f"  [Flex Attention] MaxAbsDiff: {max_abs_diff:.4f}")
    print(f"  [Flex Attention] Argmax accuracy vs baseline: {argmax_accuracy:.4f}")

    passed = max_abs_diff < FLEX_VS_SDPA_MAX_ABS_TOLERANCE and argmax_accuracy > FLEX_ARGMAX_MIN_ACCURACY
    print(f"  [Flex Attention] {'PASS' if passed else 'FAIL'}")

    del flex_model
    torch.cuda.empty_cache()
    return passed


def main():
    assert torch.cuda.is_available(), "CUDA is required to run the test suite."

    parser = argparse.ArgumentParser(description="FastPLMs test suite")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus", "dplm", "dplm2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--skip-flex", action="store_true", help="Skip flex attention tests")
    parser.add_argument("--skip-official", action="store_true", help="Skip official model loading (weight/output parity)")
    args = parser.parse_args()

    device = torch.device("cuda")
    set_seed(args.seed)
    sequences = generate_sequences(args.num_sequences, args.min_length, args.max_length, args.seed)

    specs = REPRESENTATIVE_MODELS
    if args.families is not None:
        normalized = [f.strip().lower() for f in args.families]
        specs = [s for s in specs if s.family in normalized]
    assert len(specs) > 0, "No models selected."

    print(f"Device: {device}")
    print(f"Models to test: {[s.key for s in specs]}")
    print(f"Test sequences: {len(sequences)} (lengths {args.min_length}-{args.max_length})")
    print("=" * 70)

    results: dict[str, dict[str, bool]] = {}

    for spec in specs:
        print(f"\n{'='*70}")
        print(f"Testing: {spec.key} ({spec.family})")
        print(f"  Our repo:      {spec.repo_id}")
        print(f"  Official repo:  {spec.reference_repo_id}")
        print(f"{'='*70}")

        spec_results: dict[str, bool] = {}

        if not args.skip_official and spec.reference_repo_id is not None:
            spec_results["weight_parity"] = test_weight_parity(spec, device)
            spec_results["output_parity"] = test_output_parity(spec, device, sequences)

        if not args.skip_flex:
            spec_results["flex_attention"] = test_flex_attention(spec, device, sequences)

        results[spec.key] = spec_results

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_passed = True
    for model_key, spec_results in results.items():
        for test_name, passed in spec_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {model_key:30s} {test_name:25s} {status}")
            if not passed:
                all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
