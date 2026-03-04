import torch

from e1_fastplms.modeling_e1 import E1BatchPreparer


def analyze_batch_kwargs(batch_kwargs: dict, preparer: E1BatchPreparer, sequences: list[str]) -> None:
    print("==== Batch kwargs analysis ====")

    input_ids = batch_kwargs["input_ids"]
    within_seq_position_ids = batch_kwargs["within_seq_position_ids"]
    global_position_ids = batch_kwargs["global_position_ids"]
    sequence_ids = batch_kwargs["sequence_ids"]
    labels = batch_kwargs["labels"]
    context = batch_kwargs["context"]
    context_len = batch_kwargs["context_len"]

    pad_token_id = preparer.pad_token_id
    def _shortened_list(values: list[int], max_items: int = 8) -> str:
        if len(values) <= max_items:
            return str(values)
        return str(values[:max_items] + [f"... (+{len(values) - max_items} more)"])

    assert input_ids.shape == within_seq_position_ids.shape == global_position_ids.shape == sequence_ids.shape == labels.shape
    batch_size, max_len = input_ids.shape
    assert len(context) == batch_size == len(context_len) == len(sequences)

    print(f"batch_size: {batch_size}")
    print(f"max_length: {max_len}")
    print(f"pad_token_id: {pad_token_id}")
    print(f"kwargs keys: {list(batch_kwargs.keys())}")

    for name, tensor in (
        ("input_ids", input_ids),
        ("within_seq_position_ids", within_seq_position_ids),
        ("global_position_ids", global_position_ids),
        ("sequence_ids", sequence_ids),
        ("labels", labels),
    ):
        assert isinstance(tensor, torch.Tensor)
        non_pad = (tensor != -1).sum().item()
        if tensor.numel() > 0 and tensor.dtype.is_floating_point:
            value_stats = f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
        else:
            value_stats = f"min={tensor.min().item()}, max={tensor.max().item()}"
        print()
        print(f"{name}:")
        print(f"  shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
        first_index = tuple([0] * tensor.ndim)
        print(f"  first_element={tensor[first_index].item()}")
        first_row = tensor[0, : min(8, tensor.shape[1])].tolist()
        print(f"  first_row_prefix={_shortened_list([int(x) for x in first_row], max_items=8)}")
        print(f"  non_padding_count={non_pad} / total={tensor.numel()} ({non_pad / tensor.numel() * 100:.2f}%)")
        print(f"  {value_stats}")

    print()
    print("context tokens (metadata):")
    print(f"  first_context: '{str(context[0])[:50]}'")
    print(f"  first_context_len: {context_len[0]}")
    print(f"  first_sequence: '{sequences[0]}'")
    for i, (raw_sequence, decoded_context, ctx_len, raw_ids) in enumerate(
        zip(sequences, context, context_len, sequence_ids)
    ):
        valid_len = int((raw_ids != -1).sum().item())
        ctx_len = int(ctx_len)
        print(f"  sample[{i}] raw sequence input: {raw_sequence}")
        print(f"    valid_length={valid_len}, context_len={ctx_len}, context='{decoded_context}'")

        row_input_ids = input_ids[i, :valid_len]
        row_sequence_ids = raw_ids[:valid_len]
        row_within = within_seq_position_ids[i, :valid_len]
        row_global = global_position_ids[i, :valid_len]
        row_labels = labels[i, :valid_len]

        print(f"    decoded_input_ids: {preparer.tokenizer.decode(row_input_ids.tolist(), skip_special_tokens=False)}")

        print(f"    input_id_pads: {int((row_input_ids == pad_token_id).sum().item())}")
        print(f"    sequence_id_tail: {row_sequence_ids[-5:].tolist()}")

        assert torch.equal(row_sequence_ids[torch.where(row_sequence_ids != -1)[0][0] : torch.where(row_sequence_ids != -1)[0][-1] + 1], row_sequence_ids[row_sequence_ids != -1])
        unique_sequence_ids = torch.unique(row_sequence_ids[row_sequence_ids != -1]).tolist()
        print(f"    unique sequence_ids: {unique_sequence_ids}")

        seq_boundaries = torch.where(row_sequence_ids[1:] != row_sequence_ids[:-1])[0] + 1
        seq_breaks = seq_boundaries.tolist() + [valid_len]
        seq_lens = []
        start = 0
        for end in seq_breaks:
            seq_lens.append(end - start)
            start = end
        print(f"    per-subsequence token counts (from concatenated encoding): {seq_lens}")

        context_mask = torch.arange(valid_len) < ctx_len
        context_masked = int((row_labels[context_mask] == pad_token_id).sum().item())
        target_mask = torch.arange(valid_len) >= ctx_len
        target_tokens = int((row_labels[target_mask] != pad_token_id).sum().item())
        print(f"    context tokens masked in labels: {context_masked} / {ctx_len}")
        print(f"    non-context target tokens kept: {target_tokens}")

        # Position-id behavior check
        print(f"    within_seq_position_ids unique: {torch.unique(row_within).tolist()}")
        print(f"    global_position_ids max: {int(row_global.max().item())}, min: {int(row_global.min().item())}")
        print()


def main() -> None:
    # Example batch with single-seq and multi-seq inputs.
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTFFLILV,LKQMN",
    ]

    preparer = E1BatchPreparer()
    batch_kwargs = preparer.get_batch_kwargs(sequences, device=torch.device("cpu"))

    analyze_batch_kwargs(batch_kwargs, preparer, sequences)


if __name__ == "__main__":
    main()
