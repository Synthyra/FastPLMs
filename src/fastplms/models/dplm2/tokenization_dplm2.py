"""Independent DPLM2 amino-acid and structure tokenizer.

DPLM2 stores two token tracks in one vocabulary. Amino-acid tokens use their
own boundary, unknown, and mask tokens, while structure codes use a separate
set. The generic ``cls_token`` and ``eos_token`` aliases are deliberately not
defined because a caller must choose the modality-specific boundaries.
"""

from __future__ import annotations

from typing import ClassVar
from transformers import AddedToken, EsmTokenizer, PreTrainedTokenizer


class DPLM2Tokenizer(EsmTokenizer):
    """Tokenize the official DPLM2 amino-acid and structure vocabulary.

    Input text is split against the complete pinned vocabulary. Amino-acid
    sequences may be passed as contiguous characters and structure sequences
    as whitespace-separated four-digit codes. Callers constructing a model
    input add ``aa_*`` or ``struct_*`` boundaries explicitly and then use
    ``add_special_tokens=False``. The output token IDs preserve the official
    multimodal vocabulary exactly.
    """

    SPECIAL_TOKENS_ATTRIBUTES: ClassVar[list[str]] = [
        "aa_cls_token",
        "aa_eos_token",
        "aa_unk_token",
        "aa_mask_token",
        "struct_cls_token",
        "struct_eos_token",
        "struct_unk_token",
        "struct_mask_token",
        "pad_token",
    ]
    # The official tokenizer exposes no generic sequence-boundary aliases.
    # Keeping these attributes explicitly set to None preserves that public
    # contract on Transformers v5, whose custom special-token lookup is strict.
    bos_token: ClassVar[None] = None
    cls_token: ClassVar[None] = None
    eos_token: ClassVar[None] = None
    mask_token: ClassVar[None] = None
    sep_token: ClassVar[None] = None
    unk_token: ClassVar[None] = None
    bos_token_id: ClassVar[None] = None
    cls_token_id: ClassVar[None] = None
    eos_token_id: ClassVar[None] = None
    mask_token_id: ClassVar[None] = None
    sep_token_id: ClassVar[None] = None
    unk_token_id: ClassVar[None] = None

    def __init__(
        self,
        vocab_file: str,
        aa_cls_token: str | AddedToken = "<cls_aa>",
        aa_eos_token: str | AddedToken = "<eos_aa>",
        aa_unk_token: str | AddedToken = "<unk_aa>",
        aa_mask_token: str | AddedToken = "<mask_aa>",
        struct_cls_token: str | AddedToken = "<cls_struct>",
        struct_eos_token: str | AddedToken = "<eos_struct>",
        struct_unk_token: str | AddedToken = "<unk_struct>",
        struct_mask_token: str | AddedToken = "<mask_struct>",
        pad_token: str | AddedToken = "<pad>",
        **kwargs: object,
    ) -> None:
        with open(vocab_file, encoding="utf-8") as handle:
            self.all_tokens = [line.strip() for line in handle.read().splitlines()]
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {token: token_id for token_id, token in self._id_to_token.items()}

        # EsmTokenizer would install generic ESM boundary aliases. DPLM2 has
        # modality-specific boundaries instead, so initialize the common
        # tokenizer base with only the nine official special-token fields.
        PreTrainedTokenizer.__init__(
            self,
            aa_cls_token=aa_cls_token,
            aa_eos_token=aa_eos_token,
            aa_unk_token=aa_unk_token,
            aa_mask_token=aa_mask_token,
            struct_cls_token=struct_cls_token,
            struct_eos_token=struct_eos_token,
            struct_unk_token=struct_unk_token,
            struct_mask_token=struct_mask_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)


__all__ = ["DPLM2Tokenizer"]
