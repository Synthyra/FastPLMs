"""Tokenize canonical proteins once, as lookup-table rows, and check the table against the tokenizer.

A canonical sequence is already normalized (uppercase ASCII letters), and an ESM tokenizer maps each
residue letter to one id, so a protein of `l` residues is `l + 2` ids: CLS, the residues, EOS. The
vocabulary builds a 256-entry table from the tokenizer once, then encodes a sequence by one array
lookup instead of a Python call per residue. `verify` holds the table to the tokenizer itself, so
the ids the model sees are the ids the tokenizer would have produced.

Symbols: `l` residues of one sequence, so `l + 2` token ids (row 0 CLS, rows 1..l residues, row `l + 1` EOS).
"""

from __future__ import annotations

import string
import numpy as np

from typing import Any
from numpy.typing import NDArray


UNKNOWN_ID = -1
_LETTERS = string.ascii_uppercase


def check_canonical_text(sequence: str) -> None:
    """Reject text that is not an already normalized uppercase protein; this never normalizes."""
    if not sequence or not sequence.isascii() or not sequence.isalpha() or not sequence.isupper():
        raise ValueError("Canonical feature input must be an already normalized uppercase protein.")


class ResidueVocabulary:
    """Residue letter to token id, built once from a tokenizer and verified against it."""

    def __init__(self, tokenizer: Any) -> None:
        special = set(tokenizer.all_special_ids)
        vocabulary = tokenizer.get_vocab()
        table = np.full(256, UNKNOWN_ID, dtype=np.int64)  # (256,) ASCII code to token id
        for letter in _LETTERS:
            token_id = vocabulary.get(letter)
            if token_id is not None and token_id not in special:
                table[ord(letter)] = token_id
        self.table = table
        self.cls_id = int(tokenizer.cls_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.verify(tokenizer)

    def verify(self, tokenizer: Any) -> None:
        """Hold every mapped letter to the tokenizer: `CLS, letter, EOS` must be its encoding."""
        for letter in _LETTERS:
            token_id = int(self.table[ord(letter)])
            if token_id == UNKNOWN_ID:
                continue
            encoded = tokenizer([letter], add_special_tokens=True)["input_ids"][0]
            if list(encoded) != [self.cls_id, token_id, self.eos_id]:
                raise ValueError(
                    f"Residue {letter!r} encodes as {list(encoded)}, not the table's "
                    f"{[self.cls_id, token_id, self.eos_id]}; the tokenizer is not one token per residue."
                )

    def encode(self, sequence: str) -> NDArray[np.int64]:
        """Token ids `(l + 2,)` of one canonical sequence: CLS, one id per residue, EOS."""
        check_canonical_text(sequence)
        letters = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)  # (l,)
        residues = self.table[letters]  # (l,) token ids, UNKNOWN_ID where the tokenizer lacks the letter
        if bool((residues == UNKNOWN_ID).any()):
            raise ValueError("A canonical residue has no non-special tokenizer representation.")
        ids = np.empty(len(letters) + 2, dtype=np.int64)  # (l+2,)
        ids[0], ids[-1] = self.cls_id, self.eos_id
        ids[1:-1] = residues
        return ids  # (l+2,)


__all__ = ["UNKNOWN_ID", "ResidueVocabulary", "check_canonical_text"]
