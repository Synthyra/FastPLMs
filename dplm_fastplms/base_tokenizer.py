from transformers import PreTrainedTokenizerBase


class BaseSequenceTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, sequences, **kwargs):
        raise NotImplementedError
