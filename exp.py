import time
import random
import torch
import copy
import numpy as np
from transformers import AutoModel
from tqdm.auto import tqdm


BATCH_SIZE = 4
NUM_WARMUP_BATCHES = 10
NUM_BATCHES = 100


model = ESMplusplusModel.from_pretrained("Synthyra/ESMplusplus_small")
model.attn_backend = "flex"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ThroughputChecker:
    def __init__(
        self,
        warmup_batches: int = 10,
        batch_size: int = 4,
        timed_batches: int = 100,
        min_sequence_length: int = 16,
        max_sequence_length: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_batches = warmup_batches
        self.batch_size = batch_size
        self.timed_batches = timed_batches
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def _load_model(self, model_path: str):
        load_class = AutoModel
        model = load_class.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()
        return model

    def _generate_random_sequence(self, length: int) -> str:
        return 'M' + "".join(random.choices(self.canonical_amino_acids, k=length))
    
    def _generate_random_batch(self, batch_size: int, min_length: int, max_length: int) -> list[str]:
        return [self._generate_random_sequence(random.randint(min_length, max_length)) for _ in range(batch_size)]

    @torch.inference_mode()
    def _time(self, model, tokenizer):
        def time_batches(num_batches: int, message: str):
            start_time = time.time()
            for _ in tqdm(range(num_batches), desc=message):
                batch = self._generate_random_batch(self.batch_size, self.min_sequence_length, self.max_sequence_length)
                tokenized = tokenizer(batch, return_tensors="pt", padding=True)
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                output = model(**tokenized, output_hidden_states=True)
            end_time = time.time()
            return end_time - start_time

        torch.compile(model)
        warmup_time = time_batches(self.warmup_batches, "Warmup")
        torch.cuda.synchronize()
        time_taken = time_batches(self.timed_batches, "Timed")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return time_taken

    def __call__(self, model_path: str):
        model = self._load_model(model_path)
        tokenizer = model.tokenizer
        sdpa_time = self._time(model, tokenizer)
        model.attn_backend = "flex"
        new_model = copy.deepcopy(model)
        model.cpu()
        del model
        torch.cuda.empty_cache()
        flex_time = self._time(new_model, tokenizer)
        return sdpa_time, flex_time
