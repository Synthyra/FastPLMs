import torch
import random
from torch.nn.functional import mse_loss
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM

from esm2.modeling_fastesm import FastEsmForMaskedLM
from esm2.load_official import load_official_model as load_official_esm2_model

from esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from esm_plusplus.load_official import load_official_model as load_official_esmc_model


class ComplianceChecker:
    def __init__(
        self,
        test_number_batches: int = 10,
        batch_size: int = 4,
        min_sequence_length: int = 16,
        max_sequence_length: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_number_batches = test_number_batches
        self.batch_size = batch_size
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def _load_esmc(self, from_auto_model: bool = False):
        official_model_path = "esmc-300"
        fast_model_path = "Synthyra/ESMplusplus_small"
        official_model, tokenizer = load_official_esmc_model(
            reference_repo_id=official_model_path,
            device=self.device,
            dtype=torch.float32,
        )
        load_class = AutoModelForMaskedLM if from_auto_model else ESMplusplusForMaskedLM
        fast_model = load_class.from_pretrained(
            fast_model_path,
            dtype=torch.float32,
            device_map=self.device,
            force_download=True,
        ).eval()
        fast_model.attn_backend = "sdpa"
        return official_model, fast_model, tokenizer

    def _load_esm2(self, from_auto_model: bool = False):
        official_model_path = "facebook/esm2_t6_8M_UR50D"
        fast_model_path = "Synthyra/ESM2-8M"
        official_model, tokenizer = load_official_esm2_model(
            reference_repo_id=official_model_path,
            device=self.device,
            dtype=torch.float32,
        )
        load_class = AutoModelForMaskedLM if from_auto_model else FastEsmForMaskedLM
        fast_model = load_class.from_pretrained(
            fast_model_path,
            dtype=torch.float32,
            device_map=self.device,
            force_download=True,
        ).eval()
        fast_model.attn_backend = "sdpa"
        return official_model, fast_model, tokenizer

    def _generate_random_sequence(self, length: int) -> str:
        return 'M' + "".join(random.choices(self.canonical_amino_acids, k=length))
    
    def _generate_random_batch(self, batch_size: int, min_length: int, max_length: int) -> list[str]:
        return [self._generate_random_sequence(random.randint(min_length, max_length)) for _ in range(batch_size)]

    def _weight_compliance(self, official_model, fast_model):
        for (official_name, official_param), (fast_name, fast_param) in zip(official_model.model.state_dict().items(), fast_model.state_dict().items()):
            if official_name == fast_name:
                diff = mse_loss(official_param, fast_param).item()
                if diff > 0.0:
                    print(f"{official_name}: {diff}")
                    assert diff < 1e-3, f"Parameter {official_name} has a large difference: {diff}"
            else:
                print(f"Name mismatch: {official_name} != {fast_name}")

    @torch.inference_mode()
    def _foward_compliance(self, official_model, fast_model, tokenizer, only_non_pad_tokens: bool = False):
        cumulative_logits_mse = 0
        cumulative_preds_accuracy = 0
        hidden_state_diff_dict = defaultdict(int)

        for _ in tqdm(range(self.test_number_batches)):
            batch = self._generate_random_batch(self.batch_size, self.min_sequence_length, self.max_sequence_length)
            tokenized = tokenizer(batch, return_tensors="pt", padding=True)
            attention_mask = tokenized['attention_mask'].bool()
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            official_output = official_model(**tokenized, output_hidden_states=True)
            official_hidden_states = official_output.hidden_states
            official_logits = official_output.logits.cpu()
            if only_non_pad_tokens:
                official_logits = official_logits[attention_mask]
            official_preds = official_logits.argmax(dim=-1)
            
            fast_output = fast_model(**tokenized, output_hidden_states=True)
            fast_hidden_states = fast_output.hidden_states
            fast_logits = fast_output.logits.cpu()
            if only_non_pad_tokens:
                fast_logits = fast_logits[attention_mask]
            fast_preds = fast_logits.argmax(dim=-1)

            cumulative_logits_mse += mse_loss(official_logits, fast_logits)
            cumulative_preds_accuracy += (official_preds == fast_preds).float().mean()
            assert cumulative_logits_mse < 1e-3, f"Logits MSE is too large: {cumulative_logits_mse}"
            assert cumulative_preds_accuracy > 0.95, f"Preds accuracy is too low: {cumulative_preds_accuracy}"

            for i in range(len(official_hidden_states)):
                official_state, fast_state = official_hidden_states[i], fast_hidden_states[i]
                if only_non_pad_tokens:
                    official_state, fast_state = official_state[attention_mask], fast_state[attention_mask]
                hidden_state_diff_dict[i] += mse_loss(official_state, fast_state).item()
                assert hidden_state_diff_dict[i] < 1e-3, f"Hidden state {i} MSE is too large: {hidden_state_diff_dict[i]}"

        for k, v in hidden_state_diff_dict.items():
            print(f"Hidden state {k} Avg MSE: {v / self.test_number_batches}")

        print(f"Average logits MSE: {cumulative_logits_mse / self.test_number_batches}")
        print(f"Average preds accuracy: {cumulative_preds_accuracy / self.test_number_batches}")


    def __call__(self, model_type: str = "ESMC", from_auto_model: bool = False, only_non_pad_tokens: bool = False):
        if model_type == "ESMC":
            official_model, fast_model, tokenizer = self._load_esmc(from_auto_model)
        elif model_type == "ESM2":
            official_model, fast_model, tokenizer = self._load_esm2(from_auto_model)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self._weight_compliance(official_model, fast_model)
        self._foward_compliance(official_model, fast_model, tokenizer, only_non_pad_tokens)


if __name__ == "__main__":
    checker = ComplianceChecker()
    # ESMC padding is actually incorrect, so we need to test with only non-pad tokens
    checker(model_type="ESMC", from_auto_model=False, only_non_pad_tokens=True)
    # ESM2 padding is correct, so we can test with all tokens
    checker(model_type="ESM2", from_auto_model=False, only_non_pad_tokens=False)
