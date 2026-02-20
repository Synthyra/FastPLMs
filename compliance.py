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

from e1_fastplms.load_official import load_official_model as load_official_e1_model
from e1_fastplms.modeling_e1 import E1ForMaskedLM


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

    def _load_esmc(self, from_auto_model: bool = False, force_download: bool = False):
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
            force_download=force_download,
        ).eval()
        fast_model.attn_backend = "sdpa"
        return official_model, fast_model, tokenizer

    def _load_esm2(self, from_auto_model: bool = False, force_download: bool = False):
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
            force_download=force_download,
        ).eval()
        fast_model.attn_backend = "sdpa"
        return official_model, fast_model, tokenizer

    def _load_e1(self, from_auto_model: bool = False, force_download: bool = False):
        official_model_path = "Profluent-Bio/E1-150m"
        fast_model_path = "Synthyra/Profluent-E1-150M"
        official_model, tokenizer = load_official_e1_model(
            reference_repo_id=official_model_path,
            device=self.device,
            dtype=torch.float32,
        )
        load_class = AutoModelForMaskedLM if from_auto_model else E1ForMaskedLM
        fast_model = load_class.from_pretrained(
            fast_model_path,
            dtype=torch.float32,
            device_map=self.device,
            force_download=force_download,
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
    def _foward_compliance(self, model_type: str, official_model, fast_model, tokenizer, only_non_pad_tokens: bool = False):
        cumulative_logits_mse = 0
        cumulative_preds_accuracy = 0
        hidden_state_diff_dict = defaultdict(int)

        for _ in tqdm(range(self.test_number_batches)):
            batch = self._generate_random_batch(self.batch_size, self.min_sequence_length, self.max_sequence_length)
            if model_type == "E1":
                tokenized = tokenizer.get_batch_kwargs(batch, device=self.device)
                tokenized = {
                    "input_ids": batch["input_ids"],
                    "within_seq_position_ids": batch["within_seq_position_ids"],
                    "global_position_ids": batch["global_position_ids"],
                    "sequence_ids": batch["sequence_ids"],
                    "attention_mask": (batch["sequence_ids"] != -1).long(),
                }
            else:
                tokenized = tokenizer(batch, return_tensors="pt", padding=True)
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            attention_mask = tokenized['attention_mask'].cpu().bool()

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

            for i in range(len(official_hidden_states)):
                official_state, fast_state = official_hidden_states[i], fast_hidden_states[i]
                if only_non_pad_tokens:
                    official_state, fast_state = official_state[attention_mask], fast_state[attention_mask]
                hidden_state_diff_dict[i] += mse_loss(official_state, fast_state).item()

        avg_logits_mse = cumulative_logits_mse / self.test_number_batches
        avg_preds_accuracy = cumulative_preds_accuracy / self.test_number_batches
        print(f"Average logits MSE: {avg_logits_mse}")
        print(f"Average preds accuracy: {avg_preds_accuracy}")

        if avg_logits_mse > 1e-3 or avg_preds_accuracy < 0.95:
            print("Differences were too large, printing hidden state differences for debugging...")
            for k, v in hidden_state_diff_dict.items():
                print(f"Hidden state {k} Avg MSE: {v / self.test_number_batches}")


    def __call__(
        self,
        model_type: str = "ESMC",
        force_download: bool = False,
        from_auto_model: bool = False,
        only_non_pad_tokens: bool = False,
    ):
        if model_type == "ESMC":
            official_model, fast_model, tokenizer = self._load_esmc(from_auto_model, force_download)
        elif model_type == "ESM2":
            official_model, fast_model, tokenizer = self._load_esm2(from_auto_model, force_download)
        elif model_type == "E1":
            official_model, fast_model, tokenizer = self._load_e1(from_auto_model, force_download)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self._weight_compliance(official_model, fast_model)
        self._foward_compliance(model_type, official_model, fast_model, tokenizer, only_non_pad_tokens)


if __name__ == "__main__":
    checker = ComplianceChecker()
    ONLY_NON_PAD_TOKENS = True
    FORCE_DOWNLOAD = False
    checker(model_type="ESMC", from_auto_model=False, only_non_pad_tokens=ONLY_NON_PAD_TOKENS, force_download=FORCE_DOWNLOAD)
    checker(model_type="ESM2", from_auto_model=False, only_non_pad_tokens=ONLY_NON_PAD_TOKENS, force_download=FORCE_DOWNLOAD)
    checker(model_type="E1", from_auto_model=False, only_non_pad_tokens=ONLY_NON_PAD_TOKENS, force_download=FORCE_DOWNLOAD)
