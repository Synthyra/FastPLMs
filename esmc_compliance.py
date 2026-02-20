import torch
import random
from torch.nn.functional import mse_loss
from tqdm import tqdm
from collections import defaultdict
#from transformers import AutoModelForMaskedLM

from esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from esm_plusplus.load_official import load_official_model


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TEST_NUMBER_BATCHES = 10
BATCH_SIZE = 4
MIN_SEQUENCE_LENGTH = 16
MAX_SEQUENCE_LENGTH = 64
OFFICIAL_MODEL_PATH = "esmc-300"
FAST_MODEL_PATH = "Synthyra/ESMplusplus_small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


official_model, tokenizer = load_official_model(
    reference_repo_id=OFFICIAL_MODEL_PATH,
    device=DEVICE,
    dtype=torch.float32,
)

fast_model = ESMplusplusForMaskedLM.from_pretrained(
    FAST_MODEL_PATH,
    dtype=torch.float32,
    device_map=DEVICE,
    #force_download=True
).eval()
fast_model.attn_backend = "sdpa"


for (official_name, official_param), (fast_name, fast_param) in zip(official_model.model.state_dict().items(), fast_model.state_dict().items()):
    if official_name == fast_name:
        diff = mse_loss(official_param, fast_param).item()
        if diff > 0.0:
            print(f"{official_name}: {diff}")
    else:
        print(f"Name mismatch: {official_name} != {fast_name}")


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(CANONICAL_AMINO_ACIDS, k=length))


def generate_random_batch(batch_size: int, min_length: int, max_length: int) -> list[str]:
    return [generate_random_sequence(random.randint(min_length, max_length)) for _ in range(batch_size)]


cumulative_last_hidden_state_mse = 0
cumulative_logits_mse = 0
cumulative_preds_accuracy = 0
hidden_state_diff_dict = defaultdict(int)


with torch.inference_mode():
    for _ in tqdm(range(TEST_NUMBER_BATCHES)):
        batch = generate_random_batch(BATCH_SIZE, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
        tokenized = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
        official_output = official_model(**tokenized, output_hidden_states=True)
        official_hidden_states = official_output.hidden_states
        official_logits = official_output.logits.detach().cpu()
        official_preds = official_logits.argmax(dim=-1)
        
        fast_output = fast_model(**tokenized, output_hidden_states=True)
        fast_hidden_states = fast_output.hidden_states
        fast_logits = fast_output.logits.detach().cpu()
        fast_preds = fast_logits.argmax(dim=-1)
        
        #assert torch.allclose(official_logits, fast_logits, atol=1e-3), "Logits mismatch"
        #assert torch.allclose(official_preds, fast_preds, atol=1e-3), "Preds mismatch"

        cumulative_logits_mse += mse_loss(official_logits, fast_logits)
        cumulative_preds_accuracy += (official_preds == fast_preds).float().mean()

        for i in range(len(official_hidden_states)):
            hidden_state_diff_dict[i] += mse_loss(official_hidden_states[i], fast_hidden_states[i]).item()


print(f"Average logits MSE: {cumulative_logits_mse / TEST_NUMBER_BATCHES}")
print(f"Average preds accuracy: {cumulative_preds_accuracy / TEST_NUMBER_BATCHES}")

for k, v in hidden_state_diff_dict.items():
    print(f"Hidden state {k} Avg MSE: {v / TEST_NUMBER_BATCHES}")
