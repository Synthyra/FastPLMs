import torch
import random
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import EsmForMaskedLM, EsmTokenizer
from esm2.modeling_fastesm import FastEsmForMaskedLM
from weight_parity_utils import assert_state_dict_equal


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TEST_NUMBER_BATCHES = 10
BATCH_SIZE = 4
MIN_SEQUENCE_LENGTH = 16
MAX_SEQUENCE_LENGTH = 64
OFFICIAL_MODEL_PATH = "facebook/esm2_t6_8M_UR50D"
FAST_MODEL_PATH = "Synthyra/ESM2-8M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


official_model = EsmForMaskedLM.from_pretrained(
    OFFICIAL_MODEL_PATH,
    dtype=torch.float32,
    device_map="cpu",
    force_download=True
).to(DEVICE).eval()
fast_model = FastEsmForMaskedLM.from_pretrained(
    FAST_MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cpu",
    force_download=True
).to(DEVICE).eval()
fast_model.attn_backend = "sdpa"

assert_state_dict_equal(
    reference_state_dict=official_model.state_dict(),
    candidate_state_dict=fast_model.state_dict(),
    context="Weight Parity",
)

tokenizer = EsmTokenizer.from_pretrained(OFFICIAL_MODEL_PATH)


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(CANONICAL_AMINO_ACIDS, k=length))


def generate_random_batch(batch_size: int, min_length: int, max_length: int) -> list[str]:
    return [generate_random_sequence(random.randint(min_length, max_length)) for _ in range(batch_size)]


cumulative_last_hidden_state_mse = 0
cumulative_logits_mse = 0
cumulative_preds_accuracy = 0


with torch.inference_mode():
    for _ in tqdm(range(TEST_NUMBER_BATCHES)):
        batch = generate_random_batch(BATCH_SIZE, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
        tokenized = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
        official_output = official_model(**tokenized, output_hidden_states=True)
        official_last_hidden_state = official_output.hidden_states[-1].detach().cpu()
        official_logits = official_output.logits.detach().cpu()
        official_preds = official_logits.argmax(dim=-1)
        
        fast_output = fast_model(**tokenized, output_hidden_states=True)
        fast_last_hidden_state = fast_output.hidden_states[-1].detach().cpu()
        fast_logits = fast_output.logits.detach().cpu()
        fast_preds = fast_logits.argmax(dim=-1)
        
        #assert torch.allclose(official_last_hidden_state, fast_last_hidden_state, atol=1e-3), "Last hidden state mismatch"
        #assert torch.allclose(official_logits, fast_logits, atol=1e-3), "Logits mismatch"
        #assert torch.allclose(official_preds, fast_preds, atol=1e-3), "Preds mismatch"

        cumulative_last_hidden_state_mse += mse_loss(official_last_hidden_state, fast_last_hidden_state)
        cumulative_logits_mse += mse_loss(official_logits, fast_logits)
        cumulative_preds_accuracy += (official_preds == fast_preds).float().mean()


print(f"Average last hidden state MSE: {cumulative_last_hidden_state_mse / TEST_NUMBER_BATCHES}")
print(f"Average logits MSE: {cumulative_logits_mse / TEST_NUMBER_BATCHES}")
print(f"Average preds accuracy: {cumulative_preds_accuracy / TEST_NUMBER_BATCHES}")


