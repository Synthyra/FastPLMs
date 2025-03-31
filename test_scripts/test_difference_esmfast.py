import torch
import torch.nn.functional as F
import random
import argparse
from huggingface_hub import login
from transformers import AutoModelForMaskedLM, EsmTokenizer
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='lhallee/synthyra_esm2_650_mlm')
parser.add_argument('--token', type=str, default=None)
args = parser.parse_args()

if args.token:
    login(args.token)

model_path = args.model_path
canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
length = 128
seq_count = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length))


tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
# Generate sequences first
sequences = [generate_random_sequence(length) for _ in range(seq_count)]
inputs = [tokenizer(seq, return_tensors="pt").to(device) for seq in sequences]


# Get base model outputs
base_outputs = []
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
with torch.no_grad():
    for input_batch in tqdm(inputs):
        base_outputs.append(model(**input_batch, output_hidden_states=True).hidden_states[-1].cpu())
model.cpu()
del model
torch.cuda.empty_cache()


# Get fp16 outputs
fp16_mse = 0
model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
with torch.no_grad():
    for i, input_batch in tqdm(enumerate(inputs), total=len(inputs)):
        fp16_output = model(**input_batch, output_hidden_states=True).hidden_states[-1].float().cpu()
        fp16_mse += F.mse_loss(base_outputs[i], fp16_output).item()
model.cpu()
del model
torch.cuda.empty_cache()


# Get bfloat16 outputs
bf16_mse = 0
model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
with torch.no_grad():
    for i, input_batch in tqdm(enumerate(inputs), total=len(inputs)):
        bf16_output = model(**input_batch, output_hidden_states=True).hidden_states[-1].float().cpu()
        bf16_mse += F.mse_loss(base_outputs[i], bf16_output).item()
model.cpu()
del model
torch.cuda.empty_cache()

fp16_mse /= seq_count
bf16_mse /= seq_count

print(f"Average MSE for FP16: {fp16_mse:.8f}")
print(f"Average MSE for BF16: {bf16_mse:.8f}")
print(f"{'FP16' if fp16_mse < bf16_mse else 'BF16'} has lower MSE")
