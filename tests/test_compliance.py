import torch
import torch.nn.functional as F
import random
import argparse
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoModelForMaskedLM
from tqdm.auto import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


"""
Testing if ESM++ outputs are compliant with ESMC outputs
"""


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Synthyra/ESMplusplus_small')
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
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length-3))


sequences = [generate_random_sequence(length) for _ in range(seq_count)]


if 'small' in model_path:
    esmc = ESMC.from_pretrained("esmc_300m", device=device).to(device)
else:
    esmc = ESMC.from_pretrained("esmc_600m", device=device).to(device)


# Get esmc model outputs
base_outputs = []
with torch.no_grad():
    for seq in tqdm(sequences):
        protein = ESMProtein(sequence=seq)
        protein_tensor = esmc.encode(protein)
        embeddings = esmc.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        ).embeddings.cpu()
        base_outputs.append(embeddings)
esmc.cpu()
del esmc
torch.cuda.empty_cache()


# Get plusplus outputs
total_mse = 0
model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
tokenizer = model.tokenizer
with torch.no_grad():
    for i, seq in tqdm(enumerate(sequences), total=len(sequences)):
        input = tokenizer(seq, return_tensors="pt").to(device)
        embeddings = model(**input).last_hidden_state.cpu()
        mse = F.mse_loss(base_outputs[i], embeddings).item()
        if mse > 0.0001:
            print(f"MSE for sequence {i}: {mse:.8f}")
            # Find positions where tensors differ
            diff = torch.abs(base_outputs[i] - embeddings)
            # plot diff
            plt.imshow(diff[0].detach().numpy())
            plt.show()
            
        total_mse += mse
model.cpu()
del model
torch.cuda.empty_cache()

print(f"Average MSE: {mse / seq_count}")
