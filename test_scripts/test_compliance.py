import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoModelForMaskedLM
from tqdm.auto import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


"""
Testing if ESM++ outputs are compliant with ESMC outputs
"""

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Synthyra/ESMplusplus_small')
parser.add_argument('--token', type=str, default=None)
args = parser.parse_args()

if args.token:
    login(args.token)

model_path = args.model_path
canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
length = 128
seq_count = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(42)

def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length-3))


sequences = [generate_random_sequence(length) for _ in range(seq_count)]


if 'small' in model_path:
    esmc = ESMC.from_pretrained("esmc_300m", device=device).to(device)
else:
    esmc = ESMC.from_pretrained("esmc_600m", device=device).to(device)


# Get esmc model outputs
base_outputs = []
base_logits = []
with torch.no_grad():
    for seq in tqdm(sequences):
        protein = ESMProtein(sequence=seq)
        protein_tensor = esmc.encode(protein)
        logits_result = esmc.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embeddings = logits_result.embeddings.cpu()
        logits = logits_result.logits.sequence.cpu()
        base_outputs.append(embeddings)
        base_logits.append(logits)
esmc.cpu()
del esmc
torch.cuda.empty_cache()


# Get plusplus outputs
total_mse_embeddings = 0
total_mse_logits = 0
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
tokenizer = model.tokenizer
with torch.no_grad():
    for i, seq in tqdm(enumerate(sequences), total=len(sequences)):
        input = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**input)
        embeddings = outputs.last_hidden_state.cpu()
        logits = outputs.logits.cpu()
        
        # Compare embeddings
        mse_embeddings = F.mse_loss(base_outputs[i], embeddings).item()
        # Compare logits
        mse_logits = F.mse_loss(base_logits[i], logits).item()
        
        if mse_embeddings > 0.001 or mse_logits > 0.001:
            print(f"Sequence {i}:")
            print(f"  Embeddings MSE: {mse_embeddings:.8f}")
            print(f"  Logits MSE: {mse_logits:.8f}")
            
            # Find positions where tensors differ
            diff_embeddings = torch.abs(base_outputs[i] - embeddings)
            diff_logits = torch.abs(base_logits[i] - logits)
            
            # plot diffs
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(diff_embeddings[0].detach().numpy())
            plt.title("Embeddings Difference")
            
            plt.subplot(1, 2, 2)
            plt.imshow(diff_logits[0].detach().numpy())
            plt.title("Logits Difference")
            plt.show()
            
        total_mse_embeddings += mse_embeddings
        total_mse_logits += mse_logits
model.cpu()
del model
torch.cuda.empty_cache()

print(f"Average Embeddings MSE: {total_mse_embeddings / seq_count}")
print(f"Average Logits MSE: {total_mse_logits / seq_count}")
