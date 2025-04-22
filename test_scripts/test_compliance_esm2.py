import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import EsmForMaskedLM, AutoModelForMaskedLM
from modeling_fastesm import FastEsmForMaskedLM


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
parser.add_argument('--token', type=str, default=None)
args = parser.parse_args()


if __name__ == "__main__":
    # py -m test_scripts.test_compliance_esm2
    if args.token:
        login(args.token)

    canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = 128
    seq_count = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    def generate_random_sequence(length: int) -> str:
        return 'M' + "".join(random.choices(canonical_amino_acids, k=length-3))


    sequences = [generate_random_sequence(length) for _ in range(seq_count)]


    esm2 = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
    fastesm = FastEsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
    fastesm.lm_head.load_state_dict(esm2.lm_head.state_dict())
    #fastesm = FastEsmForMaskedLM.from_pretrained('Synthyra/ESM2-650M').to(device)
    tokenizer = fastesm.tokenizer

    # Get esmc model outputs
    base_outputs = []
    base_logits = []
    with torch.no_grad():
        for seq in tqdm(sequences):
            input = tokenizer(seq, return_tensors="pt").to(device)
            outputs = esm2(**input, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].float().cpu()
            logits = outputs.logits.float().cpu()
            base_outputs.append(embeddings)
            base_logits.append(logits)
    esm2.cpu()
    del esm2
    torch.cuda.empty_cache()


    # Get plusplus outputs
    total_mse_embeddings = 0
    total_mse_logits = 0
    

    with torch.no_grad():
        for i, seq in tqdm(enumerate(sequences), total=len(sequences)):
            input = tokenizer(seq, return_tensors="pt").to(device)
            outputs = fastesm(**input)
            embeddings = outputs.last_hidden_state.float().cpu()
            logits = outputs.logits.float().cpu()
            
            # Compare embeddings
            mse_embeddings = F.mse_loss(base_outputs[i], embeddings).item()
            # Compare logits
            mse_logits = F.mse_loss(base_logits[i], logits).item()
            
            if mse_embeddings > 0.01 or mse_logits > 0.1:
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
    fastesm.cpu()
    del fastesm
    torch.cuda.empty_cache()

    print(f"Average Embeddings MSE: {total_mse_embeddings / seq_count}")
    print(f"Average Logits MSE: {total_mse_logits / seq_count}")
