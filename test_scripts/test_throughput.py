import torch
import time
import random
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForMaskedLM, EsmTokenizer
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


parser = argparse.ArgumentParser()
parser.add_argument('--model_paths', nargs='+', type=str, default=[
    #'facebook/esm2_t6_8M_UR50D',
    'Synthyra/FastESM2_650',
    'facebook/esm2_t12_35M_UR50D',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    'esmc_300m', # esmc model
    'esmc_600m', # esmc model
    'Synthyra/ESMplusplus_small',
    'Synthyra/ESMplusplus_large'
])
parser.add_argument('--token', type=str, default=None)
parser.add_argument('--test', action='store_true', help='Generate random results for testing')
args = parser.parse_args()

if args.token:
    login(args.token)

model_paths = args.model_paths
canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ESMCForEmbedding(torch.nn.Module):
    def __init__(self, esm):
        super().__init__()
        self.esm = esm
    
    def forward(self, seq):
        protein = ESMProtein(sequence=seq)
        protein_tensor = self.esm.encode(protein)
        embeddings = self.esm.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        ).embeddings.cpu()
        return embeddings


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length-3))


def generate_batch_sequences(length: int, batch_size: int, num_batches: int = 100) -> list:
    all_sequences = []
    for _ in range(num_batches):
        batch_sequences = [generate_random_sequence(length) for _ in range(batch_size)]
        all_sequences.append(batch_sequences)
    return all_sequences


def time_model(model, inputs, warmup=4):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(**inputs[0])
        
        start_time = time.time()
        for input_batch in inputs:
            _ = model(**input_batch)
        return time.time() - start_time


def time_model_esmc(model, sequences, warmup=10):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            for seq in sequences[0]:
                _ = model(seq)
        
        start_time = time.time()
        for batch in sequences:
            for seq in batch:
                _ = model(seq)
        return time.time() - start_time


def get_gpu_memory():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB


# Test different sequence lengths and batch sizes
lengths = [32, 64, 128, 256, 512, 1024, 2048]
batch_sizes = [1, 2, 4, 8, 16, 32]
num_batches = 16
results = []

if not args.test:
    # Generate all test sequences first
    all_sequences = {}
    for length in lengths:
        for batch_size in batch_sizes:
            print(f"\nGenerating sequences for length={length}, batch_size={batch_size}")
            all_sequences[(length, batch_size)] = generate_batch_sequences(length, batch_size, num_batches)

    # Test each model
    for model_path in model_paths:
        print(f"\nTesting {model_path}...")
        if 'esmc' in model_path.lower():
            esm = ESMC.from_pretrained(model_path, device=device).to(device)
            model = ESMCForEmbedding(esm).to(device)
            tokenizer = None
        elif 'synthyra' in model_path.lower():
            model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
            tokenizer = model.tokenizer
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
            tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        
        for length in lengths:
            for batch_size in batch_sizes:
                print(f"\nTesting length={length}, batch_size={batch_size}")
                sequences = all_sequences[(length, batch_size)]
                
                torch.cuda.reset_peak_memory_stats()
                if isinstance(model, ESMCForEmbedding):
                    model_time = time_model_esmc(model, sequences)
                else:
                    inputs = [tokenizer(batch_seq, padding=True, return_tensors="pt").to(device) for batch_seq in sequences]
                    model_time = time_model(model, inputs)
                model_memory = get_gpu_memory()
                
                results.append({
                    'Model': model_path,
                    'Length': length,
                    'Batch Size': batch_size,
                    'Time': model_time,
                    'Memory': model_memory
                })
                print(f"Time: {model_time:.2f}s, memory: {model_memory:.0f}MB")
                torch.cuda.empty_cache()

        model.cpu()
        del model
        torch.cuda.empty_cache()
else:
    # Generate random test results
    for model_path in model_paths:
        for length in lengths:
            for batch_size in batch_sizes:
                # Generate random time between 0.1 and 10 seconds, scaling with length and batch size
                model_time = random.uniform(0.1, 10) * (length/2) * (batch_size/1)
                # Generate random memory between 100 and 5000 MB, scaling with length and batch size
                model_memory = random.uniform(100, 5000) * (length/2) * (batch_size/1)
                
                results.append({
                    'Model': model_path,
                    'Length': length,
                    'Batch Size': batch_size,
                    'Time': model_time,
                    'Memory': model_memory
                })
                print(f"Generated random - Time: {model_time:.2f}s, memory: {model_memory:.0f}MB")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('model_benchmarks.csv', index=False)

# Create visualization for throughput
num_batch_sizes = len(batch_sizes)
plt.figure(figsize=(15, 5 * num_batch_sizes))

for i, batch_size in enumerate(batch_sizes):
    plt.subplot(num_batch_sizes, 1, i + 1)
    for model_path in model_paths:
        model_results = [(r['Length'], r['Time']) for r in results 
                        if r['Model'] == model_path and r['Batch Size'] == batch_size]
        if model_results:
            lengths, times = zip(*model_results)
            throughput = [batch_size * len * num_batches / time for len, time in zip(lengths, times)]
            plt.plot(lengths, throughput, marker='o', label=model_path)

    plt.title(f'Model Throughput vs Sequence Length (Batch Size = {batch_size})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Throughput (tokens/second)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

plt.tight_layout()
plt.savefig('model_throughput.png', bbox_inches='tight', dpi=300)
plt.close()

# Create visualization for memory usage
plt.figure(figsize=(15, 5 * num_batch_sizes))

for i, batch_size in enumerate(batch_sizes):
    plt.subplot(num_batch_sizes, 1, i + 1)
    for model_path in model_paths:
        model_results = [(r['Length'], r['Memory']) for r in results 
                        if r['Model'] == model_path and r['Batch Size'] == batch_size]
        if model_results:
            lengths, memory = zip(*model_results)
            plt.plot(lengths, memory, marker='o', label=model_path)

    plt.title(f'GPU Memory Usage vs Sequence Length (Batch Size = {batch_size})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

plt.tight_layout()
plt.savefig('model_memory.png', bbox_inches='tight', dpi=300)
plt.close()
