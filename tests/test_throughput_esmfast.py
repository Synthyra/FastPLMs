import torch
import time
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from transformers import EsmForMaskedLM, AutoModelForMaskedLM, EsmTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='lhallee/synthyra_esm2_650_mlm')
parser.add_argument('--token', type=str, default=None)
args = parser.parse_args()

if args.token:
    login(args.token)

model_path = args.model_path
canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length-1))


def time_model(model, inputs, warmup=10):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(**inputs[0])
        
        start_time = time.time()
        for input_batch in inputs:
            _ = model(**input_batch)
        return time.time() - start_time


def get_gpu_memory():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB


tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

# Test different sequence lengths and batch sizes
#lengths = [128, 256, 512, 1024, 2048]
#batch_sizes = [1, 4, 16, 32]
lengths = [8, 16]
batch_sizes = [1, 2]


results = []

# Generate all test sequences first
all_test_inputs = {}
for length in lengths:
    for batch_size in batch_sizes:
        print(f"\nGenerating sequences for length={length}, batch_size={batch_size}")
        all_sequences = []
        for _ in range(100):
            batch_sequences = [generate_random_sequence(length) for _ in range(batch_size)]
            all_sequences.append(batch_sequences)
        inputs = [tokenizer(sequences, padding=True, return_tensors="pt").to(device) for sequences in all_sequences]
        all_test_inputs[(length, batch_size)] = inputs

# Test ESM model in fp32
print("\nTesting ESM model in FP32...")
model = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
for length in lengths:
    for batch_size in batch_sizes:
        print(f"\nTesting length={length}, batch_size={batch_size}")
        inputs = all_test_inputs[(length, batch_size)]
        
        torch.cuda.reset_peak_memory_stats()
        esm_time = time_model(model, inputs)
        esm_memory = get_gpu_memory()
        
        results.append({
            'Length': length,
            'Batch Size': batch_size,
            'ESM Time': esm_time,
            'ESM Memory': esm_memory
        })
        print(f"ESM FP32 time: {esm_time:.2f}s, memory: {esm_memory:.0f}MB")
        torch.cuda.empty_cache()

model.cpu()
del model
torch.cuda.empty_cache()

# Test FastESM in fp16
print("\nTesting FastESM model in FP16...")
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
for i, (length, batch_size) in enumerate([(l,b) for l in lengths for b in batch_sizes]):
    print(f"\nTesting length={length}, batch_size={batch_size}")
    inputs = all_test_inputs[(length, batch_size)]
    
    torch.cuda.reset_peak_memory_stats()
    fast_time = time_model(model, inputs)
    fast_memory = get_gpu_memory()
    
    results[i].update({
        'FastESM Time': fast_time,
        'FastESM Memory': fast_memory,
        'Speedup': results[i]['ESM Time']/fast_time
    })
    print(f"FastESM FP16 time: {fast_time:.2f}s, memory: {fast_memory:.0f}MB")
    print(f"Speedup: {results[i]['Speedup']:.2f}x")
    torch.cuda.empty_cache()

model.cpu()
del model
torch.cuda.empty_cache()

# Create plots
plt.figure(figsize=(15, 10))

# Speedup heatmap
plt.subplot(221)
speedup_data = [[r['Speedup'] for r in results if r['Length']==l] for l in lengths]
sns.heatmap(speedup_data, 
            xticklabels=batch_sizes,
            yticklabels=lengths,
            annot=True, 
            fmt='.2f',
            cmap='viridis')
plt.title('Speedup (ESM/FastESM)')
plt.xlabel('Batch Size')
plt.ylabel('Sequence Length')

# Absolute times line plot
plt.subplot(222)
for length in lengths:
    length_results = [r for r in results if r['Length']==length]
    plt.plot([r['Batch Size'] for r in length_results],
             [r['FastESM Time'] for r in length_results],
             label=f'Length {length}',
             marker='o')

plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.title('FastESM Processing Time')
plt.legend()
plt.xscale('log')
plt.yscale('log')

# ESM Memory heatmap
plt.subplot(223)
memory_data = [[r['ESM Memory'] for r in results if r['Length']==l] for l in lengths]
sns.heatmap(memory_data,
            xticklabels=batch_sizes,
            yticklabels=lengths,
            annot=True,
            fmt='.0f',
            cmap='viridis')
plt.title('ESM FP32 Memory Usage (MB)')
plt.xlabel('Batch Size')
plt.ylabel('Sequence Length')

# FastESM Memory heatmap
plt.subplot(224)
memory_data = [[r['FastESM Memory'] for r in results if r['Length']==l] for l in lengths]
sns.heatmap(memory_data,
            xticklabels=batch_sizes,
            yticklabels=lengths,
            annot=True,
            fmt='.0f',
            cmap='viridis')
plt.title('FastESM FP16 Memory Usage (MB)')
plt.xlabel('Batch Size')
plt.ylabel('Sequence Length')

plt.tight_layout()
plt.savefig('throughput_results.png')
plt.close()

print("\nPlot saved as throughput_results.png")
