import torch
import random
import numpy as np
import sqlite3
from transformers import AutoModelForMaskedLM


canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(canonical_amino_acids, k=length))


sequences = [generate_random_sequence(random.randint(4, 8)) for _ in range(100)]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_large", trust_remote_code=True, torch_dtype=torch.float16).to(device)
print(model)

embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=16, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=False, # return dictionary of sequences and embeddings
)

count = 0
for k, v in embeddings.items():
    print(k)
    print(v.dtype, v.shape)
    count += 1
    if count > 10:
        break

db_path = 'embeddings.db'
    
_ = model.embed_dataset(
    sequences=sequences,
    batch_size=2,
    max_len=512,
    full_embeddings=False,
    full_precision=False,
    pooling_type='cls',
    num_workers=0,
    sql=True,
    sql_db_path=db_path,
)

# Verify database contents
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check number of sequences
c.execute('SELECT COUNT(*) FROM embeddings')
db_count = c.fetchone()[0]
print(f"\nNumber of sequences in database: {db_count}")

count = 0
for seq in sequences:
    c.execute('SELECT embedding FROM embeddings WHERE sequence = ?', (seq,))
    result = c.fetchone()
    assert result is not None, f"Sequence {seq} not found in database"
    if count < 10:
        embedding = np.frombuffer(result[0], dtype=np.float32)
        print(seq)
        print(f"Embedding shape: {embedding.shape}")
    count += 1
conn.close()
