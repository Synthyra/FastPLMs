import torch
import random
import numpy as np
import sqlite3

from typing import List
from transformers import AutoModel
from esm2.modeling_fastesm import FastEsmModel
from esm_plusplus.modeling_esm_plusplus import ESMplusplusModel
from e1.modeling_e1 import E1Model


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(CANONICAL_AMINO_ACIDS, k=length))


if __name__ == "__main__":
    # py -m test_scripts.test_embedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_embedding(model, sequences: List[str], name: str):
        embeddings = model.embed_dataset(
            sequences=sequences,
            tokenizer=model.tokenizer if hasattr(model, 'tokenizer') else None,
            sql=False, # return dictionary of sequences and embeddings
            save=False,
        )

        count = 0
        for k, v in embeddings.items():
            print(k)
            print(v.dtype, v.shape)
            count += 1
            if count > 5:
                break

        embeddings = model.embed_dataset(
            sequences=sequences,
            tokenizer=model.tokenizer if hasattr(model, 'tokenizer') else None,
            full_embeddings=True,
            sql=False, # return dictionary of sequences and embeddings
            save=False,
        )

        count = 0
        for k, v in embeddings.items():
            print(k)
            print(v.dtype, v.shape)
            count += 1
            if count > 5:
                break

        db_path = f'embeddings_{name}.db'
        _ = model.embed_dataset(
            sequences=sequences,
            tokenizer=model.tokenizer if hasattr(model, 'tokenizer') else None,
            pooling_types=['cls', 'mean'],
            sql=True,
            sql_db_path=db_path,
            save=True,
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
        
        # Make sure to close the connection before attempting to delete the file
        c.close()
        conn.close()

    print("Testing E1 model...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = E1Model.from_pretrained("Synthyra/Profluent-E1-150M", dtype=torch.bfloat16).to(device)
    print(model)
    test_embedding(model, sequences, 'e1')

    print("Testing FastESM model...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = FastEsmModel.from_pretrained("Synthyra/ESM2-8M", dtype=torch.float16).to(device)
    print(model)
    test_embedding(model, sequences, 'fastesm')

    print("Testing ESM++ model...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = ESMplusplusModel.from_pretrained("Synthyra/ESMplusplus_small", dtype=torch.float16).to(device)
    print(model)
    test_embedding(model, sequences, 'esmplusplus')

    print("Testing E1 model with AutoModel...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = AutoModel.from_pretrained("Synthyra/Profluent-E1-150M", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    print(model)
    test_embedding(model, sequences, 'e1_auto')

    print("Testing FastESM model with AutoModel...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = AutoModel.from_pretrained("Synthyra/ESM2-8M", dtype=torch.float16, trust_remote_code=True).to(device)
    print(model)
    test_embedding(model, sequences, 'fastesm_auto')

    print("Testing ESM++ model with AutoModel...")
    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", dtype=torch.float16, trust_remote_code=True).to(device)
    print(model)
    test_embedding(model, sequences, 'esmplusplus_auto')