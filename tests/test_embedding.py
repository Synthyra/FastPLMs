import torch
import random
import numpy as np
import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_fastesm import FastEsmModel
from modeling_esm_plusplus import ESMplusplusModel


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(CANONICAL_AMINO_ACIDS, k=length))


if __name__ == "__main__":
    # py tests/test_embedding.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_embedding(model, sequences):
        embeddings = model.embed_dataset(
            sequences=sequences,
            tokenizer=model.tokenizer,
            sql=False, # return dictionary of sequences and embeddings
            save=False,
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
            tokenizer=model.tokenizer,
            pooling_types=['cls', 'mean'],
            sql=True,
            sql_db_path=db_path,
            save=False,
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

    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = FastEsmModel.from_pretrained("Synthyra/ESM2-8M", torch_dtype=torch.float16).to(device)
    print(model)
    test_embedding(model, sequences)

    sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
    model = ESMplusplusModel.from_pretrained("Synthyra/ESMplusplus_small", torch_dtype=torch.float16).to(device)
    print(model)
    test_embedding(model, sequences)
