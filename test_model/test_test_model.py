import random
import torch
from transformers import AutoModel


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int) -> str:
    return 'M' + "".join(random.choices(CANONICAL_AMINO_ACIDS, k=length))


model = AutoModel.from_pretrained('lhallee/test_auto_model', trust_remote_code=True).eval()
tokenizer = model.tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

sequences = [generate_random_sequence(random.randint(4, 16)) for _ in range(100)]
embeddings = model.embed_dataset(
    sequences=sequences,
    tokenizer=model.tokenizer,
    sql=False, # return dictionary of sequences and embeddings
    save=False,
)


for i, (k, v) in enumerate(embeddings.items()):
    print(i, k, v.shape)
    if i >= 10:
        break