import torch


VOCAB_SIZE = 64
PAD_TOKEN = 0

input_ids_1 = torch.randint(0, VOCAB_SIZE, (1, 10))
input_ids_2 = torch.randint(0, VOCAB_SIZE, (1, 10))

input_ids_2[:,:-5] = PAD_TOKEN

batch = torch.cat([input_ids_1, input_ids_2], dim=0)

seq_id = batch != PAD_TOKEN

mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
mask = mask.unsqueeze(1)
print(mask)
print(mask.shape)

correct_mask = 