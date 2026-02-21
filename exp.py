import torch

VOCAB_SIZE = 64
PAD_TOKEN = 0

input_ids_1 = torch.randint(0, VOCAB_SIZE, (1, 6))
input_ids_2 = torch.randint(0, VOCAB_SIZE, (1, 6))

input_ids_2[:,-3:] = PAD_TOKEN

batch = torch.cat([input_ids_1, input_ids_2], dim=0)

seq_id = batch != PAD_TOKEN
print("2D attention mask:")
print(seq_id)

print("4D attention mask from ESM repo:")
mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
mask = mask.unsqueeze(1)
print(mask)
print(mask.shape)

print("A correct 4D attention mask:")
correct_mask = seq_id[:, None, :, None] & seq_id[:, None, None, :]
print(correct_mask)
print(correct_mask.shape)