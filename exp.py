import torch
from transformers import AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', trust_remote_code=True, dtype=torch.bfloat16).eval().to(device)

sequences = ['MPRTEIN', 'MSEQWENCE']
batch = model.prep_tokens.get_batch_kwargs(sequences, device=device)

output = model(**batch, output_hidden_states=True) # get all hidden states with output_hidden_states=True
print(output.logits.shape) # language modeling logits, (batch_size, seq_len, vocab_size), (2, 11, 34)
print(output.last_hidden_state.shape) # last hidden state of the model, (batch_size, seq_len, hidden_size), (2, 11, 768)
print(output.loss)
print(len(output.hidden_states)) # all hidden states if you passed output_hidden_states=True (in tuple)