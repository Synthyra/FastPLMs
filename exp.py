from dplm_fastplms.dplm2 import DPLM2ForMaskedLM


model = DPLM2ForMaskedLM.from_pretrained('airkingbd/dplm2_150m')
tokenizer = model.tokenizer
#tokenizer = AutoTokenizer.from_pretrained('airkingbd/dplm2_150m')

print(tokenizer)
print(model)


seq = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
input_ids = tokenizer.encode(seq, add_special_tokens=True)
decoded = tokenizer.decode(input_ids)
print(decoded)