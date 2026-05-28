---
library_name: transformers
tags:
  - biology
  - esm
  - protein
  - protein-language-model
  - masked-language-modeling
---

# ESM++ 6B

[ESM++](https://github.com/Synthyra/FastPLMs) is a Hugging Face compatible implementation of [Biohub ESMC](https://biohub.ai/esm/protein) ([license](https://github.com/Biohub/esm/blob/main/LICENSE.md)).
This checkpoint corresponds to the 6 billion parameter ESMC model released as [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B).

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = model.tokenizer

sequences = ["MPRTEIN", "MSEQWENCE"]
inputs = tokenizer(sequences, padding=True, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

output = model(**inputs)
print(output.logits.shape)
print(output.last_hidden_state.shape)
```

To load the Biohub source weights directly through FastPLMs:

```python
from fastplms.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM

model = ESMplusplusForMaskedLM.from_pretrained_esm("esmc-6b")
```

## Citation

```bibtex
@misc{candido2026language,
  title  = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan
            and Lin, Zeming and Verkuil, Robert and Wu, Bryan and Lee, Jin Sub
            and Bruguera, Elise S. and Keval, Jehan A. and Kopylov, Mykhailo
            and Pak, John E. and Wu, Wesley and Thomas, Neil and Mataraso, Samson
            and Hsu, Alvin and Trotman-Grant, Ashton C. and Fatras, Kilian
            and dos Santos Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil
            and Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth
            and Sitwala, Hrishita and Yu, Yue and Wiggert, Marius
            and Carlin, Dylan Alexander and Goering, Anthony W. and Blazejewski, Tomasz
            and Sandora, McCullen and Hla, Michael and Jia, Tina Z.
            and Kloker, Leon H. and Sofroniew, Nicholas J. and Uehara, Masatoshi
            and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S.
            and Sercu, Tom and Rives, Alexander},
  year   = {2026},
  url    = {https://biohub.ai/papers/esm_protein.pdf},
  note   = {Preprint}
}
```
