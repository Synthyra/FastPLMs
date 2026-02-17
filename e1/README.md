# Profluent-E1

This repository contains the code for the [Profluent E1](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1) family of models - our best in class single sequence and retrieval augmented protein representation models. They are designed to be drop-in replacement for ESM family of models. See Section on [licenses](#licenses) for the license details. 

## Available Models

| Model Name | Model Parameters | HuggingFace Link                                                      |
| ---------- | ---------------- | --------------------------------------------------------------------- |
| E1-150m    | 150M             | [Profluent-Bio/E1-150m](https://huggingface.co/Profluent-Bio/E1-150m) |
| E1-300m    | 300M             | [Profluent-Bio/E1-300m](https://huggingface.co/Profluent-Bio/E1-300m) |
| E1-600m    | 600M             | [Profluent-Bio/E1-600m](https://huggingface.co/Profluent-Bio/E1-600m) |

## Installation

To use the code in this repository, install the dependencies using the following command:

```bash
git clone https://github.com/Profluent-AI/E1.git
cd E1 && pip3 install -e .
```

If you are using GPUs with cuda capability 8.0 or higher (Ampere architecture or higher), we also recommend installing the `flash-attn` package:

```bash
pip3 install flash-attn --no-build-isolation
```

## Compute Requirements

While the models can be run on both CPU and GPU, we recommend using a GPU for faster inference.
In addition, the model was trained with BF16 precision, so we recommend using a GPU with BF16 support (CUDA Capability 8.0 or higher; for example, NVIDIA L40, A100, H100, etc.) since that also allows the inference performance to improve using flash attention.

## Interactive Usage

The model weights are hosted on Hugging Face and will be downloaded automatically in the following code. The model can be used in both single sequence and retrieval augmented mode. You can use `?` as mask token. To use the model in retrieval augmented mode, you can prepend your query with homolog sequences separated by commas.

```python
import torch

from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM

model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-300m").to("cuda:0")
model.eval()

sequences = ["AAAAA?C", "MFCATEEKL,MCCASDF,MFCC?SEF"]
batch_preparer = E1BatchPreparer()
batch = batch_preparer.get_batch_kwargs(sequences, device="cuda:0")

with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
    outputs = model(
        input_ids=batch["input_ids"],
        within_seq_position_ids=batch["within_seq_position_ids"],
        global_position_ids=batch["global_position_ids"],
        sequence_ids=batch["sequence_ids"],
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )

logits: torch.Tensor = outputs.logits  # (B, L, V)
embeddings: torch.Tensor = outputs.embeddings  # (B, L, E)

print(logits)
print(embeddings)

# Boolean Selectors of shape (B, L) to get relevant tokens from logits/embeddings
# last_sequence_selector: True for tokens that are part of the last sequence (including boundary tokens) in case of multi-sequence input.
last_sequence_selector = batch["sequence_ids"] == batch["sequence_ids"].max(dim=1)[0][:, None]
# residue_selector: True for tokens that are part of the input sequence i.e not boundary tokens like 1, 2, <bos>, <eos>, <pad>, etc.
residue_selector = ~(batch_preparer.get_boundary_token_mask(batch["input_ids"]))
# last_sequence_residue_selector: True for residues that are part of the last sequence (excluding boundary tokens)
last_sequence_residue_selector = last_sequence_selector & residue_selector

# Will yield embeddings for ["AAAAA?C", "MFCC?SEF"] while throwing away embeddings for the boundary tokens
# and for homologous sequences in the second instance. Can do similar for logits.
last_sequence_embeddings = [embeddings[i, last_sequence_residue_selector[i]] for i in range(embeddings.shape[0])]
```

See the [cookbook/basic.ipynb](cookbook/basic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profluent-AI/E1/blob/main/cookbook/basic.ipynb) file for a more complete example. We also provide a notebook for [embedding analysis](cookbook/embedding.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profluent-AI/E1/blob/main/cookbook/embedding.ipynb) to demonstrate how to use the model to get sequence embeddings using `E1Predictor` class.

You can also use the [cookbook/zero_shot_fitness_prediction.ipynb](cookbook/zero_shot_fitness_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profluent-AI/E1/blob/main/cookbook/zero_shot_fitness_prediction.ipynb) file to predict the fitness of zero-shot substitution mutants against a wild type parent.

For performing in-silico site saturation mutagenesis using masked-marginal scoring method, see the [cookbook/site_saturation_mutagenesis.ipynb](cookbook/site_saturation_mutagenesis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profluent-AI/E1/blob/main/cookbook/site_saturation_mutagenesis.ipynb) file. We score all possible single mutants of a given parent sequence in both single sequence and retrieval augmented mode.

## Comparison to other models

| Average Spearman on Substitution Assays in Protein Gym Benchmark | Unsupervised Contact Map Prediction on CAMEO |
| ---------------------------------------------------------------- | -------------------------------------------- |
| ![Protein Gym Results](assets/protein_gym.svg)                   | ![CAMEO Results](assets/cameo.svg)           |

## Licenses

Your use of the Profluent-E1 model code is governed by the Apache License 2.0, while your use of the Profluent-E1 model weights and the full release of the Profluent-E1 model is governed by a similarly permissive license with additional attribution requirements - see the [NOTICE](NOTICE) file for details. You can use, share, and modify Profluent-E1 for free, but you must follow our [ATTRIBUTION](ATTRIBUTION) guidelines to give credit, include the license when you share and follow some other basic rules. Profluent is not responsible for what you build, and may terminate your rights to use Profluent-E1 if you breach the license.

1. Code in [src/E1/model/flash_attention_utils.py](src/E1/model/flash_attention_utils.py) is adapted from [flash-attention](https://github.com/Dao-AILab/flash-attention) project under BSD-3-Clause license.
