### Embedding Mixin + Pooler
import os
import sqlite3
import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


class Pooler:
    def __init__(self, pooling_types: List[str]):
        self.pooling_types = pooling_types
        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'norm': self.norm_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'parti': self._pool_parti,
        }

    def _create_pooled_matrices_across_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        maxed_attentions = torch.max(attentions, dim=1)[0]
        return maxed_attentions

    def _page_rank(self, attention_matrix, personalization=None, nstart=None, prune_type="top_k_outdegree"):
        # Run PageRank on the attention matrix converted to a graph.
        # Raises exceptions if the graph doesn't match the token sequence or has no edges.
        # Returns the PageRank scores for each token node.
        G = self._convert_to_graph(attention_matrix)
        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.")
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(G, alpha=0.85, tol=1e-06, weight='weight', personalization=personalization, nstart=nstart, max_iter=100)

    def _convert_to_graph(self, matrix):
        # Convert a matrix (e.g., attention scores) to a directed graph using networkx.
        # Each element in the matrix represents a directed edge with a weight.
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        return G

    def _calculate_importance_weights(self, dict_importance, attention_mask: Optional[torch.Tensor] = None):
        # Remove keys where attention_mask is 0
        if attention_mask is not None:
            for k in list(dict_importance.keys()):
                if attention_mask[k] == 0:
                    del dict_importance[k]

        #dict_importance[0] # remove cls
        #dict_importance[-1] # remove eos
        total = sum(dict_importance.values())
        return np.array([v / total for _, v in dict_importance.items()])

    def _pool_parti(self, emb: torch.Tensor, attentions: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        maxed_attentions = self._create_pooled_matrices_across_layers(attentions).numpy()
        # emb is (b, L, d), maxed_attentions is (b, L, L)
        emb_pooled = []
        for e, a, mask in zip(emb, maxed_attentions, attention_mask):
            dict_importance = self._page_rank(a)
            importance_weights = self._calculate_importance_weights(dict_importance, mask)
            num_tokens = int(mask.sum().item())
            emb_pooled.append(np.average(e[:num_tokens], weights=importance_weights, axis=0))
        pooled = torch.tensor(np.array(emb_pooled))
        return pooled

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).max(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).median(dim=1).values
    
    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            # Compute variance correctly over non-masked positions, then take sqrt
            var = self.var_pooling(emb, attention_mask, **kwargs)
            return torch.sqrt(var)
    
    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            # Correctly compute variance over only non-masked positions
            attention_mask = attention_mask.unsqueeze(-1)  # (b, L, 1)
            # Compute mean over non-masked positions
            mean = (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
            mean = mean.unsqueeze(1)  # (b, 1, d)
            # Compute squared differences from mean, only over non-masked positions
            squared_diff = (emb - mean) ** 2  # (b, L, d)
            # Sum squared differences over non-masked positions and divide by count
            var = (squared_diff * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
            return var

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        return emb[:, 0, :]

    def __call__(
            self,
            emb: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attentions: Optional[torch.Tensor] = None
        ): # [mean, max]
        final_emb = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb=emb, attention_mask=attention_mask, attentions=attentions)) # (b, d)
        return torch.cat(final_emb, dim=-1) # (b, n_pooling_types * d)


class ProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer: PreTrainedTokenizerBase) -> Callable[[list[str]], dict[str, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(sequences, return_tensors="pt", padding='longest')
    return _collate_fn


class EmbeddingMixin:
    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        sequences = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                sequences.append(row[0])
        return set(sequences)

    def _ensure_embeddings_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "sequence TEXT PRIMARY KEY, "
            "embedding BLOB NOT NULL, "
            "shape TEXT, "
            "dtype TEXT"
            ")"
        )
        cursor.execute("PRAGMA table_info(embeddings)")
        rows = cursor.fetchall()
        column_names = [row[1] for row in rows]
        if "shape" not in column_names:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN shape TEXT")
        if "dtype" not in column_names:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT")
        conn.commit()

    def load_embeddings_from_pth(self, save_path: str) -> dict[str, torch.Tensor]:
        assert os.path.exists(save_path), f"Embedding file does not exist: {save_path}"
        payload = torch.load(save_path, map_location="cpu", weights_only=True)
        assert isinstance(payload, dict), "Expected .pth embeddings file to contain a dictionary."
        for sequence, tensor in payload.items():
            assert isinstance(sequence, str), "Expected embedding dictionary keys to be sequences (str)."
            assert isinstance(tensor, torch.Tensor), "Expected embedding dictionary values to be tensors."
        return payload

    def load_embeddings_from_db(self, db_path: str, sequences: Optional[List[str]] = None) -> dict[str, torch.Tensor]:
        assert os.path.exists(db_path), f"Embedding database does not exist: {db_path}"
        loaded: dict[str, torch.Tensor] = {}
        with sqlite3.connect(db_path) as conn:
            self._ensure_embeddings_table(conn)
            cursor = conn.cursor()
            if sequences is None:
                cursor.execute("SELECT sequence, embedding, shape, dtype FROM embeddings")
            else:
                if len(sequences) == 0:
                    return loaded
                placeholders = ",".join(["?"] * len(sequences))
                cursor.execute(
                    f"SELECT sequence, embedding, shape, dtype FROM embeddings WHERE sequence IN ({placeholders})",
                    tuple(sequences),
                )

            rows = cursor.fetchall()
            for row in rows:
                sequence = row[0]
                embedding_bytes = row[1]
                shape_text = row[2]
                dtype_text = row[3]
                assert shape_text is not None, "Missing shape metadata in embeddings table."
                assert dtype_text is not None, "Missing dtype metadata in embeddings table."
                shape_values = [int(value) for value in shape_text.split(",") if len(value) > 0]
                assert len(shape_values) > 0, f"Invalid shape metadata for sequence: {sequence}"
                expected_size = int(np.prod(shape_values))
                np_dtype = np.dtype(dtype_text)
                array = np.frombuffer(embedding_bytes, dtype=np_dtype)
                assert array.size == expected_size, f"Shape mismatch while reading sequence: {sequence}"
                reshaped = array.copy().reshape(tuple(shape_values))
                loaded[sequence] = torch.from_numpy(reshaped)
        return loaded

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
        **kwargs,
    ) -> Optional[dict[str, torch.Tensor]]:
        """
        Embed a dataset of protein sequences.

        Supports two modes:
        - Tokenizer mode (ESM2/ESM++): provide `tokenizer`, `_embed(input_ids, attention_mask)` is used.
        - Sequence mode (E1): pass `tokenizer=None`, `_embed(sequences, return_attention_mask=True, **kwargs)` is used.
        """
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        pooler = Pooler(pooling_types) if not full_embeddings else None
        tokenizer_mode = tokenizer is not None
        if tokenizer_mode:
            collate_fn = build_collator(tokenizer)
            device = self.device
        else:
            collate_fn = None
            device = None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if full_embeddings or residue_embeddings.ndim == 2:
                return residue_embeddings
            return pooler(residue_embeddings, attention_mask)

        def iter_batches(to_embed: List[str]):
            if tokenizer_mode:
                assert collate_fn is not None
                assert device is not None
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    residue_embeddings = self._embed(input_ids, attention_mask)
                    yield seqs, residue_embeddings, attention_mask
            else:
                for batch_start in tqdm(range(0, len(to_embed), batch_size), desc='Embedding batches'):
                    seqs = to_embed[batch_start:batch_start + batch_size]
                    batch_output = self._embed(seqs, return_attention_mask=True, **kwargs)
                    assert isinstance(batch_output, tuple), "Sequence mode _embed must return (last_hidden_state, attention_mask)."
                    assert len(batch_output) == 2, "Sequence mode _embed must return exactly two values."
                    residue_embeddings, attention_mask = batch_output
                    assert isinstance(attention_mask, torch.Tensor), "Sequence mode _embed must return attention_mask as a torch.Tensor."
                    yield seqs, residue_embeddings, attention_mask

        if sql:
            conn = sqlite3.connect(sql_db_path)
            self._ensure_embeddings_table(conn)
            c = conn.cursor()
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                with torch.no_grad():
                    for i, (seqs, residue_embeddings, attention_mask) in enumerate(iter_batches(to_embed)):
                        embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if full_embeddings:
                                emb = emb[mask.bool()].reshape(-1, hidden_size)
                            emb_np = emb.cpu().numpy()
                            emb_shape = ",".join([str(dim) for dim in emb_np.shape])
                            emb_dtype = str(emb_np.dtype)
                            c.execute(
                                "INSERT OR REPLACE INTO embeddings (sequence, embedding, shape, dtype) VALUES (?, ?, ?, ?)",
                                (seq, emb_np.tobytes(), emb_shape, emb_dtype),
                            )
                        if tokenizer_mode and (i + 1) % 100 == 0:
                            conn.commit()
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = self.load_embeddings_from_pth(save_path)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            with torch.no_grad():
                for seqs, residue_embeddings, attention_mask in iter_batches(to_embed):
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        if full_embeddings:
                            emb = emb[mask.bool()].reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict


import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Dict, Any
from einops import rearrange
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, EsmTokenizer
from transformers import initialization as init
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers.models.esm.modeling_esm import (
    EsmIntermediate,
    EsmOutput,
    EsmPooler,
    EsmLMHead,
    EsmSelfOutput,
    EsmClassificationHead,
    EsmContactPredictionHead,
    EsmEmbeddings,
    RotaryEmbedding,
)
try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None


def get_attention_mask(
    attn_backend: str, 
    batch_size: int, 
    seq_len: int, 
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if attention_mask is None:
        token_attention_mask = torch.ones((batch_size, seq_len), device=device).bool() 
    else:
        token_attention_mask = attention_mask.bool()
    
    if attn_backend == "flex":
        assert create_block_mask is not None, "Flex attention backend requested but torch.create_block_mask is unavailable."

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return token_attention_mask[batch_idx, q_idx] & token_attention_mask[batch_idx, kv_idx]

        flex_block_mask = create_block_mask(
            mask_mod,
            batch_size,
            1,
            seq_len,
            seq_len,
            device=device,
        )
        extended_attention_mask = None
    else:
        flex_block_mask = None
        extended_attention_mask = token_attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)

    return extended_attention_mask, flex_block_mask


@dataclass
class EsmMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class FastEsmConfig(PretrainedConfig):
    model_type = "fast_esm"
    def __init__(
        self,
        vocab_size: int = None,
        mask_token_id: int = None,
        pad_token_id: int = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1026,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        emb_layer_norm_before: bool = None,
        token_dropout: bool = True,
        attn_backend: str = "sdpa",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.emb_layer_norm_before = emb_layer_norm_before
        self.tie_word_embeddings = False
        self.token_dropout = token_dropout
        self.attn_backend = attn_backend

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionar y of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        return output


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: Optional[str] = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.scale = self.attention_head_size**-0.5

        self.dropout_prob = config.attention_probs_dropout_prob
        self.attn_backend = config.attn_backend
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for self attention.
        
        Args:
            hidden_states: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_layer = query_layer * self.scale

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if output_attentions:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (b, h, l, l)
            attention_scores = attention_scores.masked_fill(attention_mask.logical_not(), float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout_prob > 0 and self.training:
                attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer, attention_probs
        else:
            if self.attn_backend == "flex":
                assert flex_attention is not None, "Flex attention backend requested but torch.flex_attention is unavailable."
                assert query_layer.dtype in (torch.float16, torch.bfloat16), f"Flex attention backend requires float16 or bfloat16, got {query_layer.dtype}."
                assert flex_block_mask is not None, "Flex attention backend requires a block mask"
                context_layer = flex_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    block_mask=flex_block_mask,
                    scale=1.0, # applied before rotary
                )
            else:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    scale=1.0 # applied before rotary
                )
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer
        

class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for attention layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            flex_block_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = self_outputs
            attention_output = self.output(attention_output, hidden_states)
            return attention_output, attention_weights
        else:
            attention_output = self_outputs
            return self.output(attention_output, hidden_states)


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for transformer layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            flex_block_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = attention_outputs
        else:
            attention_output = attention_outputs
            attention_weights = None

        layer_output = self.feed_forward_chunk(attention_output)
        
        if output_attentions:
            return layer_output, attention_weights
        return layer_output

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass for transformer encoder.
        
        Args:
            hidden_states: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            BaseModelOutputWithPastAndCrossAttentions containing model outputs
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    flex_block_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    flex_block_mask,
                    output_attentions,
                )

            if output_attentions:
                hidden_states, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_states = layer_outputs

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FastEsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FastEsmConfig
    base_model_prefix = "fastesm"
    supports_gradient_checkpointing = True
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    all_tied_weights_keys = {}

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, EsmLMHead):
            init.zeros_(module.bias)
        elif isinstance(module, EsmEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, RotaryEmbedding):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim))
            init.copy_(module.inv_freq, inv_freq)

    def get_output_embeddings(self):
        # NOTE: get_output_embeddings() must return None to prevent accidental weight tying.
        # See e.g. https://github.com/huggingface/transformers/pull/39339#discussion_r2219126400
        return None


class FAST_ESM_ENCODER(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        attention_mask, flex_block_mask = get_attention_mask(
            attn_backend=self.config.attn_backend,
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            device=input_ids.device,
            attention_mask=attention_mask,
        )
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            flex_block_mask=flex_block_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        return encoder_outputs.last_hidden_state

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attns = self(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        attention_mask, flex_block_mask = get_attention_mask(
            attn_backend=self.config.attn_backend,
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            device=input_ids.device,
            attention_mask=attention_mask,
        )
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            flex_block_mask=flex_block_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.last_hidden_state

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FastEsmModel(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.esm = FAST_ESM_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        assert input_ids or inputs_embeds, "You have to specify either input_ids or inputs_embeds"
        assert not (input_ids and inputs_embeds), "You cannot specify both input_ids and inputs_embeds at the same time"
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForMaskedLM(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple, EsmMaskedLMOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return EsmMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForSequenceClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.squeeze(), labels.squeeze())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForTokenClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
