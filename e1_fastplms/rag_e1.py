"""
MSA-based retrieval augmentation for E1 protein language models.
Supports zero-shot PPLL scoring (ensemble over context configurations)
and embedding extraction with RAG context.
"""

import os
import hashlib
import pickle
import tarfile
import logging
from typing import Optional
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .modeling_e1 import (
    E1ForMaskedLM,
    DataPrepConfig,
    Pooler
)
from .e1_utils import (
    ContextSpecification,
    sample_multiple_contexts,
    E1Predictor
)

logger = logging.getLogger(__name__)

# Default context configurations as described in E1 preprint
DEFAULT_MAX_CONTEXT_TOKENS = [6144, 12288, 24576]
DEFAULT_SIMILARITY_THRESHOLDS = [1.0, 0.95, 0.9, 0.7, 0.5]

# Single context prompt for embedding; settings from E1 preprint contact prediction case study
DEFAULT_EMBED_MAX_TOKENS = 8192
DEFAULT_EMBED_SIMILARITY = 0.95


# Context helpers

def get_context_id(max_tokens: int, sim_threshold: float) -> str:
    """Generate a context ID string from token budget and similarity threshold."""
    return f"identity_{sim_threshold}_tokens_{max_tokens}"


def build_context_specifications(
    max_context_tokens: list[int] = None,
    similarity_thresholds: list[float] = None,
    min_query_similarity: float = 0.3,
) -> list[tuple[ContextSpecification, str]]:
    """Build all (token budget, similarity threshold) context configurations with their IDs."""
    if max_context_tokens is None:
        max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    if similarity_thresholds is None:
        similarity_thresholds = DEFAULT_SIMILARITY_THRESHOLDS

    specs = []
    for max_tokens in max_context_tokens:
        for sim_threshold in similarity_thresholds:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=max_tokens,
                max_query_similarity=sim_threshold,
                min_query_similarity=min_query_similarity,
                neighbor_similarity_lower_bound=0.8,
            )
            context_id = get_context_id(max_tokens, sim_threshold)
            specs.append((spec, context_id))

    return specs


def sample_contexts_for_msa(
    a3m_path: str,
    context_specs: list[tuple[ContextSpecification, str]],
    seed: int = 42,
) -> dict[str, str]:
    """Sample all contexts from an MSA file. Returns dict mapping context_id to context string."""
    specs_only = [spec for spec, _ in context_specs]
    context_ids = [ctx_id for _, ctx_id in context_specs]

    contexts, _ = sample_multiple_contexts(
        msa_path=a3m_path,
        context_specifications=specs_only,
        seed=seed,
    )
    return dict(zip(context_ids, contexts))


# MSA loading

def get_query_from_a3m(path: str) -> str:
    """Read the first (query) sequence from an a3m file, stripping insertions and gaps."""
    header_found = False
    seq_parts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header_found:
                    # We've hit the second header, stop
                    break
                header_found = True
                continue
            seq_parts.append(line)

    raw_seq = "".join(seq_parts)
    # Remove lowercase (insertions in a3m format) and gaps
    ungapped = "".join(c for c in raw_seq if c.isupper() or c == "-")
    ungapped = ungapped.replace("-", "")
    return ungapped


def load_msa_dir(msa_dir: str) -> dict[str, str]:
    """Load all .a3m files from a directory. Returns dict mapping query sequences to a3m paths."""
    msa_lookup = {}
    a3m_files = list(Path(msa_dir).rglob("*.a3m"))

    if not a3m_files:
        raise FileNotFoundError(f"No .a3m files found in {msa_dir}")

    for a3m_path in tqdm(a3m_files, desc="Loading MSAs"):
        try:
            query_seq = get_query_from_a3m(str(a3m_path))
            msa_lookup[query_seq] = str(a3m_path)
        except Exception as e:
            logger.warning(f"Failed to parse {a3m_path}: {e}")

    logger.info(f"Loaded {len(msa_lookup)} MSAs from {msa_dir}")
    return msa_lookup


def load_msa_from_hf(
    hf_path: str,
    cache_dir: str = None,
    token: str = None,
) -> dict[str, str]:
    """Download and load MSAs from a HuggingFace dataset repo (expects .a3m files or tar.gz)."""
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "protify_msa")
    os.makedirs(cache_dir, exist_ok=True)

    local_dir = os.path.join(cache_dir, hf_path.replace("/", "_"))

    if not os.path.exists(local_dir) or not any(Path(local_dir).rglob("*.a3m")):
        logger.info(f"Downloading MSAs from {hf_path}...")
        local_dir = snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
        )

        # Extract any tar.gz archives
        for tar_path in Path(local_dir).rglob("*.tar.gz"):
            logger.info(f"Extracting {tar_path}...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(tar_path.parent)

    return load_msa_dir(local_dir)


def get_msa_for_sequence(
    sequence: str,
    msa_lookup: dict[str, str],
) -> Optional[str]:
    """Find the a3m path for a sequence. Falls back to fuzzy match (>95% identity) for mutants."""
    if sequence in msa_lookup:
        return msa_lookup[sequence]

    # Try fuzzy matching for mutant sequences
    best_match_path = None
    best_identity = 0.0
    for query_seq, a3m_path in msa_lookup.items():
        if abs(len(query_seq) - len(sequence)) > 10:
            continue
        # Simple identity computation between two ungapped sequences
        min_len = min(len(query_seq), len(sequence))
        if min_len == 0:
            continue
        matches = sum(a == b for a, b in zip(query_seq[:min_len], sequence[:min_len]))
        identity = matches / min_len
        if identity > best_identity:
            best_identity = identity
            best_match_path = a3m_path

    if best_identity >= 0.95:
        return best_match_path

    return None


# Context caching

class ContextCache:
    """Disk cache for sampled MSA contexts."""

    def __init__(self, cache_dir: str, specs_hash: str, seed: int):
        self.cache_dir = cache_dir
        self.specs_hash = specs_hash
        self.seed = seed
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{safe_key}_seed{self.seed}_{self.specs_hash}.pkl")

    def get(self, key: str) -> Optional[dict[str, str]]:
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, key: str, contexts: dict[str, str]) -> None:
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(contexts, f)


def compute_ppll(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> float:
    """Mean probability of correct tokens from logits (higher = better, max = 1.0)."""
    if token_ids.device != logits.device:
        token_ids = token_ids.to(logits.device)

    if logits.shape[0] != token_ids.shape[0]:
        raise ValueError(
            f"Logits length {logits.shape[0]} != token_ids length {token_ids.shape[0]}"
        )

    probs = logits.softmax(dim=-1)
    token_probs = probs.gather(dim=1, index=token_ids.unsqueeze(1)).squeeze(1)
    return token_probs.mean().item()


class E1RAGPredictor:
    """E1 predictor with MSA-based retrieval augmentation. Supports ensemble PPLL scoring
    and embedding extraction with RAG context."""

    def __init__(
        self,
        model: E1ForMaskedLM,
        max_batch_tokens: int = 131072,
        cache_size: int = 1,
        context_cache_dir: Optional[str] = None,
        seed: int = 42,
        max_context_tokens: list[int] = None,
        similarity_thresholds: list[float] = None,
        min_query_similarity: float = 0.3,
    ):
        self.model = model
        self.model.eval()
        self.max_batch_tokens = max_batch_tokens
        self.seed = seed

        # Initialize E1Predictor for context-aware batching and KV caching
        self.predictor = E1Predictor(
            model=self.model,
            data_prep_config=DataPrepConfig(remove_X_tokens=True),
            max_batch_tokens=max_batch_tokens,
            fields_to_save=["logits"],
            save_masked_positions_only=False,
            keep_predictions_in_gpu=False,
            use_cache=True,
            cache_size=cache_size,
        )
        self.vocab = self.predictor.batch_preparer.tokenizer.get_vocab()

        # Build context specifications
        self.max_context_tokens = max_context_tokens if max_context_tokens is not None else DEFAULT_MAX_CONTEXT_TOKENS
        self.similarity_thresholds = similarity_thresholds if similarity_thresholds is not None else DEFAULT_SIMILARITY_THRESHOLDS
        self.min_query_similarity = min_query_similarity
        self.context_specs = build_context_specifications(
            max_context_tokens=self.max_context_tokens,
            similarity_thresholds=self.similarity_thresholds,
            min_query_similarity=self.min_query_similarity,
        )

        self._context_cache = None
        if context_cache_dir is not None:
            _key = repr((sorted(self.max_context_tokens), sorted(self.similarity_thresholds), self.min_query_similarity))
            specs_hash = hashlib.md5(_key.encode()).hexdigest()[:8]
            self._context_cache = ContextCache(context_cache_dir, specs_hash, seed)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = None,
        **kwargs,
    ) -> "E1RAGPredictor":
        """Load a pretrained E1 model and create a RAG predictor."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = E1ForMaskedLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model = model.to(device)
        model.eval()

        return cls(model=model, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    # MSA loading helpers

    def load_msa_dir(self, msa_dir: str) -> dict[str, str]:
        """Load all .a3m files from a directory, keyed by query sequence -> a3m path."""
        return load_msa_dir(msa_dir)

    def load_msa_from_hf(self, hf_path: str, **kwargs) -> dict[str, str]:
        """Download and load MSAs from HuggingFace."""
        return load_msa_from_hf(hf_path, **kwargs)

    def sample_contexts(
        self,
        a3m_path: str,
        seed: Optional[int] = None,
    ) -> dict[str, str]:
        """Sample all context configurations from an MSA file. Returns context_id -> context string."""
        seed = seed if seed is not None else self.seed

        if self._context_cache is not None:
            cached = self._context_cache.get(a3m_path)
            if cached is not None:
                return cached

        contexts = sample_contexts_for_msa(a3m_path, self.context_specs, seed=seed)

        if self._context_cache is not None:
            self._context_cache.put(a3m_path, contexts)

        return contexts

    # Pooling

    def _pool_hidden_states(
        self,
        hidden_list: list[torch.Tensor],
        pooling_types: list[str],
    ) -> torch.Tensor:
        """Pool variable-length hidden states via Pooler. Returns (batch, n_types * hidden_dim)."""
        pooler = Pooler(pooling_types)

        # Pad to common length and create attention mask
        max_len = max(h.shape[0] for h in hidden_list)
        hidden_dim = hidden_list[0].shape[1]
        batch_size = len(hidden_list)

        padded = torch.zeros(batch_size, max_len, hidden_dim, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, device=self.device)

        for i, h in enumerate(hidden_list):
            seq_len = h.shape[0]
            padded[i, :seq_len] = h
            attention_mask[i, :seq_len] = 1.0

        return pooler(padded, attention_mask)

    @torch.inference_mode()
    def _forward_for_embedding(
        self,
        sequences: list[str],
        context: Optional[str] = None,
    ) -> list[torch.Tensor]:
        """Run E1Predictor with token_embeddings output. Returns (seq_len, hidden_dim) per sequence."""
        embed_predictor = E1Predictor(
            model=self.model,
            data_prep_config=DataPrepConfig(remove_X_tokens=True),
            max_batch_tokens=self.max_batch_tokens,
            fields_to_save=["token_embeddings"],
            save_masked_positions_only=False,
            keep_predictions_in_gpu=True,
            use_cache=False,  # Don't use KV cache for embeddings
            cache_size=1,
        )

        num_seqs = len(sequences)
        context_seqs = None
        if context:
            context_seqs = {"embed_ctx": context}

        predictions = list(embed_predictor.predict(
            sequences=sequences,
            sequence_ids=list(range(num_seqs)),
            context_seqs=context_seqs,
        ))

        predictions.sort(key=lambda p: p["id"])

        result = []
        for pred in predictions:
            result.append(pred["token_embeddings"])

        return result

    # Scoring

    @torch.inference_mode()
    def score_ppll(
        self,
        sequences: list[str],
        a3m_path: str,
        ensemble: bool = True,
        seed: Optional[int] = None,
        progress: bool = True,
    ) -> list[float]:
        """Score sequences via PPLL ensemble over all context configurations.

        Returns one score per sequence (higher = better).
        """
        contexts = self.sample_contexts(a3m_path, seed=seed)

        num_seqs = len(sequences)
        num_contexts = len(contexts)

        # Pre-compute token IDs for each sequence
        seq_token_ids: list[torch.Tensor] = [
            torch.tensor([self.vocab[aa] for aa in seq], device=self.device)
            for seq in sequences
        ]

        # Map context_id to index for aggregation
        context_id_to_idx = {ctx_id: i for i, ctx_id in enumerate(contexts.keys())}
        all_scores = torch.zeros(num_seqs, num_contexts, device=self.device)

        # Process one context prompt at a time (allows KV cache reuse within context)
        context_iter = list(contexts.items())
        if progress:
            context_iter_wrapped = tqdm(context_iter, desc="Scoring with contexts", leave=False)
        else:
            context_iter_wrapped = context_iter

        for ctx_id, ctx_str in context_iter_wrapped:
            ctx_idx = context_id_to_idx[ctx_id]
            single_context = {ctx_id: ctx_str}

            predictions = list(self.predictor.predict(
                sequences=sequences,
                sequence_ids=list(range(num_seqs)),
                context_seqs=single_context,
            ))

            for pred in predictions:
                seq_idx = pred["id"]
                logits = pred["logits"]

                ppll_score = compute_ppll(logits, seq_token_ids[seq_idx])
                all_scores[seq_idx, ctx_idx] = ppll_score

            # Clear KV cache after each context prompt is processed
            if self.predictor.kv_cache is not None:
                self.predictor.kv_cache.reset()

        if ensemble:
            return all_scores.mean(dim=1).tolist()
        return all_scores.tolist()

    # Embedding extraction

    @torch.inference_mode()
    def embed(
        self,
        sequences: list[str],
        a3m_path: Optional[str] = None,
        context: Optional[str] = None,
        pooling_types: Optional[list[str]] = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings, optionally with RAG context from an .a3m file.

        Use pooling_types for multi-type Pooler output, pooling for simple mean/cls,
        or matrix_embed=True for per-residue tensors.
        """
        if a3m_path is not None and context is None:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=DEFAULT_EMBED_MAX_TOKENS,
                max_query_similarity=DEFAULT_EMBED_SIMILARITY,
                min_query_similarity=0.3,
            )
            contexts, _ = sample_multiple_contexts(
                msa_path=a3m_path,
                context_specifications=[spec],
                seed=self.seed,
            )
            context = contexts[0] if contexts else None

        hidden_list = self._forward_for_embedding(sequences, context=context)

        if matrix_embed:
            return hidden_list

        if pooling_types is not None:
            return self._pool_hidden_states(hidden_list, pooling_types)

        embeddings = []
        for hidden in hidden_list:
            if pooling == "mean":
                embeddings.append(hidden.mean(dim=0))
            elif pooling == "cls":
                embeddings.append(hidden[0])
            else:
                embeddings.append(hidden)

        if pooling in ("mean", "cls"):
            return torch.stack(embeddings)
        return embeddings

    @torch.inference_mode()
    def embed_dataset(
        self,
        sequences: list[str],
        msa_lookup: Optional[dict[str, str]] = None,
        msa_dir: Optional[str] = None,
        msa_hf_path: Optional[str] = None,
        batch_size: int = 2,
        max_len: int = 2048,
        pooling_types: Optional[list[str]] = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
        embed_dtype: torch.dtype = torch.bfloat16,
        embed_max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
        embed_similarity: float = DEFAULT_EMBED_SIMILARITY,
        progress: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Embed a dataset of sequences with RAG context.

        Provide MSA source via msa_lookup, msa_dir, or msa_hf_path.
        Returns dict mapping sequences to embedding tensors.
        """
        # Load MSA lookup if not provided
        if msa_lookup is None:
            if msa_dir is not None:
                msa_lookup = load_msa_dir(msa_dir)
            elif msa_hf_path is not None:
                msa_lookup = load_msa_from_hf(msa_hf_path)
            else:
                logger.warning("No MSA source provided. Embedding without RAG context.")
                msa_lookup = {}

        # Deduplicate and truncate, preserving original->truncated mapping for keying
        truncated_map = {seq: seq[:max_len] for seq in sequences}
        unique_seqs = list(set(truncated_map.values()))
        unique_seqs = sorted(unique_seqs, key=len, reverse=True)

        # Pre-sample contexts for all sequences using a single context prompt
        context_map: dict[str, Optional[str]] = {}
        spec = ContextSpecification(
            max_num_samples=511,
            max_token_length=embed_max_tokens,
            max_query_similarity=embed_similarity,
            min_query_similarity=0.3,
        )
        n_with_context = 0
        for seq in unique_seqs:
            a3m_path = get_msa_for_sequence(seq, msa_lookup)
            if a3m_path is not None:
                contexts, _ = sample_multiple_contexts(
                    msa_path=a3m_path,
                    context_specifications=[spec],
                    seed=self.seed,
                )
                context_map[seq] = contexts[0] if contexts else None
                if context_map[seq]:
                    n_with_context += 1
            else:
                context_map[seq] = None

        if progress:
            logger.info(f"RAG context available for {n_with_context}/{len(unique_seqs)} sequences")

        # Group sequences by context to enable batching with shared context
        context_groups: dict[Optional[str], list[str]] = defaultdict(list)
        for seq in unique_seqs:
            ctx = context_map[seq]
            context_groups[ctx].append(seq)

        embeddings_dict: dict[str, torch.Tensor] = {}

        total_batches = sum(
            (len(seqs) + batch_size - 1) // batch_size
            for seqs in context_groups.values()
        )
        pbar = tqdm(total=total_batches, desc="Embedding with RAG", disable=not progress)

        # Determine pooling behavior
        use_pooler = pooling_types is not None and not matrix_embed

        for ctx, ctx_seqs in context_groups.items():
            for i in range(0, len(ctx_seqs), batch_size):
                batch_seqs = ctx_seqs[i:i + batch_size]
                hidden_list = self._forward_for_embedding(batch_seqs, context=ctx)

                if matrix_embed:
                    # Per-residue embeddings, no pooling
                    for seq, hidden in zip(batch_seqs, hidden_list):
                        embeddings_dict[seq] = hidden.to(embed_dtype).cpu()
                elif use_pooler:
                    # Use Pooler for multi-type pooling
                    pooled = self._pool_hidden_states(hidden_list, pooling_types)
                    for j, seq in enumerate(batch_seqs):
                        embeddings_dict[seq] = pooled[j].to(embed_dtype).cpu()
                else:
                    # Simple pooling
                    for seq, hidden in zip(batch_seqs, hidden_list):
                        if pooling == "mean":
                            emb = hidden.mean(dim=0)
                        elif pooling == "cls":
                            emb = hidden[0]
                        else:
                            emb = hidden
                        embeddings_dict[seq] = emb.to(embed_dtype).cpu()

                pbar.update(1)

        pbar.close()

        # return embeddings keyed by full-length seqs
        result = {}
        for seq in sequences:
            trunc = truncated_map[seq]
            if trunc in embeddings_dict:
                result[seq] = embeddings_dict[trunc]
        return result

def score_with_msa(
    sequences: list[str],
    a3m_path: str,
    model_name: str = "Synthyra/Profluent-E1-600M",
    seed: Optional[int] = None,
    progress: bool = True,
) -> list[float]:
    """Score sequences using PPLL with MSA context."""
    predictor = E1RAGPredictor.from_pretrained(model_name)
    return predictor.score_ppll(sequences, a3m_path=a3m_path, seed=seed, progress=progress)


def embed_with_msa(
    sequences: list[str],
    a3m_path: str = None,
    msa_dir: str = None,
    model_name: str = "Synthyra/Profluent-E1-600M",
    pooling_types: Optional[list[str]] = None,
    pooling: str = "mean",
    matrix_embed: bool = False,
) -> dict[str, torch.Tensor]:
    """Embed sequences with MSA context."""
    predictor = E1RAGPredictor.from_pretrained(model_name)
    if a3m_path is not None:
        query_seq = get_query_from_a3m(a3m_path)
        msa_lookup = {query_seq: a3m_path}
    else:
        msa_lookup = None
    return predictor.embed_dataset(
        sequences, msa_lookup=msa_lookup, msa_dir=msa_dir,
        pooling_types=pooling_types, pooling=pooling, matrix_embed=matrix_embed,
    )
