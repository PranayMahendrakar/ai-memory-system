"""
Vector Memory module for the AI Memory System.

Provides embedding-based semantic similarity search over stored memories.
Supports pluggable embedding backends (local TF-IDF by default,
or external providers like OpenAI, Sentence Transformers).
"""

import json
import math
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .models import VectorEntry, SearchResult


# --------------------------------------------------------------------------- #
#  Built-in embedding backends                                                  #
# --------------------------------------------------------------------------- #

class TFIDFEmbedder:
    """
    Lightweight TF-IDF based embedder that works with zero dependencies.
    Uses a vocabulary built from stored documents.
    Suitable for development/testing. For production use a transformer model.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def fit(self, documents: List[str]) -> None:
        """Build vocabulary and IDF from a corpus of documents."""
        self._doc_count = len(documents)
        df: Dict[str, int] = {}
        term_freq: Dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
            for token in self._tokenize(doc):
                term_freq[token] = term_freq.get(token, 0) + 1

        # Select top vocab_size terms by document frequency
        sorted_terms = sorted(df.keys(), key=lambda t: df[t], reverse=True)
        selected = sorted_terms[:self.vocab_size]
        self._vocab = {term: idx for idx, term in enumerate(selected)}
        self._idf = {
            term: math.log((self._doc_count + 1) / (df[term] + 1)) + 1
            for term in selected
        }

    def embed(self, text: str) -> List[float]:
        """Convert text to a TF-IDF vector."""
        tokens = self._tokenize(text)
        if not tokens or not self._vocab:
            return [0.0] * max(len(self._vocab), 1)

        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1 / len(tokens)

        vector = [0.0] * len(self._vocab)
        for term, tf_val in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                idf_val = self._idf.get(term, 1.0)
                vector[idx] = tf_val * idf_val

        # L2 normalize
        norm = math.sqrt(sum(v ** 2 for v in vector)) or 1.0
        return [v / norm for v in vector]

    def to_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "vocab": self._vocab,
            "idf": self._idf,
            "doc_count": self._doc_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TFIDFEmbedder":
        obj = cls(vocab_size=data.get("vocab_size", 1000))
        obj._vocab = data.get("vocab", {})
        obj._idf = data.get("idf", {})
        obj._doc_count = data.get("doc_count", 0)
        return obj


class HashEmbedder:
    """
    Feature hashing embedder — no vocabulary needed, deterministic.
    O(1) per token, dimension controlled by hash_size.
    Good for production use when no external embedding service is available.
    """

    def __init__(self, hash_size: int = 512):
        self.hash_size = hash_size

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.hash_size
        if not tokens:
            return vector

        for token in tokens:
            h = hash(token) % self.hash_size
            sign = 1 if (hash(token + "_sign") % 2 == 0) else -1
            vector[h] += sign / len(tokens)

        norm = math.sqrt(sum(v ** 2 for v in vector)) or 1.0
        return [v / norm for v in vector]

    def to_dict(self) -> Dict:
        return {"hash_size": self.hash_size}

    @classmethod
    def from_dict(cls, data: Dict) -> "HashEmbedder":
        return cls(hash_size=data.get("hash_size", 512))


# --------------------------------------------------------------------------- #
#  VectorMemory                                                                 #
# --------------------------------------------------------------------------- #

class VectorMemory:
    """
    Vector-based semantic memory with similarity search.

    Stores embeddings alongside content and supports cosine similarity
    retrieval. Pluggable embedding function allows use of any model
    (OpenAI, Sentence Transformers, Cohere, etc.).

    Example:
        # Default (hash embedder, no dependencies)
        vm = VectorMemory(storage_path="./memory_store")
        vm.add(user_id="u1", content="Python is a great scripting language")
        vm.add(user_id="u1", content="I love writing machine learning code")
        results = vm.search(user_id="u1", query="programming languages", top_k=5)

        # With OpenAI embeddings
        import openai
        def openai_embed(text):
            r = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return r["data"][0]["embedding"]
        vm = VectorMemory(embed_fn=openai_embed)
    """

    def __init__(
        self,
        storage_path: str = "./ai_memory_store",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        embedder_type: str = "hash",  # "hash" or "tfidf"
        hash_size: int = 512,
        similarity_threshold: float = 0.1,
    ):
        """
        Initialize VectorMemory.

        Args:
            storage_path: Base directory for persisting vector entries.
            embed_fn: Custom embedding function. Takes text, returns List[float].
                      If None, uses the built-in embedder (hash or tfidf).
            embedder_type: Built-in embedder to use if embed_fn is None.
            hash_size: Dimension for HashEmbedder (ignored for tfidf).
            similarity_threshold: Minimum cosine similarity to include in results.
        """
        self.storage_path = Path(storage_path) / "vectors"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self._custom_embed_fn = embed_fn

        # Built-in embedder
        if embed_fn is None:
            if embedder_type == "tfidf":
                self._embedder = TFIDFEmbedder()
            else:
                self._embedder = HashEmbedder(hash_size=hash_size)
        else:
            self._embedder = None

        self._cache: Dict[str, List[VectorEntry]] = {}

    # --------------------------------------------------------------------- #
    #  Core operations                                                        #
    # --------------------------------------------------------------------- #

    def add(
        self,
        user_id: str,
        content: str,
        source_type: str = "memory",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorEntry:
        """
        Embed and store content for a user.

        Args:
            user_id: User to store the vector for.
            content: Text content to embed and store.
            source_type: Type of source ('memory', 'conversation', 'preference').
            source_id: Optional ID linking back to source object.
            metadata: Additional metadata.

        Returns:
            The created VectorEntry.
        """
        embedding = self._embed(content)
        entry = VectorEntry(
            content=content,
            embedding=embedding,
            user_id=user_id,
            source_type=source_type,
            source_id=source_id,
            metadata=metadata or {},
        )
        entries = self._load_user_entries(user_id)
        entries.append(entry)
        self._cache[user_id] = entries
        self._persist_user_entries(user_id, entries)

        # Retrain TF-IDF if using it
        if isinstance(self._embedder, TFIDFEmbedder):
            self._retrain_tfidf(user_id, entries)

        return entry

    def add_batch(
        self,
        user_id: str,
        items: List[Dict[str, Any]],
    ) -> List[VectorEntry]:
        """
        Add multiple entries at once (more efficient for bulk loading).

        Args:
            user_id: User ID.
            items: List of dicts with 'content' and optional 'source_type',
                   'source_id', 'metadata' keys.

        Returns:
            List of created VectorEntry objects.
        """
        # For TF-IDF, pre-fit on all new documents
        if isinstance(self._embedder, TFIDFEmbedder):
            existing = self._load_user_entries(user_id)
            all_docs = [e.content for e in existing] + [item["content"] for item in items]
            self._embedder.fit(all_docs)

        results = []
        for item in items:
            entry = self.add(
                user_id=user_id,
                content=item["content"],
                source_type=item.get("source_type", "memory"),
                source_id=item.get("source_id"),
                metadata=item.get("metadata"),
            )
            results.append(entry)
        return results

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        source_type: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Semantic similarity search over stored entries.

        Args:
            user_id: User to search for.
            query: Query text to find similar entries for.
            top_k: Maximum number of results.
            source_type: Optional filter by source type.
            min_score: Minimum similarity score (overrides instance threshold).

        Returns:
            List of SearchResult objects sorted by cosine similarity (desc).
        """
        threshold = min_score if min_score is not None else self.similarity_threshold
        entries = self._load_user_entries(user_id)

        if not entries:
            return []

        # For TF-IDF, ensure it's fitted
        if isinstance(self._embedder, TFIDFEmbedder) and not self._embedder._vocab:
            self._embedder.fit([e.content for e in entries])

        query_vec = self._embed(query)
        if all(v == 0.0 for v in query_vec):
            return []

        if source_type:
            entries = [e for e in entries if e.source_type == source_type]

        # Score all entries
        scored: List[Tuple[VectorEntry, float]] = []
        for entry in entries:
            if not entry.embedding:
                continue
            sim = self._cosine_similarity(query_vec, entry.embedding)
            if sim >= threshold:
                scored.append((entry, sim))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry, score in scored[:top_k]:
            entry_copy = VectorEntry(**{**entry.to_dict(), "score": score,
                                        "created_at": entry.created_at,
                                        "embedding": entry.embedding})
            results.append(SearchResult(entry=entry_copy, score=score, result_type="vector"))

        return results

    def delete(self, user_id: str, entry_id: str) -> bool:
        """Delete a vector entry by ID."""
        entries = self._load_user_entries(user_id)
        new_entries = [e for e in entries if e.entry_id != entry_id]
        if len(new_entries) == len(entries):
            return False
        self._cache[user_id] = new_entries
        self._persist_user_entries(user_id, new_entries)
        return True

    def delete_by_source(self, user_id: str, source_id: str) -> int:
        """Delete all vector entries linked to a source ID."""
        entries = self._load_user_entries(user_id)
        new_entries = [e for e in entries if e.source_id != source_id]
        removed = len(entries) - len(new_entries)
        if removed > 0:
            self._cache[user_id] = new_entries
            self._persist_user_entries(user_id, new_entries)
        return removed

    def clear_user(self, user_id: str) -> int:
        """Clear all vector entries for a user."""
        entries = self._load_user_entries(user_id)
        count = len(entries)
        self._cache[user_id] = []
        self._persist_user_entries(user_id, [])
        return count

    def get_all(self, user_id: str) -> List[VectorEntry]:
        """Get all vector entries for a user."""
        return list(self._load_user_entries(user_id))

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get vector store statistics for a user."""
        entries = self._load_user_entries(user_id)
        if not entries:
            return {"total": 0}

        dim = len(entries[0].embedding) if entries[0].embedding else 0
        type_counts: Dict[str, int] = {}
        for e in entries:
            type_counts[e.source_type] = type_counts.get(e.source_type, 0) + 1

        return {
            "total": len(entries),
            "embedding_dimension": dim,
            "by_source_type": type_counts,
            "embedder": type(self._embedder).__name__ if self._embedder else "custom",
        }

    # --------------------------------------------------------------------- #
    #  Embedding helpers                                                      #
    # --------------------------------------------------------------------- #

    def _embed(self, text: str) -> List[float]:
        """Embed text using the configured embedder."""
        if self._custom_embed_fn is not None:
            return self._custom_embed_fn(text)
        return self._embedder.embed(text)

    def _retrain_tfidf(self, user_id: str, entries: List[VectorEntry]) -> None:
        """Retrain TF-IDF embedder and re-embed all entries for user."""
        if not isinstance(self._embedder, TFIDFEmbedder):
            return
        docs = [e.content for e in entries]
        if len(docs) < 5:
            return  # Too few docs to train meaningfully
        self._embedder.fit(docs)
        # Re-embed all entries
        for entry in entries:
            entry.embedding = self._embedder.embed(entry.content)
        self._persist_user_entries(user_id, entries)

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            # Pad shorter vector
            max_len = max(len(vec_a), len(vec_b))
            vec_a = vec_a + [0.0] * (max_len - len(vec_a))
            vec_b = vec_b + [0.0] * (max_len - len(vec_b))

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a ** 2 for a in vec_a)) or 1e-9
        norm_b = math.sqrt(sum(b ** 2 for b in vec_b)) or 1e-9
        return dot / (norm_a * norm_b)

    # --------------------------------------------------------------------- #
    #  Persistence                                                            #
    # --------------------------------------------------------------------- #

    def _get_user_file(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return self.storage_path / f"{safe_id}.json"

    def _load_user_entries(self, user_id: str) -> List[VectorEntry]:
        if user_id in self._cache:
            return self._cache[user_id]

        file_path = self._get_user_file(user_id)
        entries: List[VectorEntry] = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    entries.append(VectorEntry.from_dict(item))
            except (json.JSONDecodeError, KeyError):
                pass

        self._cache[user_id] = entries
        return entries

    def _persist_user_entries(self, user_id: str, entries: List[VectorEntry]) -> None:
        file_path = self._get_user_file(user_id)
        data = [e.to_dict() for e in entries]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
