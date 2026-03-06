"""
Long-Term Memory module for the AI Memory System.

Provides persistent storage and retrieval of memories across sessions,
with importance scoring, decay, and consolidation features.
"""

import json
import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .models import Memory, MemoryType, SearchResult


class LongTermMemory:
    """
    Manages persistent long-term memories for users.

    Memories are stored in a JSON-based local store by default (pluggable).
    Supports importance scoring, temporal decay, and tag-based retrieval.

    Example:
        ltm = LongTermMemory(storage_path="./memory_store")
        mem = ltm.store(
            user_id="user_123",
            content="User prefers dark mode UI",
            memory_type=MemoryType.SEMANTIC,
            importance_score=0.8,
            tags=["ui", "preferences"]
        )
        results = ltm.retrieve(user_id="user_123", query="UI preferences", top_k=5)
    """

    def __init__(
        self,
        storage_path: str = "./ai_memory_store",
        decay_factor: float = 0.01,
        max_memories_per_user: int = 10000,
    ):
        """
        Initialize LongTermMemory.

        Args:
            storage_path: Directory path for persisting memories.
            decay_factor: Rate at which importance decays over time (per day).
            max_memories_per_user: Maximum memories per user before pruning.
        """
        self.storage_path = Path(storage_path) / "long_term"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.decay_factor = decay_factor
        self.max_memories_per_user = max_memories_per_user
        self._cache: Dict[str, Dict[str, Memory]] = {}

    # --------------------------------------------------------------------- #
    #  Core CRUD                                                              #
    # --------------------------------------------------------------------- #

    def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
        source_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a new long-term memory."""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            source_session_id=source_session_id,
            metadata=metadata or {},
        )
        self._save_memory(memory)
        # Prune if over limit
        self._maybe_prune(user_id)
        return memory

    def get(self, user_id: str, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        memories = self._load_user_memories(user_id)
        memory = memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            self._save_memory(memory)
        return memory

    def update(
        self,
        user_id: str,
        memory_id: str,
        content: Optional[str] = None,
        importance_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """Update an existing memory."""
        memory = self.get(user_id, memory_id)
        if not memory:
            return None
        if content is not None:
            memory.content = content
        if importance_score is not None:
            memory.importance_score = max(0.0, min(1.0, importance_score))
        if tags is not None:
            memory.tags = tags
        if metadata is not None:
            memory.metadata.update(metadata)
        memory.updated_at = datetime.utcnow()
        self._save_memory(memory)
        return memory

    def delete(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory."""
        memories = self._load_user_memories(user_id)
        if memory_id not in memories:
            return False
        del memories[memory_id]
        if user_id in self._cache:
            self._cache[user_id] = memories
        self._persist_user_memories(user_id, memories)
        return True

    def clear_user(self, user_id: str) -> int:
        """Clear all memories for a user. Returns count deleted."""
        memories = self._load_user_memories(user_id)
        count = len(memories)
        self._cache[user_id] = {}
        self._persist_user_memories(user_id, {})
        return count

    # --------------------------------------------------------------------- #
    #  Retrieval                                                              #
    # --------------------------------------------------------------------- #

    def retrieve(
        self,
        user_id: str,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
    ) -> List[SearchResult]:
        """
        Retrieve memories matching criteria, ranked by effective importance.

        Args:
            user_id: User to retrieve memories for.
            query: Optional keyword query (simple substring match).
            memory_type: Filter by memory type.
            tags: Filter by tags (any match).
            top_k: Maximum results to return.
            min_importance: Minimum importance threshold.
            since: Only return memories created after this datetime.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        memories = list(self._load_user_memories(user_id).values())

        # Apply filters
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set.intersection(set(m.tags))]
        if since:
            memories = [m for m in memories if m.created_at >= since]
        if min_importance > 0.0:
            memories = [m for m in memories if m.importance_score >= min_importance]

        # Score memories
        scored: List[Tuple[Memory, float]] = []
        for memory in memories:
            score = self._compute_score(memory, query)
            scored.append((memory, score))

        # Sort and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for memory, score in scored[:top_k]:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            self._save_memory(memory)
            results.append(SearchResult(entry=memory, score=score, result_type="memory"))

        return results

    def get_all(self, user_id: str) -> List[Memory]:
        """Get all memories for a user."""
        return list(self._load_user_memories(user_id).values())

    def get_by_tags(self, user_id: str, tags: List[str]) -> List[Memory]:
        """Get memories with any of the specified tags."""
        tag_set = set(tags)
        return [m for m in self.get_all(user_id) if tag_set.intersection(set(m.tags))]

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        memories = self.get_all(user_id)
        if not memories:
            return {"total": 0}

        type_counts: Dict[str, int] = defaultdict(int)
        for m in memories:
            type_counts[m.memory_type.value] += 1

        return {
            "total": len(memories),
            "by_type": dict(type_counts),
            "avg_importance": sum(m.importance_score for m in memories) / len(memories),
            "most_accessed": max(memories, key=lambda m: m.access_count).memory_id,
            "oldest": min(memories, key=lambda m: m.created_at).memory_id,
            "newest": max(memories, key=lambda m: m.created_at).memory_id,
        }

    # --------------------------------------------------------------------- #
    #  Scoring & Decay                                                        #
    # --------------------------------------------------------------------- #

    def _compute_score(self, memory: Memory, query: Optional[str] = None) -> float:
        """
        Compute effective importance score with temporal decay and query relevance.
        Score = importance * recency_factor * query_relevance
        """
        days_old = (datetime.utcnow() - memory.created_at).days
        recency_factor = math.exp(-self.decay_factor * days_old)
        base_score = memory.importance_score * recency_factor

        if query:
            query_lower = query.lower()
            content_lower = memory.content.lower()
            # Simple keyword relevance boost
            words = query_lower.split()
            matches = sum(1 for w in words if w in content_lower)
            relevance = matches / max(len(words), 1)
            # Also check tags
            tag_match = any(query_lower in tag.lower() for tag in memory.tags)
            base_score *= (1 + relevance + (0.2 if tag_match else 0))

        return round(base_score, 6)

    def consolidate(self, user_id: str, similarity_threshold: float = 0.85) -> int:
        """
        Consolidate similar memories by merging duplicates.
        Returns the number of memories removed.

        Note: This uses simple content similarity (Jaccard). For production,
        use vector embeddings via VectorMemory for better consolidation.
        """
        memories = self.get_all(user_id)
        merged = 0
        skip_ids = set()

        for i, m1 in enumerate(memories):
            if m1.memory_id in skip_ids:
                continue
            for m2 in memories[i + 1:]:
                if m2.memory_id in skip_ids:
                    continue
                if self._jaccard_similarity(m1.content, m2.content) >= similarity_threshold:
                    # Keep the one with higher importance, merge tags
                    keep, remove = (m1, m2) if m1.importance_score >= m2.importance_score else (m2, m1)
                    keep.tags = list(set(keep.tags + remove.tags))
                    keep.importance_score = min(1.0, keep.importance_score + 0.05)
                    keep.updated_at = datetime.utcnow()
                    self._save_memory(keep)
                    self.delete(user_id, remove.memory_id)
                    skip_ids.add(remove.memory_id)
                    merged += 1

        return merged

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    # --------------------------------------------------------------------- #
    #  Persistence                                                            #
    # --------------------------------------------------------------------- #

    def _get_user_file(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return self.storage_path / f"{safe_id}.json"

    def _load_user_memories(self, user_id: str) -> Dict[str, Memory]:
        if user_id in self._cache:
            return self._cache[user_id]

        file_path = self._get_user_file(user_id)
        memories: Dict[str, Memory] = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    memory = Memory.from_dict(item)
                    memories[memory.memory_id] = memory
            except (json.JSONDecodeError, KeyError):
                pass

        self._cache[user_id] = memories
        return memories

    def _save_memory(self, memory: Memory) -> None:
        memories = self._load_user_memories(memory.user_id)
        memories[memory.memory_id] = memory
        self._cache[memory.user_id] = memories
        self._persist_user_memories(memory.user_id, memories)

    def _persist_user_memories(self, user_id: str, memories: Dict[str, Memory]) -> None:
        file_path = self._get_user_file(user_id)
        data = [m.to_dict() for m in memories.values()]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _maybe_prune(self, user_id: str) -> None:
        memories = self._load_user_memories(user_id)
        if len(memories) <= self.max_memories_per_user:
            return
        # Remove lowest importance memories
        sorted_mems = sorted(memories.values(), key=lambda m: self._compute_score(m))
        to_remove = len(memories) - self.max_memories_per_user
        for mem in sorted_mems[:to_remove]:
            self.delete(user_id, mem.memory_id)
