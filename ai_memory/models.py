"""
Data models for the AI Memory System.
Defines core data structures used across all memory modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid


class MessageRole(str, Enum):
    """Role of a participant in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryType(str, Enum):
    """Type/category of a memory entry."""
    EPISODIC = "episodic"       # Specific events or interactions
    SEMANTIC = "semantic"       # General facts and knowledge
    PROCEDURAL = "procedural"   # How to do things
    EMOTIONAL = "emotional"     # Emotional context and sentiment


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: MessageRole
    content: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            tokens=data.get("tokens"),
        )


@dataclass
class Memory:
    """Represents a long-term memory entry."""
    content: str
    memory_type: MemoryType
    user_id: str
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    source_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, MemoryType) else self.memory_type,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "tags": self.tags,
            "source_session_id": self.source_session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else datetime.utcnow(),
            importance_score=data.get("importance_score", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            tags=data.get("tags", []),
            source_session_id=data.get("source_session_id"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class UserPreference:
    """Represents a learned user preference."""
    user_id: str
    category: str
    key: str
    value: Any
    preference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 0.5
    observation_count: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_id": self.preference_id,
            "user_id": self.user_id,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        return cls(
            preference_id=data.get("preference_id", str(uuid.uuid4())),
            user_id=data["user_id"],
            category=data["category"],
            key=data["key"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            observation_count=data.get("observation_count", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VectorEntry:
    """Represents a vector-embedded memory entry for similarity search."""
    content: str
    embedding: List[float]
    user_id: str
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    score: Optional[float] = None
    source_type: str = "memory"
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "embedding": self.embedding,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "score": self.score,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorEntry":
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            content=data["content"],
            embedding=data["embedding"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            score=data.get("score"),
            source_type=data.get("source_type", "memory"),
            source_id=data.get("source_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Result from a memory search operation."""
    entry: Any
    score: float
    result_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict() if hasattr(self.entry, "to_dict") else str(self.entry),
            "score": self.score,
            "result_type": self.result_type,
        }
