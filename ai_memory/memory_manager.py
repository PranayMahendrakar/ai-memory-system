"""
MemoryManager — unified orchestrator for the AI Memory System.

Single entry point that coordinates LongTermMemory, ConversationHistory,
PreferenceLearner, and VectorMemory. Designed to be used as a drop-in
memory layer for any LLM application.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .long_term_memory import LongTermMemory
from .conversation_history import ConversationHistory
from .preference_learner import PreferenceLearner
from .vector_memory import VectorMemory
from .models import (
    Memory, MemoryType, Message, MessageRole,
    UserPreference, VectorEntry, SearchResult
)


class MemoryManager:
    """
    Unified memory manager for LLM applications.

    Coordinates all four memory subsystems and provides a high-level API
    for the most common memory operations. The recommended way to use
    ai_memory in application code.

    Example:
        from ai_memory import MemoryManager, MessageRole, MemoryType

        mm = MemoryManager(user_id="user_123", storage_path="./memory")

        # Record a conversation turn
        mm.add_user_message("Hello, my name is Alex!")
        mm.add_assistant_message("Hi Alex! How can I help you today?")

        # Store an important fact
        mm.remember("Alex lives in San Francisco", MemoryType.SEMANTIC,
                    importance=0.9, tags=["personal", "location"])

        # Get context for next LLM call
        messages = mm.get_context()

        # Search for relevant memories
        results = mm.search("where does the user live?")

        # Get a fully assembled prompt context
        context = mm.build_context(query="Tell me about the user")
    """

    def __init__(
        self,
        user_id: str,
        storage_path: str = "./ai_memory_store",
        session_id: Optional[str] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        auto_learn_preferences: bool = True,
        auto_vectorize_memories: bool = True,
        context_window_size: int = 20,
        context_max_tokens: Optional[int] = None,
    ):
        """
        Initialize MemoryManager for a specific user.

        Args:
            user_id: The user this memory manager is for.
            storage_path: Base directory for all memory storage.
            session_id: Current session ID. Auto-generated if None.
            embed_fn: Custom embedding function for vector memory.
            auto_learn_preferences: Auto-detect preferences from messages.
            auto_vectorize_memories: Auto-add memories to vector store.
            context_window_size: Default conversation window size.
            context_max_tokens: Optional token budget for context windows.
        """
        self.user_id = user_id
        self.storage_path = storage_path
        self.session_id = session_id or self._generate_session_id()
        self.auto_learn_preferences = auto_learn_preferences
        self.auto_vectorize_memories = auto_vectorize_memories
        self.context_window_size = context_window_size
        self.context_max_tokens = context_max_tokens

        # Initialize subsystems
        self.long_term = LongTermMemory(storage_path=storage_path)
        self.history = ConversationHistory(storage_path=storage_path)
        self.preferences = PreferenceLearner(storage_path=storage_path)
        self.vectors = VectorMemory(storage_path=storage_path, embed_fn=embed_fn)

    # --------------------------------------------------------------------- #
    #  Conversation                                                           #
    # --------------------------------------------------------------------- #

    def add_user_message(
        self,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Record a user message in conversation history.
        Automatically detects preferences if auto_learn_preferences is True.
        """
        sid = session_id or self.session_id
        msg = self.history.add_user_message(
            session_id=sid, content=content, user_id=self.user_id,
            metadata=metadata
        )

        if self.auto_learn_preferences:
            self.preferences.detect_from_message(
                user_id=self.user_id, content=content, session_id=sid
            )

        return msg

    def add_assistant_message(
        self,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Record an assistant message in conversation history."""
        sid = session_id or self.session_id
        return self.history.add_assistant_message(
            session_id=sid, content=content, metadata=metadata
        )

    def add_system_message(
        self,
        content: str,
        session_id: Optional[str] = None,
    ) -> Message:
        """Add a system message to the current session."""
        sid = session_id or self.session_id
        return self.history.add_system_message(session_id=sid, content=content)

    def get_context(
        self,
        session_id: Optional[str] = None,
        max_messages: Optional[int] = None,
        as_dicts: bool = True,
    ) -> Union[List[Message], List[Dict[str, str]]]:
        """
        Get the current conversation context window.

        Args:
            session_id: Session to get context for (defaults to current).
            max_messages: Override default window size.
            as_dicts: If True, returns OpenAI-compatible dicts.

        Returns:
            List of Message objects or dicts.
        """
        sid = session_id or self.session_id
        if as_dicts:
            return self.history.to_openai_format(
                sid,
                max_messages=max_messages or self.context_window_size,
                max_tokens=self.context_max_tokens,
            )
        return self.history.get_context_window(
            sid,
            max_messages=max_messages or self.context_window_size,
            max_tokens=self.context_max_tokens,
        )

    # --------------------------------------------------------------------- #
    #  Long-Term Memory                                                       #
    # --------------------------------------------------------------------- #

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Store an important piece of information in long-term memory.
        Also adds to vector store if auto_vectorize_memories is True.

        Args:
            content: The information to remember.
            memory_type: Type of memory (SEMANTIC, EPISODIC, etc.).
            importance: Importance score 0.0-1.0.
            tags: Optional tags for categorization.
            metadata: Additional metadata.

        Returns:
            The created Memory object.
        """
        memory = self.long_term.store(
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
            importance_score=importance,
            tags=tags,
            source_session_id=self.session_id,
            metadata=metadata,
        )

        if self.auto_vectorize_memories:
            self.vectors.add(
                user_id=self.user_id,
                content=content,
                source_type="memory",
                source_id=memory.memory_id,
                metadata={"memory_type": memory_type.value, "tags": tags or []},
            )

        return memory

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """
        Recall relevant long-term memories using keyword matching.

        Args:
            query: What to search for.
            top_k: Maximum memories to return.
            memory_type: Optional filter by type.
            tags: Optional filter by tags.
            min_importance: Minimum importance threshold.

        Returns:
            List of SearchResult objects.
        """
        return self.long_term.retrieve(
            user_id=self.user_id,
            query=query,
            memory_type=memory_type,
            tags=tags,
            top_k=top_k,
            min_importance=min_importance,
        )

    # --------------------------------------------------------------------- #
    #  Vector / Semantic Search                                               #
    # --------------------------------------------------------------------- #

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantic similarity search across all vectorized content.

        Args:
            query: Query text.
            top_k: Number of results.
            source_type: Optional filter ('memory', 'conversation', etc).

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        return self.vectors.search(
            user_id=self.user_id,
            query=query,
            top_k=top_k,
            source_type=source_type,
        )

    # --------------------------------------------------------------------- #
    #  Preferences                                                            #
    # --------------------------------------------------------------------- #

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific user preference value."""
        return self.preferences.get(self.user_id, category, key, default)

    def set_preference(
        self,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.9,
    ) -> UserPreference:
        """Explicitly set a user preference."""
        return self.preferences.assert_preference(
            self.user_id, category, key, value, confidence
        )

    def get_preference_profile(self, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Get the full preference profile for the user."""
        return self.preferences.get_profile(self.user_id, min_confidence)

    def get_personalization_hint(self) -> str:
        """Get a system prompt fragment based on user preferences."""
        return self.preferences.build_system_prompt_hint(self.user_id)

    # --------------------------------------------------------------------- #
    #  Unified Context Building                                               #
    # --------------------------------------------------------------------- #

    def build_context(
        self,
        query: Optional[str] = None,
        include_memories: bool = True,
        include_preferences: bool = True,
        include_conversation: bool = True,
        memory_top_k: int = 3,
        vector_top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a rich context dict for use in LLM prompt construction.

        Returns a structured dict with:
        - conversation: recent messages
        - memories: relevant long-term memories
        - vector_results: semantically similar entries
        - preferences: user preference profile
        - personalization_hint: preference-based system prompt hint

        Args:
            query: Optional query for semantic/memory retrieval.
            include_memories: Include long-term memory results.
            include_preferences: Include user preferences.
            include_conversation: Include conversation history.
            memory_top_k: Max memory results to include.
            vector_top_k: Max vector results to include.
            session_id: Session to pull conversation from.

        Returns:
            Dict with all context components.
        """
        context: Dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": session_id or self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if include_conversation:
            context["conversation"] = self.get_context(
                session_id=session_id, as_dicts=True
            )

        if include_memories and query:
            ltm_results = self.recall(query, top_k=memory_top_k)
            context["memories"] = [r.to_dict() for r in ltm_results]

            vec_results = self.search(query, top_k=vector_top_k)
            context["vector_results"] = [r.to_dict() for r in vec_results]

        if include_preferences:
            context["preferences"] = self.get_preference_profile()
            hint = self.get_personalization_hint()
            if hint:
                context["personalization_hint"] = hint

        return context

    def build_system_prompt(
        self,
        base_prompt: str = "You are a helpful AI assistant.",
        query: Optional[str] = None,
        memory_top_k: int = 3,
    ) -> str:
        """
        Build a personalized system prompt with memory context injected.

        Args:
            base_prompt: Base system prompt to enhance.
            query: Optional query for relevant memory retrieval.
            memory_top_k: Max memories to inject.

        Returns:
            Enhanced system prompt string.
        """
        parts = [base_prompt]

        # Personalization hint
        hint = self.get_personalization_hint()
        if hint:
            parts.append(f"\nUser preferences: {hint}")

        # Relevant memories
        if query:
            memories = self.recall(query, top_k=memory_top_k)
            if memories:
                mem_texts = [f"- {r.entry.content}" for r in memories]
                parts.append("\nRelevant context about this user:\n" + "\n".join(mem_texts))

        return "\n".join(parts)

    # --------------------------------------------------------------------- #
    #  Session Management                                                     #
    # --------------------------------------------------------------------- #

    def new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        self.session_id = session_id or self._generate_session_id()
        return self.session_id

    def get_session_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for the current or specified session."""
        return self.history.get_session_stats(session_id or self.session_id)

    def get_user_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for the current user."""
        return {
            "user_id": self.user_id,
            "long_term_memory": self.long_term.get_stats(self.user_id),
            "vector_memory": self.vectors.get_stats(self.user_id),
            "preferences_count": len(self.preferences.get_all(self.user_id)),
            "session_id": self.session_id,
        }

    # --------------------------------------------------------------------- #
    #  Utilities                                                              #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _generate_session_id() -> str:
        import uuid
        return f"sess_{uuid.uuid4().hex[:12]}"

    def __repr__(self) -> str:
        return (
            f"MemoryManager(user_id={self.user_id!r}, "
            f"session_id={self.session_id!r})"
        )
