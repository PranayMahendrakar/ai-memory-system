"""
Conversation History module for the AI Memory System.

Manages per-session and cross-session conversation tracking with
windowing, summarization hooks, and token budget management.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .models import Message, MessageRole


class ConversationHistory:
    """
    Stores and retrieves conversation messages per session.

    Supports windowed context for LLM prompt construction,
    session persistence, and optional summarization callbacks.

    Example:
        history = ConversationHistory(storage_path="./memory_store")
        history.add_message(session_id="sess_1", role=MessageRole.USER,
                            content="Hello!", user_id="user_123")
        history.add_message(session_id="sess_1", role=MessageRole.ASSISTANT,
                            content="Hi there! How can I help?")
        ctx = history.get_context_window(session_id="sess_1", max_messages=10)
    """

    def __init__(
        self,
        storage_path: str = "./ai_memory_store",
        max_session_messages: int = 1000,
        default_window_size: int = 20,
        summarize_callback: Optional[Callable[[List[Message]], str]] = None,
    ):
        """
        Initialize ConversationHistory.

        Args:
            storage_path: Base directory for persisting conversations.
            max_session_messages: Hard cap per session before rolling.
            default_window_size: Default number of messages in context window.
            summarize_callback: Optional async/sync function to summarize
                                 messages when the window is compressed.
        """
        self.storage_path = Path(storage_path) / "conversations"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_session_messages = max_session_messages
        self.default_window_size = default_window_size
        self.summarize_callback = summarize_callback
        # In-memory cache: {session_id: deque[Message]}
        self._cache: Dict[str, deque] = {}

    # --------------------------------------------------------------------- #
    #  Adding Messages                                                        #
    # --------------------------------------------------------------------- #

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        user_id: Optional[str] = None,
        tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to the conversation session.

        Args:
            session_id: Unique identifier for the conversation session.
            role: MessageRole (USER, ASSISTANT, SYSTEM, TOOL).
            content: The message text.
            user_id: Optional user ID for multi-user contexts.
            tokens: Token count for this message.
            metadata: Additional metadata dict.

        Returns:
            The created Message object.
        """
        message = Message(
            role=role,
            content=content,
            session_id=session_id,
            tokens=tokens,
            metadata={**(metadata or {}), **({"user_id": user_id} if user_id else {})},
        )
        history = self._get_session(session_id)
        history.append(message)

        # Roll over if too long
        while len(history) > self.max_session_messages:
            history.popleft()

        self._cache[session_id] = history
        self._persist_session(session_id, history)
        return message

    def add_system_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a system message (instructions/persona) to the session."""
        return self.add_message(
            session_id=session_id,
            role=MessageRole.SYSTEM,
            content=content,
            metadata=metadata,
        )

    def add_user_message(self, session_id: str, content: str, user_id: Optional[str] = None, **kwargs) -> Message:
        """Convenience: add a user message."""
        return self.add_message(session_id, MessageRole.USER, content, user_id=user_id, **kwargs)

    def add_assistant_message(self, session_id: str, content: str, **kwargs) -> Message:
        """Convenience: add an assistant message."""
        return self.add_message(session_id, MessageRole.ASSISTANT, content, **kwargs)

    # --------------------------------------------------------------------- #
    #  Retrieval                                                              #
    # --------------------------------------------------------------------- #

    def get_context_window(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Message]:
        """
        Get the most recent messages for LLM context.

        Retrieves up to max_messages from the end of the history,
        optionally bounded by a token budget.

        Args:
            session_id: Session to retrieve context for.
            max_messages: Max number of messages (defaults to default_window_size).
            max_tokens: Optional token budget cap.
            include_system: Whether to include system messages.

        Returns:
            Ordered list of Message objects (oldest first).
        """
        max_messages = max_messages or self.default_window_size
        history = list(self._get_session(session_id))

        if not include_system:
            history = [m for m in history if m.role != MessageRole.SYSTEM]

        # Take last N messages
        window = history[-max_messages:]

        # Apply token budget if specified
        if max_tokens is not None:
            selected = []
            used_tokens = 0
            for msg in reversed(window):
                msg_tokens = msg.tokens or self._estimate_tokens(msg.content)
                if used_tokens + msg_tokens > max_tokens:
                    break
                selected.insert(0, msg)
                used_tokens += msg_tokens
            window = selected

        return window

    def get_session_messages(self, session_id: str) -> List[Message]:
        """Get all messages in a session."""
        return list(self._get_session(session_id))

    def get_last_n(self, session_id: str, n: int) -> List[Message]:
        """Get the last N messages from a session."""
        history = list(self._get_session(session_id))
        return history[-n:]

    def get_messages_by_role(self, session_id: str, role: MessageRole) -> List[Message]:
        """Get all messages from a session with a specific role."""
        return [m for m in self._get_session(session_id) if m.role == role]

    def get_message_by_id(self, session_id: str, message_id: str) -> Optional[Message]:
        """Get a specific message by ID."""
        for msg in self._get_session(session_id):
            if msg.message_id == message_id:
                return msg
        return None

    def search(self, session_id: str, query: str, case_sensitive: bool = False) -> List[Message]:
        """Search messages in a session by content substring."""
        q = query if case_sensitive else query.lower()
        return [
            m for m in self._get_session(session_id)
            if q in (m.content if case_sensitive else m.content.lower())
        ]

    def iter_sessions(self) -> Iterator[str]:
        """Iterate over all stored session IDs."""
        for f in self.storage_path.glob("*.json"):
            yield f.stem

    # --------------------------------------------------------------------- #
    #  Session Management                                                     #
    # --------------------------------------------------------------------- #

    def clear_session(self, session_id: str) -> int:
        """Clear all messages in a session. Returns number of messages deleted."""
        history = self._get_session(session_id)
        count = len(history)
        self._cache[session_id] = deque()
        self._persist_session(session_id, deque())
        return count

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its persisted data."""
        if session_id in self._cache:
            del self._cache[session_id]
        file_path = self._get_session_file(session_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def fork_session(self, source_session_id: str, new_session_id: str) -> str:
        """
        Fork a session — copy all messages into a new session ID.
        Useful for branching conversations.
        """
        messages = self.get_session_messages(source_session_id)
        new_history = deque(messages)
        self._cache[new_session_id] = new_history
        self._persist_session(new_session_id, new_history)
        return new_session_id

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation session."""
        history = list(self._get_session(session_id))
        if not history:
            return {"session_id": session_id, "total_messages": 0}

        role_counts: Dict[str, int] = {}
        total_tokens = 0
        for msg in history:
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            role_counts[role] = role_counts.get(role, 0) + 1
            total_tokens += msg.tokens or self._estimate_tokens(msg.content)

        return {
            "session_id": session_id,
            "total_messages": len(history),
            "by_role": role_counts,
            "total_tokens_estimate": total_tokens,
            "first_message_at": history[0].timestamp.isoformat(),
            "last_message_at": history[-1].timestamp.isoformat(),
            "duration_seconds": (history[-1].timestamp - history[0].timestamp).total_seconds(),
        }

    # --------------------------------------------------------------------- #
    #  LLM-ready formatting                                                   #
    # --------------------------------------------------------------------- #

    def to_openai_format(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Format context window as OpenAI-compatible message dicts.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        window = self.get_context_window(session_id, max_messages, max_tokens)
        return [
            {"role": m.role.value if isinstance(m.role, MessageRole) else m.role, "content": m.content}
            for m in window
        ]

    def to_plain_text(self, session_id: str, separator: str = "\n") -> str:
        """Format entire session as plain text (role: content)."""
        lines = []
        for msg in self._get_session(session_id):
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            lines.append(f"{role.upper()}: {msg.content}")
        return separator.join(lines)

    # --------------------------------------------------------------------- #
    #  Utilities                                                              #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def _get_session_file(self, session_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() else "_" for c in session_id)
        return self.storage_path / f"{safe_id}.json"

    def _get_session(self, session_id: str) -> deque:
        if session_id in self._cache:
            return self._cache[session_id]

        file_path = self._get_session_file(session_id)
        history: deque = deque()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    history.append(Message.from_dict(item))
            except (json.JSONDecodeError, KeyError):
                pass

        self._cache[session_id] = history
        return history

    def _persist_session(self, session_id: str, history: deque) -> None:
        file_path = self._get_session_file(session_id)
        data = [msg.to_dict() for msg in history]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
