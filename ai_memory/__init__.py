"""
ai_memory - A persistent AI Memory System for LLMs.

A reusable Python library providing long-term memory, conversation history,
user preference learning, and vector memory for AI applications.
"""

__version__ = "0.1.0"
__author__ = "PranayMahendrakar"
__license__ = "MIT"

from ai_memory.memory_manager import MemoryManager
from ai_memory.long_term_memory import LongTermMemory
from ai_memory.conversation_history import ConversationHistory
from ai_memory.preference_learner import PreferenceLearner
from ai_memory.vector_memory import VectorMemory
from ai_memory.models import Message, Memory, UserPreference, VectorEntry

__all__ = [
      "MemoryManager",
      "LongTermMemory",
      "ConversationHistory",
      "PreferenceLearner",
      "VectorMemory",
      "Message",
      "Memory",
      "UserPreference",
      "VectorEntry",
]
