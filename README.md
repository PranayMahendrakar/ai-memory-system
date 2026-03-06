# 🧠 ai-memory-system

> **Persistent AI Memory System for LLMs** — a reusable Python library providing long-term memory, conversation history, user preference learning, and vector memory for AI applications.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Long-Term Memory** | Persistent facts with importance scoring, temporal decay, tag filtering, and consolidation |
| 💬 **Conversation History** | Per-session message tracking, context windowing, token budgets, and OpenAI-compatible formatting |
| 🎯 **User Preference Learning** | Auto-detects preferences from conversations via pattern matching + Bayesian confidence updates |
| 🔍 **Vector Memory** | Semantic similarity search with pluggable embedding backends (TF-IDF, Hash, OpenAI, Sentence Transformers) |
| 🎛️ **MemoryManager** | Unified orchestrator — one import, one object, all memory features wired together |
| 💾 **Zero Dependencies** | Pure Python stdlib. No mandatory external packages. Extras available for power users. |
| 🔌 **Pluggable** | Swap embedding functions, storage backends, or extend any module independently |

---

## 🚀 Quick Start

### Installation

```bash
# From PyPI (when published)
pip install ai-memory-system

# From source
git clone https://github.com/PranayMahendrakar/ai-memory-system.git
cd ai-memory-system
pip install -e .

# With optional extras (OpenAI embeddings, Sentence Transformers, etc.)
pip install -e ".[all]"
```

### 30-Second Example

```python
from ai_memory import MemoryManager, MemoryType

# Initialize for a user
mm = MemoryManager(user_id="user_123", storage_path="./memory_store")

# Record a conversation
mm.add_user_message("Hi! My name is Alex, I'm a senior Python developer.")
mm.add_assistant_message("Nice to meet you, Alex! How can I help you today?")

# Store an important fact with high importance
mm.remember(
    "Alex is a senior Python developer who loves ML",
    memory_type=MemoryType.SEMANTIC,
    importance=0.9,
    tags=["professional", "python", "ml"]
)

# Get context for next LLM call
messages = mm.get_context()  # OpenAI-compatible list of dicts

# Recall relevant memories
results = mm.recall("Python developer skills", top_k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.entry.content}")

# Build a personalized system prompt
prompt = mm.build_system_prompt(
    base_prompt="You are a helpful coding assistant.",
    query="Python best practices"
)
print(prompt)
```

---

## 📦 Architecture

```
ai_memory/
├── __init__.py              # Public API
├── models.py                # Data models (Message, Memory, UserPreference, VectorEntry)
├── memory_manager.py        # 🎛️  Unified orchestrator (main entry point)
├── long_term_memory.py      # 🧠 Persistent long-term memories
├── conversation_history.py  # 💬 Session-based conversation tracking
├── preference_learner.py    # 🎯 User preference detection & storage
└── vector_memory.py         # 🔍 Embedding-based semantic search
```

### Module Overview

**`MemoryManager`** — The recommended entry point. Wires all subsystems together and exposes a clean high-level API. One object to rule them all.

**`LongTermMemory`** — Stores facts, events, and knowledge with importance scores, temporal decay (memories fade over time), and consolidation (merges similar memories). Supports keyword-based retrieval with Jaccard scoring.

**`ConversationHistory`** — Manages per-session message history with context windowing, token budget tracking, session forking, and OpenAI/Anthropic-compatible message formatting.

**`PreferenceLearner`** — Automatically detects user preferences from conversation patterns (format, tone, domain). Uses Bayesian-style confidence updating — the more a preference is observed, the higher its confidence. Provides `build_system_prompt_hint()` for injecting personalization.

**`VectorMemory`** — Embedding-based semantic similarity search. Ships with two zero-dependency built-in embedders (HashEmbedder and TFIDFEmbedder). Plug in OpenAI, Sentence Transformers, or any custom function.

---

## 📖 Detailed Usage

### Long-Term Memory

```python
from ai_memory import LongTermMemory
from ai_memory.models import MemoryType

ltm = LongTermMemory(storage_path="./store")

# Store memories
mem = ltm.store(
    user_id="u1",
    content="User prefers detailed technical explanations",
    memory_type=MemoryType.SEMANTIC,
    importance_score=0.8,
    tags=["communication", "style"]
)

# Retrieve with keyword search
results = ltm.retrieve(user_id="u1", query="technical style", top_k=5)

# Filter by type and tags
episodic = ltm.retrieve(user_id="u1", memory_type=MemoryType.EPISODIC)
tagged = ltm.get_by_tags(user_id="u1", tags=["communication"])

# Update importance
ltm.update(user_id="u1", memory_id=mem.memory_id, importance_score=0.95)

# Consolidate duplicates
merged_count = ltm.consolidate(user_id="u1", similarity_threshold=0.8)

# Statistics
stats = ltm.get_stats(user_id="u1")
print(stats)  # {"total": 5, "by_type": {...}, "avg_importance": 0.73}
```

### Conversation History

```python
from ai_memory import ConversationHistory
from ai_memory.models import MessageRole

history = ConversationHistory(storage_path="./store")

# Add messages
history.add_system_message("sess_1", "You are a Python expert.")
history.add_user_message("sess_1", "What is a decorator?", user_id="u1")
history.add_assistant_message("sess_1", "A decorator is...")

# Get context window (last N messages, respecting token budget)
window = history.get_context_window("sess_1", max_messages=10, max_tokens=4000)

# Get OpenAI-compatible format
messages = history.to_openai_format("sess_1")
# [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

# Search within a session
matches = history.search("sess_1", query="decorator")

# Fork a session (for branching conversations)
new_id = history.fork_session("sess_1", "sess_1_branch")

# Session stats
stats = history.get_session_stats("sess_1")
print(stats["total_tokens_estimate"])
```

### User Preference Learning

```python
from ai_memory import PreferenceLearner

learner = PreferenceLearner(storage_path="./store")

# Auto-detect from messages (pattern matching)
prefs = learner.detect_from_message(
    user_id="u1",
    content="Can you keep answers short and use bullet points? I prefer Python."
)
# Detected: format/response_length=concise, format/use_bullet_points=True,
#           domain/programming_language=python

# Auto-detect from full conversation
prefs = learner.detect_from_conversation(user_id="u1", messages=[
    {"role": "user", "content": "Please be brief"},
    {"role": "user", "content": "I prefer formal tone"},
])

# Explicitly assert a preference (high confidence)
learner.assert_preference(user_id="u1", category="format", key="language", value="Python")

# Query preferences
lang = learner.get(user_id="u1", category="domain", key="programming_language")
fmt_prefs = learner.get_category(user_id="u1", category="format")
profile = learner.get_profile(user_id="u1", min_confidence=0.6)

# Generate system prompt personalization hint
hint = learner.build_system_prompt_hint(user_id="u1")
# "Keep responses concise. Use bullet points when listing items. Prefer Python for code examples."
```

### Vector Memory

```python
from ai_memory import VectorMemory

# Default (HashEmbedder — zero dependencies)
vm = VectorMemory(storage_path="./store")

# Add content
vm.add(user_id="u1", content="Python list comprehensions are fast and readable")
vm.add(user_id="u1", content="Neural networks learn from data using backpropagation")
vm.add(user_id="u1", content="Cooking pasta requires boiling water")

# Semantic search
results = vm.search(user_id="u1", query="programming best practices", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.entry.content}")

# Batch add (efficient for bulk loading)
vm.add_batch(user_id="u1", items=[
    {"content": "Fact 1", "source_type": "memory"},
    {"content": "Fact 2", "metadata": {"topic": "python"}},
])

# Use OpenAI embeddings
import openai
def openai_embed(text: str):
    response = openai.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

vm_openai = VectorMemory(storage_path="./store", embed_fn=openai_embed)

# Use Sentence Transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vm_st = VectorMemory(storage_path="./store", embed_fn=lambda t: model.encode(t).tolist())
```

### MemoryManager (Unified API)

```python
from ai_memory import MemoryManager, MemoryType

mm = MemoryManager(
    user_id="user_123",
    storage_path="./memory_store",
    auto_learn_preferences=True,   # Auto-detect prefs from messages
    auto_vectorize_memories=True,  # Auto-add memories to vector store
    context_window_size=20,        # Default context window
)

# Full conversation with auto-everything
mm.add_user_message("I prefer short Python code examples")
mm.add_assistant_message("Got it! Here's a short example...")
mm.remember("User is a Python developer", MemoryType.SEMANTIC, importance=0.9)

# Build full context for LLM
context = mm.build_context(
    query="Python programming",
    include_memories=True,
    include_preferences=True,
    include_conversation=True,
    memory_top_k=3,
    vector_top_k=3,
)
# {
#   "conversation": [...],
#   "memories": [...],
#   "vector_results": [...],
#   "preferences": {...},
#   "personalization_hint": "Keep responses concise. Prefer Python..."
# }

# Build personalized system prompt
system_prompt = mm.build_system_prompt(
    base_prompt="You are a helpful assistant.",
    query="Python best practices",
    memory_top_k=3,
)

# Multi-session support
mm.new_session()  # Start fresh session
mm.add_user_message("New session message")
stats = mm.get_user_stats()
```

---

## 🔧 Configuration

### Storage

All memory is persisted to a directory structure under `storage_path`:

```
memory_store/
├── long_term/      # {user_id}.json — long-term memories
├── conversations/  # {session_id}.json — conversation histories
├── preferences/    # {user_id}.json — learned preferences
└── vectors/        # {user_id}.json — vector embeddings
```

### Custom Embedding Function

```python
# Bring your own embedding function
def my_embed(text: str) -> list[float]:
    # Call your embedding API or local model
    return [...]

mm = MemoryManager(user_id="u1", embed_fn=my_embed)
```

### Memory Decay

Long-term memories fade in importance over time using exponential decay:

```python
ltm = LongTermMemory(
    storage_path="./store",
    decay_factor=0.01,  # 1% importance decay per day
)
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ai_memory --cov-report=html

# Run specific test class
pytest tests/test_memory_system.py::TestMemoryManager -v
```

---

## 🛣️ Roadmap

- [ ] Async support (asyncio-native API)
- [ ] SQLite storage backend
- [ ] Redis storage backend
- [ ] FAISS/Chroma vector store integration
- [ ] Memory importance auto-scoring via LLM
- [ ] LangChain / LlamaIndex integration adapters
- [ ] Memory export/import (JSON, YAML)
- [ ] Multi-user isolation & tenancy controls
- [ ] REST API server mode
- [ ] PyPI package publication

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/PranayMahendrakar/ai-memory-system.git
cd ai-memory-system
pip install -e ".[dev]"
# Make changes, write tests
pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with ❤️ by [PranayMahendrakar](https://github.com/PranayMahendrakar). Inspired by the need for persistent, personalized AI experiences without vendor lock-in.
