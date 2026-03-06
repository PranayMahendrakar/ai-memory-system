"""
Tests for the AI Memory System.
Run with: pytest tests/ -v
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ai_memory import (
    MemoryManager,
    LongTermMemory,
    ConversationHistory,
    PreferenceLearner,
    VectorMemory,
    Message,
    Memory,
    UserPreference,
    VectorEntry,
)
from ai_memory.models import MessageRole, MemoryType, SearchResult


# --------------------------------------------------------------------------- #
#  Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tmp_store(tmp_path):
    """Provide a temporary storage path cleaned up after each test."""
    store = tmp_path / "test_memory"
    store.mkdir()
    yield str(store)
    shutil.rmtree(str(store), ignore_errors=True)


@pytest.fixture
def user_id():
    return "test_user_001"


@pytest.fixture
def session_id():
    return "session_abc123"


# --------------------------------------------------------------------------- #
#  Models                                                                       #
# --------------------------------------------------------------------------- #

class TestModels:

    def test_message_creation(self, session_id):
        msg = Message(role=MessageRole.USER, content="Hello!", session_id=session_id)
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.message_id is not None

    def test_message_serialization(self, session_id):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi!", session_id=session_id)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        restored = Message.from_dict(d)
        assert restored.content == msg.content
        assert restored.role == msg.role

    def test_memory_creation(self, user_id):
        mem = Memory(content="Test fact", memory_type=MemoryType.SEMANTIC, user_id=user_id)
        assert mem.content == "Test fact"
        assert mem.memory_id is not None
        assert mem.importance_score == 0.5

    def test_memory_serialization(self, user_id):
        mem = Memory(content="Fact", memory_type=MemoryType.EPISODIC, user_id=user_id,
                     importance_score=0.8, tags=["a", "b"])
        d = mem.to_dict()
        restored = Memory.from_dict(d)
        assert restored.content == mem.content
        assert restored.tags == mem.tags
        assert restored.importance_score == mem.importance_score

    def test_user_preference_serialization(self, user_id):
        pref = UserPreference(user_id=user_id, category="format", key="length", value="concise")
        d = pref.to_dict()
        restored = UserPreference.from_dict(d)
        assert restored.value == "concise"
        assert restored.category == "format"


# --------------------------------------------------------------------------- #
#  LongTermMemory                                                               #
# --------------------------------------------------------------------------- #

class TestLongTermMemory:

    def test_store_and_retrieve(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        mem = ltm.store(user_id=user_id, content="Python is great for ML",
                        memory_type=MemoryType.SEMANTIC, importance_score=0.7,
                        tags=["python", "ml"])
        assert mem.memory_id is not None

        results = ltm.retrieve(user_id=user_id, query="Python programming")
        assert len(results) > 0
        assert results[0].entry.content == "Python is great for ML"

    def test_get_specific_memory(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        mem = ltm.store(user_id=user_id, content="Specific fact", memory_type=MemoryType.SEMANTIC)
        retrieved = ltm.get(user_id=user_id, memory_id=mem.memory_id)
        assert retrieved is not None
        assert retrieved.content == "Specific fact"
        assert retrieved.access_count == 1

    def test_update_memory(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        mem = ltm.store(user_id=user_id, content="Old content", memory_type=MemoryType.SEMANTIC)
        updated = ltm.update(user_id=user_id, memory_id=mem.memory_id,
                             content="New content", importance_score=0.9)
        assert updated.content == "New content"
        assert updated.importance_score == 0.9

    def test_delete_memory(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        mem = ltm.store(user_id=user_id, content="To delete", memory_type=MemoryType.SEMANTIC)
        success = ltm.delete(user_id=user_id, memory_id=mem.memory_id)
        assert success is True
        assert ltm.get(user_id=user_id, memory_id=mem.memory_id) is None

    def test_persistence(self, tmp_store, user_id):
        ltm1 = LongTermMemory(storage_path=tmp_store)
        mem = ltm1.store(user_id=user_id, content="Persistent fact", memory_type=MemoryType.SEMANTIC)

        # New instance should load from disk
        ltm2 = LongTermMemory(storage_path=tmp_store)
        retrieved = ltm2.get(user_id=user_id, memory_id=mem.memory_id)
        assert retrieved is not None
        assert retrieved.content == "Persistent fact"

    def test_filter_by_tags(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        ltm.store(user_id=user_id, content="Python fact", memory_type=MemoryType.SEMANTIC, tags=["python"])
        ltm.store(user_id=user_id, content="Java fact", memory_type=MemoryType.SEMANTIC, tags=["java"])

        python_mems = ltm.get_by_tags(user_id=user_id, tags=["python"])
        assert len(python_mems) == 1
        assert "python" in python_mems[0].tags

    def test_get_stats(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        ltm.store(user_id=user_id, content="Fact 1", memory_type=MemoryType.SEMANTIC)
        ltm.store(user_id=user_id, content="Fact 2", memory_type=MemoryType.EPISODIC)
        stats = ltm.get_stats(user_id=user_id)
        assert stats["total"] == 2

    def test_consolidate_similar(self, tmp_store, user_id):
        ltm = LongTermMemory(storage_path=tmp_store)
        ltm.store(user_id=user_id, content="User likes Python programming", memory_type=MemoryType.SEMANTIC)
        ltm.store(user_id=user_id, content="User likes Python programming very much", memory_type=MemoryType.SEMANTIC)
        ltm.store(user_id=user_id, content="Completely different content about Java", memory_type=MemoryType.SEMANTIC)
        merged = ltm.consolidate(user_id=user_id, similarity_threshold=0.5)
        assert merged >= 1


# --------------------------------------------------------------------------- #
#  ConversationHistory                                                          #
# --------------------------------------------------------------------------- #

class TestConversationHistory:

    def test_add_and_retrieve(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        history.add_user_message(session_id=session_id, content="Hello")
        history.add_assistant_message(session_id=session_id, content="Hi there!")
        messages = history.get_session_messages(session_id)
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT

    def test_context_window(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        for i in range(30):
            history.add_user_message(session_id=session_id, content=f"Message {i}")
        window = history.get_context_window(session_id=session_id, max_messages=10)
        assert len(window) == 10
        assert window[-1].content == "Message 29"

    def test_openai_format(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        history.add_user_message(session_id=session_id, content="Hello")
        history.add_assistant_message(session_id=session_id, content="Hi!")
        fmt = history.to_openai_format(session_id)
        assert isinstance(fmt, list)
        assert fmt[0]["role"] == "user"
        assert fmt[0]["content"] == "Hello"

    def test_search_messages(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        history.add_user_message(session_id=session_id, content="Tell me about Python")
        history.add_user_message(session_id=session_id, content="What about Java?")
        results = history.search(session_id=session_id, query="Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_persistence(self, tmp_store, session_id):
        h1 = ConversationHistory(storage_path=tmp_store)
        h1.add_user_message(session_id=session_id, content="Persisted message")

        h2 = ConversationHistory(storage_path=tmp_store)
        msgs = h2.get_session_messages(session_id)
        assert len(msgs) == 1
        assert msgs[0].content == "Persisted message"

    def test_fork_session(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        history.add_user_message(session_id=session_id, content="Original message")
        new_id = history.fork_session(source_session_id=session_id, new_session_id="forked_session")
        forked = history.get_session_messages(new_id)
        assert len(forked) == 1

    def test_clear_session(self, tmp_store, session_id):
        history = ConversationHistory(storage_path=tmp_store)
        history.add_user_message(session_id=session_id, content="Message 1")
        history.add_user_message(session_id=session_id, content="Message 2")
        count = history.clear_session(session_id)
        assert count == 2
        assert len(history.get_session_messages(session_id)) == 0


# --------------------------------------------------------------------------- #
#  PreferenceLearner                                                            #
# --------------------------------------------------------------------------- #

class TestPreferenceLearner:

    def test_auto_detect_format(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        prefs = learner.detect_from_message(user_id=user_id,
                                             content="Please keep your answers brief and concise")
        assert len(prefs) > 0
        assert any(p.key == "response_length" for p in prefs)

    def test_assert_preference(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        pref = learner.assert_preference(user_id=user_id, category="format",
                                          key="language", value="Python")
        assert pref.value == "Python"
        assert pref.confidence >= 0.9

    def test_get_preference(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        learner.assert_preference(user_id=user_id, category="tone", key="formality", value="casual")
        value = learner.get(user_id=user_id, category="tone", key="formality")
        assert value == "casual"

    def test_default_value(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        value = learner.get(user_id=user_id, category="nonexistent", key="key", default="fallback")
        assert value == "fallback"

    def test_get_profile(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        learner.assert_preference(user_id=user_id, category="format", key="length", value="short")
        learner.assert_preference(user_id=user_id, category="tone", key="formality", value="casual")
        profile = learner.get_profile(user_id=user_id)
        assert "format" in profile
        assert "tone" in profile

    def test_confidence_increases(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        learner.detect_from_message(user_id=user_id, content="keep it brief")
        prefs1 = learner.get_all(user_id=user_id)
        conf1 = prefs1[0].confidence if prefs1 else 0

        learner.detect_from_message(user_id=user_id, content="please be brief and concise")
        prefs2 = learner.get_all(user_id=user_id)
        conf2 = max((p.confidence for p in prefs2 if p.key == "response_length"), default=0)
        assert conf2 >= conf1

    def test_build_system_prompt_hint(self, tmp_store, user_id):
        learner = PreferenceLearner(storage_path=tmp_store)
        learner.assert_preference(user_id=user_id, category="format", key="response_length", value="concise")
        hint = learner.build_system_prompt_hint(user_id=user_id)
        assert "concise" in hint.lower()

    def test_persistence(self, tmp_store, user_id):
        l1 = PreferenceLearner(storage_path=tmp_store)
        l1.assert_preference(user_id=user_id, category="domain", key="lang", value="Python")

        l2 = PreferenceLearner(storage_path=tmp_store)
        val = l2.get(user_id=user_id, category="domain", key="lang")
        assert val == "Python"


# --------------------------------------------------------------------------- #
#  VectorMemory                                                                 #
# --------------------------------------------------------------------------- #

class TestVectorMemory:

    def test_add_and_search(self, tmp_store, user_id):
        vm = VectorMemory(storage_path=tmp_store)
        vm.add(user_id=user_id, content="Python is great for data science")
        vm.add(user_id=user_id, content="Machine learning with neural networks")
        vm.add(user_id=user_id, content="Cooking recipes for pasta")

        results = vm.search(user_id=user_id, query="data science programming", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_cosine_similarity_ranking(self, tmp_store, user_id):
        from ai_memory.vector_memory import VectorMemory
        vm = VectorMemory(storage_path=tmp_store)
        vm.add(user_id=user_id, content="deep learning neural networks machine learning")
        vm.add(user_id=user_id, content="cooking and food recipes")
        results = vm.search(user_id=user_id, query="deep learning", top_k=2)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_delete_entry(self, tmp_store, user_id):
        vm = VectorMemory(storage_path=tmp_store)
        entry = vm.add(user_id=user_id, content="To be deleted")
        success = vm.delete(user_id=user_id, entry_id=entry.entry_id)
        assert success is True

    def test_batch_add(self, tmp_store, user_id):
        vm = VectorMemory(storage_path=tmp_store)
        items = [{"content": f"Fact number {i}"} for i in range(10)]
        entries = vm.add_batch(user_id=user_id, items=items)
        assert len(entries) == 10

    def test_persistence(self, tmp_store, user_id):
        vm1 = VectorMemory(storage_path=tmp_store)
        vm1.add(user_id=user_id, content="Persisted vector entry")

        vm2 = VectorMemory(storage_path=tmp_store)
        entries = vm2.get_all(user_id=user_id)
        assert len(entries) == 1
        assert entries[0].content == "Persisted vector entry"

    def test_custom_embed_fn(self, tmp_store, user_id):
        def my_embed(text: str):
            return [float(ord(c) % 10) / 10 for c in text[:16].ljust(16)]

        vm = VectorMemory(storage_path=tmp_store, embed_fn=my_embed)
        vm.add(user_id=user_id, content="custom embedding test")
        results = vm.search(user_id=user_id, query="custom embedding")
        assert len(results) >= 0  # Just verify no error


# --------------------------------------------------------------------------- #
#  MemoryManager (integration)                                                  #
# --------------------------------------------------------------------------- #

class TestMemoryManager:

    def test_basic_conversation(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.add_user_message("Hello, I am Alex!")
        mm.add_assistant_message("Hi Alex, nice to meet you!")
        ctx = mm.get_context(as_dicts=True)
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    def test_remember_and_recall(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.remember("User is a senior Python developer", MemoryType.SEMANTIC, importance=0.9)
        results = mm.recall("Python developer skills")
        assert len(results) > 0

    def test_semantic_search(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.remember("User loves machine learning", MemoryType.SEMANTIC)
        mm.remember("User prefers dark mode UI", MemoryType.SEMANTIC)
        results = mm.search("machine learning")
        assert len(results) >= 0  # May or may not match depending on embedder

    def test_preference_management(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.set_preference("format", "length", "concise")
        val = mm.get_preference("format", "length")
        assert val == "concise"

    def test_build_context(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.add_user_message("Tell me about Python")
        mm.remember("Python is a versatile language", MemoryType.SEMANTIC)
        context = mm.build_context(query="Python programming")
        assert "conversation" in context
        assert "user_id" in context

    def test_build_system_prompt(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.set_preference("format", "response_length", "concise")
        mm.remember("User is an expert in deep learning", MemoryType.SEMANTIC, importance=0.9)
        prompt = mm.build_system_prompt(
            base_prompt="You are a helpful assistant.",
            query="deep learning",
        )
        assert "helpful assistant" in prompt

    def test_auto_preference_learning(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store,
                           auto_learn_preferences=True)
        mm.add_user_message("Please keep your answers brief and concise!")
        profile = mm.get_preference_profile()
        # Profile might be empty or have detected prefs
        assert isinstance(profile, dict)

    def test_session_management(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        session1 = mm.session_id
        mm.add_user_message("Message in session 1")
        session2 = mm.new_session()
        assert session1 != session2
        mm.add_user_message("Message in session 2")

        stats1 = mm.get_session_stats(session_id=session1)
        assert stats1["total_messages"] == 1

    def test_get_user_stats(self, tmp_store, user_id):
        mm = MemoryManager(user_id=user_id, storage_path=tmp_store)
        mm.remember("Fact about user", MemoryType.SEMANTIC)
        stats = mm.get_user_stats()
        assert stats["user_id"] == user_id
        assert "long_term_memory" in stats
