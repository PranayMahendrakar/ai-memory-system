"""
User Preference Learner module for the AI Memory System.

Automatically detects and stores user preferences from conversation patterns,
explicit feedback, and behavioral signals. Confidence-weighted with
Bayesian-style updating.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import UserPreference


# --------------------------------------------------------------------------- #
#  Built-in pattern detectors                                                   #
# --------------------------------------------------------------------------- #

PREFERENCE_PATTERNS: List[Dict] = [
    # Language / format preferences
    {"category": "format", "key": "response_length",
     "patterns": [r"\b(short|brief|concise|tl;?dr)\b"], "value": "concise"},
    {"category": "format", "key": "response_length",
     "patterns": [r"\b(detailed|comprehensive|thorough|in.depth)\b"], "value": "detailed"},
    {"category": "format", "key": "use_bullet_points",
     "patterns": [r"\b(bullet.?points?|list.?format|use.?lists?)\b"], "value": True},
    {"category": "format", "key": "use_code_examples",
     "patterns": [r"\b(show.?code|code.?example|with.?code|show.?me.?how)\b"], "value": True},
    # Tone preferences
    {"category": "tone", "key": "formality",
     "patterns": [r"\b(formal|professional|business)\b"], "value": "formal"},
    {"category": "tone", "key": "formality",
     "patterns": [r"\b(casual|informal|friendly|conversational)\b"], "value": "casual"},
    {"category": "tone", "key": "humor",
     "patterns": [r"\b(funny|humorous|jokes?|witty|playful)\b"], "value": True},
    # Domain preferences
    {"category": "domain", "key": "programming_language",
     "patterns": [r"\b(python|javascript|typescript|java|go|rust|c\+\+|ruby)\b"], "value": None, "capture_group": 1},
    {"category": "domain", "key": "framework",
     "patterns": [r"\b(react|vue|angular|django|flask|fastapi|express|spring)\b"], "value": None, "capture_group": 1},
    # Explicit preferences
    {"category": "explicit", "key": "likes",
     "patterns": [r"i.?(?:really.?)?(?:like|love|prefer|enjoy).{0,30}"], "value": None, "capture_group": 0},
    {"category": "explicit", "key": "dislikes",
     "patterns": [r"i.?(?:don.?t.?like|hate|dislike|avoid|prefer.?not).{0,30}"], "value": None, "capture_group": 0},
]


class PreferenceLearner:
    """
    Learns and manages user preferences from conversation interactions.

    Supports automatic detection via pattern matching, manual assertion,
    and confidence-weighted Bayesian updating over repeated observations.

    Example:
        learner = PreferenceLearner(storage_path="./memory_store")

        # Auto-detect from message
        prefs = learner.detect_from_message(user_id="u1",
                    content="Please keep answers brief and use bullet points")

        # Manually assert a preference
        learner.assert_preference(user_id="u1", category="format",
                                  key="language", value="Python")

        # Get preferences for a category
        fmt_prefs = learner.get_category(user_id="u1", category="format")
    """

    def __init__(
        self,
        storage_path: str = "./ai_memory_store",
        min_confidence_threshold: float = 0.3,
        confidence_increment: float = 0.1,
        max_preferences_per_user: int = 500,
    ):
        """
        Initialize PreferenceLearner.

        Args:
            storage_path: Base directory for persisting preferences.
            min_confidence_threshold: Minimum confidence to store preference.
            confidence_increment: How much confidence rises per observation.
            max_preferences_per_user: Hard cap on stored preferences.
        """
        self.storage_path = Path(storage_path) / "preferences"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence_threshold
        self.confidence_increment = confidence_increment
        self.max_preferences = max_preferences_per_user
        self._cache: Dict[str, Dict[str, UserPreference]] = {}

    # --------------------------------------------------------------------- #
    #  Detection                                                              #
    # --------------------------------------------------------------------- #

    def detect_from_message(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> List[UserPreference]:
        """
        Auto-detect preferences from a message using built-in patterns.

        Args:
            user_id: User to attribute preferences to.
            content: Message content to analyze.
            session_id: Optional source session ID for traceability.

        Returns:
            List of UserPreference objects that were detected/updated.
        """
        detected = []
        content_lower = content.lower()

        for pattern_def in PREFERENCE_PATTERNS:
            for pattern in pattern_def["patterns"]:
                match = re.search(pattern, content_lower, re.IGNORECASE)
                if match:
                    value = pattern_def["value"]
                    # If capture_group, extract matched text as value
                    if pattern_def.get("capture_group") is not None and value is None:
                        try:
                            value = match.group(pattern_def["capture_group"]).strip()
                        except IndexError:
                            value = match.group(0).strip()

                    pref = self._upsert_preference(
                        user_id=user_id,
                        category=pattern_def["category"],
                        key=pattern_def["key"],
                        value=value,
                        confidence_delta=self.confidence_increment,
                        source_session_id=session_id,
                    )
                    if pref:
                        detected.append(pref)
                    break  # Only match once per pattern_def

        return detected

    def detect_from_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> List[UserPreference]:
        """
        Detect preferences from a list of conversation messages.

        Args:
            user_id: User to attribute preferences to.
            messages: List of dicts with 'role' and 'content' keys.
            session_id: Optional session ID for traceability.

        Returns:
            All preferences detected across the conversation.
        """
        all_detected = []
        # Only analyze user messages
        for msg in messages:
            if msg.get("role") in ("user", "USER"):
                prefs = self.detect_from_message(user_id, msg["content"], session_id)
                all_detected.extend(prefs)
        return all_detected

    # --------------------------------------------------------------------- #
    #  Manual Management                                                       #
    # --------------------------------------------------------------------- #

    def assert_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.9,
        metadata: Optional[Dict] = None,
    ) -> UserPreference:
        """
        Explicitly set a user preference with high confidence.

        Use this for explicit user-provided preferences
        (e.g., from a settings page or direct statement).
        """
        return self._upsert_preference(
            user_id=user_id,
            category=category,
            key=key,
            value=value,
            confidence_override=confidence,
            metadata=metadata,
        )

    def retract_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Optional[Any] = None,
    ) -> int:
        """
        Remove a preference or reduce its confidence.

        If value is specified, only removes preference with that value.
        Returns number of preferences removed.
        """
        prefs = self._load_user_preferences(user_id)
        to_delete = []
        for pref_id, pref in prefs.items():
            if pref.category == category and pref.key == key:
                if value is None or pref.value == value:
                    to_delete.append(pref_id)

        for pref_id in to_delete:
            del prefs[pref_id]

        self._cache[user_id] = prefs
        self._persist_user_preferences(user_id, prefs)
        return len(to_delete)

    # --------------------------------------------------------------------- #
    #  Retrieval                                                              #
    # --------------------------------------------------------------------- #

    def get(
        self,
        user_id: str,
        category: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get the highest-confidence value for a specific preference key.

        Returns:
            The preference value, or default if not found.
        """
        candidates = self.get_by_key(user_id, category, key)
        if not candidates:
            return default
        best = max(candidates, key=lambda p: p.confidence)
        return best.value

    def get_by_key(self, user_id: str, category: str, key: str) -> List[UserPreference]:
        """Get all preferences for a category/key pair."""
        prefs = self._load_user_preferences(user_id)
        return [p for p in prefs.values() if p.category == category and p.key == key]

    def get_category(self, user_id: str, category: str) -> Dict[str, Any]:
        """
        Get all preferences in a category as a flat key->value dict.
        Returns highest-confidence value for each key.
        """
        prefs = self._load_user_preferences(user_id)
        category_prefs: Dict[str, List[UserPreference]] = {}
        for pref in prefs.values():
            if pref.category == category:
                category_prefs.setdefault(pref.key, []).append(pref)

        result = {}
        for key, candidates in category_prefs.items():
            best = max(candidates, key=lambda p: p.confidence)
            result[key] = best.value
        return result

    def get_all(self, user_id: str, min_confidence: float = 0.0) -> List[UserPreference]:
        """Get all preferences for a user above minimum confidence."""
        prefs = list(self._load_user_preferences(user_id).values())
        if min_confidence > 0:
            prefs = [p for p in prefs if p.confidence >= min_confidence]
        return sorted(prefs, key=lambda p: p.confidence, reverse=True)

    def get_profile(self, user_id: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Get a structured preference profile for a user.

        Returns:
            Dict keyed by category, containing key->value pairs.
        """
        prefs = self.get_all(user_id, min_confidence=min_confidence)
        profile: Dict[str, Dict[str, Any]] = {}
        for pref in prefs:
            profile.setdefault(pref.category, {})[pref.key] = pref.value
        return profile

    def build_system_prompt_hint(self, user_id: str) -> str:
        """
        Build a short system prompt fragment based on known user preferences.
        Useful for injecting personalization into LLM system messages.

        Returns:
            A human-readable string describing the user's preferences.
        """
        profile = self.get_profile(user_id, min_confidence=0.6)
        if not profile:
            return ""

        hints = []
        fmt = profile.get("format", {})
        tone = profile.get("tone", {})
        domain = profile.get("domain", {})

        if fmt.get("response_length"):
            hints.append(f"Keep responses {fmt['response_length']}.")
        if fmt.get("use_bullet_points"):
            hints.append("Use bullet points when listing items.")
        if fmt.get("use_code_examples"):
            hints.append("Include code examples when helpful.")
        if tone.get("formality"):
            hints.append(f"Use a {tone['formality']} tone.")
        if domain.get("programming_language"):
            hints.append(f"Prefer {domain['programming_language']} for code examples.")

        return " ".join(hints)

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                       #
    # --------------------------------------------------------------------- #

    def _upsert_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        confidence_delta: float = 0.0,
        confidence_override: Optional[float] = None,
        source_session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[UserPreference]:
        """Insert or update a preference, returning the resulting object."""
        prefs = self._load_user_preferences(user_id)

        # Find existing preference with same category/key/value
        existing = None
        for pref in prefs.values():
            if pref.category == category and pref.key == key and pref.value == value:
                existing = pref
                break

        if existing:
            existing.observation_count += 1
            if confidence_override is not None:
                existing.confidence = confidence_override
            else:
                existing.confidence = min(1.0, existing.confidence + confidence_delta)
            existing.updated_at = datetime.utcnow()
            if metadata:
                existing.metadata.update(metadata)
            result = existing
        else:
            initial_confidence = confidence_override if confidence_override is not None else (
                self.min_confidence + confidence_delta
            )
            if initial_confidence < self.min_confidence:
                return None

            result = UserPreference(
                user_id=user_id,
                category=category,
                key=key,
                value=value,
                confidence=initial_confidence,
                metadata={**(metadata or {}), **({"source_session_id": source_session_id} if source_session_id else {})},
            )
            prefs[result.preference_id] = result

        self._cache[user_id] = prefs
        self._persist_user_preferences(user_id, prefs)
        return result

    def _get_user_file(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return self.storage_path / f"{safe_id}.json"

    def _load_user_preferences(self, user_id: str) -> Dict[str, UserPreference]:
        if user_id in self._cache:
            return self._cache[user_id]

        file_path = self._get_user_file(user_id)
        prefs: Dict[str, UserPreference] = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    pref = UserPreference.from_dict(item)
                    prefs[pref.preference_id] = pref
            except (json.JSONDecodeError, KeyError):
                pass

        self._cache[user_id] = prefs
        return prefs

    def _persist_user_preferences(self, user_id: str, prefs: Dict[str, UserPreference]) -> None:
        file_path = self._get_user_file(user_id)
        data = [p.to_dict() for p in prefs.values()]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
