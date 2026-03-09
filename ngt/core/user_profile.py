"""
UserProfile — структурированная память о пользователе.

Хранит слоты (age, city, diet, allergies, ...) с историей изменений,
confidence и валидацией. Интегрируется с NGTMemoryLLMWrapper для
приоритетной инжекции в system prompt.

Использование:
    profile = UserProfile()
    profile.extract_and_update("мне 30 лет", confidence=1.0)
    profile.extract_and_update("живу в Берлине", confidence=1.0)
    print(profile.as_prompt_block())
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Slot definitions ─────────────────────────────────────────────────

@dataclass
class SlotChange:
    """One historical change to a slot."""
    old_value: Any
    new_value: Any
    timestamp: float
    reason: str  # "user_explicit", "fragment_merged", "correction", "conflict_resolved"


@dataclass
class ProfileSlot:
    """A single profile fact with metadata."""
    value: Any = None
    confidence: float = 0.0
    updated_at: float = 0.0
    source: str = ""  # "user_explicit", "fragment_merged", "inferred"
    history: List[SlotChange] = field(default_factory=list)

    @property
    def is_set(self) -> bool:
        return self.value is not None

    def update(self, new_value: Any, confidence: float, source: str, reason: str = "update") -> bool:
        """Update slot value. Returns True if value actually changed."""
        if self.value == new_value:
            # Same value — just bump confidence if higher
            if confidence > self.confidence:
                self.confidence = confidence
            return False

        now = time.time()
        if self.is_set:
            self.history.append(SlotChange(
                old_value=self.value,
                new_value=new_value,
                timestamp=now,
                reason=reason,
            ))
        self.value = new_value
        self.confidence = confidence
        self.updated_at = now
        self.source = source
        return True


# ── Slot validators ──────────────────────────────────────────────────

def _validate_age(value: Any, current: Optional[Any]) -> Tuple[bool, Optional[str]]:
    """Validate age: must be 0-150, cannot decrease (unless correction)."""
    try:
        age = int(value)
    except (ValueError, TypeError):
        return False, "age must be a number"
    if age < 0 or age > 150:
        return False, f"age {age} out of range 0-150"
    # Age can only increase naturally (but corrections allowed)
    return True, None


def _validate_city(value: Any, current: Optional[Any]) -> Tuple[bool, Optional[str]]:
    """City must be a non-empty string."""
    if not value or not str(value).strip():
        return False, "city cannot be empty"
    return True, None


def _validate_name(value: Any, current: Optional[Any]) -> Tuple[bool, Optional[str]]:
    """Name must be a non-empty string."""
    if not value or not str(value).strip():
        return False, "name cannot be empty"
    return True, None


# ── Extraction patterns ──────────────────────────────────────────────

# Each pattern: (compiled regex, slot_name, group_index or callable, confidence_bonus)
# confidence_bonus is added to base confidence for explicit patterns

_EXTRACT_PATTERNS: List[Tuple[re.Pattern, str, Any, float]] = [
    # Age
    (re.compile(r'\b(?:мне|me|i am|i\'m|my age is)\s+(\d{1,3})\s*(?:лет|год|года|years?\s*old|years?)\b', re.I), "age", 1, 0.0),
    (re.compile(r'\b(\d{1,3})\s*(?:лет|год|года|years?\s*old)\b', re.I), "age", 1, -0.1),
    (re.compile(r'\b(?:i\'m|i am)\s+(\d{1,3})(?:\.|,|\s|$)', re.I), "age", 1, -0.2),

    # Name
    (re.compile(r'\b(?:меня зовут|my name is|i\'m called|зовите меня)\s+([A-ZА-ЯЁ][a-zа-яё]+)\b', re.I), "name", 1, 0.0),
    (re.compile(r'\b(?:hi,?\s+)?i\'m\s+([A-Z][a-z]{1,20})(?:\.|,|\s|$)(?!\s*(?:allergic|also|just|really|not|a\s|an\s|the\s|\d))', re.I), "name", 1, -0.1),

    # City
    (re.compile(r'\b(?:живу в|я из|i live in|i\'m from|i am from|based in|located in)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)(?:\s*[,.]|\s+and\b|\s*$)', re.I), "city", 1, 0.0),
    (re.compile(r'\b(?:moved to|relocated to|переехал в|перееха(?:ла|л) в)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)\b', re.I), "city", 1, 0.0),

    # Diet
    (re.compile(r'\b(?:я\s+)?(вегетарианец|вегетарианка|веган|веганка|vegetarian|vegan)\b', re.I), "diet", 1, 0.0),
    (re.compile(r'\b(?:i am|i\'m|я)\s+(vegetarian|vegan|pescatarian|omnivore)\b', re.I), "diet", 1, 0.0),

    # Allergies (accumulative)
    (re.compile(r'\b(?:аллергия на|allergic to|allergy to|i have .* allergy)\s+(.+?)(?:\s+(?:too|either|as well))?(?:\.|,|!|$)', re.I), "allergies", 1, 0.0),
    (re.compile(r'\b(?:не переношу|intolerant to|can\'t eat|cannot eat)\s+(.+?)(?:\s+(?:too|either|as well))?(?:\.|,|!|$)', re.I), "allergies", 1, -0.1),

    # Work
    (re.compile(r'\b(?:работаю|я работаю|i work as|i\'m a|my job is|my profession is)\s+(.+?)(?:\.|,|$)', re.I), "work", 1, 0.0),

    # Family
    (re.compile(r'\b(?:я\s+)?(женат|замужем|married|single|divorced|холост|разведён|разведена)\b', re.I), "family_status", 1, 0.0),

    # Explicit memory commands
    (re.compile(r'\b(?:запомни|remember|имей в виду|учти|важно что)\s*[:\-—]?\s*(.+)', re.I), "_explicit_fact", 1, 0.3),

    # Correction signals
    (re.compile(r'\b(?:я ошибся|ошибся|опечатался|опечатка|i was wrong|correction|actually|на самом деле|исправ)\b', re.I), "_correction_signal", 0, 0.0),
]

# Slots that accumulate values instead of replacing
_ACCUMULATIVE_SLOTS = {"allergies"}

# Slot validator map
_VALIDATORS: Dict[str, Any] = {
    "age": _validate_age,
    "city": _validate_city,
    "name": _validate_name,
}


# ── UserProfile ──────────────────────────────────────────────────────

class UserProfile:
    """Structured user profile with conflict resolution and confidence tracking."""

    SLOT_NAMES = ("name", "age", "city", "diet", "allergies", "work", "family_status")

    def __init__(self):
        self.slots: Dict[str, ProfileSlot] = {
            name: ProfileSlot() for name in self.SLOT_NAMES
        }
        # Explicit facts that don't fit slots
        self.explicit_facts: List[Dict] = []
        # Correction mode flag
        self._correction_mode = False
        self._correction_mode_until = 0.0

    def get(self, slot_name: str) -> Optional[Any]:
        """Get current value of a slot."""
        slot = self.slots.get(slot_name)
        return slot.value if slot and slot.is_set else None

    def set_slot(
        self,
        slot_name: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "user_explicit",
        force: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Set a slot value with validation and conflict resolution.

        Returns (success, message).
        """
        if slot_name not in self.slots:
            return False, f"unknown slot: {slot_name}"

        slot = self.slots[slot_name]

        # Accumulative slots (allergies) — add to list
        if slot_name in _ACCUMULATIVE_SLOTS:
            current_list = slot.value or []
            if isinstance(current_list, str):
                current_list = [current_list]
            new_item = str(value).strip().lower()
            if new_item not in [x.lower() for x in current_list]:
                current_list.append(str(value).strip())
                return slot.update(current_list, confidence, source, reason="accumulate"), None
            return False, "already known"

        # Validate
        validator = _VALIDATORS.get(slot_name)
        if validator and not force:
            valid, msg = validator(value, slot.value)
            if not valid:
                return False, msg

        # Conflict resolution
        reason = "update"
        if slot.is_set and slot.value != value:
            if self._correction_mode or force:
                reason = "correction"
            elif slot_name == "age":
                # Age-specific logic: can increase by 1-2 naturally
                try:
                    old_age = int(slot.value)
                    new_age = int(value)
                    if new_age < old_age:
                        # Age decreased — possible typo correction
                        if confidence >= slot.confidence:
                            reason = "correction"
                        else:
                            return False, f"age decreased from {old_age} to {new_age} — needs confirmation"
                    elif new_age - old_age > 5:
                        reason = "correction"
                    else:
                        reason = "natural_update"
                except (ValueError, TypeError):
                    reason = "correction"
            else:
                reason = "conflict_resolved"

        changed = slot.update(value, confidence, source, reason=reason)
        return changed, None

    def extract_and_update(self, text: str, confidence: float = 1.0, source: str = "user_explicit") -> List[Dict]:
        """Extract profile facts from text and update slots.

        Returns list of {slot, value, action, message} dicts.
        """
        results = []
        normalized = text.strip()
        if not normalized:
            return results

        # Check for correction signals
        for pattern, slot_name, group_idx, conf_bonus in _EXTRACT_PATTERNS:
            if slot_name == "_correction_signal":
                if pattern.search(normalized):
                    self._correction_mode = True
                    self._correction_mode_until = time.time() + 60.0  # 60 sec window
                    results.append({
                        "slot": "_correction",
                        "value": None,
                        "action": "correction_mode_on",
                        "message": "Correction mode activated for 60s",
                    })
                    break

        # Expire correction mode
        if self._correction_mode and time.time() > self._correction_mode_until:
            self._correction_mode = False

        # Extract slot values
        for pattern, slot_name, group_idx, conf_bonus in _EXTRACT_PATTERNS:
            if slot_name.startswith("_"):
                # Handle explicit facts
                if slot_name == "_explicit_fact":
                    match = pattern.search(normalized)
                    if match:
                        fact_text = match.group(group_idx).strip()
                        if fact_text and len(fact_text) > 3:
                            self.explicit_facts.append({
                                "text": fact_text,
                                "confidence": confidence + conf_bonus,
                                "timestamp": time.time(),
                            })
                            results.append({
                                "slot": "explicit_fact",
                                "value": fact_text,
                                "action": "stored",
                                "message": None,
                            })
                continue

            match = pattern.search(normalized)
            if match:
                raw_value = match.group(group_idx).strip()

                # Type coercion
                if slot_name == "age":
                    try:
                        raw_value = int(raw_value)
                    except ValueError:
                        continue

                effective_confidence = min(1.0, max(0.0, confidence + conf_bonus))
                changed, msg = self.set_slot(
                    slot_name,
                    raw_value,
                    confidence=effective_confidence,
                    source=source,
                    force=self._correction_mode,
                )

                action = "updated" if changed else ("unchanged" if msg is None else "rejected")
                results.append({
                    "slot": slot_name,
                    "value": raw_value,
                    "action": action,
                    "message": msg,
                })

        return results

    def as_prompt_block(self) -> str:
        """Format profile as a block for system prompt injection."""
        lines = []
        for name in self.SLOT_NAMES:
            slot = self.slots[name]
            if slot.is_set:
                display_value = slot.value
                if isinstance(display_value, list):
                    display_value = ", ".join(str(v) for v in display_value)
                conf_marker = "" if slot.confidence >= 0.8 else " (unconfirmed)"
                lines.append(f"  - {name}: {display_value}{conf_marker}")

        for fact in self.explicit_facts[-5:]:  # last 5 explicit facts
            conf_marker = "" if fact["confidence"] >= 0.8 else " (unconfirmed)"
            lines.append(f"  - fact: {fact['text']}{conf_marker}")

        if not lines:
            return ""

        return (
            "\n\n[USER PROFILE — structured facts, highest priority]\n"
            + "\n".join(lines)
            + "\n[END USER PROFILE]\n"
        )

    def as_dict(self) -> Dict:
        """Serialize profile to dict."""
        result = {}
        for name in self.SLOT_NAMES:
            slot = self.slots[name]
            if slot.is_set:
                result[name] = {
                    "value": slot.value,
                    "confidence": slot.confidence,
                    "source": slot.source,
                    "history_count": len(slot.history),
                }
        if self.explicit_facts:
            result["explicit_facts"] = self.explicit_facts[-10:]
        return result

    def needs_confirmation(self, slot_name: str) -> bool:
        """Check if a slot value needs user confirmation."""
        slot = self.slots.get(slot_name)
        if not slot or not slot.is_set:
            return False
        return slot.confidence < 0.8

    def confirmation_questions(self) -> List[str]:
        """Generate confirmation questions for low-confidence slots."""
        questions = []
        for name in self.SLOT_NAMES:
            slot = self.slots[name]
            if slot.is_set and slot.confidence < 0.8:
                if name == "age":
                    questions.append(f"You mentioned you are {slot.value} years old — is that correct?")
                elif name == "city":
                    questions.append(f"You mentioned you live in {slot.value} — is that right?")
                elif name == "diet":
                    questions.append(f"You said you are {slot.value} — correct?")
                else:
                    questions.append(f"Is it correct that your {name} is {slot.value}?")
        return questions
