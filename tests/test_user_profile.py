"""
Тесты для UserProfile — структурированного извлечения фактов о пользователе.

Ключевой фокус: city-паттерны с дефисами (Санкт-Петербург, Ростов-на-Дону,
Rio-de-Janeiro) — основные кейсы российского продукта.

Известное ограничение (зафиксировано комментарием в user_profile.py):
захватывается словоформа из текста («Петербурге», не «Петербург») —
нормализация падежей вне скоупа regex, это задача для NER/лемматизации.

Также покрываются age / name / allergies — раньше у них было ноль тестов.
"""

import pytest

from ngt.core.user_profile import UserProfile


@pytest.fixture
def profile():
    return UserProfile()


# ============ City: дефисы, словосочетания, стоп-слова ============

class TestCityExtraction:

    @pytest.mark.parametrize("text,expected_city", [
        ("живу в Санкт-Петербурге", "Санкт-Петербурге"),
        ("я из Ростов-на-Дону", "Ростов-на-Дону"),
        ("переехал в Нижний Новгород", "Нижний Новгород"),
        ("I live in New York and work remotely", "New York"),
        ("i'm from Rio-de-Janeiro", "Rio-de-Janeiro"),
    ])
    def test_city_cases_from_spec(self, profile, text, expected_city):
        """Кейсы из ТЗ: дефис-сегменты + многословные города + стоп на 'and'."""
        profile.extract_and_update(text, confidence=1.0)
        assert profile.get("city") == expected_city, \
            f"text={text!r} → city={profile.get('city')!r}, ожидалось {expected_city!r}"

    def test_simple_city_still_works(self, profile):
        """Регрессия: простые однословные города без дефисов."""
        profile.extract_and_update("I live in Berlin", confidence=1.0)
        assert profile.get("city") == "Berlin"

    def test_city_stops_before_russian_and(self, profile):
        """Стоп перед русским 'и'."""
        profile.extract_and_update("живу в Москве и работаю удалённо", confidence=1.0)
        assert profile.get("city") == "Москве"

    def test_moved_to_hyphenated(self, profile):
        """Паттерн 'moved to' тоже понимает дефисы."""
        profile.extract_and_update("I moved to Saint-Petersburg.", confidence=1.0)
        assert profile.get("city") == "Saint-Petersburg"


# ============ Age ============

class TestAgeExtraction:

    def test_age_russian(self, profile):
        profile.extract_and_update("мне 30 лет", confidence=1.0)
        assert profile.get("age") == 30

    def test_age_english(self, profile):
        profile.extract_and_update("I am 42 years old", confidence=1.0)
        assert profile.get("age") == 42

    def test_age_out_of_range_rejected(self, profile):
        profile.extract_and_update("мне 200 лет", confidence=1.0)
        assert profile.get("age") is None

    def test_age_natural_increase_allowed(self, profile):
        profile.extract_and_update("мне 30 лет", confidence=1.0)
        profile.extract_and_update("мне 31 год", confidence=1.0)
        assert profile.get("age") == 31

    def test_age_decrease_with_equal_confidence_is_correction(self, profile):
        """Уменьшение возраста при confidence >= текущей трактуется как correction."""
        profile.extract_and_update("мне 35 лет", confidence=1.0)
        profile.extract_and_update("мне 33 года", confidence=1.0)
        assert profile.get("age") == 33


# ============ Name ============

class TestNameExtraction:

    def test_name_russian(self, profile):
        profile.extract_and_update("Меня зовут Антон", confidence=1.0)
        assert profile.get("name") == "Антон"

    def test_name_english(self, profile):
        profile.extract_and_update("My name is Alice", confidence=1.0)
        assert profile.get("name") == "Alice"

    def test_im_allergic_is_not_a_name(self, profile):
        """'I'm allergic ...' не должно становиться именем (negative lookahead)."""
        profile.extract_and_update("I'm allergic to penicillin", confidence=1.0)
        assert profile.get("name") is None


# ============ Allergies (accumulative slot) ============

class TestAllergiesExtraction:

    def test_single_allergy(self, profile):
        profile.extract_and_update("I'm allergic to penicillin.", confidence=1.0)
        allergies = profile.get("allergies")
        assert allergies is not None
        assert any("penicillin" in a.lower() for a in allergies)

    def test_allergies_accumulate(self, profile):
        profile.extract_and_update("аллергия на пенициллин.", confidence=1.0)
        profile.extract_and_update("аллергия на орехи.", confidence=1.0)
        allergies = profile.get("allergies")
        assert len(allergies) == 2

    def test_duplicate_allergy_not_added_twice(self, profile):
        profile.extract_and_update("allergic to peanuts.", confidence=1.0)
        profile.extract_and_update("allergic to peanuts.", confidence=1.0)
        assert len(profile.get("allergies")) == 1


# ============ Prompt block / serialization sanity ============

class TestProfileOutput:

    def test_as_prompt_block_contains_city(self, profile):
        profile.extract_and_update("живу в Санкт-Петербурге", confidence=1.0)
        block = profile.as_prompt_block()
        assert "Санкт-Петербурге" in block
        assert "USER PROFILE" in block

    def test_as_dict_roundtrip(self, profile):
        profile.extract_and_update("мне 30 лет", confidence=1.0)
        profile.extract_and_update("я из Ростов-на-Дону", confidence=1.0)
        d = profile.as_dict()
        assert d["age"]["value"] == 30
        assert d["city"]["value"] == "Ростов-на-Дону"

    def test_empty_profile_empty_block(self, profile):
        assert profile.as_prompt_block() == ""
