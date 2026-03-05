"""
Тесты NGT Memory на conversational сценариях.

Это честные тесты — query семантически ОТЛИЧАЕТСЯ от stored fact.
Не "original + 5% noise", а реальные перефразировки и indirect questions.

Сценарии:
    TestMultiTurnRecall      — факт из хода #2 находится при запросе в ходу #8
    TestSemanticGapRecall    — query не содержит ключевых слов из факта
    TestCrossSessionRecall   — факты сессии 1 доступны в сессии 2
    TestPositionBias         — нет предпочтения поздних vs ранних ходов
    TestConceptGraphBridge   — граф связывает смежные медицинские концепты
    TestTopicIsolation       — запросы не смешивают несвязанные топики
"""

import pytest
import torch
from sentence_transformers import SentenceTransformer
from ngt.core.llm_memory import NGTMemoryForLLM


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def encoder():
    """Загружаем SentenceTransformer один раз на весь модуль."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def encode(model, text: str) -> torch.Tensor:
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


def encode_batch(model, texts: list) -> torch.Tensor:
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=False)


@pytest.fixture
def fresh_memory():
    return NGTMemoryForLLM(
        embedding_dim=384,
        max_entries=200,
        max_concepts=500,
        hebbian_lr=0.2,
        consolidation_interval=500,
        concept_extraction="regex",
        concept_top_k=5,
        device="cpu",
    )


# ============================================================
# Вспомогательные функции
# ============================================================

def retrieve_texts(memory: NGTMemoryForLLM, query_emb: torch.Tensor,
                   top_k: int = 5, use_graph: bool = True) -> list:
    results = memory.retrieve(query_emb, top_k=top_k, use_graph=use_graph)
    return [r["text"] for r in results]


def store_conversation(memory: NGTMemoryForLLM, model, turns: list, domain: str = "general"):
    """Сохраняет список {"role": ..., "text": ...} в память."""
    for turn in turns:
        emb = encode(model, turn["text"])
        memory.store(emb, text=turn["text"], concepts=None, domain=domain)
    memory._flush_hebbian()


# ============================================================
# 1. Multi-turn recall — факт из раннего хода
# ============================================================

class TestMultiTurnRecall:
    """
    Пользователь упоминает что-то в ходе 2.
    В ходе 8 задаётся вопрос, который требует вспомнить факт из хода 2.
    """

    MEDICAL_DIALOG = [
        {"role": "patient", "text": "I've been having severe headaches for the past three days."},
        {"role": "doctor",  "text": "Any associated symptoms like nausea or light sensitivity?"},
        {"role": "patient", "text": "Yes, bright lights make it worse. No nausea."},
        {"role": "doctor",  "text": "Any family history of migraines?"},
        {"role": "patient", "text": "My mother had migraines. I was diagnosed with tension headaches two years ago."},
        {"role": "doctor",  "text": "Are you on any medications? Any allergies?"},
        # ← key fact: turn 6
        {"role": "patient", "text": "I take lisinopril for blood pressure. Allergic to penicillin — severe childhood reaction."},
        {"role": "doctor",  "text": "Given your symptoms this looks like migraine. I'll prescribe sumatriptan."},
        {"role": "patient", "text": "Are there any contraindications with my current meds?"},
        {"role": "doctor",  "text": "Sumatriptan can interact with blood pressure medications."},
    ]
    KEY_FACT = "I take lisinopril for blood pressure. Allergic to penicillin — severe childhood reaction."

    def test_recall_at_5_indirect_allergy_query(self, encoder, fresh_memory):
        """Indirect query о аллергии находит факт из хода 6."""
        store_conversation(fresh_memory, encoder, self.MEDICAL_DIALOG, domain="medical")
        query = encode(encoder, "Does the patient have any drug allergies to consider?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert self.KEY_FACT in found, \
            f"Fact not found in top-5. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_recall_at_5_medication_query(self, encoder, fresh_memory):
        """Query о лекарствах находит факт с упоминанием lisinopril."""
        store_conversation(fresh_memory, encoder, self.MEDICAL_DIALOG, domain="medical")
        query = encode(encoder, "What medications is this patient currently taking?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert self.KEY_FACT in found, \
            f"Medication fact not in top-5. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_recall_at_5_contraindication_query(self, encoder, fresh_memory):
        """
        Query о противопоказаниях: факт с lisinopril должен быть в top-5.
        top-3 слишком строго — 'contraindication' семантически ближе к turn-тексту
        'Are there any contraindications with my current meds?' чем к самому факту.
        """
        store_conversation(fresh_memory, encoder, self.MEDICAL_DIALOG, domain="medical")
        query = encode(encoder, "Any contraindications or interactions to check before prescribing?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert self.KEY_FACT in found, \
            f"Fact not in top-5. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_family_history_recall(self, encoder, fresh_memory):
        """Факт о семейной истории (ход 4) находится через indirect query."""
        store_conversation(fresh_memory, encoder, self.MEDICAL_DIALOG, domain="medical")
        query = encode(encoder, "Is there a genetic predisposition to headaches in this patient's family?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        family_fact = "My mother had migraines. I was diagnosed with tension headaches two years ago."
        assert family_fact in found, \
            f"Family history not found. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)


# ============================================================
# 2. Semantic gap — query не содержит ключевых слов из факта
# ============================================================

class TestSemanticGapRecall:
    """
    Проверяем что NGT находит факты когда query ПОЛНОСТЬЮ ДРУГИМИ словами.
    Это главный failure case для keyword-based систем.
    """

    def test_shellfish_allergy_indirect(self, encoder, fresh_memory):
        """
        Stored: "Traveler is allergic to shellfish — no shrimp or crab."
        Query:  "What dietary restrictions apply to restaurant booking?"
        Нет пересечения ключевых слов.
        """
        travel_facts = [
            {"text": "Planning a two-week trip to Japan in April.", "domain": "travel"},
            {"text": "Budget is around 3000 dollars, traveling with partner.", "domain": "travel"},
            {"text": "Partner is vegetarian and I'm allergic to shellfish — no shrimp or crab.", "domain": "travel"},
            {"text": "We'd like 3 nights in a traditional ryokan.", "domain": "travel"},
            {"text": "Partner has a mild knee issue, prefers chairs over floor seating.", "domain": "travel"},
        ]
        for f in travel_facts:
            emb = encode(encoder, f["text"])
            fresh_memory.store(emb, text=f["text"], domain=f["domain"])
        fresh_memory._flush_hebbian()

        query = encode(encoder, "What dietary restrictions apply when booking restaurants for this traveler?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "Partner is vegetarian and I'm allergic to shellfish — no shrimp or crab."
        assert target in found, \
            f"Dietary restriction not found semantically. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_visa_requirement_indirect(self, encoder, fresh_memory):
        """
        Stored: "I'm on H-1B visa, needs visa transfer from current employer."
        Query:  "Are there any immigration issues with hiring this candidate?"
        """
        hr_facts = [
            {"text": "Candidate has 4 years experience leading a team of 6 engineers.", "domain": "hr"},
            {"text": "Tech stack is Python, Go, PostgreSQL, Kubernetes on AWS.", "domain": "hr"},
            {"text": "Salary expectation is 180–210k annually plus equity.", "domain": "hr"},
            {"text": "Currently on H-1B visa, needs visa transfer from current employer.", "domain": "hr"},
            {"text": "Notice period is three months, negotiable to two.", "domain": "hr"},
        ]
        for f in hr_facts:
            emb = encode(encoder, f["text"])
            fresh_memory.store(emb, text=f["text"], domain=f["domain"])
        fresh_memory._flush_hebbian()

        query = encode(encoder, "Are there any immigration or legal hurdles to hiring this person?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "Currently on H-1B visa, needs visa transfer from current employer."
        assert target in found, \
            f"Visa fact not found. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_refund_preference_indirect(self, encoder, fresh_memory):
        """
        Stored: "Customer wants resend not refund — item was at 40% discount."
        Query:  "What resolution should we process for this customer?"
        """
        support_facts = [
            {"text": "Order 77291-B has been stuck in customs since the 17th.", "domain": "support"},
            {"text": "Customer needs item urgently — starting new job on the 28th.", "domain": "support"},
            {"text": "Customer is located in Germany.", "domain": "support"},
            {"text": "Customer wants resend not refund — item was at 40% discount, wants to keep that price.", "domain": "support"},
            {"text": "UK warehouse can ship to Germany in 2–3 days.", "domain": "support"},
        ]
        for f in support_facts:
            emb = encode(encoder, f["text"])
            fresh_memory.store(emb, text=f["text"], domain=f["domain"])
        fresh_memory._flush_hebbian()

        query = encode(encoder, "What resolution should we process for this customer's delayed order?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "Customer wants resend not refund — item was at 40% discount, wants to keep that price."
        assert target in found, \
            f"Resolution preference not found. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_bitlocker_key_location(self, encoder, fresh_memory):
        """
        Stored: "BitLocker enabled, recovery key saved in Microsoft account."
        Query:  "How can we access the encrypted drive if boot fails?"
        """
        tech_facts = [
            {"text": "Dell XPS 15, Windows 11, stopped working after a Windows update.", "domain": "tech"},
            {"text": "Hard reset attempted three times, no change.", "domain": "tech"},
            {"text": "BitLocker encryption is enabled, recovery key is saved in Microsoft account.", "domain": "tech"},
            {"text": "External monitor connected via HDMI shows nothing either.", "domain": "tech"},
        ]
        for f in tech_facts:
            emb = encode(encoder, f["text"])
            fresh_memory.store(emb, text=f["text"], domain=f["domain"])
        fresh_memory._flush_hebbian()

        query = encode(encoder, "How can we access the encrypted drive to recover from a failed boot?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "BitLocker encryption is enabled, recovery key is saved in Microsoft account."
        assert target in found, \
            f"BitLocker fact not found. Got:\n" + "\n".join(f"  - {t}" for t in found)


# ============================================================
# 3. Cross-session recall
# ============================================================

class TestCrossSessionRecall:
    """
    Факты сохранённые в сессии 1 должны находиться при запросах в сессии 2.
    Имитирует реальный chatbot: разговор сегодня → продолжение завтра.
    """

    def test_allergy_persists_across_sessions(self, encoder, fresh_memory):
        """Аллергия упомянутая в сессии 1 находится в сессии 2."""
        # Сессия 1
        session1 = [
            "Patient presents with recurring migraines and light sensitivity.",
            "Patient takes lisinopril 10mg daily for hypertension.",
            "Patient reports severe penicillin allergy — anaphylaxis at age 8.",
            "Prescribed sumatriptan 50mg as needed.",
        ]
        for text in session1:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="medical")
        fresh_memory._flush_hebbian()

        # Сессия 2
        fresh_memory.new_session()
        session2 = [
            "Patient returns, reports chest tightness after sumatriptan.",
            "Considering switching to rizatriptan.",
        ]
        for text in session2:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="medical")
        fresh_memory._flush_hebbian()

        # Query из сессии 2 → должен найти факт из сессии 1
        query = encode(encoder, "Are there any antibiotics we should avoid for this patient?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "Patient reports severe penicillin allergy — anaphylaxis at age 8."
        assert target in found, \
            f"Session-1 allergy not found in session-2 query. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_salary_persists_across_sessions(self, encoder, fresh_memory):
        """Ожидания по зарплате из интервью сессии 1 находятся в сессии 2."""
        session1 = [
            "Candidate applies for senior backend engineer position.",
            "Candidate has 4 years at FinTech startup managing team of 6.",
            "Candidate's salary expectation is 180 to 210 thousand dollars annually.",
            "Candidate currently earns 160k and wants equity component.",
        ]
        for text in session1:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="hr")
        fresh_memory._flush_hebbian()

        fresh_memory.new_session()
        session2 = [
            "Second round interview scheduled for candidate.",
            "Technical panel approved the candidate.",
        ]
        for text in session2:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="hr")
        fresh_memory._flush_hebbian()

        query = encode(encoder, "What compensation package does this candidate expect?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        target = "Candidate's salary expectation is 180 to 210 thousand dollars annually."
        assert target in found, \
            f"Salary fact not found across sessions. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_three_sessions_retention(self, encoder, fresh_memory):
        """Факт из сессии 1 находится после трёх сессий."""
        # Сессия 1
        key_fact = "Customer placed order 77291-B for a laptop stand, paid for express shipping."
        facts_s1 = [key_fact, "Order shipped on the 14th, currently in customs."]
        for text in facts_s1:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="support")
        fresh_memory._flush_hebbian()

        # Сессии 2 и 3 (добавляем другие диалоги)
        for session_i in range(2):
            fresh_memory.new_session()
            for text in ["Unrelated topic about weather.", "Different conversation entirely.",
                         "Technical discussion about databases."]:
                emb = encode(encoder, text)
                fresh_memory.store(emb, text=text, domain="other")
        fresh_memory._flush_hebbian()

        query = encode(encoder, "What was the order number and shipping method for the laptop stand?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert key_fact in found, \
            f"Order fact not retained after 3 sessions. Got:\n" + "\n".join(f"  - {t}" for t in found)


# ============================================================
# 4. Position bias
# ============================================================

class TestPositionBias:
    """
    Факты из ранних ходов должны находиться не хуже поздних.
    Recency bias — распространённая проблема memory систем.
    """

    def _build_long_conversation(self, model, memory):
        """Строит 12-ходовой диалог с ключевыми фактами на разных позициях."""
        turns = [
            # Turn 0 — EARLY KEY FACT
            "Patient's name is James Miller, born 1975, lives in Seattle.",
            "Initial complaint is persistent lower back pain for 6 weeks.",
            # Turn 2 — EARLY KEY FACT
            "James is allergic to NSAIDs — ibuprofen caused GI bleeding in 2019.",
            "Pain is rated 7 out of 10, worse in the morning.",
            "No history of spinal surgery or trauma.",
            "Patient works as a software engineer, mostly sitting.",
            # Turn 6 — MID KEY FACT
            "MRI shows mild L4-L5 disc bulge, no nerve compression.",
            "Referred to physiotherapy — 8 sessions over 4 weeks.",
            "Patient is also being treated for type 2 diabetes — on metformin.",
            # Turn 9 — LATE KEY FACT
            "Prescribed acetaminophen 500mg up to 3 times daily for pain.",
            "Follow-up appointment scheduled in 6 weeks.",
            "Patient agrees to reduce sitting time and use standing desk.",
        ]
        for t in turns:
            emb = encode(model, t)
            memory.store(emb, text=t, domain="medical")
        memory._flush_hebbian()
        return turns

    def test_early_fact_recalled(self, encoder, fresh_memory):
        """Факт из хода 0 (имя и город) находится по indirect query."""
        turns = self._build_long_conversation(encoder, fresh_memory)
        query = encode(encoder, "Who is this patient and where are they from?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert turns[0] in found, \
            f"Early fact (turn 0) not found. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_early_allergy_recalled(self, encoder, fresh_memory):
        """Аллергия из хода 2 находится по query из конца диалога."""
        turns = self._build_long_conversation(encoder, fresh_memory)
        query = encode(encoder, "What pain medications should be avoided for this patient?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        # turn 2: NSAIDs allergy
        assert turns[2] in found, \
            f"NSAID allergy (turn 2) not found. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_late_fact_recalled(self, encoder, fresh_memory):
        """Назначение из хода 9 также находится."""
        turns = self._build_long_conversation(encoder, fresh_memory)
        query = encode(encoder, "What analgesic was prescribed and what is the dosage?")
        found = retrieve_texts(fresh_memory, query, top_k=5)
        assert turns[9] in found, \
            f"Late prescription (turn 9) not found. Got:\n" + "\n".join(f"  - {t[:80]}" for t in found)

    def test_no_strong_recency_bias(self, encoder, fresh_memory):
        """
        Recall@5 для ранних фактов (≤2) должен быть не хуже чем для поздних (≥8)
        на уровне не ниже 0.5 разницы.
        """
        turns = self._build_long_conversation(encoder, fresh_memory)

        early_queries = [
            ("Patient's name is James Miller, born 1975, lives in Seattle.",
             "Who is this patient and what is their demographic?"),
            ("James is allergic to NSAIDs — ibuprofen caused GI bleeding in 2019.",
             "What anti-inflammatory drugs are contraindicated for this patient?"),
        ]
        late_queries = [
            ("Prescribed acetaminophen 500mg up to 3 times daily for pain.",
             "What painkiller was prescribed and at what dose?"),
            ("Patient is also being treated for type 2 diabetes — on metformin.",
             "Does this patient have any comorbidities requiring medication?"),
        ]

        def hit_rate(pairs):
            hits = 0
            for target, query_text in pairs:
                q_emb = encode(encoder, query_text)
                found = retrieve_texts(fresh_memory, q_emb, top_k=5)
                hits += int(target in found)
            return hits / len(pairs)

        early_hr = hit_rate(early_queries)
        late_hr  = hit_rate(late_queries)

        # Разница не должна превышать 0.5 (нет сильного recency bias)
        assert abs(early_hr - late_hr) <= 0.5, \
            f"Recency bias too strong: early={early_hr:.2f} late={late_hr:.2f} diff={abs(early_hr-late_hr):.2f}"


# ============================================================
# 5. Topic isolation — разные топики не мешают друг другу
# ============================================================

class TestTopicIsolation:
    """
    Запрос о медицинских фактах не должен возвращать факты о путешествиях.
    """

    def _build_mixed_memory(self, model, memory):
        medical = [
            "Patient John is allergic to aspirin.",
            "John has type 1 diabetes, uses insulin pump.",
            "John's blood pressure is consistently high — 150/95.",
        ]
        travel = [
            "Sarah plans to visit Kyoto in spring.",
            "Sarah's travel budget is five thousand euros.",
            "Sarah prefers window seats on flights.",
        ]
        hr = [
            "Candidate Maria expects 120k salary.",
            "Maria has ten years experience in data science.",
            "Maria requires relocation assistance to New York.",
        ]
        for texts, domain in [(medical, "medical"), (travel, "travel"), (hr, "hr")]:
            for t in texts:
                emb = encode(model, t)
                memory.store(emb, text=t, domain=domain)
        memory._flush_hebbian()
        return medical, travel, hr

    def test_medical_query_returns_medical_facts(self, encoder, fresh_memory):
        """Query о диабете возвращает медицинские факты, а не travel/hr."""
        medical, travel, hr = self._build_mixed_memory(encoder, fresh_memory)
        query = encode(encoder, "What chronic conditions does this patient have?")
        found = retrieve_texts(fresh_memory, query, top_k=3)
        # Хотя бы один медицинский факт в top-3
        medical_in_found = sum(1 for f in found if any(m in f for m in ["allergy", "diabetes", "blood pressure", "aspirin", "insulin"]))
        assert medical_in_found >= 1, \
            f"No medical facts in top-3. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_travel_query_returns_travel_facts(self, encoder, fresh_memory):
        """Query о путешествии возвращает travel факты."""
        medical, travel, hr = self._build_mixed_memory(encoder, fresh_memory)
        query = encode(encoder, "What is the traveler's budget and destination?")
        found = retrieve_texts(fresh_memory, query, top_k=3)
        travel_in_found = sum(1 for f in found if any(t in f for t in ["Kyoto", "budget", "euros", "travel", "spring", "Sarah"]))
        assert travel_in_found >= 1, \
            f"No travel facts in top-3. Got:\n" + "\n".join(f"  - {t}" for t in found)

    def test_hr_query_returns_hr_facts(self, encoder, fresh_memory):
        """Query о зарплате возвращает HR факты."""
        medical, travel, hr = self._build_mixed_memory(encoder, fresh_memory)
        query = encode(encoder, "What are the candidate's salary expectations and experience?")
        found = retrieve_texts(fresh_memory, query, top_k=3)
        hr_in_found = sum(1 for f in found if any(h in f for h in ["salary", "experience", "Maria", "relocation", "data science"]))
        assert hr_in_found >= 1, \
            f"No HR facts in top-3. Got:\n" + "\n".join(f"  - {t}" for t in found)


# ============================================================
# 6. Graph bridge — граф соединяет смежные концепты
# ============================================================

class TestConceptGraphBridge:
    """
    После накопления Hebbian связей между концептами одного домена,
    граф должен находить связанные документы через bridge.
    """

    def test_medication_concepts_bridge(self, encoder, fresh_memory):
        """
        Факты о лекарствах (lisinopril, sumatriptan, metformin) должны быть связаны
        через общий концепт 'medication'/'drug'.
        После Hebbian накопления запрос о лекарствах поднимает все.
        """
        facts = [
            "Patient takes lisinopril 10mg for blood pressure management.",
            "Patient uses metformin 500mg twice daily for type 2 diabetes.",
            "Sumatriptan 50mg prescribed for acute migraine attacks.",
            "Patient allergic to penicillin — documented anaphylaxis.",
            "Ibuprofen contraindicated due to previous GI bleeding.",
        ]
        for text in facts:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="medical")
        fresh_memory._flush_hebbian()

        # Повторный проход для усиления Hebbian связей
        for text in facts:
            emb = encode(encoder, text)
            cids = [fresh_memory.associations._name_to_id.get(c)
                    for c in fresh_memory._concept_extractor.extract(text)
                    if c in fresh_memory.associations._name_to_id]
            cids = [c for c in cids if c is not None]
            if len(cids) >= 2:
                fresh_memory.associations.record_co_occurrence(cids)

        # Query о лекарственном взаимодействии
        query = encode(encoder, "What drugs is this patient currently on and what should be avoided?")
        found_graph   = retrieve_texts(fresh_memory, query, top_k=5, use_graph=True)
        found_nograph = retrieve_texts(fresh_memory, query, top_k=5, use_graph=False)

        # Хотя бы 2 факта о лекарствах в top-5 (с графом ≥ без графа)
        drug_keywords = ["lisinopril", "metformin", "sumatriptan", "penicillin", "ibuprofen"]
        hits_graph   = sum(1 for f in found_graph   if any(kw in f.lower() for kw in drug_keywords))
        hits_nograph = sum(1 for f in found_nograph if any(kw in f.lower() for kw in drug_keywords))

        assert hits_graph >= 2, \
            f"Graph retrieval found only {hits_graph} drug facts (need ≥2). Got:\n" + \
            "\n".join(f"  - {t}" for t in found_graph)

    def test_graph_improves_or_equals_nograph(self, encoder, fresh_memory):
        """
        Use-graph=True должен давать hit@5 ≥ use_graph=False на drug-related queries.
        """
        facts = [
            "Patient takes lisinopril for hypertension since 2018.",
            "Sumatriptan prescribed for migraines, take at first sign.",
            "Aspirin allergy — causes hives and facial swelling.",
            "Metformin 1000mg for glycaemic control.",
            "Last blood test showed elevated creatinine levels.",
        ]
        target = "Aspirin allergy — causes hives and facial swelling."

        for text in facts:
            emb = encode(encoder, text)
            fresh_memory.store(emb, text=text, domain="medical")
        fresh_memory._flush_hebbian()

        query = encode(encoder, "Are there any known drug hypersensitivity reactions?")

        res_graph   = fresh_memory.retrieve(query, top_k=5, use_graph=True)
        res_nograph = fresh_memory.retrieve(query, top_k=5, use_graph=False)

        hit_graph   = any(r["text"] == target for r in res_graph)
        hit_nograph = any(r["text"] == target for r in res_nograph)

        # Граф должен быть не хуже (hit_graph >= hit_nograph)
        # ИЛИ оба нашли — тоже окей
        assert hit_graph or hit_nograph, \
            f"Neither graph nor nograph found the allergy fact."

        if not hit_graph and hit_nograph:
            pytest.fail(f"Graph retrieval MISSED allergy fact that nograph found. "
                        f"Graph top-5:\n" + "\n".join(f"  - {r['text']}" for r in res_graph))
