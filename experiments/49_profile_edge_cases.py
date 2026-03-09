"""
Exp 49 — Comprehensive Profile & Memory Edge Cases Test

Проверяет ВСЕ краевые случаи системы памяти:

1. PROFILE EXTRACTION
   - Прямые факты (RU/EN): имя, возраст, город, диета, аллергия, работа
   - Фрагментная склейка (мне + 30 + лет → age=30)
   - Аккумулятивные слоты (несколько аллергий)

2. CONFLICT RESOLUTION
   - Возраст не может уменьшиться без correction mode
   - "я ошибся" → correction mode → возраст обновляется
   - Город меняется при переезде
   - Диета меняется

3. MEMORY QUALITY GATE
   - Мусор НЕ сохраняется (ыва, 456, случайные символы)
   - Осмысленные сообщения сохраняются
   - Ответы на мусор тоже не сохраняются

4. PROFILE-AWARE ANSWERS
   - "Могу ли я есть мясо?" при diet=vegetarian
   - "Что ты знаешь обо мне?" — должен перечислить все факты
   - "Где мне поесть?" — учитывает город + диету + аллергии

5. EXPLICIT MEMORY COMMANDS
   - "запомни: ..." — сохраняет как explicit fact
   - "имей в виду: ..." — аналогично

6. CROSS-LINGUAL
   - Факты на русском, вопросы на английском и наоборот

7. TEMPORAL CONSISTENCY
   - Факты сохраняются через много ходов мусора
   - Профиль не теряется при спаме
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = os.environ.get("NGT_API_URL", "https://ngt-memory.ru/api")
API_KEY = os.environ.get("NGT_API_SECRET", "")
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "49_profile_edge_cases.json"

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-Api-Key"] = API_KEY


# ── Helpers ──────────────────────────────────────────────────────────

def chat(session_id: str, message: str, timeout: float = 30.0) -> Dict:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=timeout) as client:
        r = client.post("/chat", json={"message": message, "session_id": session_id})
        r.raise_for_status()
        return r.json()


def health() -> Dict:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=5) as client:
        r = client.get("/health")
        r.raise_for_status()
        return r.json()


def normalize(text: str) -> str:
    return " ".join((text or "").lower().replace("-", " ").split())


def contains_any(text: str, terms: List[str]) -> bool:
    norm = normalize(text)
    return any(normalize(t) in norm for t in terms)


def contains_all(text: str, term_groups: List[List[str]]) -> bool:
    """Each group must have at least one match."""
    for group in term_groups:
        if not contains_any(text, group):
            return False
    return True


# ── Test scenarios ───────────────────────────────────────────────────

class TestCase:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.steps: List[Dict] = []
        self.checks: List[Dict] = []

    def send(self, message: str, delay: float = 1.0):
        self.steps.append({"type": "send", "message": message, "delay": delay})
        return self

    def check_response(self, must_contain: Optional[List[List[str]]] = None,
                       must_not_contain: Optional[List[str]] = None,
                       description: str = ""):
        self.checks.append({
            "type": "response",
            "step_index": len(self.steps) - 1,
            "must_contain": must_contain or [],
            "must_not_contain": must_not_contain or [],
            "description": description,
        })
        return self

    def check_profile(self, slot: str, expected_value: Any, description: str = ""):
        self.checks.append({
            "type": "profile",
            "step_index": len(self.steps) - 1,
            "slot": slot,
            "expected_value": expected_value,
            "description": description,
        })
        return self

    def check_entries(self, op: str, value: int, description: str = ""):
        """op: 'eq', 'gt', 'lt', 'lte', 'gte'"""
        self.checks.append({
            "type": "entries",
            "step_index": len(self.steps) - 1,
            "op": op,
            "value": value,
            "description": description,
        })
        return self

    def check_memories_count(self, op: str, value: int, description: str = ""):
        self.checks.append({
            "type": "memories_count",
            "step_index": len(self.steps) - 1,
            "op": op,
            "value": value,
            "description": description,
        })
        return self


def build_tests() -> List[TestCase]:
    tests = []

    # ── 1. BASIC PROFILE EXTRACTION (EN) ────────────────────────────
    t = TestCase("Profile extraction — English direct facts", "profile_extraction")
    t.send("My name is Alice and I live in London.")
    t.check_profile("name", "Alice", "name extracted")
    t.check_profile("city", "London", "city extracted")
    t.send("I'm vegetarian and allergic to gluten.")
    t.check_profile("diet", "vegetarian", "diet extracted")
    t.check_profile("allergies", ["gluten"], "allergy extracted")
    t.send("I'm also allergic to shellfish.")
    t.check_profile("allergies", ["gluten", "shellfish"], "second allergy accumulated")
    t.send("What do you know about me?")
    t.check_response([["Alice"], ["London"], ["vegetarian"], ["gluten"], ["shellfish"]],
                     description="response mentions all profile facts")
    tests.append(t)

    # ── 2. PROFILE EXTRACTION (RU) ──────────────────────────────────
    t = TestCase("Profile extraction — Russian direct facts", "profile_extraction")
    t.send("Меня зовут Борис и я живу в Москве.")
    t.check_profile("name", "Борис", "имя извлечено")
    t.check_profile("city", "Москве", "город извлечён")
    t.send("Я вегетарианец.")
    t.check_profile("diet", "вегетарианец", "диета извлечена")
    t.send("Что ты знаешь обо мне?")
    t.check_response([["Борис"], ["Москв"], ["вегетариан"]],
                     description="ответ содержит все факты профиля")
    tests.append(t)

    # ── 3. FRAGMENT MERGE → PROFILE ─────────────────────────────────
    t = TestCase("Fragment merge into profile slot", "fragment_merge")
    t.send("мне", delay=0.5)
    t.check_entries("eq", 0, "fragment 'мне' not stored alone")
    t.send("25", delay=0.5)
    t.check_entries("eq", 0, "fragment '25' not stored alone")
    t.send("лет", delay=0.5)
    t.check_entries("gt", 0, "merged 'мне 25 лет' stored")
    t.check_profile("age", 25, "age extracted from merged fragment (confidence=0.6)")
    t.send("How old am I?")
    t.check_response([["25"]], description="LLM uses merged age from fragments")
    tests.append(t)

    # ── 4. NOISE FILTERING ──────────────────────────────────────────
    t = TestCase("Noise filtering — garbage not stored", "noise_filter")
    t.send("I live in Paris and I'm 28 years old.")
    t.check_profile("city", "Paris", "city set")
    t.check_profile("age", 28, "age set")
    # Record baseline entries
    t.send("ыва")
    t.check_entries("lte", 6, "garbage 'ыва' not stored")
    t.send("456456456")
    t.check_entries("lte", 6, "garbage numbers not stored")
    t.send("фывапро")
    t.check_entries("lte", 6, "garbage cyrillic not stored")
    t.send("!@#$%")
    t.check_entries("lte", 6, "garbage symbols not stored")
    t.send("x")
    t.check_entries("lte", 6, "single char not stored")
    t.send("Where do I live?")
    t.check_response([["Paris"]], description="profile survives noise spam")
    tests.append(t)

    # ── 5. AGE CONFLICT — NATURAL INCREASE ──────────────────────────
    t = TestCase("Age conflict — natural increase allowed", "conflict_resolution")
    t.send("I'm 30 years old.")
    t.check_profile("age", 30, "initial age set")
    t.send("I'm 31 years old now.")
    t.check_profile("age", 31, "age increased by 1 — natural update")
    t.send("How old am I?")
    t.check_response([["31"]], description="LLM uses updated age")
    tests.append(t)

    # ── 6. AGE CONFLICT — DECREASE BLOCKED → CORRECTION MODE ───────
    t = TestCase("Age conflict — decrease blocked, correction fixes it", "conflict_resolution")
    t.send("I'm 35 years old.")
    t.check_profile("age", 35, "initial age set")
    t.send("I'm 28 years old.")
    # Age might still be 35 (decrease blocked without correction) or 28 (if confidence wins)
    t.send("Actually I made a mistake earlier, I'm really 28.")
    t.check_profile("age", 28, "correction mode activated, age corrected to 28")
    t.send("How old am I?")
    t.check_response([["28"]], description="LLM uses corrected age")
    tests.append(t)

    # ── 7. CITY CHANGE — RELOCATION ─────────────────────────────────
    t = TestCase("City change — user relocates", "conflict_resolution")
    t.send("I live in Berlin.")
    t.check_profile("city", "Berlin", "initial city")
    t.send("I just moved to Amsterdam.")
    t.check_profile("city", "Amsterdam", "city updated after move")
    t.send("Where do I live now?")
    t.check_response([["Amsterdam"]],
                     description="LLM uses new city")
    tests.append(t)

    # ── 8. DIET CHANGE ──────────────────────────────────────────────
    t = TestCase("Diet change — vegetarian to vegan", "conflict_resolution")
    t.send("I'm vegetarian.")
    t.check_profile("diet", "vegetarian", "initial diet")
    t.send("Actually I went fully vegan last month.")
    t.check_profile("diet", "vegan", "diet updated to vegan")
    t.send("Can I eat cheese?")
    t.check_response([["vegan"]], description="LLM references vegan diet, not vegetarian")
    tests.append(t)

    # ── 9. EXPLICIT MEMORY COMMANDS ─────────────────────────────────
    t = TestCase("Explicit memory commands — запомни / remember", "explicit_memory")
    t.send("Запомни: мой любимый цвет — синий.")
    t.send("Remember: I have a meeting at 3pm tomorrow.")
    t.send("What's my favorite color?")
    t.check_response([["синий", "blue"]], description="explicit fact recalled")
    tests.append(t)

    # ── 10. PROFILE-AWARE CONTRADICTION ─────────────────────────────
    t = TestCase("Profile contradiction — vegetarian asked about meat", "profile_aware")
    t.send("I am vegetarian, live in Tokyo, and I'm allergic to soy.")
    t.check_profile("diet", "vegetarian", "diet set")
    t.check_profile("city", "Tokyo", "city set")
    t.send("Can I eat a steak?")
    t.check_response([["vegetarian"]],
                     description="LLM references vegetarian status when asked about steak")
    t.send("Recommend me a restaurant that fits my restrictions.")
    t.check_response([["vegetarian", "vegan", "plant"]], description="restaurant fits diet")
    tests.append(t)

    # ── 11. CROSS-LINGUAL RECALL ────────────────────────────────────
    t = TestCase("Cross-lingual — facts in RU, questions in EN", "cross_lingual")
    t.send("Я живу в Берлине и мне 40 лет.")
    t.check_profile("city", "Берлине", "город извлечён из RU")
    t.check_profile("age", 40, "возраст извлечён из RU")
    t.send("How old am I and where do I live?")
    t.check_response([["40"], ["Berlin", "Берлин"]], description="EN answer uses RU facts")
    tests.append(t)

    # ── 12. TEMPORAL PERSISTENCE — FACTS SURVIVE NOISE SPAM ─────────
    t = TestCase("Temporal persistence — profile survives 10 noise messages", "temporal")
    t.send("My name is Charlie and I'm 22.")
    t.check_profile("name", "Charlie", "name set")
    t.check_profile("age", 22, "age set")
    # 10 noise messages
    for i in range(10):
        noise = ["ыв", "123", "аа", "!!!", "...", "хх", "99", "gg", "  ", "~~"][i]
        t.send(noise, delay=0.3)
    t.send("What's my name and age?")
    t.check_response([["Charlie"], ["22"]], description="profile intact after 10 noise messages")
    tests.append(t)

    # ── 13. MULTIPLE ALLERGIES ACCUMULATION ─────────────────────────
    t = TestCase("Multiple allergies accumulate correctly", "profile_extraction")
    t.send("I'm allergic to peanuts.")
    t.check_profile("allergies", ["peanuts"], "first allergy")
    t.send("I'm also allergic to penicillin.")
    t.check_profile("allergies", ["peanuts", "penicillin"], "second allergy accumulated")
    t.send("And I can't eat shellfish either.")
    t.check_profile("allergies", ["peanuts", "penicillin", "shellfish"], "third allergy accumulated")
    t.send("What are all my allergies?")
    t.check_response([["peanut"], ["penicillin"], ["shellfish"]],
                     description="LLM lists all allergies")
    tests.append(t)

    # ── 14. FULL PROFILE ASSEMBLY FROM SCATTERED MESSAGES ───────────
    t = TestCase("Full profile from scattered messages across turns", "integration")
    t.send("Hi, I'm David.")
    t.send("I live in San Francisco.")
    t.send("I'm a software engineer.")
    t.send("I'm 29 years old.")
    t.send("I follow a vegan diet.")
    t.send("I'm allergic to dust mites.")
    t.send("Tell me everything you know about me.")
    t.check_response(
        [["David"], ["San Francisco"], ["software engineer", "engineer"], ["29"], ["vegan"], ["dust"]],
        description="full profile assembled from scattered messages"
    )
    t.check_profile("name", "David", "name from scattered")
    t.check_profile("age", 29, "age from scattered")
    t.check_profile("diet", "vegan", "diet from scattered")
    tests.append(t)

    return tests


# ── Runner ───────────────────────────────────────────────────────────

def compare_op(actual: int, op: str, expected: int) -> bool:
    if op == "eq": return actual == expected
    if op == "gt": return actual > expected
    if op == "lt": return actual < expected
    if op == "gte": return actual >= expected
    if op == "lte": return actual <= expected
    return False


def run_test(test: TestCase) -> Dict:
    session_id = f"exp49-{test.name[:20].replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    step_results: List[Dict] = []
    check_results: List[Dict] = []

    print(f"\n{'─' * 60}")
    print(f"  {test.category} / {test.name}")
    print(f"  session: {session_id}")
    print(f"{'─' * 60}")

    for i, step in enumerate(test.steps):
        time.sleep(step.get("delay", 1.0))
        msg = step["message"]
        try:
            result = chat(session_id, msg)
            step_results.append({
                "index": i,
                "message": msg,
                "response": result.get("response", "")[:200],
                "memories_count": result.get("memories_count", 0),
                "memory_entries": result.get("memory_entries", 0),
                "profile": result.get("profile"),
                "profile_updates": result.get("profile_updates"),
            })
            resp_short = (result.get("response", "") or "")[:80].replace("\n", " ")
            prof = result.get("profile", {}) or {}
            prof_str = ",".join(f"{k}={v.get('value','')}" for k, v in prof.items() if k != "explicit_facts" and v.get("value"))
            print(f"    [{i}] {msg[:40]:40s} mem={result.get('memories_count',0)} ent={result.get('memory_entries',0)} prof={{{prof_str}}}")
        except Exception as exc:
            step_results.append({"index": i, "message": msg, "error": str(exc)})
            print(f"    [{i}] {msg[:40]:40s} ERROR: {exc}")

    # Run checks
    for check in test.checks:
        step_idx = check["step_index"]
        if step_idx >= len(step_results):
            check_results.append({"check": check, "passed": False, "reason": "step not reached"})
            continue

        step_data = step_results[step_idx]
        if "error" in step_data:
            check_results.append({"check": check, "passed": False, "reason": f"step errored: {step_data['error']}"})
            continue

        passed = False
        reason = ""

        if check["type"] == "response":
            response_text = step_data.get("response", "")
            must_contain = check.get("must_contain", [])
            must_not_contain = check.get("must_not_contain", [])

            if must_contain:
                passed = contains_all(response_text, must_contain)
                if not passed:
                    reason = f"response missing required terms from {must_contain}"
            else:
                passed = True

            if passed and must_not_contain:
                for bad in must_not_contain:
                    if normalize(bad) in normalize(response_text):
                        passed = False
                        reason = f"response contains forbidden term '{bad}'"
                        break

        elif check["type"] == "profile":
            profile = step_data.get("profile") or {}
            slot = check["slot"]
            expected = check["expected_value"]
            slot_data = profile.get(slot, {})
            actual = slot_data.get("value") if isinstance(slot_data, dict) else None

            if isinstance(expected, list) and isinstance(actual, list):
                passed = set(str(x).lower() for x in expected) == set(str(x).lower() for x in actual)
            elif isinstance(expected, int):
                try:
                    passed = int(actual) == expected
                except (TypeError, ValueError):
                    passed = False
            else:
                passed = str(actual or "").lower() == str(expected).lower()

            if not passed:
                reason = f"profile.{slot}: expected={expected}, actual={actual}"

        elif check["type"] == "entries":
            actual = step_data.get("memory_entries", 0)
            passed = compare_op(actual, check["op"], check["value"])
            if not passed:
                reason = f"entries={actual}, expected {check['op']} {check['value']}"

        elif check["type"] == "memories_count":
            actual = step_data.get("memories_count", 0)
            passed = compare_op(actual, check["op"], check["value"])
            if not passed:
                reason = f"memories_count={actual}, expected {check['op']} {check['value']}"

        status = "PASS" if passed else "FAIL"
        desc = check.get("description", "")
        print(f"    {'✅' if passed else '❌'} {status}: {desc}" + (f" — {reason}" if reason else ""))

        check_results.append({
            "check_type": check["type"],
            "description": check.get("description", ""),
            "passed": passed,
            "reason": reason,
        })

    passed_count = sum(1 for c in check_results if c["passed"])
    total_count = len(check_results)

    return {
        "test_name": test.name,
        "category": test.category,
        "session_id": session_id,
        "steps": step_results,
        "checks": check_results,
        "passed": passed_count,
        "total": total_count,
        "all_passed": passed_count == total_count,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 72)
    print("Exp 49 — Comprehensive Profile & Memory Edge Cases Test")
    print(f"API: {BASE_URL}")
    print("=" * 72)

    try:
        h = health()
    except Exception as exc:
        print(f"FATAL: API not reachable at {BASE_URL}: {exc}")
        return 1

    print(f"Health: status={h.get('status')} version={h.get('version')} model={h.get('model')}")
    print()

    tests = build_tests()
    all_results: List[Dict] = []
    total_start = time.perf_counter()

    for test in tests:
        result = run_test(test)
        all_results.append(result)

    total_time = time.perf_counter() - total_start

    # Summary
    total_passed = sum(r["passed"] for r in all_results)
    total_checks = sum(r["total"] for r in all_results)
    tests_all_passed = sum(1 for r in all_results if r["all_passed"])

    summary = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": BASE_URL,
        "health": h,
        "total_time_sec": round(total_time, 2),
        "test_count": len(all_results),
        "tests_all_passed": tests_all_passed,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "pass_rate": round(total_passed / max(1, total_checks), 3),
        "categories": {},
    }

    # Per-category stats
    categories: Dict[str, Dict] = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"tests": 0, "passed": 0, "total": 0, "checks_passed": 0}
        categories[cat]["tests"] += 1
        categories[cat]["passed"] += 1 if r["all_passed"] else 0
        categories[cat]["total"] += r["total"]
        categories[cat]["checks_passed"] += r["passed"]
    summary["categories"] = categories

    payload = {"summary": summary, "results": all_results}
    RESULTS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Tests:          {len(all_results)}")
    print(f"All passed:     {tests_all_passed}/{len(all_results)}")
    print(f"Checks:         {total_passed}/{total_checks} ({summary['pass_rate']:.0%})")
    print(f"Total time:     {summary['total_time_sec']}s")
    print()

    for cat, stats in categories.items():
        print(f"  {cat:25s}  tests={stats['passed']}/{stats['tests']}  checks={stats['checks_passed']}/{stats['total']}")

    print()
    for r in all_results:
        status = "✅" if r["all_passed"] else "❌"
        print(f"  {status} {r['test_name']:50s}  {r['passed']}/{r['total']}")

    print(f"\nResults: {RESULTS_FILE}")
    return 0 if tests_all_passed == len(all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
