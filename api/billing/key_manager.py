"""
Key Manager — хранение API-ключей и usage tracking в SQLite.

Таблицы:
  api_keys   — ключ, email, план, stripe IDs, active
  usage      — ключ, дата, кол-во запросов, токены
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

from api.billing.models import ApiKeyRecord, PlanTier, PLAN_LIMITS, UsageRecord


DEFAULT_DB_PATH = Path("data/billing.db")


class KeyManager:
    """Управление API-ключами и лимитами."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    key_prefix TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'free',
                    stripe_customer_id TEXT,
                    stripe_subscription_id TEXT,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    key_hash TEXT NOT NULL,
                    date TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (key_hash, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keys_email
                ON api_keys(user_email)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keys_stripe
                ON api_keys(stripe_customer_id)
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    # ── Key CRUD ──────────────────────────────────────────────────────────

    def create_key(
        self,
        user_email: str,
        plan: PlanTier = PlanTier.FREE,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
    ) -> str:
        """Создаёт новый API-ключ. Возвращает plaintext ключ (показать пользователю 1 раз)."""
        raw_key = f"ngt_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:8]

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO api_keys
                   (key_hash, key_prefix, user_email, plan,
                    stripe_customer_id, stripe_subscription_id, created_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (key_hash, key_prefix, user_email, plan.value,
                 stripe_customer_id, stripe_subscription_id,
                 datetime.utcnow().isoformat()),
            )
        return raw_key

    def validate_key(self, key: str) -> Optional[ApiKeyRecord]:
        """Проверяет ключ. Возвращает запись или None если невалидный."""
        key_hash = self._hash_key(key)
        with self._conn() as conn:
            row = conn.execute(
                """SELECT key_prefix, user_email, plan,
                          stripe_customer_id, stripe_subscription_id,
                          created_at, is_active
                   FROM api_keys WHERE key_hash = ?""",
                (key_hash,),
            ).fetchone()

        if not row or not row[6]:  # is_active == 0
            return None

        return ApiKeyRecord(
            key=row[0] + "...",  # только prefix
            user_email=row[1],
            plan=PlanTier(row[2]),
            stripe_customer_id=row[3],
            stripe_subscription_id=row[4],
            created_at=datetime.fromisoformat(row[5]),
            is_active=bool(row[6]),
        )

    def update_plan(
        self,
        stripe_customer_id: str,
        new_plan: PlanTier,
        subscription_id: Optional[str] = None,
    ) -> bool:
        """Обновляет план по stripe_customer_id."""
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE api_keys SET plan = ?, stripe_subscription_id = ?
                   WHERE stripe_customer_id = ? AND is_active = 1""",
                (new_plan.value, subscription_id, stripe_customer_id),
            )
        return cursor.rowcount > 0

    def deactivate_key(self, stripe_customer_id: str) -> bool:
        """Деактивирует ключ (при отмене подписки)."""
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE api_keys SET is_active = 0
                   WHERE stripe_customer_id = ? AND is_active = 1""",
                (stripe_customer_id,),
            )
        return cursor.rowcount > 0

    def reactivate_key(self, stripe_customer_id: str) -> bool:
        """Реактивирует ключ."""
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE api_keys SET is_active = 1
                   WHERE stripe_customer_id = ?""",
                (stripe_customer_id,),
            )
        return cursor.rowcount > 0

    def get_key_by_email(self, email: str) -> Optional[str]:
        """Возвращает key_prefix для email (для поддержки)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT key_prefix FROM api_keys WHERE user_email = ? AND is_active = 1",
                (email,),
            ).fetchone()
        return row[0] if row else None

    # ── Usage Tracking ────────────────────────────────────────────────────

    def check_limit(self, key: str) -> tuple[bool, int, int]:
        """
        Проверяет лимит. Возвращает (allowed, used_today, daily_limit).
        """
        record = self.validate_key(key)
        if not record:
            return False, 0, 0

        daily_limit = PLAN_LIMITS[record.plan]
        used = self._get_usage_today(key)
        return used < daily_limit, used, daily_limit

    def increment_usage(self, key: str, tokens_in: int = 0, tokens_out: int = 0):
        """Инкрементирует usage для ключа."""
        key_hash = self._hash_key(key)
        today = date.today().isoformat()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO usage (key_hash, date, request_count, tokens_in, tokens_out)
                   VALUES (?, ?, 1, ?, ?)
                   ON CONFLICT(key_hash, date)
                   DO UPDATE SET
                     request_count = request_count + 1,
                     tokens_in = tokens_in + ?,
                     tokens_out = tokens_out + ?""",
                (key_hash, today, tokens_in, tokens_out, tokens_in, tokens_out),
            )

    def _get_usage_today(self, key: str) -> int:
        key_hash = self._hash_key(key)
        today = date.today().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT request_count FROM usage WHERE key_hash = ? AND date = ?",
                (key_hash, today),
            ).fetchone()
        return row[0] if row else 0

    def get_usage_stats(self, key: str, days: int = 30) -> List[UsageRecord]:
        """Возвращает статистику использования за N дней."""
        key_hash = self._hash_key(key)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, request_count, tokens_in, tokens_out
                   FROM usage WHERE key_hash = ?
                   ORDER BY date DESC LIMIT ?""",
                (key_hash, days),
            ).fetchall()
        return [
            UsageRecord(
                api_key=key[:8] + "...",
                date=r[0],
                request_count=r[1],
                tokens_in=r[2],
                tokens_out=r[3],
            )
            for r in rows
        ]
