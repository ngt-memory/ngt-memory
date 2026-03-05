"""
Billing models — Pydantic схемы для тарифов, ключей и usage tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PlanTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Лимиты запросов в день для каждого тарифа
PLAN_LIMITS = {
    PlanTier.FREE: 100,
    PlanTier.PRO: 10_000,
    PlanTier.ENTERPRISE: 1_000_000,  # практически безлимит
}

PLAN_NAMES = {
    PlanTier.FREE: "Free",
    PlanTier.PRO: "Pro ($29/mo)",
    PlanTier.ENTERPRISE: "Enterprise ($99/mo)",
}


class ApiKeyRecord(BaseModel):
    """Запись API-ключа в базе."""
    key: str
    user_email: str
    plan: PlanTier = PlanTier.FREE
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class UsageRecord(BaseModel):
    """Дневной usage для ключа."""
    api_key: str
    date: str  # YYYY-MM-DD
    request_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class BillingWebhookEvent(BaseModel):
    """Входящее событие от Stripe webhook."""
    event_type: str
    customer_email: str
    customer_id: str
    subscription_id: Optional[str] = None
    plan: Optional[PlanTier] = None


class ApiKeyResponse(BaseModel):
    """Ответ при создании/получении ключа."""
    api_key: str
    plan: str
    daily_limit: int
    used_today: int
    remaining_today: int
