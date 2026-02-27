"""
Stripe Webhook Handler — обработка событий от Stripe.

Поддерживаемые события:
  - checkout.session.completed  → создание API-ключа
  - customer.subscription.updated → смена тарифа
  - customer.subscription.deleted → деактивация ключа
  - invoice.payment_failed       → предупреждение
"""

import os
import logging
from typing import Optional

import stripe
from fastapi import APIRouter, Request, HTTPException

from api.billing.key_manager import KeyManager
from api.billing.models import PlanTier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

# Stripe инициализация
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Price ID → PlanTier маппинг (заполнить из Stripe Dashboard)
PRICE_TO_PLAN = {
    os.environ.get("STRIPE_PRICE_FREE", "price_free"): PlanTier.FREE,
    os.environ.get("STRIPE_PRICE_PRO", "price_pro"): PlanTier.PRO,
    os.environ.get("STRIPE_PRICE_ENTERPRISE", "price_enterprise"): PlanTier.ENTERPRISE,
}

# Singleton KeyManager
_key_manager: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager


def _resolve_plan(price_id: str) -> PlanTier:
    """Определяет тариф по Stripe price_id."""
    return PRICE_TO_PLAN.get(price_id, PlanTier.FREE)


# ── Webhook endpoint ─────────────────────────────────────────────────────────

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Принимает события от Stripe.
    Stripe подписывает каждый webhook — мы верифицируем подпись.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Верификация подписи
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError:
            logger.warning("Stripe webhook: invalid signature")
            raise HTTPException(status_code=400, detail="Invalid signature")
        except Exception as e:
            logger.error(f"Stripe webhook error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Dev mode — без верификации
        import json
        event = json.loads(payload)
        logger.warning("Stripe webhook: signature verification DISABLED (dev mode)")

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    logger.info(f"Stripe event: {event_type}")

    km = get_key_manager()

    # ── checkout.session.completed ────────────────────────────────────────
    if event_type == "checkout.session.completed":
        customer_email = data.get("customer_details", {}).get("email", "")
        customer_id = data.get("customer", "")
        subscription_id = data.get("subscription", "")

        # Определяем план из line_items
        plan = PlanTier.FREE
        if subscription_id:
            try:
                sub = stripe.Subscription.retrieve(subscription_id)
                price_id = sub["items"]["data"][0]["price"]["id"]
                plan = _resolve_plan(price_id)
            except Exception as e:
                logger.error(f"Failed to retrieve subscription: {e}")

        # Создаём API-ключ
        api_key = km.create_key(
            user_email=customer_email,
            plan=plan,
            stripe_customer_id=customer_id,
            stripe_subscription_id=subscription_id,
        )

        logger.info(f"New API key created for {customer_email}, plan={plan.value}")

        # TODO: отправить email с ключом через SendGrid/Resend
        # Пока логируем (в продакшне нужен email-сервис)
        logger.info(f"API Key (send to user): {api_key[:12]}...")

        return {"status": "ok", "action": "key_created"}

    # ── customer.subscription.updated ─────────────────────────────────────
    elif event_type == "customer.subscription.updated":
        customer_id = data.get("customer", "")
        subscription_id = data.get("id", "")
        price_id = data.get("items", {}).get("data", [{}])[0].get("price", {}).get("id", "")
        new_plan = _resolve_plan(price_id)

        km.update_plan(customer_id, new_plan, subscription_id)
        logger.info(f"Plan updated: customer={customer_id}, plan={new_plan.value}")

        return {"status": "ok", "action": "plan_updated"}

    # ── customer.subscription.deleted ─────────────────────────────────────
    elif event_type == "customer.subscription.deleted":
        customer_id = data.get("customer", "")

        # Даунгрейд на Free вместо полной деактивации
        km.update_plan(customer_id, PlanTier.FREE)
        logger.info(f"Subscription cancelled: customer={customer_id}, downgraded to FREE")

        return {"status": "ok", "action": "downgraded_to_free"}

    # ── invoice.payment_failed ────────────────────────────────────────────
    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer", "")
        logger.warning(f"Payment failed: customer={customer_id}")
        # Можно отправить email-уведомление
        return {"status": "ok", "action": "payment_failed_logged"}

    # ── Прочие события — игнорируем ───────────────────────────────────────
    else:
        return {"status": "ok", "action": "ignored"}


# ── Checkout Session Creation (для фронтенда) ────────────────────────────────

@router.post("/create-checkout")
async def create_checkout_session(request: Request):
    """
    Создаёт Stripe Checkout Session.
    Фронтенд редиректит пользователя на Stripe для оплаты.
    """
    body = await request.json()
    price_id = body.get("price_id")
    success_url = body.get("success_url", "https://ngt-memory.dev/success")
    cancel_url = body.get("cancel_url", "https://ngt-memory.dev/pricing")

    if not price_id:
        raise HTTPException(status_code=400, detail="price_id required")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except Exception as e:
        logger.error(f"Checkout creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Usage endpoint ────────────────────────────────────────────────────────────

@router.get("/usage")
async def get_usage(request: Request):
    """Возвращает статистику использования для API-ключа."""
    api_key = request.headers.get("X-Api-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-Api-Key header required")

    km = get_key_manager()
    record = km.validate_key(api_key)
    if not record:
        raise HTTPException(status_code=401, detail="Invalid API key")

    allowed, used, limit = km.check_limit(api_key)
    stats = km.get_usage_stats(api_key, days=30)

    return {
        "plan": record.plan.value,
        "daily_limit": limit,
        "used_today": used,
        "remaining_today": max(0, limit - used),
        "history": [s.dict() for s in stats],
    }
