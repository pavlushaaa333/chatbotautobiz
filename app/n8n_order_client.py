from __future__ import annotations

import logging
from typing import Any

import requests

from app.config import N8N_ORDER_WEBHOOK_TIMEOUT_SECONDS, N8N_ORDER_WEBHOOK_URL


logger = logging.getLogger(__name__)


def send_order_draft(payload: dict[str, Any]) -> dict[str, Any]:
    if not N8N_ORDER_WEBHOOK_URL:
        return {
            "ok": False,
            "error_type": "missing_webhook_url",
            "message": "N8N_ORDER_WEBHOOK_URL is not configured.",
        }

    draft_id = payload.get("draft_id")
    try:
        response = requests.post(
            N8N_ORDER_WEBHOOK_URL,
            json=payload,
            timeout=N8N_ORDER_WEBHOOK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.warning("Order draft webhook timed out | draft_id=%s", draft_id)
        return {"ok": False, "error_type": "timeout", "message": str(exc)}
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "Order draft webhook failed | draft_id=%s | status_code=%s",
            draft_id,
            status_code,
        )
        return {
            "ok": False,
            "error_type": "http_or_network_error",
            "status_code": status_code,
            "message": str(exc),
        }

    logger.info(
        "Order draft webhook submitted | draft_id=%s | status_code=%s",
        draft_id,
        response.status_code,
    )
    body: Any = None
    try:
        body = response.json()
    except ValueError:
        body = response.text[:500]
    return {"ok": True, "status_code": response.status_code, "body": body}

