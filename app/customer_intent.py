"""Bounded customer order transport; Core owns prices, stock and approval."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

CONTRACT = "autobiz.customer-order-intent.v1"


def enabled() -> bool:
    return os.getenv("AUTOBIZ_CUSTOMER_INGRESS_ENABLED", "").strip().lower() == "true"


def text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} is missing or exceeds its limit")
    if re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError(f"{field} contains control characters")
    return value.strip()


def build_customer_intent(
    draft: Mapping[str, Any], conversation_id: str
) -> dict[str, Any]:
    account = text(
        os.getenv("AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE", ""),
        "external_account_reference",
        255,
    )
    reference = text(
        draft.get("draft_order_id") or draft.get("draft_id"),
        "external_order_reference",
        255,
    )
    conversation = text(conversation_id, "customer_conversation_id", 255)
    source_items = draft.get("items")
    if not isinstance(source_items, list) or not 1 <= len(source_items) <= 20:
        raise ValueError("items must contain 1 to 20 entries")
    items = []
    for raw in source_items:
        if not isinstance(raw, Mapping):
            raise ValueError("each item must be an object")
        quantity = raw.get("quantity")
        if type(quantity) is not int or not 1 <= quantity <= 1000:
            raise ValueError("quantity must be an integer from 1 to 1000")
        # No display colour/size inference: send the exact catalog variant.
        variant = raw.get("variant") or "default"
        items.append(
            {
                "sku": text(raw.get("sku"), "sku", 50),
                "variant": text(variant, "variant", 100),
                "quantity": quantity,
            }
        )
    source_customer = draft.get("customer") or {}
    if not isinstance(source_customer, Mapping):
        raise ValueError("customer must be an object")
    customer = {
        key: text(source_customer.get(key) or draft.get(alias), key, maximum)
        for key, alias, maximum in (
            ("name", "customer_name", 255),
            ("phone", "phone", 50),
            ("address", "address", 1000),
        )
    }
    payload = {
        "contract": CONTRACT,
        "surface": "telegram",
        "external_account_reference": account,
        "provider_event_id": reference,
        "external_order_reference": reference,
        "customer_conversation_id": conversation,
        "items": items,
        "customer": customer,
    }
    note = source_customer.get("note")
    if note:
        payload["note"] = text(note, "note", 1000)
    return payload


def validate_receipt(value: Any, payload: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("status") != "DRAFT_CREATED":
        raise ValueError("Core has not confirmed draft creation")
    if value.get("external_order_reference") != payload["external_order_reference"]:
        raise ValueError("Core receipt does not match this request")
    from uuid import UUID

    for field in ("draft_order_id", "run_id", "event_id"):
        identifier = value.get(field)
        if not isinstance(identifier, str):
            raise ValueError("Core receipt identifier must be a UUID string")
        UUID(identifier)
    if value.get("draft_status") != "draft":
        raise ValueError("Core receipt is not a draft awaiting review")


def customer_envelope(
    intent: Mapping[str, Any], secret: str, *, created: int | None = None
) -> dict[str, Any]:
    import hashlib
    import hmac
    import json
    import time

    if len(secret.encode("utf-8")) < 32:
        raise ValueError("customer credential must contain at least 32 bytes")
    timestamp = int(time.time()) if created is None else created
    raw = json.dumps(
        intent,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    base = b"autobiz.customer-order-intent.v1\n" + str(timestamp).encode() + b"\n" + raw
    return {
        "intent": dict(intent),
        "customer_auth": {
            "created": timestamp,
            "signature": hmac.new(
                secret.encode("utf-8"), base, hashlib.sha256
            ).hexdigest(),
        },
    }
