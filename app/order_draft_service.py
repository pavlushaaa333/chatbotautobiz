from __future__ import annotations

import itertools
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Mapping, MutableMapping
from typing import Any


import requests

from app.config import active_shop_id
from app.normalizer import extract_color, extract_size, normalize_text, strip_accents

logger = logging.getLogger(__name__)

ORDER_STATE_COLLECTING = "collecting"
ORDER_STATE_AWAITING_CONFIRMATION = "awaiting_customer_confirmation"
ORDER_STATE_SUBMITTING = "submitting_to_n8n"
ORDER_STATE_PENDING_SHOP_APPROVAL = "pending_shop_approval"
ORDER_STATE_SUBMISSION_FAILED = "submission_failed"
ORDER_STATE_CANCELLED = "cancelled"
ORDER_STATE_LEGACY_SUBMITTED = "submitted_to_n8n"

ORDER_PENDING_ACTIONS = {
    "choose_order_product",
    "choose_order_color",
    "choose_order_size",
    "choose_order_color_and_size",
    "choose_order_quantity",
    "collect_customer_name",
    "collect_customer_phone",
    "collect_customer_address",
    "collect_customer_note",
    "choose_payment_method",
    "confirm_order_draft",
    "edit_order_draft",
}

ORDER_DRAFT_RESET_KEYS = {
    "order_draft",
    "submission_failed",
    "last_submission_failed",
    "submitted_to_n8n",
    "submission_in_progress",
    "awaiting_customer_confirmation",
    "pending_shop_approval",
}

_DRAFT_COUNTER = itertools.count(1)

_NUMBER_WORDS = {
    "mot": 1,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bon": 4,
    "bốn": 4,
    "tu": 4,
    "tư": 4,
    "nam": 5,
    "năm": 5,
    "sau": 6,
    "sáu": 6,
    "bay": 7,
    "bảy": 7,
    "tam": 8,
    "tám": 8,
    "chin": 9,
    "chín": 9,
    "muoi": 10,
    "mười": 10,
}

ORDER_CONFIRMATION_EXACT_TEXTS = {
    "xac nhan",
    "dung roi",
    "thong tin dung roi",
    "chot don",
    "dat don",
    "gui don",
    "minh dong y",
    "ok xac nhan",
    "oke chot",
}
ORDER_CONFIRMATION_NEGATIVE_PATTERNS = [
    r"\b(?:khong|chua)\s+xac\s+nhan\b",
    r"\b(?:dung|đung)\s+chot\b",
    r"\bchua\s+dung\b",
    r"\bsai\s+roi\b",
    r"\bdoi\s+(?:lai|size|mau|dia\s+chi|so\s+luong|so\s+dien\s+thoai|sdt)\b",
    r"\b(?:huy|khong\s+mua\s+nua|bo\s+don|dung\s+don)\b",
    r"\bxac\s+nhan\b.*\b(?:con\s+hang|con\s+khong|gia|ton\s+kho)\b",
    r"\bshop\s+xac\s+nhan\b",
]


@dataclass(slots=True)
class DraftValidationResult:
    ok: bool
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SubmissionResult:
    ok: bool
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    payload: dict[str, Any] | None = None


def text_key(text: str | None) -> str:
    return strip_accents(normalize_text(text or ""))


def new_draft_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"DRAFT-{now:%Y%m%d}-{next(_DRAFT_COUNTER):04d}"


def _default_shop_id() -> str | None:
    return (
        active_shop_id()
        or os.getenv("SHOP_ID", "").strip()
        or os.getenv("AUTOBIZ_SHOP_ID", "").strip()
        or None
    )


def blank_order_draft(draft_id: str | None = None) -> dict[str, Any]:
    resolved_draft_id = draft_id or new_draft_id()
    return {
        "draft_id": resolved_draft_id,
        "draft_order_id": resolved_draft_id,
        "shop_id": _default_shop_id(),
        "customer_chat_id": None,
        "status": ORDER_STATE_COLLECTING,
        "submitted_to_n8n": False,
        "submission_in_progress": False,
        "submitted_at": None,
        "items": [
            {
                "product_id": None,
                "product_name": None,
                "sku": None,
                "color": None,
                "size": None,
                "quantity": 1,
                "unit_price": None,
                "variant_stock": None,
            }
        ],
        "customer": {
            "name": None,
            "phone": None,
            "address": None,
            "note": None,
        },
        "payment_method": None,
        "subtotal": None,
        "shipping_fee": None,
        "total": None,
    }


def ensure_order_draft(memory: dict[str, Any]) -> dict[str, Any]:
    draft = memory.get("order_draft")
    if not isinstance(draft, dict) or draft.get("status") in {
        ORDER_STATE_CANCELLED,
        ORDER_STATE_LEGACY_SUBMITTED,
    }:
        draft = blank_order_draft()
        memory["order_draft"] = draft
    draft.setdefault("items", [blank_order_draft()["items"][0]])
    if not draft["items"]:
        draft["items"] = [blank_order_draft()["items"][0]]
    draft.setdefault(
        "customer", {"name": None, "phone": None, "address": None, "note": None}
    )
    draft.setdefault("draft_order_id", draft.get("draft_id") or new_draft_id())
    draft.setdefault("draft_id", draft.get("draft_order_id"))
    if not draft.get("shop_id"):
        draft["shop_id"] = _default_shop_id()
    draft.setdefault("customer_chat_id", None)
    draft.setdefault("status", ORDER_STATE_COLLECTING)
    draft.setdefault("submitted_to_n8n", False)
    draft.setdefault("submission_in_progress", False)
    draft.setdefault("submitted_at", None)
    return draft


def active_order_draft(memory: dict[str, Any] | None) -> dict[str, Any] | None:
    if not memory:
        return None
    draft = memory.get("order_draft")
    if not isinstance(draft, dict):
        return None
    if draft.get("status") in {ORDER_STATE_CANCELLED, ORDER_STATE_LEGACY_SUBMITTED}:
        return None
    return draft


def clear_draft(
    conversation_id: str,
    store: MutableMapping[str, dict[str, Any]] | None = None,
) -> bool:
    if store is None:
        from app.service import CONVERSATION_STORE

        store = CONVERSATION_STORE

    memory = store.get(conversation_id)
    if not isinstance(memory, MutableMapping):
        return False

    had_draft = isinstance(memory.get("order_draft"), dict)
    for key in ORDER_DRAFT_RESET_KEYS:
        memory.pop(key, None)
    return had_draft


def first_item(draft: dict[str, Any]) -> dict[str, Any]:
    items = draft.setdefault("items", [])
    if not items:
        items.append(blank_order_draft()["items"][0])
    return items[0]


def set_item_product(item: dict[str, Any], product: dict[str, Any]) -> None:
    item.update(
        {
            "product_id": product.get("product_id"),
            "product_name": product.get("product_name"),
            "unit_price": product.get("effective_price_vnd"),
            "sku": None,
            "color": None,
            "size": None,
            "variant_stock": None,
        }
    )


def set_item_variant(
    item: dict[str, Any], variant: dict[str, Any], product: dict[str, Any] | None = None
) -> None:
    if product:
        item["unit_price"] = product.get("effective_price_vnd")
    variant_size = variant.get("size")
    canonical_size = extract_size(str(variant_size or "")) or variant_size
    item.update(
        {
            "sku": variant.get("sku"),
            "color": variant.get("color"),
            "size": canonical_size,
            "variant_stock": int(variant.get("stock") or 0),
        }
    )


def clear_item_variant(
    item: dict[str, Any], *, color: str | None = None, size: str | None = None
) -> None:
    item.update(
        {
            "sku": None,
            "color": color,
            "size": size,
            "variant_stock": None,
        }
    )


def recalculate_totals(draft: dict[str, Any]) -> None:
    subtotal = 0
    for item in draft.get("items") or []:
        unit_price = item.get("unit_price")
        quantity = item.get("quantity")
        if unit_price is None or quantity is None:
            continue
        subtotal += int(unit_price) * int(quantity)
    draft["subtotal"] = subtotal if subtotal > 0 else None
    draft["shipping_fee"] = None
    draft["total"] = subtotal if subtotal > 0 else None


def extract_quantity(text: str | None) -> int | None:
    key = text_key(text)
    if not key:
        return None

    match = re.search(
        r"\b(\d{1,3})\s*(?:cai|chiec|san pham|sp|ao|mau|bo)\b",
        key,
    )
    if match:
        return int(match.group(1))

    match = re.search(
        r"\b(mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi)\s*(?:cai|chiec|san pham|sp|ao|mau|bo)\b",
        key,
    )
    if match:
        return _NUMBER_WORDS.get(match.group(1))

    if re.fullmatch(r"\d{1,3}", key):
        return int(key)
    if key in _NUMBER_WORDS:
        return _NUMBER_WORDS[key]
    return None


def extract_order_color_size(text: str | None) -> tuple[str | None, str | None]:
    normalized = normalize_text(text or "")
    return extract_color(normalized), extract_size(normalized)


def normalize_phone(text: str | None) -> str:
    return re.sub(r"\D+", "", text or "")


def is_valid_phone(phone: str | None) -> bool:
    phone = phone or ""
    return len(phone) == 10 and phone.startswith("0") and phone.isdigit()


def extract_phone_candidate(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(?:0|\+?84)[\d\s.\-]{8,16}\d", text)
    if not match:
        return None
    phone = normalize_phone(match.group(0))
    if phone.startswith("84") and len(phone) == 11:
        phone = "0" + phone[2:]
    return phone


def _strip_customer_label(text: str) -> str:
    value = text.strip(" ,.-")
    patterns = [
        r"^(?:ten minh la|ten toi la|nguoi nhan la|minh la|toi la)\s+",
        r"^(?:tên mình là|tên tôi là|người nhận là|mình là|tôi là)\s+",
    ]
    key = text_key(value)
    for pattern in patterns:
        if re.match(pattern, key):
            words = value.split()
            return " ".join(words[3:]).strip() if len(words) > 3 else value
    for prefix in [
        "Tên mình là ",
        "Tên tôi là ",
        "Người nhận là ",
        "Mình là ",
        "Tôi là ",
        "ten minh la ",
        "ten toi la ",
        "nguoi nhan la ",
        "minh la ",
        "toi la ",
    ]:
        if value.lower().startswith(prefix.lower()):
            return value[len(prefix) :].strip()
    return value


def extract_customer_bundle(text: str | None) -> dict[str, str | None]:
    raw = (text or "").strip()
    result: dict[str, str | None] = {"name": None, "phone": None, "address": None}
    if not raw:
        return result

    phone = extract_phone_candidate(raw)
    if phone:
        result["phone"] = phone

    parts = [part.strip() for part in re.split(r"[\n,]+", raw) if part.strip()]
    if phone and len(parts) >= 2:
        phone_index = next(
            (
                index
                for index, part in enumerate(parts)
                if normalize_phone(part) == phone
            ),
            None,
        )
        if phone_index is not None:
            before = [
                part
                for part in parts[:phone_index]
                if not extract_phone_candidate(part)
            ]
            after = [
                part
                for part in parts[phone_index + 1 :]
                if not extract_phone_candidate(part)
            ]
            if before:
                result["name"] = _strip_customer_label(" ".join(before))
            if after:
                result["address"] = ", ".join(after)
            return result

    if len(parts) >= 3:
        result["name"] = _strip_customer_label(parts[0])
        result["phone"] = normalize_phone(parts[1])
        result["address"] = ", ".join(parts[2:])
        return result

    if not phone:
        result["name"] = _strip_customer_label(raw)
    return result


def normalize_payment_method(text: str | None) -> str | None:
    key = text_key(text)
    if not key:
        return None
    if re.search(
        r"\b(?:cod|nhan hang|luc nhan|khi nhan|tra tien luc nhan|tra tien khi nhan)\b",
        key,
    ):
        return "cod"
    if re.search(r"\b(?:chuyen khoan|bank transfer|transfer|ngan hang)\b", key):
        return "bank_transfer"
    return None


def is_order_confirmation_message(text: str | None) -> bool:
    key = text_key(text)
    if not key:
        return False
    if any(re.search(pattern, key) for pattern in ORDER_CONFIRMATION_NEGATIVE_PATTERNS):
        return False
    if key in ORDER_CONFIRMATION_EXACT_TEXTS:
        return True
    return bool(
        re.fullmatch(r"(?:ok|oke)\s+(?:xac\s+nhan|chot(?:\s+don)?)", key)
        or re.fullmatch(
            r"(?:minh\s+)?(?:dong\s+y|xac\s+nhan|chot\s+don|dat\s+don|gui\s+don)", key
        )
        or re.fullmatch(r"thong\s+tin\s+dung(?:\s+roi)?", key)
    )


def is_retry(text: str | None) -> bool:
    key = text_key(text)
    return key in {"gui lai", "thu lai", "retry"} or bool(
        re.search(r"\b(?:gui lai|thu lai|retry)\b", key)
    )


def is_cancel(text: str | None) -> bool:
    key = text_key(text)
    return bool(
        re.search(
            r"\b(?:thoi.*khong mua|huy don|bo don|khong dat nua|khong mua nua|huy)\b",
            key,
        )
    )


def looks_like_browsing_shift(text: str | None) -> bool:
    key = text_key(text)
    return bool(
        re.search(
            r"\b(?:xem mau khac|cho toi xem|cho minh xem|tim mau khac|mau khac|san pham khac|tu van lai|doi san pham)\b",
            key,
        )
    )


def product_reference_index(text: str | None, count: int) -> int | None:
    key = text_key(text)
    if count <= 0:
        return None
    if re.search(r"\b(?:mau|cai|san pham)\s+(?:dau tien|so 1|1)\b", key):
        return 0
    if re.search(r"\b(?:mau|cai|san pham)\s+(?:thu hai|so 2|2)\b", key):
        return 1 if count >= 2 else None
    if re.search(r"\b(?:mau|cai|san pham)\s+(?:thu ba|so 3|3)\b", key):
        return 2 if count >= 3 else None
    if re.search(r"\b(?:mau|cai|san pham)\s+cuoi\b", key):
        return count - 1
    return None


def requested_current_product(text: str | None) -> bool:
    key = text_key(text)
    return bool(
        re.search(
            r"\b(?:mau nay|cai nay|san pham nay|lay\s+\d+\s+cai|lay mot cai|chot mau nay)\b",
            key,
        )
    )


def missing_order_action(draft: dict[str, Any]) -> str | None:
    item = first_item(draft)
    if not item.get("product_id"):
        return "choose_order_product"
    if not item.get("color") and not item.get("size"):
        return "choose_order_color_and_size"
    if not item.get("color"):
        return "choose_order_color"
    if not item.get("size"):
        return "choose_order_size"
    if not item.get("sku") or item.get("variant_stock") is None:
        return "choose_order_color_and_size"
    if not item.get("quantity"):
        return "choose_order_quantity"
    customer = draft.get("customer") or {}
    if not customer.get("name"):
        return "collect_customer_name"
    if not customer.get("phone"):
        return "collect_customer_phone"
    if not customer.get("address"):
        return "collect_customer_address"
    if not draft.get("payment_method"):
        return "choose_payment_method"
    return None


def money(vnd: int | float | None) -> str:
    if vnd is None:
        return "chưa rõ"
    return f"{int(vnd):,}".replace(",", ".") + "đ"


def payment_label(payment_method: str | None) -> str:
    if payment_method == "cod":
        return "COD"
    if payment_method == "bank_transfer":
        return "Chuyển khoản"
    return "Chưa chọn"


def build_summary(draft: dict[str, Any]) -> str:
    item = first_item(draft)
    customer = draft.get("customer") or {}
    note = customer.get("note") or "Không có"
    lines = [
        "Dạ, bạn kiểm tra giúp mình đơn nháp:",
        "",
        f"- Sản phẩm: {item.get('product_name') or 'Chưa chọn'}",
        f"- SKU: {item.get('sku') or 'Chưa chọn'}",
        f"- Màu: {item.get('color') or 'Chưa chọn'}",
        f"- Size: {item.get('size') or 'Chưa chọn'}",
        f"- Số lượng: {item.get('quantity') or 1}",
        f"- Đơn giá: {money(item.get('unit_price'))}",
        f"- Tạm tính: {money(draft.get('subtotal'))}",
        f"- Người nhận: {customer.get('name') or 'Chưa có'}",
        f"- Số điện thoại: {customer.get('phone') or 'Chưa có'}",
        f"- Địa chỉ: {customer.get('address') or 'Chưa có'}",
        f"- Thanh toán: {payment_label(draft.get('payment_method'))}",
        f"- Ghi chú: {note}",
        "- Phí vận chuyển: Shop sẽ xác nhận sau",
        "",
        "Bạn kiểm tra thông tin đã đúng chưa ạ? Nếu đúng, hãy nhắn “Xác nhận”. "
        "Nếu cần sửa, bạn có thể nhắn ví dụ “Đổi địa chỉ...” hoặc “Đổi size...”.",
    ]
    return "\n".join(lines)


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def validate_draft_order(draft: Mapping[str, Any]) -> DraftValidationResult:
    missing: list[str] = []
    errors: list[str] = []
    customer = draft.get("customer") or {}
    items = list(draft.get("items") or [])

    required_top_level = {
        "shop_id": draft.get("shop_id"),
        "customer_chat_id": draft.get("customer_chat_id"),
        "customer_name": draft.get("customer_name") or customer.get("name"),
        "phone": draft.get("phone") or customer.get("phone"),
        "address": draft.get("address") or customer.get("address"),
        "payment_method": draft.get("payment_method"),
    }
    for field_name, value in required_top_level.items():
        if _is_blank(value):
            missing.append(field_name)

    if not items:
        missing.append("items")

    expected_total = 0
    for index, item in enumerate(items):
        prefix = f"items[{index}]"
        for field_name in ["sku", "product_name", "quantity", "unit_price"]:
            if _is_blank(item.get(field_name)):
                missing.append(f"{prefix}.{field_name}")

        try:
            quantity = int(item.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0
        try:
            unit_price = int(float(item.get("unit_price") or 0))
        except (TypeError, ValueError):
            unit_price = -1

        if quantity <= 0:
            errors.append(f"{prefix}.quantity must be > 0")
        if unit_price < 0:
            errors.append(f"{prefix}.unit_price must be >= 0")
        if quantity > 0 and unit_price >= 0:
            expected_total += quantity * unit_price

    subtotal = draft.get("subtotal")
    total = draft.get("total")
    if subtotal is None:
        missing.append("subtotal")
    if total is None:
        missing.append("total")
    else:
        try:
            if int(total) != expected_total:
                errors.append("total must equal unit_price * quantity")
        except (TypeError, ValueError):
            errors.append("total must be numeric")

    return DraftValidationResult(
        ok=not missing and not errors,
        missing_fields=list(dict.fromkeys(missing)),
        errors=errors,
    )


def build_n8n_payload(
    draft: dict[str, Any],
    *,
    conversation_id: str,
    channel: str = "telegram",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(timezone.utc)
    customer = draft.get("customer") or {}
    payload_items = []
    for item in draft.get("items") or []:
        quantity = int(item.get("quantity") or 0)
        unit_price = int(float(item.get("unit_price") or 0))
        variant = item.get("variant")
        if not variant:
            variant = (
                " / ".join(
                    str(value)
                    for value in [item.get("color"), item.get("size")]
                    if str(value or "").strip()
                )
                or None
            )
        payload_items.append(
            {
                "product_id": item.get("product_id"),
                "product_name": item.get("product_name"),
                "sku": item.get("sku"),
                "variant": variant,
                "color": item.get("color"),
                "size": item.get("size"),
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": quantity * unit_price,
                "stock_at_customer_confirmation": item.get("variant_stock"),
            }
        )
    return {
        "event": "draft_order_confirmed",
        "draft_order_id": draft.get("draft_order_id") or draft.get("draft_id"),
        "draft_id": draft.get("draft_id"),
        "shop_id": draft.get("shop_id"),
        "customer_chat_id": draft.get("customer_chat_id") or conversation_id,
        "conversation_id": conversation_id,
        "channel": channel,
        "customer_name": draft.get("customer_name") or customer.get("name"),
        "phone": draft.get("phone") or customer.get("phone"),
        "address": draft.get("address") or customer.get("address"),
        "items": payload_items,
        "subtotal": draft.get("subtotal"),
        "shipping_fee": draft.get("shipping_fee"),
        "total": draft.get("total"),
        "payment_method": draft.get("payment_method"),
        "note": customer.get("note"),
        "source": "telegram_customer_bot",
        "customer_confirmed": True,
        "status": "pending_owner_review",
        "created_at": created_at.isoformat(),
    }


def _webhook_url() -> str:
    return (
        os.getenv("N8N_ORDER_WEBHOOK_URL", "").strip()
        or os.getenv("N8N_DRAFT_ORDER_WEBHOOK_URL", "").strip()
    )


def _webhook_secret() -> str:
    return (
        os.getenv("N8N_ORDER_WEBHOOK_SECRET", "").strip()
        or os.getenv("N8N_DRAFT_ORDER_WEBHOOK_SECRET", "").strip()
        or os.getenv("AUTOBIZ_WEBHOOK_SECRET", "").strip()
    )


def _webhook_timeout_seconds() -> int:
    raw_value = (
        os.getenv("N8N_ORDER_WEBHOOK_TIMEOUT_SECONDS", "").strip()
        or os.getenv("N8N_DRAFT_ORDER_WEBHOOK_TIMEOUT_SECONDS", "").strip()
    )
    try:
        return int(raw_value) if raw_value else 15
    except ValueError:
        return 15


def submit_draft_order_to_n8n(draft_order: Mapping[str, Any]) -> SubmissionResult:
    payload = build_n8n_payload(
        dict(draft_order),
        conversation_id=str(draft_order.get("customer_chat_id") or ""),
    )
    url = _webhook_url()
    if not url:
        logger.warning("Draft order webhook URL is not configured.")
        return SubmissionResult(
            ok=False,
            error_type="missing_webhook_url",
            error_message="Draft order webhook URL is not configured.",
            payload=payload,
        )

    try:
        headers = {
            "Content-Type": "application/json",
        }

        secret = _webhook_secret()
        if secret:
            headers["x-autobiz-secret"] = secret

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=_webhook_timeout_seconds(),
        )
        status_code = response.status_code
        response.raise_for_status()
        logger.info(
            "Draft order webhook submitted successfully. status_code=%s", status_code
        )
        return SubmissionResult(ok=True, status_code=status_code, payload=payload)
    except requests.Timeout as exc:
        logger.warning("Draft order webhook timed out: %s", exc.__class__.__name__)
        return SubmissionResult(
            ok=False,
            error_type="timeout",
            error_message=str(exc),
            payload=payload,
        )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        logger.warning(
            "Draft order webhook returned HTTP error. status_code=%s error_type=%s",
            status_code,
            exc.__class__.__name__,
        )
        return SubmissionResult(
            ok=False,
            status_code=status_code,
            error_type="http_error",
            error_message=str(exc),
            payload=payload,
        )
    except requests.RequestException as exc:
        logger.warning("Draft order webhook failed: %s", exc.__class__.__name__)
        return SubmissionResult(
            ok=False,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            payload=payload,
        )


def masked_order_draft(draft: dict[str, Any] | None) -> dict[str, Any] | None:
    if not draft:
        return None
    masked = deepcopy(draft)
    phone = ((masked.get("customer") or {}).get("phone") or "").strip()
    if len(phone) >= 7:
        masked["customer"]["phone"] = f"{phone[:4]}***{phone[-3:]}"
    return masked
