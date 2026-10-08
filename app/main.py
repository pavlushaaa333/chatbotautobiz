from __future__ import annotations

import hmac
import os
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ParseRequest,
    RAGRebuildRequest,
    RAGSearchRequest,
    SearchRequest,
)

app = FastAPI(title="AutoBiz MVP Source Open", version="0.1.0")


def _service():
    from app.service import get_service
    try:
        return get_service()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class OrderStatusPayload(BaseModel):
    event: str
    draft_order_id: str
    shop_id: str | None = None
    customer_chat_id: str
    customer_name: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    total: int | None = None
    payment_method: str | None = None
    status: str | None = None


def _webhook_secret() -> str:
    return (
        os.getenv("AUTOBIZ_WEBHOOK_SECRET", "").strip()
        or os.getenv("N8N_ORDER_WEBHOOK_SECRET", "").strip()
    )


def _telegram_bot_token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _normalize_telegram_chat_id(customer_chat_id: str) -> str:
    if customer_chat_id.startswith("telegram_"):
        return customer_chat_id.replace("telegram_", "", 1)
    return customer_chat_id


def _money(value: int | float | None) -> str:
    if value is None:
        return "chưa rõ"
    return f"{int(value):,}".replace(",", ".") + "đ"


def _payment_label(payment_method: str | None) -> str:
    if payment_method == "cod":
        return "COD"
    if payment_method == "bank_transfer":
        return "Chuyển khoản"
    return payment_method or "Chưa rõ"


def _build_order_approved_message(payload: OrderStatusPayload) -> str:
    item = payload.items[0] if payload.items else {}

    product_name = item.get("product_name") or "Sản phẩm"
    variant = item.get("variant") or "Chưa rõ"
    quantity = item.get("quantity") or 1

    return (
        "Dạ đơn hàng của bạn đã được shop xác nhận ạ 🎉\n\n"
        f"Mã đơn: {payload.draft_order_id}\n"
        f"Sản phẩm: {product_name}\n"
        f"Phân loại: {variant}\n"
        f"Số lượng: {quantity}\n"
        f"Tổng tiền: {_money(payload.total)}\n"
        f"Thanh toán: {_payment_label(payload.payment_method)}\n\n"
        "Shop sẽ chuẩn bị hàng và liên hệ giao sớm cho bạn ạ."
    )


def _build_order_rejected_message(payload: OrderStatusPayload) -> str:
    return (
        "Dạ đơn hàng của bạn chưa được shop xác nhận ạ.\n\n"
        f"Mã đơn: {payload.draft_order_id}\n"
        "Shop sẽ kiểm tra lại và liên hệ bạn nếu cần đổi sản phẩm hoặc thông tin đơn hàng."
    )


def _build_order_stock_failed_message(payload: OrderStatusPayload) -> str:
    item = payload.items[0] if payload.items else {}

    product_name = item.get("product_name") or "Sản phẩm"
    variant = item.get("variant") or "Chưa rõ"
    quantity = item.get("quantity") or 1

    return (
        "Dạ rất tiếc, đơn hàng của bạn chưa thể xác nhận vì sản phẩm hiện không còn đủ số lượng trong kho ạ.\n\n"
        f"Mã đơn: {payload.draft_order_id}\n"
        f"Sản phẩm: {product_name}\n"
        f"Phân loại: {variant}\n"
        f"Số lượng bạn đặt: {quantity}\n\n"
        "Shop sẽ liên hệ để đổi mẫu hoặc xử lý lại đơn cho bạn."
    )


def _send_telegram_message(chat_id: str, text: str) -> dict[str, Any]:
    token = _telegram_bot_token()
    if not token:
        raise HTTPException(status_code=500, detail="Missing TELEGRAM_BOT_TOKEN")

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Telegram sendMessage failed",
        ) from None

    if not isinstance(result, dict) or result.get("ok") is not True:
        raise HTTPException(status_code=502, detail="Telegram delivery was not confirmed")
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
def parse(request: ParseRequest) -> dict:
    return _service().parse(request.message)


@app.post("/search-products")
def search_products(request: SearchRequest) -> dict:
    products = _service().search_products(request.criteria, top_k=request.top_k)
    return {
        "products": products,
        "product_ids": [product["product_id"] for product in products],
    }


@app.post("/rag/rebuild")
def rebuild_rag(request: RAGRebuildRequest = RAGRebuildRequest()) -> dict:
    return _service().rebuild_rag_index(force_rebuild=request.force_rebuild)


@app.post("/rag/search")
def search_rag(request: RAGSearchRequest) -> dict:
    results = _service().search_rag(
        request.query,
        top_k=request.top_k,
        filters=request.filters,
    )
    return {"results": results}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    return _service().chat(
        request.message,
        top_k=request.top_k,
        conversation_id=request.conversation_id,
    )


@app.post("/webhooks/order-status")
def order_status_webhook(
    payload: OrderStatusPayload,
    x_autobiz_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    if os.getenv("AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED", "").lower() != "true":
        raise HTTPException(status_code=410, detail="Use authenticated customer-order-results")
    expected_secret = _webhook_secret()

    if not expected_secret:
        raise HTTPException(status_code=503, detail="Webhook authentication is unavailable")
    if not hmac.compare_digest(x_autobiz_secret or "", expected_secret):
        raise HTTPException(status_code=401, detail="Invalid x-autobiz-secret")

    chat_id = _normalize_telegram_chat_id(payload.customer_chat_id)

    if payload.event == "order_approved":
        message = _build_order_approved_message(payload)
    elif payload.event == "order_rejected":
        message = _build_order_rejected_message(payload)
    elif payload.event in {"order_stock_failed", "order_not_enough_stock"}:
        message = _build_order_stock_failed_message(payload)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported event: {payload.event}"
        )

    telegram_result = _send_telegram_message(chat_id, message)

    return {
        "ok": True,
        "sent": True,
        "event": payload.event,
        "draft_order_id": payload.draft_order_id,
        "customer_chat_id": payload.customer_chat_id,
        "telegram_chat_id": chat_id,
        "telegram_result": telegram_result,
    }


@app.post("/webhooks/customer-order-results")
async def customer_order_result(request: Request):
    from app.customer_result import AuthenticationUnavailable, deliver_result
    raw = await request.body()
    if len(raw) > 8192:
        raise HTTPException(status_code=413, detail="Customer result too large")
    try:
        return deliver_result(await request.json(), _send_telegram_message)
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Customer result rejected") from None
    except AuthenticationUnavailable:
        raise HTTPException(status_code=503, detail="Customer result authentication unavailable") from None
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Customer result requires reconciliation") from None
