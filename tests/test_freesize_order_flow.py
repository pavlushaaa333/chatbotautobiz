from __future__ import annotations

from typing import Any

from app.normalizer import extract_size
from app.order_draft_service import (
    ensure_order_draft,
    extract_order_color_size,
    first_item,
    set_item_product,
)
from app.service import CONVERSATION_STORE, ChatbotService


PRODUCT = {
    "product_id": "P-FREE",
    "product_name": "\u00c1o s\u01a1 mi tr\u1eafng n\u1eef",
    "effective_price_vnd": 189000,
}


class FakeRagService:
    def status(self) -> dict[str, Any]:
        return {}


class FakeVariantSearchEngine:
    def __init__(self, *, has_freesize_variant: bool = True) -> None:
        self.has_freesize_variant = has_freesize_variant
        self.calls: list[dict[str, Any]] = []

    def check_variant_availability(self, criteria: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(criteria))
        if (
            self.has_freesize_variant
            and criteria.get("color") == "tr\u1eafng"
            and criteria.get("size") == "Free size"
        ):
            return {
                "status": "in_stock",
                "product": PRODUCT,
                "variant": {
                    "sku": "SKU-WHITE-FREE",
                    "color": "Tr\u1eafng",
                    "size": "Free size",
                    "stock": 4,
                    "status": "in_stock",
                },
                "alternatives": [],
            }
        return {
            "status": "variant_not_found",
            "product": PRODUCT,
            "variant": None,
            "alternatives": [],
        }


def _service(search_engine: FakeVariantSearchEngine) -> ChatbotService:
    service = ChatbotService.__new__(ChatbotService)
    service.active_shop_id = "test-shop"
    service.product_data_source = "csv"
    service.search_engine = search_engine
    service.rag_service = FakeRagService()
    return service


def _context_with_product() -> dict[str, Any]:
    context: dict[str, Any] = {}
    draft = ensure_order_draft(context)
    set_item_product(first_item(draft), PRODUCT)
    return context


def test_extract_size_supports_freesize_aliases() -> None:
    assert extract_size("freesize") == "Freesize"
    assert extract_size("free size") == "Freesize"
    assert extract_size("size free") == "Freesize"
    assert extract_size("FS") == "Freesize"
    assert extract_size("size M") == "M"
    assert extract_size("size S") == "S"
    assert extract_size("size L") == "L"
    assert extract_size("size XL") == "XL"


def test_extract_order_color_size_with_freesize() -> None:
    assert extract_order_color_size("ok, l\u1ea5y tr\u1eafng freesize") == (
        "tr\u1eafng",
        "Freesize",
    )


def test_order_variant_freesize_in_stock_goes_to_quantity() -> None:
    search_engine = FakeVariantSearchEngine(has_freesize_variant=True)
    service = _service(search_engine)
    conversation_id = "test-freesize-in-stock"
    CONVERSATION_STORE.pop(conversation_id, None)
    context = _context_with_product()

    result = service._handle_order_variant_input(
        conversation_id,
        context,
        {"intent": "order_draft"},
        "ok, l\u1ea5y tr\u1eafng freesize",
    )
    item = result["order_draft"]["items"][0]

    assert search_engine.calls[0]["size"] == "Freesize"
    assert result["pending_action"] == "choose_order_quantity"
    assert item["size"] == "Freesize"
    assert item["color"] == "Tr\u1eafng"
    assert item["sku"] == "SKU-WHITE-FREE"
    assert item["variant_stock"] == 4
    assert "size n\u00e0o" not in result["reply"].lower()
    assert "bao nhi\u00eau" in result["reply"].lower()


def test_order_variant_freesize_not_found_does_not_assign_sku() -> None:
    search_engine = FakeVariantSearchEngine(has_freesize_variant=False)
    service = _service(search_engine)
    conversation_id = "test-freesize-not-found"
    CONVERSATION_STORE.pop(conversation_id, None)
    context = _context_with_product()

    result = service._handle_order_variant_input(
        conversation_id,
        context,
        {"intent": "order_draft"},
        "ok, l\u1ea5y tr\u1eafng freesize",
    )
    item = result["order_draft"]["items"][0]

    assert result["criteria"]["availability"]["status"] != "in_stock"
    assert item["sku"] is None
    assert "tr\u1eafng" in result["reply"].lower()
    assert "Freesize" in result["reply"]
