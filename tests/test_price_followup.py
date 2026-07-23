from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from app.parser import parse_customer_message
from app.search_engine import PostgresProductSearchEngine, ProductSearchEngine
from app.service import CONVERSATION_STORE, ChatbotService


AO_SO_MI = "\u00c1o s\u01a1 mi"
NU = "n\u1eef"
DI_CHOI = "\u0111i ch\u01a1i"


class _DummyRag:
    def status(self) -> dict[str, Any]:
        return {}

    def retrieve_contexts(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


class _FakePostgresRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def search_products(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return self.rows


def _product_row(
    product_id: str,
    product_name: str,
    price: int,
    *,
    product_type: str = AO_SO_MI,
    gender: str = NU,
    use_case: str = DI_CHOI,
) -> dict[str, object]:
    text = " ".join([product_name, product_type, gender, use_case])
    return {
        "product_id": product_id,
        "product_name": product_name,
        "category_code": "Fashion",
        "category_name": "Fashion",
        "product_type": product_type,
        "brand": "AutoBiz",
        "tags": f"{product_type}|{gender}|{use_case}",
        "short_description": product_name,
        "shop_id": "test-shop",
        "sku": product_id,
        "color": "",
        "size": "",
        "quantity": 5,
        "available_qty": 5,
        "price_vnd": price,
        "sale_price_vnd": price,
        "sale_price": price,
        "selling_price": price,
        "effective_price_vnd": price,
        "stock_total": 5,
        "listing_status": "active",
        "inventory_status": "active",
        "status": "active",
        "source": "test",
        "gender": gender,
        "use_case": use_case,
        "style": "",
        "rating": 4.5,
        "sold_30d": 1,
        "search_text": text,
    }


def _variants() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "product_id",
            "size",
            "color",
            "available_qty",
            "status",
            "listing_status",
            "inventory_status",
        ]
    )


def _service(products: pd.DataFrame) -> ChatbotService:
    service = ChatbotService.__new__(ChatbotService)
    service.catalog_source = "csv"
    service.product_data_source = "csv"
    service.active_shop_id = None
    service.data_store = SimpleNamespace(synonyms={})
    service.search_engine = ProductSearchEngine(products, _variants())
    service.rag_service = _DummyRag()
    return service


def test_parse_below_price() -> None:
    criteria = parse_customer_message("d\u01b0\u1edbi 300k")

    assert criteria["budget_max_vnd"] == 300000
    assert criteria.get("budget_min_vnd") is None
    assert criteria["price_direction"] == "below"


def test_parse_above_price() -> None:
    criteria = parse_customer_message("tr\u00ean 300k")

    assert criteria["budget_min_vnd"] == 300000
    assert criteria.get("budget_max_vnd") is None
    assert criteria["price_direction"] == "above"


def test_parse_at_least_price() -> None:
    criteria = parse_customer_message("t\u1eeb 300k tr\u1edf l\u00ean")

    assert criteria["budget_min_vnd"] == 300000
    assert criteria.get("budget_max_vnd") is None
    assert criteria["price_direction"] == "at_least"


def test_gate_does_not_treat_nao_as_out_of_scope_product() -> None:
    service = _service(
        pd.DataFrame([_product_row("P1", "\u00c1o s\u01a1 mi tr\u1eafng n\u1eef", 189000)])
    )

    criteria = service.parse("c\u00f3 c\u00e1i n\u00e0o tr\u00ean 300k kh\u00f4ng")

    assert criteria["catalog_coverage"] != "unsupported"
    assert "n\u00e0o" not in criteria.get("out_of_scope_items", [])
    assert criteria.get("requested_product_group") in {None, ""}


def test_search_filters_price_above_exclusively() -> None:
    engine = ProductSearchEngine(
        pd.DataFrame(
            [
                _product_row("A", "Product A", 189000),
                _product_row("B", "Product B", 299000),
                _product_row("C", "Product C", 301000),
                _product_row("D", "Product D", 499000),
            ]
        ),
        _variants(),
    )

    results = engine.search(
        {
            "category_code": "Fashion",
            "budget_min_vnd": 300000,
            "price_direction": "above",
        },
        top_k=10,
    )

    assert [product["product_name"] for product in results] == ["Product C", "Product D"]


def test_search_filters_price_at_least_inclusively() -> None:
    engine = ProductSearchEngine(
        pd.DataFrame(
            [
                _product_row("B", "Product B", 299000),
                _product_row("C", "Product C", 300000),
                _product_row("D", "Product D", 499000),
            ]
        ),
        _variants(),
    )

    results = engine.search(
        {
            "category_code": "Fashion",
            "budget_min_vnd": 300000,
            "price_direction": "at_least",
        },
        top_k=10,
    )

    assert [product["product_name"] for product in results] == ["Product C", "Product D"]


def test_postgres_engine_filters_price_above_after_repository_fetch() -> None:
    engine = PostgresProductSearchEngine(
        _FakePostgresRepo(
            [
                {"product_id": "A", "product_name": "Product A", "price": 189000, "available_qty": 5},
                {"product_id": "B", "product_name": "Product B", "price": 299000, "available_qty": 5},
                {"product_id": "C", "product_name": "Product C", "price": 301000, "available_qty": 5},
                {"product_id": "D", "product_name": "Product D", "price": 499000, "available_qty": 5},
            ]
        )
    )

    results = engine.search(
        {"budget_min_vnd": 300000, "price_direction": "above"},
        top_k=10,
    )

    assert [product["product_name"] for product in results] == ["Product C", "Product D"]


def test_price_followup_merges_memory_and_replaces_old_budget_direction() -> None:
    service = _service(
        pd.DataFrame([_product_row("P1", "\u00c1o s\u01a1 mi tr\u1eafng n\u1eef", 189000)])
    )
    conversation_id = "price-followup-test"
    CONVERSATION_STORE.pop(conversation_id, None)

    for message in [
        "T\u00f4i mu\u1ed1n t\u00ecm \u00e1o s\u01a1 mi",
        "cho n\u1eef nh\u00e9",
        "\u0111i ch\u01a1i",
        "d\u01b0\u1edbi 300k",
    ]:
        service.chat(message, conversation_id=conversation_id)

    result = service.chat(
        "c\u00f3 c\u00e1i n\u00e0o tr\u00ean 300k kh\u00f4ng",
        conversation_id=conversation_id,
    )
    criteria = result["criteria"]
    memory = result["conversation_memory"]

    assert criteria["intent"] == "conversation_followup"
    assert criteria["product_type"] == AO_SO_MI
    assert criteria["product_types"] == [AO_SO_MI]
    assert criteria["gender"] == NU
    assert criteria["use_case"] == DI_CHOI
    assert criteria["budget_min_vnd"] == 300000
    assert criteria.get("budget_max_vnd") is None
    assert criteria["price_direction"] == "above"
    assert criteria["catalog_coverage"] == "supported"
    assert criteria.get("out_of_scope_items") == []
    assert memory["budget_min_vnd"] == 300000
    assert memory.get("budget_max_vnd") is None
    assert "tr\u00ean 300.000\u0111" in result["reply"]
    assert "d\u01b0\u1edbi 300.000\u0111" not in result["reply"]


def test_direct_price_query_searches_catalog_without_context() -> None:
    service = _service(
        pd.DataFrame(
            [
                _product_row("A", "Product A", 189000),
                _product_row("C", "Product C", 301000),
                _product_row("D", "Product D", 499000),
            ]
        )
    )

    result = service.chat(
        "shop c\u00f3 s\u1ea3n ph\u1ea9m n\u00e0o tr\u00ean 300k kh\u00f4ng",
        conversation_id="direct-price-query",
    )

    assert result["criteria"]["catalog_coverage"] != "unsupported"
    assert result["criteria"]["budget_min_vnd"] == 300000
    assert result["criteria"]["price_direction"] == "above"
    assert [product["product_name"] for product in result["products"]] == [
        "Product C",
        "Product D",
    ]
