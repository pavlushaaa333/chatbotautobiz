from __future__ import annotations

import pandas as pd

from app.response_generator import build_reply
from app.search_engine import ProductSearchEngine
from app.service import CONVERSATION_STORE, ChatbotService


CHAN_VAY = "Ch\u00e2n v\u00e1y"
VAY = "V\u00e1y"
QUAN_KAKI = "Qu\u1ea7n kaki"
CHAN_VAY_NAME = "Ch\u00e2n v\u00e1y be c\u00f4ng s\u1edf"
VAY_NAME = "V\u00e1y \u0111en d\u00e1ng A"
SECOND_VAY_NAME = "V\u00e1y xanh c\u00f4ng s\u1edf"
AO_SO_MI_NAME = "\u00c1o s\u01a1 mi tr\u1eafng n\u1eef"
BLAZER_NAME = "Blazer kem n\u1eef"


def _product_row(
    product_id: str,
    product_name: str,
    product_type: str,
    stock_total: int,
    *,
    color: str = "",
    price: int = 299000,
    gender: str = "",
    use_case: str = "",
    style: str = "",
) -> dict[str, object]:
    tags = "|".join(
        item for item in [product_type, color, gender, use_case, style] if item
    )
    search_text = " ".join(
        item
        for item in [product_name, product_type, color, gender, use_case, style]
        if item
    )
    return {
        "product_id": product_id,
        "product_name": product_name,
        "category_code": "Fashion",
        "category_name": "Fashion",
        "product_type": product_type,
        "brand": "AutoBiz",
        "tags": tags,
        "short_description": product_name,
        "shop_id": "test-shop",
        "sku": product_id,
        "color": color,
        "size": "",
        "quantity": stock_total,
        "available_qty": stock_total,
        "price_vnd": price,
        "sale_price_vnd": price,
        "sale_price": price,
        "selling_price": price,
        "effective_price_vnd": price,
        "stock_total": stock_total,
        "listing_status": "active",
        "inventory_status": "active",
        "status": "active",
        "source": "test",
        "gender": gender,
        "use_case": use_case,
        "style": style,
        "rating": 4.5,
        "sold_30d": 1,
        "search_text": search_text,
    }


def _products(extra_rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows = [
        _product_row(
            "P001",
            CHAN_VAY_NAME,
            CHAN_VAY,
            0,
            color="be",
            price=249000,
            gender="n\u1eef",
            use_case="\u0111i l\u00e0m",
        ),
        _product_row(
            "P002",
            VAY_NAME,
            VAY,
            8,
            color="\u0111en",
            price=299000,
            gender="n\u1eef",
            use_case="\u0111i l\u00e0m",
        ),
        _product_row(
            "P003",
            AO_SO_MI_NAME,
            "\u00c1o s\u01a1 mi",
            4,
            color="tr\u1eafng",
            price=189000,
            gender="n\u1eef",
            use_case="\u0111i l\u00e0m",
        ),
        _product_row(
            "P004",
            BLAZER_NAME,
            "Blazer",
            6,
            color="kem",
            price=499000,
            gender="n\u1eef",
            use_case="\u0111i l\u00e0m",
        ),
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return pd.DataFrame(rows)


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


def _service(products: pd.DataFrame | None = None) -> ChatbotService:
    service = ChatbotService.__new__(ChatbotService)
    service.product_data_source = "csv"
    service.search_engine = ProductSearchEngine(
        products if products is not None else _products(), _variants()
    )
    return service


def _without_rag(service: ChatbotService) -> None:
    service._attach_rag_contexts = lambda *_args, **_kwargs: []


def _conversation_followup(message: str) -> dict[str, object]:
    return {
        "raw_message": message,
        "normalized_message": message,
        "intent": "conversation_followup",
    }


def _seed_selected_vay_context(conversation_id: str) -> None:
    CONVERSATION_STORE[conversation_id] = {
        "category_code": "Fashion",
        "product_type": VAY,
        "last_product_type": VAY,
        "gender": "n\u1eef",
        "use_case": "\u0111i l\u00e0m",
        "style": "\u0111i l\u00e0m",
        "budget_max_vnd": 300000,
        "selected_product_id": "P002",
        "selected_variant_sku": "P002",
        "last_product_ids": ["P002"],
        "rejected_product_ids": [],
    }


def _alternative_criteria(message: str = "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c") -> dict[str, object]:
    return {
        "raw_message": message,
        "normalized_message": message,
        "intent": "request_alternative_product",
        "response_mode": "alternative_product",
    }


def _parsed_catalog_browsing(message: str) -> dict[str, object]:
    return {
        "raw_message": message,
        "normalized_message": message,
        "intent": "product_search",
        "response_mode": "catalog_browsing",
        "category_code": None,
        "product_type": None,
        "product_types": [],
        "must_be_in_stock": True,
        "need_clarification": True,
        "clarification_questions": [
            "B\u1ea1n mu\u1ed1n t\u00ecm nh\u00f3m s\u1ea3n ph\u1ea9m n\u00e0o trong catalog hi\u1ec7n t\u1ea1i c\u1ee7a shop?"
        ],
    }


def _seed_browse_available_context(service: ChatbotService, conversation_id: str) -> None:
    criteria = {
        "intent": "product_search",
        "category_code": "Fashion",
        "product_type": CHAN_VAY,
        "product_types": [CHAN_VAY],
        "response_mode": "product_type_out_of_stock",
    }
    pending_action = service._store_pending_action(conversation_id, criteria, [])
    assert pending_action == "browse_available_products"


def test_empty_search_exact_type_out_of_stock_response() -> None:
    service = _service()
    criteria = {
        "category_code": "Fashion",
        "product_type": CHAN_VAY,
        "product_types": [CHAN_VAY],
    }

    products = service.search_products(criteria, top_k=5)
    products = service._classify_empty_product_search(criteria, products)
    reply = build_reply(criteria, products)

    assert products == []
    assert criteria["response_mode"] == "product_type_out_of_stock"
    assert [(item["product_name"], item["product_type"], item["stock_total"]) for item in criteria["out_of_stock_products"]] == [
        (CHAN_VAY_NAME, CHAN_VAY, 0)
    ]
    assert "h\u1ebft h\u00e0ng" in reply.lower()
    assert "n\u1edbi ng\u00e2n s\u00e1ch" not in reply.lower()
    assert "\u0111\u1ed5i m\u00e0u/size" not in reply.lower()
    assert VAY_NAME not in reply


def test_empty_search_exact_type_not_found_response() -> None:
    service = _service()
    criteria = {
        "category_code": "Fashion",
        "product_type": QUAN_KAKI,
        "product_types": [QUAN_KAKI],
    }

    products = service.search_products(criteria, top_k=5)
    products = service._classify_empty_product_search(criteria, products)
    reply = build_reply(criteria, products)

    assert products == []
    assert criteria["response_mode"] == "product_type_not_found"
    assert "shop ch\u01b0a c\u00f3" in reply.lower()


def test_empty_search_exact_type_has_stock_but_no_match_response() -> None:
    service = _service()
    criteria = {
        "category_code": "Fashion",
        "product_type": VAY,
        "product_types": [VAY],
        "color": "\u0111\u1ecf",
    }

    products = service._classify_empty_product_search(criteria, [])
    reply = build_reply(criteria, products)

    assert products == []
    assert criteria["response_mode"] == "product_search_no_match"
    assert "\u0111\u1ed5i m\u00e0u" in reply.lower()
    assert "h\u1ebft h\u00e0ng" not in reply.lower()


def test_exact_vay_still_returns_in_stock_product() -> None:
    service = _service()
    criteria = {
        "category_code": "Fashion",
        "product_type": VAY,
        "product_types": [VAY],
    }

    products = service.search_products(criteria, top_k=5)
    products = service._classify_empty_product_search(criteria, products)

    assert [(item["product_name"], item["product_type"], item["stock_total"]) for item in products] == [
        (VAY_NAME, VAY, 8)
    ]
    assert criteria.get("response_mode") != "product_type_out_of_stock"


def test_alternative_product_returns_second_same_type_when_available() -> None:
    products = _products(
        [
            _product_row(
                "P005",
                SECOND_VAY_NAME,
                VAY,
                3,
                color="xanh",
                price=289000,
                gender="n\u1eef",
                use_case="\u0111i l\u00e0m",
            )
        ]
    )
    service = _service(products)
    _without_rag(service)
    conversation_id = "test-alt-same-type-found"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_selected_vay_context(conversation_id)

    result = service._handle_alternative_product(
        "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c",
        top_k=5,
        conversation_id=conversation_id,
        criteria=_alternative_criteria(),
    )

    product_names = [product["product_name"] for product in result["products"]]
    assert SECOND_VAY_NAME in product_names
    assert VAY_NAME not in product_names
    assert result["pending_action"] != "confirm_cross_type_alternative"
    assert result["criteria"]["product_type"] == VAY
    assert result["conversation_memory"]["product_type"] == VAY


def test_alternative_product_asks_before_cross_type_when_same_type_not_found() -> None:
    service = _service()
    _without_rag(service)
    conversation_id = "test-alt-same-type-missing"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_selected_vay_context(conversation_id)

    result = service._handle_alternative_product(
        "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c",
        top_k=5,
        conversation_id=conversation_id,
        criteria=_alternative_criteria(),
    )

    assert result["products"] == []
    assert result["criteria"]["response_mode"] == "same_type_alternative_not_found"
    assert result["pending_action"] == "confirm_cross_type_alternative"
    assert result["criteria"]["product_type"] == VAY
    assert result["criteria"]["style"] is None
    reply = result["reply"].lower()
    assert "ch\u01b0a c\u00f3 th\u00eam m\u1eabu v\u00e1y kh\u00e1c" in reply
    assert "product_search_no_match" not in reply
    assert "to\u00e0n b\u1ed9 nh\u00f3m v\u00e1y h\u1ebft h\u00e0ng" not in reply


def test_cross_type_alternative_followup_keeps_need_and_excludes_rejected() -> None:
    service = _service()
    _without_rag(service)
    conversation_id = "test-alt-cross-type-accepted"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_selected_vay_context(conversation_id)
    service._handle_alternative_product(
        "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c",
        top_k=5,
        conversation_id=conversation_id,
        criteria=_alternative_criteria(),
    )
    service.parse = _conversation_followup

    result = service._handle_pending_followup(
        "c\u00f3",
        top_k=5,
        conversation_id=conversation_id,
    )

    assert result is not None
    assert result["criteria"]["response_mode"] == "cross_type_alternative"
    assert result["criteria"]["product_type"] is None
    assert result["criteria"]["product_types"] == []
    product_names = [product["product_name"] for product in result["products"]]
    assert AO_SO_MI_NAME in product_names
    assert VAY_NAME not in product_names
    assert BLAZER_NAME not in product_names
    memory = result["conversation_memory"]
    assert "product_type" not in memory
    assert "last_product_type" not in memory
    assert memory.get("selected_product_id") != "P002"
    assert memory["gender"] == "n\u1eef"
    assert memory["use_case"] == "\u0111i l\u00e0m"
    assert memory["budget_max_vnd"] == 300000
    assert "P002" in memory["rejected_product_ids"]


def test_cross_type_alternative_decline_clears_pending_without_search() -> None:
    service = _service()
    _without_rag(service)
    conversation_id = "test-alt-cross-type-declined"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_selected_vay_context(conversation_id)
    service._handle_alternative_product(
        "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c",
        top_k=5,
        conversation_id=conversation_id,
        criteria=_alternative_criteria(),
    )
    service.parse = _conversation_followup
    service.search_products = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("cross-type search should not run on decline")
    )

    result = service._handle_pending_followup(
        "kh\u00f4ng",
        top_k=5,
        conversation_id=conversation_id,
    )

    assert result is not None
    assert result["pending_action"] is None
    assert result["products"] == []
    assert result["conversation_memory"].get("pending_action") is None
    assert result["conversation_memory"]["use_case"] == "\u0111i l\u00e0m"


def test_alternative_product_drops_duplicate_style_before_search() -> None:
    service = _service()
    _without_rag(service)
    conversation_id = "test-alt-duplicate-style"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_selected_vay_context(conversation_id)
    captured: list[dict[str, object]] = []

    def fake_search(criteria: dict[str, object], top_k: int = 5) -> list[dict[str, object]]:
        captured.append(dict(criteria))
        return []

    service.search_products = fake_search

    service._handle_alternative_product(
        "cho t\u00f4i xem s\u1ea3n ph\u1ea9m kh\u00e1c",
        top_k=5,
        conversation_id=conversation_id,
        criteria={
            **_alternative_criteria(),
            "product_type": VAY,
            "product_types": [VAY],
            "use_case": "\u0111i l\u00e0m",
            "style": "\u0111i l\u00e0m",
        },
    )

    assert captured
    assert captured[0]["style"] is None
    assert captured[0]["use_case"] == "\u0111i l\u00e0m"


def test_browse_available_products_followup_full_question() -> None:
    service = _service()
    conversation_id = "test-browse-after-oos-full"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_browse_available_context(service, conversation_id)
    message = "c\u00f3, c\u00f2n s\u1ea3n ph\u1ea9m n\u00e0o c\u00f2n h\u00e0ng k?"
    service.parse = lambda _: _parsed_catalog_browsing(message)

    result = service._handle_pending_followup(message, top_k=5, conversation_id=conversation_id)

    assert result is not None
    assert result["criteria"]["response_mode"] == "catalog_browsing"
    assert result["criteria"]["must_be_in_stock"] is True
    product_names = [product["product_name"] for product in result["products"]]
    assert AO_SO_MI_NAME in product_names
    assert BLAZER_NAME in product_names
    assert VAY_NAME in product_names
    assert CHAN_VAY_NAME not in product_names
    memory = result["conversation_memory"]
    assert memory.get("last_product_type") != CHAN_VAY
    assert memory.get("product_type") != CHAN_VAY
    assert memory.get("last_product_ids")


def test_browse_available_products_followup_affirmative_only() -> None:
    service = _service()
    conversation_id = "test-browse-after-oos-yes"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_browse_available_context(service, conversation_id)
    service.parse = lambda message: {
        "raw_message": message,
        "normalized_message": message,
        "intent": "conversation_followup",
    }

    result = service._handle_pending_followup("c\u00f3", top_k=5, conversation_id=conversation_id)

    assert result is not None
    assert result["criteria"]["response_mode"] == "catalog_browsing"
    assert result["products"]
    assert "nh\u00f3m s\u1ea3n ph\u1ea9m n\u00e0o" not in result["reply"].lower()
    assert CHAN_VAY_NAME not in [product["product_name"] for product in result["products"]]


def test_direct_catalog_browsing_without_context_searches_products() -> None:
    service = _service()
    conversation_id = "test-direct-browse-available"
    CONVERSATION_STORE.pop(conversation_id, None)
    message = "shop c\u00f2n s\u1ea3n ph\u1ea9m n\u00e0o c\u00f2n h\u00e0ng?"
    service.parse = lambda _: _parsed_catalog_browsing(message)
    service._handle_order_message = lambda *_args, **_kwargs: None
    service._prepare_contextual_criteria = lambda _message, _conversation_id, criteria: criteria
    service._attach_rag_contexts = lambda *_args, **_kwargs: []
    service._rerank_products_with_rag = lambda products, _rag_contexts: products

    result = service.chat(message, top_k=5, conversation_id=conversation_id)

    assert result["criteria"]["response_mode"] == "catalog_browsing"
    assert result["criteria"]["need_clarification"] is False
    product_names = [product["product_name"] for product in result["products"]]
    assert AO_SO_MI_NAME in product_names
    assert BLAZER_NAME in product_names
    assert VAY_NAME in product_names
    assert CHAN_VAY_NAME not in product_names


def test_browse_available_products_decline_clears_pending_action() -> None:
    service = _service()
    conversation_id = "test-browse-after-oos-decline"
    CONVERSATION_STORE.pop(conversation_id, None)
    _seed_browse_available_context(service, conversation_id)
    service.parse = lambda message: {
        "raw_message": message,
        "normalized_message": message,
        "intent": "conversation_followup",
    }
    service.search_products = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("catalog search should not run on decline")
    )

    result = service._handle_pending_followup("kh\u00f4ng", top_k=5, conversation_id=conversation_id)

    assert result is not None
    assert result["pending_action"] is None
    assert result["products"] == []
    assert result["conversation_memory"].get("pending_action") is None
