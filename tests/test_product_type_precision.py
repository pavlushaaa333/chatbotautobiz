from __future__ import annotations

import pandas as pd

from app.catalog_gate import CatalogDomainGate
from app.search_engine import ProductSearchEngine


CHAN_VAY = "Ch\u00e2n v\u00e1y"
VAY = "V\u00e1y"
AO_SO_MI = "\u00c1o s\u01a1 mi"
CHAN_VAY_NAME = "Ch\u00e2n v\u00e1y be c\u00f4ng s\u1edf"
VAY_NAME = "V\u00e1y \u0111en d\u00e1ng A"


def _product_row(
    product_id: str,
    product_name: str,
    product_type: str,
    stock_total: int,
) -> dict[str, object]:
    return {
        "product_id": product_id,
        "product_name": product_name,
        "category_code": "Fashion",
        "category_name": "Fashion",
        "product_type": product_type,
        "brand": "AutoBiz",
        "tags": product_type,
        "short_description": product_name,
        "shop_id": "test-shop",
        "sku": product_id,
        "color": "",
        "size": "",
        "quantity": stock_total,
        "available_qty": stock_total,
        "price_vnd": 299000,
        "sale_price_vnd": 299000,
        "sale_price": 299000,
        "selling_price": 299000,
        "effective_price_vnd": 299000,
        "stock_total": stock_total,
        "listing_status": "active",
        "inventory_status": "active",
        "status": "active",
        "source": "test",
        "gender": "",
        "use_case": "",
        "style": "",
        "rating": 4.5,
        "sold_30d": 1,
        "search_text": f"{product_name} {product_type}",
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


def _dress_products() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _product_row("P001", CHAN_VAY_NAME, CHAN_VAY, 0),
            _product_row("P002", VAY_NAME, VAY, 8),
        ]
    )


def _catalog_products() -> pd.DataFrame:
    return pd.concat(
        [
            _dress_products(),
            pd.DataFrame([_product_row("P003", "\u00c1o s\u01a1 mi tr\u1eafng", AO_SO_MI, 5)]),
        ],
        ignore_index=True,
    )


def test_catalog_gate_prefers_chan_vay_over_vay_parent_match() -> None:
    gate = CatalogDomainGate(_catalog_products(), _variants())

    criteria = gate.apply(
        {
            "raw_message": "t mu\u1ed1n mua ch\u00e2n v\u00e1y",
            "normalized_message": "t mu\u1ed1n mua ch\u00e2n v\u00e1y",
            "category_code": "Fashion",
            "product_type": CHAN_VAY,
            "product_types": [CHAN_VAY, VAY],
        }
    )

    assert criteria["product_type"] == CHAN_VAY
    assert criteria["product_types"] == [CHAN_VAY]
    assert criteria["catalog_gate"]["product_types"] == [CHAN_VAY]


def test_catalog_gate_keeps_standalone_vay() -> None:
    gate = CatalogDomainGate(_catalog_products(), _variants())

    criteria = gate.apply(
        {
            "raw_message": "t mu\u1ed1n mua v\u00e1y",
            "normalized_message": "t mu\u1ed1n mua v\u00e1y",
            "category_code": "Fashion",
            "product_type": VAY,
            "product_types": [VAY],
        }
    )

    assert criteria["product_types"] == [VAY]


def test_catalog_gate_keeps_independent_product_types() -> None:
    gate = CatalogDomainGate(_catalog_products(), _variants())

    criteria = gate.apply(
        {
            "raw_message": "t\u00f4i mu\u1ed1n xem v\u00e1y v\u00e0 \u00e1o s\u01a1 mi",
            "normalized_message": "t\u00f4i mu\u1ed1n xem v\u00e1y v\u00e0 \u00e1o s\u01a1 mi",
            "category_code": "Fashion",
            "product_types": [VAY, AO_SO_MI],
        }
    )

    assert criteria["product_types"] == [VAY, AO_SO_MI]


def test_search_exact_product_type_wins_over_broad_product_types() -> None:
    engine = ProductSearchEngine(_dress_products(), _variants())

    results = engine.search(
        {
            "category_code": "Fashion",
            "product_type": CHAN_VAY,
            "product_types": [CHAN_VAY, VAY],
            "apparel_intent": "v\u00e1y",
        },
        top_k=5,
    )

    assert results == []


def test_search_returns_vay_when_vay_is_exact_request() -> None:
    engine = ProductSearchEngine(_dress_products(), _variants())

    results = engine.search(
        {
            "category_code": "Fashion",
            "product_type": VAY,
            "product_types": [VAY],
        },
        top_k=5,
    )

    assert [product["product_name"] for product in results] == [VAY_NAME]
