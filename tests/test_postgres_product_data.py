from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from app.product_repository import (
    PRODUCT_BY_SKU_SQL_ENV,
    PRODUCT_SEARCH_SQL_ENV,
    ProductRepository,
    ProductRepositoryDatabaseError,
)
from app.rag_service import AutoBizRAGService
from app.response_generator import build_model_context
from app.search_engine import PostgresProductSearchEngine


class FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.sql: str | None = None
        self.params: dict[str, Any] | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.sql = sql
        self.params = params

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_obj = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_obj


def repository_with_rows(rows: list[dict[str, Any]]) -> tuple[ProductRepository, FakeCursor]:
    cursor = FakeCursor(rows)
    repo = ProductRepository(
        connection_factory=lambda: FakeConnection(cursor),
        sql_templates={
            PRODUCT_SEARCH_SQL_ENV: "select * from real_products where name ilike %(keyword_pattern)s",
            PRODUCT_BY_SKU_SQL_ENV: "select * from real_products where sku = %(sku)s",
        },
        column_map={
            "product_id": "db_product_id",
            "product_name": "db_product_name",
            "sku": "db_sku",
            "price": "db_price",
            "available_qty": "db_available_qty",
            "product_status": "db_status",
        },
    )
    return repo, cursor


def test_repository_search_uses_parameters_and_normalizes_product() -> None:
    repo, cursor = repository_with_rows(
        [
            {
                "db_product_id": 10,
                "db_product_name": "Áo thun basic",
                "db_sku": "SKU-WHITE-S",
                "color": "trắng",
                "size": "S",
                "db_price": Decimal("199000"),
                "db_available_qty": 7,
                "db_status": "active",
            }
        ]
    )

    products = repo.search_products(keyword="áo thun", color="trắng", size="S", limit=3)

    assert cursor.params is not None
    assert cursor.params["keyword_pattern"] == "%áo thun%"
    assert cursor.params["color"] == "trắng"
    assert cursor.params["size"] == "S"
    assert products == [
        {
            "product_id": "10",
            "product_name": "Áo thun basic",
            "sku": "SKU-WHITE-S",
            "variant": "trắng / S",
            "color": "trắng",
            "size": "S",
            "price": 199000,
            "available_qty": 7,
            "product_status": "active",
            "price_vnd": 199000,
            "sale_price_vnd": 199000,
            "effective_price_vnd": 199000,
            "stock_total": 7,
            "stock": 7,
            "status": "active",
            "variant_status": "in_stock",
        }
    ]


def test_repository_get_product_by_sku() -> None:
    repo, cursor = repository_with_rows(
        [
            {
                "db_product_id": "P1",
                "db_product_name": "Áo sơ mi",
                "db_sku": "SKU-BLUE-M",
                "db_price": 299000,
                "db_available_qty": 2,
                "db_status": "active",
            }
        ]
    )

    product = repo.get_product_by_sku("SKU-BLUE-M")

    assert cursor.params is not None
    assert cursor.params["sku"] == "SKU-BLUE-M"
    assert product is not None
    assert product["sku"] == "SKU-BLUE-M"
    assert product["available_qty"] == 2


class FakeRepo:
    def __init__(self, rows: list[dict[str, Any]], variants: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows
        self.variants = variants if variants is not None else rows

    def search_products(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    def get_product_by_id(self, product_id: str, **_kwargs: Any) -> dict[str, Any] | None:
        return next((row for row in self.rows if row.get("product_id") == product_id), None)

    def get_product_by_sku(self, sku: str, **_kwargs: Any) -> dict[str, Any] | None:
        return next((row for row in self.rows if row.get("sku") == sku), None)

    def get_available_variants(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return self.variants


def test_postgres_engine_checks_color_size_and_in_stock() -> None:
    engine = PostgresProductSearchEngine(
        FakeRepo(
            [
                {
                    "product_id": "P1",
                    "product_name": "Áo thun basic",
                    "sku": "SKU-WHITE-S",
                    "color": "trắng",
                    "size": "S",
                    "price": 199000,
                    "available_qty": 4,
                    "product_status": "active",
                }
            ]
        )
    )

    availability = engine.check_variant_availability(
        {"raw_message": "Áo thun basic trắng S còn không?", "color": "trắng", "size": "S"}
    )

    assert availability["status"] == "in_stock"
    assert availability["variant"]["sku"] == "SKU-WHITE-S"
    assert availability["variant"]["stock"] == 4


def test_postgres_engine_reports_out_of_stock() -> None:
    engine = PostgresProductSearchEngine(
        FakeRepo(
            [
                {
                    "product_id": "P1",
                    "product_name": "Áo thun basic",
                    "sku": "SKU-WHITE-S",
                    "color": "trắng",
                    "size": "S",
                    "price": 199000,
                    "available_qty": 0,
                    "product_status": "active",
                }
            ]
        )
    )

    availability = engine.check_variant_availability({"color": "trắng", "size": "S"})

    assert availability["status"] == "out_of_stock"
    assert availability["variant"]["stock"] == 0


def test_postgres_engine_reports_no_database_result() -> None:
    engine = PostgresProductSearchEngine(FakeRepo([]))

    assert engine.search({"raw_message": "áo blazer"}, top_k=5) == []
    assert engine.check_variant_availability({"raw_message": "áo blazer"})["status"] == "product_not_found"


class BrokenRepo(FakeRepo):
    def search_products(self, **_kwargs: Any) -> list[dict[str, Any]]:
        raise ProductRepositoryDatabaseError("database down")


def test_postgres_engine_reports_database_error() -> None:
    engine = PostgresProductSearchEngine(BrokenRepo([]))

    products = engine.search({"raw_message": "áo thun"}, top_k=5)
    availability = engine.check_variant_availability({"raw_message": "áo thun"})

    assert products == []
    assert engine.last_error == "database down"
    assert availability["status"] == "database_error"


def test_stale_rag_product_context_is_filtered_in_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRODUCT_DATA_SOURCE", "postgres")
    service = AutoBizRAGService()

    contexts = service._format_contexts(
        [
            {
                "id": "product:OLD",
                "content": "Giá cũ 100000 tồn kho 99",
                "metadata": {"doc_type": "product", "product_id": "OLD"},
                "distance": 0.1,
            },
            {
                "id": "policy:cod",
                "content": "Shop hỗ trợ COD",
                "metadata": {"doc_type": "policy"},
                "distance": 0.2,
            },
        ]
    )

    assert [context["id"] for context in contexts] == ["policy:cod"]


def test_model_context_separates_database_product_data_from_rag() -> None:
    context = build_model_context(
        user_message="Áo thun trắng còn không?",
        product_data_from_database=[
            {"product_name": "Áo thun basic", "sku": "SKU-WHITE-S", "price": 199000, "available_qty": 4}
        ],
        rag_context=[{"content_preview": "RAG cũ nói tồn kho 99"}],
    )

    assert context["PRODUCT_DATA_FROM_DATABASE"][0]["price"] == 199000
    assert "Không được lấy giá hoặc tồn kho từ RAG_CONTEXT." in context["RULES"]
