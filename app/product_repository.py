from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from decimal import Decimal
from typing import Any

from app.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    get_connection,
)

SAFE_PRODUCT_DATA_ERROR_REPLY = (
    "Hiện tại hệ thống chưa kiểm tra được dữ liệu sản phẩm. "
    "Bạn vui lòng thử lại sau."
)

PRODUCT_SEARCH_SQL_ENV = "PRODUCT_SEARCH_SQL"
PRODUCT_BY_SKU_SQL_ENV = "PRODUCT_BY_SKU_SQL"
PRODUCT_BY_ID_SQL_ENV = "PRODUCT_BY_ID_SQL"
PRODUCT_INVENTORY_SQL_ENV = "PRODUCT_INVENTORY_SQL"
PRODUCT_VARIANTS_SQL_ENV = "PRODUCT_VARIANTS_SQL"
PRODUCT_COLUMN_MAP_ENV = "PRODUCT_COLUMN_MAP"

DEFAULT_PRODUCT_SEARCH_SQL = """
SELECT
    COALESCE(product_id, id) AS product_id,
    product_name,
    sku,
    variant,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 1)), '') AS color,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 2)), '') AS size,
    selling_price AS price,
    quantity AS available_qty,
    COALESCE(listing_status, status) AS product_status
FROM public.inventory
WHERE status = 'active'
  AND (
      %(shop_id)s::uuid IS NULL
      OR shop_id = %(shop_id)s::uuid
  )
  AND (
      %(keyword_pattern)s::text IS NULL
      OR product_name ILIKE %(keyword_pattern)s::text
      OR sku ILIKE %(keyword_pattern)s::text
      OR variant ILIKE %(keyword_pattern)s::text
  )
  AND (
      %(color)s::text IS NULL
      OR variant ILIKE '%%' || %(color)s::text || '%%'
  )
  AND (
      %(size)s::text IS NULL
      OR variant ILIKE '%%' || %(size)s::text || '%%'
  )
  AND (
      %(max_price)s::bigint IS NULL
      OR selling_price <= %(max_price)s::bigint
  )
  AND (
      %(must_be_in_stock)s::boolean IS NOT TRUE
      OR quantity > 0
  )
ORDER BY
    CASE WHEN quantity > 0 THEN 0 ELSE 1 END,
    quantity DESC,
    product_name ASC
LIMIT %(limit)s
"""


DEFAULT_PRODUCT_BY_SKU_SQL = """
SELECT
    COALESCE(product_id, id) AS product_id,
    product_name,
    sku,
    variant,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 1)), '') AS color,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 2)), '') AS size,
    selling_price AS price,
    quantity AS available_qty,
    COALESCE(listing_status, status) AS product_status
FROM public.inventory
WHERE sku = %(sku)s::text
  AND (
      %(shop_id)s::uuid IS NULL
      OR shop_id = %(shop_id)s::uuid
  )
LIMIT 1
"""


DEFAULT_PRODUCT_BY_ID_SQL = """
SELECT
    COALESCE(product_id, id) AS product_id,
    product_name,
    sku,
    variant,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 1)), '') AS color,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 2)), '') AS size,
    selling_price AS price,
    quantity AS available_qty,
    COALESCE(listing_status, status) AS product_status
FROM public.inventory
WHERE COALESCE(product_id, id) = %(product_id)s::uuid
  AND (
      %(shop_id)s::uuid IS NULL
      OR shop_id = %(shop_id)s::uuid
  )
LIMIT 1
"""


DEFAULT_PRODUCT_INVENTORY_SQL = """
SELECT
    COALESCE(product_id, id) AS product_id,
    product_name,
    sku,
    variant,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 1)), '') AS color,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 2)), '') AS size,
    selling_price AS price,
    quantity AS available_qty,
    COALESCE(listing_status, status) AS product_status
FROM public.inventory
WHERE (
        %(sku)s::text IS NULL
        OR sku = %(sku)s::text
      )
  AND (
        %(product_id)s::uuid IS NULL
        OR COALESCE(product_id, id) = %(product_id)s::uuid
      )
  AND (
        %(color)s::text IS NULL
        OR variant ILIKE '%%' || %(color)s::text || '%%'
      )
  AND (
        %(size)s::text IS NULL
        OR variant ILIKE '%%' || %(size)s::text || '%%'
      )
  AND (
        %(shop_id)s::uuid IS NULL
        OR shop_id = %(shop_id)s::uuid
      )
ORDER BY quantity DESC
"""


DEFAULT_PRODUCT_VARIANTS_SQL = """
SELECT
    COALESCE(product_id, id) AS product_id,
    product_name,
    sku,
    variant,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 1)), '') AS color,
    NULLIF(TRIM(SPLIT_PART(variant, '/', 2)), '') AS size,
    selling_price AS price,
    quantity AS available_qty,
    COALESCE(listing_status, status) AS product_status
FROM public.inventory
WHERE product_name = (
    SELECT product_name
    FROM public.inventory
    WHERE COALESCE(product_id, id) = %(product_id)s::uuid
      AND (
          %(shop_id)s::uuid IS NULL
          OR shop_id = %(shop_id)s::uuid
      )
    LIMIT 1
)
  AND (
      %(shop_id)s::uuid IS NULL
      OR shop_id = %(shop_id)s::uuid
  )
  AND (
      %(color)s::text IS NULL
      OR variant ILIKE '%%' || %(color)s::text || '%%'
  )
  AND (
      %(size)s::text IS NULL
      OR variant ILIKE '%%' || %(size)s::text || '%%'
  )
  AND (
      %(only_in_stock)s::boolean IS NOT TRUE
      OR quantity > 0
  )
ORDER BY variant ASC
LIMIT %(limit)s
"""


DEFAULT_SQL_BY_ENV = {
    PRODUCT_SEARCH_SQL_ENV: DEFAULT_PRODUCT_SEARCH_SQL,
    PRODUCT_BY_SKU_SQL_ENV: DEFAULT_PRODUCT_BY_SKU_SQL,
    PRODUCT_BY_ID_SQL_ENV: DEFAULT_PRODUCT_BY_ID_SQL,
    PRODUCT_INVENTORY_SQL_ENV: DEFAULT_PRODUCT_INVENTORY_SQL,
    PRODUCT_VARIANTS_SQL_ENV: DEFAULT_PRODUCT_VARIANTS_SQL,
}


OPTIONAL_COMPATIBILITY_FIELDS = (
    "category_code",
    "category_name",
    "product_type",
    "brand",
    "tags",
    "short_description",
    "effective_price_vnd",
    "sale_price_vnd",
    "sale_price",
    "selling_price",
    "price_vnd",
    "rating",
    "sold_30d",
)

COMMON_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "product_id": ("product_id", "id", "product_uuid"),
    "product_name": ("product_name", "name", "title"),
    "sku": ("sku", "variant_sku"),
    "variant": ("variant", "variant_name", "option_name"),
    "color": ("color", "colour"),
    "size": ("size", "variant_size"),
    "price": (
        "sale_price",
        "sale_price_vnd",
        "selling_price",
        "effective_price_vnd",
        "price_vnd",
        "price",
    ),
    "available_qty": ("available_qty", "stock", "stock_qty", "quantity", "stock_total"),
    "product_status": ("product_status", "status", "product_state"),
}


class ProductRepositoryError(RuntimeError):
    """Base error for product data access."""


class ProductRepositoryConfigurationError(ProductRepositoryError):
    """Raised when SQL templates or column mappings are not configured."""


class ProductRepositoryDatabaseError(ProductRepositoryError):
    """Raised when PostgreSQL returns an error."""


ConnectionFactory = Callable[[], AbstractContextManager[Any]]


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        value = float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def _int_value(value: Any) -> int:
    number = _number(value)
    if number is None:
        return 0
    return int(number)


def _load_column_map() -> dict[str, str]:
    raw = os.getenv(PRODUCT_COLUMN_MAP_ENV, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProductRepositoryConfigurationError(
            f"{PRODUCT_COLUMN_MAP_ENV} must be valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ProductRepositoryConfigurationError(
            f"{PRODUCT_COLUMN_MAP_ENV} must be a JSON object."
        )
    return {str(key): str(value) for key, value in payload.items() if value}


def _status_values(env_name: str, defaults: set[str]) -> set[str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return defaults
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


class ProductRepository:
    def __init__(
        self,
        *,
        connection_factory: ConnectionFactory = get_connection,
        sql_templates: Mapping[str, str] | None = None,
        column_map: Mapping[str, str] | None = None,
    ) -> None:
        self.connection_factory = connection_factory
        self.sql_templates = dict(sql_templates or {})
        self.column_map = dict(column_map or _load_column_map())
        self.active_statuses = _status_values(
            "PRODUCT_ACTIVE_STATUSES",
            {"active", "available", "published", "in_stock", "true", "1"},
        )
        self.inactive_statuses = _status_values(
            "PRODUCT_INACTIVE_STATUSES",
            {"inactive", "deleted", "archived", "disabled", "draft"},
        )

    def search_products(
        self,
        *,
        keyword: str | None = None,
        product_type: str | None = None,
        category_code: str | None = None,
        color: str | None = None,
        size: str | None = None,
        max_price: int | float | None = None,
        must_be_in_stock: bool | None = None,
        shop_id: str | int | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        params = self._query_params(
            keyword=keyword,
            product_type=product_type,
            category_code=category_code,
            color=color,
            size=size,
            max_price=max_price,
            must_be_in_stock=must_be_in_stock,
            shop_id=shop_id,
            limit=limit,
        )
        rows = self._fetch_all(PRODUCT_SEARCH_SQL_ENV, params)
        return self._normalize_rows(rows)

    def get_product_by_sku(
        self,
        sku: str,
        *,
        shop_id: str | int | None = None,
    ) -> dict[str, Any] | None:
        rows = self._normalize_rows(
            self._fetch_all(
                PRODUCT_BY_SKU_SQL_ENV,
                {"sku": sku, "shop_id": shop_id, "limit": 1},
            )
        )
        return rows[0] if rows else None

    def get_product_by_id(
        self,
        product_id: str,
        *,
        shop_id: str | int | None = None,
    ) -> dict[str, Any] | None:
        rows = self._normalize_rows(
            self._fetch_all(
                PRODUCT_BY_ID_SQL_ENV,
                {"product_id": product_id, "shop_id": shop_id, "limit": 1},
            )
        )
        return rows[0] if rows else None

    def check_inventory(
        self,
        *,
        sku: str | None = None,
        product_id: str | None = None,
        color: str | None = None,
        size: str | None = None,
        shop_id: str | int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._fetch_all(
            PRODUCT_INVENTORY_SQL_ENV,
            {
                "sku": sku,
                "product_id": product_id,
                "color": color,
                "size": size,
                "shop_id": shop_id,
            },
        )
        return self._normalize_rows(rows)

    def get_available_variants(
        self,
        *,
        product_id: str,
        color: str | None = None,
        size: str | None = None,
        shop_id: str | int | None = None,
        only_in_stock: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self._fetch_all(
            PRODUCT_VARIANTS_SQL_ENV,
            {
                "product_id": product_id,
                "color": color,
                "size": size,
                "shop_id": shop_id,
                "only_in_stock": only_in_stock,
                "limit": limit,
            },
        )
        return self._normalize_rows(rows)

    def _query_params(
        self,
        *,
        keyword: str | None,
        product_type: str | None,
        category_code: str | None,
        color: str | None,
        size: str | None,
        max_price: int | float | None,
        must_be_in_stock: bool | None,
        shop_id: str | int | None,
        limit: int,
    ) -> dict[str, Any]:
        keyword = (keyword or "").strip() or None
        return {
            "keyword": keyword,
            "keyword_pattern": f"%{keyword}%" if keyword else None,
            "product_type": product_type,
            "category_code": category_code,
            "color": color,
            "size": size,
            "max_price": max_price,
            "must_be_in_stock": must_be_in_stock,
            "shop_id": shop_id,
            "limit": max(1, int(limit)),
            "active_statuses": tuple(sorted(self.active_statuses)),
        }

    def _sql(self, env_name: str) -> str:
        configured_sql = (
            self.sql_templates.get(env_name) or os.getenv(env_name, "")
        ).strip()

        if configured_sql:
            return configured_sql

        default_sql = DEFAULT_SQL_BY_ENV.get(env_name)

        if default_sql:
            return default_sql.strip()

        raise ProductRepositoryConfigurationError(
            f"Missing SQL configuration for {env_name}."
        )

    def _fetch_all(
        self, sql_env: str, params: Mapping[str, Any]
    ) -> list[Mapping[str, Any]]:
        sql = self._sql(sql_env)
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, dict(params))
                    rows = cursor.fetchall()
        except ProductRepositoryConfigurationError:
            raise
        except DatabaseConfigurationError as exc:
            raise ProductRepositoryConfigurationError(str(exc)) from exc
        except DatabaseConnectionError as exc:
            raise ProductRepositoryDatabaseError(str(exc)) from exc
        except Exception as exc:
            raise ProductRepositoryDatabaseError(
                f"PostgreSQL query failed for {sql_env}: {exc.__class__.__name__}"
            ) from exc
        return [dict(row) for row in rows]

    def _field_value(self, row: Mapping[str, Any], canonical_field: str) -> Any:
        mapped = self.column_map.get(canonical_field, canonical_field)
        if mapped in row:
            return row[mapped]
        for alias in COMMON_COLUMN_ALIASES.get(canonical_field, (canonical_field,)):
            if alias in row:
                return row[alias]
        return None

    def _normalize_rows(self, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        products = [self._normalize_row(row) for row in rows]
        products = [product for product in products if self._is_valid_product(product)]
        indexed_products = list(enumerate(products))
        indexed_products.sort(
            key=lambda item: (
                self._is_active_status(item[1].get("product_status")) is False,
                item[0],
            )
        )
        return [product for _, product in indexed_products]

    def _normalize_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        product_id = _clean_text(self._field_value(row, "product_id"))
        product_name = _clean_text(self._field_value(row, "product_name")) or ""
        sku = _clean_text(self._field_value(row, "sku")) or ""
        color = _clean_text(self._field_value(row, "color"))
        size = _clean_text(self._field_value(row, "size"))
        variant = _clean_text(self._field_value(row, "variant"))
        if not variant and (color or size):
            variant = " / ".join(part for part in [color, size] if part)
        price = _number(self._field_value(row, "price"))
        available_qty = _int_value(self._field_value(row, "available_qty"))
        product_status = _clean_text(self._field_value(row, "product_status"))
        variant_status = "in_stock" if available_qty > 0 else "out_of_stock"

        payload: dict[str, Any] = {
            "product_id": product_id,
            "product_name": product_name,
            "sku": sku,
            "variant": variant,
            "color": color,
            "size": size,
            "price": price,
            "available_qty": available_qty,
            "product_status": product_status,
            "price_vnd": price,
            "sale_price_vnd": price,
            "effective_price_vnd": price,
            "stock_total": available_qty,
            "stock": available_qty,
            "status": product_status,
            "variant_status": variant_status,
        }

        for field in OPTIONAL_COMPATIBILITY_FIELDS:
            value = row.get(self.column_map.get(field, field), row.get(field))
            if value is not None:
                payload[field] = value

        return payload

    def _is_valid_product(self, product: Mapping[str, Any]) -> bool:
        if not _clean_text(product.get("product_name")):
            return False
        status = _clean_text(product.get("product_status"))
        if status and status.lower() in self.inactive_statuses:
            return False
        return True

    def _is_active_status(self, status: Any) -> bool:
        text = _clean_text(status)
        if text is None:
            return False
        return text.lower() in self.active_statuses


def _criteria_keyword(criteria: Mapping[str, Any] | None) -> str | None:
    if not criteria:
        return None
    for field in ["requested_product_name", "raw_message", "normalized_message"]:
        value = _clean_text(criteria.get(field))
        if value:
            return value
    keywords = [
        str(item) for item in criteria.get("keywords") or [] if str(item).strip()
    ]
    return " ".join(keywords) if keywords else None


def search_products(
    criteria: Mapping[str, Any] | None = None,
    *,
    keyword: str | None = None,
    product_type: str | None = None,
    category_code: str | None = None,
    color: str | None = None,
    size: str | None = None,
    max_price: int | float | None = None,
    must_be_in_stock: bool | None = None,
    shop_id: str | int | None = None,
    limit: int = 5,
    repository: ProductRepository | None = None,
) -> list[dict[str, Any]]:
    criteria = criteria or {}
    repo = repository or ProductRepository()
    return repo.search_products(
        keyword=keyword or _criteria_keyword(criteria),
        product_type=product_type or criteria.get("product_type"),
        category_code=category_code or criteria.get("category_code"),
        color=color or criteria.get("color"),
        size=size or criteria.get("size"),
        max_price=(
            max_price if max_price is not None else criteria.get("budget_max_vnd")
        ),
        must_be_in_stock=(
            must_be_in_stock
            if must_be_in_stock is not None
            else criteria.get("must_be_in_stock")
        ),
        shop_id=shop_id or criteria.get("shop_id"),
        limit=limit,
    )


def get_product_by_sku(
    sku: str,
    *,
    shop_id: str | int | None = None,
    repository: ProductRepository | None = None,
) -> dict[str, Any] | None:
    return (repository or ProductRepository()).get_product_by_sku(sku, shop_id=shop_id)


def check_inventory(
    *,
    sku: str | None = None,
    product_id: str | None = None,
    color: str | None = None,
    size: str | None = None,
    shop_id: str | int | None = None,
    repository: ProductRepository | None = None,
) -> list[dict[str, Any]]:
    return (repository or ProductRepository()).check_inventory(
        sku=sku,
        product_id=product_id,
        color=color,
        size=size,
        shop_id=shop_id,
    )


def get_available_variants(
    *,
    product_id: str,
    color: str | None = None,
    size: str | None = None,
    shop_id: str | int | None = None,
    only_in_stock: bool = False,
    limit: int = 20,
    repository: ProductRepository | None = None,
) -> list[dict[str, Any]]:
    return (repository or ProductRepository()).get_available_variants(
        product_id=product_id,
        color=color,
        size=size,
        shop_id=shop_id,
        only_in_stock=only_in_stock,
        limit=limit,
    )
