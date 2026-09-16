from __future__ import annotations

import hashlib
import re
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import CSV_DIR, active_shop_id, catalog_source, include_simulated_data
from app.database import get_connection
from app.normalizer import (
    detect_product_type,
    normalize_color,
    normalize_text,
    strip_accents,
)


@dataclass(slots=True)
class DataStore:
    products: pd.DataFrame
    variants: pd.DataFrame
    inventory: pd.DataFrame
    synonyms: pd.DataFrame
    policies: pd.DataFrame
    source: str = "csv"
    active_shop_id: str | None = None
    database_row_count: int = 0
    logical_product_count: int = 0
    variant_count: int = 0
    sellable_variant_count: int = 0

    @property
    def synonym_map(self) -> dict[str, str]:
        if self.synonyms.empty:
            return {}
        return {
            str(row.raw_term).strip().lower(): str(row.normalized_term).strip().lower()
            for row in self.synonyms.itertuples(index=False)
            if str(row.raw_term).strip() and str(row.normalized_term).strip()
        }


PRODUCT_FILES = {
    "products": "simulated_products.csv",
    "variants": "simulated_product_variants.csv",
    "inventory": "simulated_inventory.csv",
}

SUPPORT_FILES = {
    "synonyms": "simulated_synonym_dictionary.csv",
    "policies": "simulated_shop_policies.csv",
}

SELLABLE_LISTING_STATUSES = {"active", "low_stock"}
DATABASE_PRICE_FIELDS = (
    "sale_price",
    "sale_price_vnd",
    "selling_price",
    "effective_price_vnd",
    "price_vnd",
)
DATABASE_INVENTORY_SQL = """
SELECT
    id,
    shop_id,
    sku,
    product_name,
    variant,
    quantity,
    low_stock_threshold,
    updated_at,
    status,
    shopee_item_id,
    shopee_model_id,
    listing_status,
    product_id,
    cost_price,
    selling_price
FROM public.inventory
WHERE (
        %(shop_id)s::uuid IS NULL
        OR shop_id = %(shop_id)s::uuid
      )
ORDER BY product_name ASC, variant ASC, sku ASC
"""


def _read_csv(csv_dir: Path, filename: str) -> pd.DataFrame:
    path = csv_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file dataset bắt buộc: {path}")
    return pd.read_csv(path, encoding="utf-8")


def _as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _valid_price(value: Any) -> int | None:
    try:
        price = int(float(value))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _database_price(row: dict[str, Any]) -> int | None:
    for field in DATABASE_PRICE_FIELDS:
        price = _valid_price(row.get(field))
        if price is not None:
            return price
    return None


def _key(text: Any) -> str:
    return strip_accents(normalize_text(str(text or ""))).lower()


def _display_text(text: Any) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    return value[:1].upper() + value[1:]


def _parse_variant(variant: Any) -> tuple[str | None, str | None]:
    value = _clean_text(variant)
    if not value:
        return None, None
    parts = [part.strip() for part in value.split("/") if part.strip()]
    if len(parts) >= 2:
        return _display_text(normalize_color(parts[0]) or parts[0]), _display_text(
            parts[1]
        )
    if len(parts) == 1:
        part_key = _key(parts[0])
        if part_key in {"s", "m", "l", "xl", "freesize", "free size"}:
            return None, _display_text(parts[0])
        return _display_text(normalize_color(parts[0]) or parts[0]), None
    return None, None


def _stable_logical_product_id(
    shop_id: str | None, product_id: Any, product_name: str
) -> str:
    raw_product_id = _clean_text(product_id)
    if raw_product_id:
        return raw_product_id
    product_key = re.sub(r"\W+", "-", _key(product_name)).strip("-")
    digest = hashlib.sha1(f"{shop_id or ''}:{product_key}".encode("utf-8")).hexdigest()[
        :16
    ]
    return f"db_{digest}"


def _infer_product_type(product_name: str) -> str:
    name_key = _key(product_name)
    if "chan vay" in name_key:
        return "Chân váy"
    if "ao so mi" in name_key or re.search(r"(?<!\w)so mi(?!\w)", name_key):
        return "Áo sơ mi"
    if "blazer" in name_key or "ao khoac" in name_key:
        return "Áo khoác"
    if "vay chu a" in name_key or "vay" in name_key or "dam" in name_key:
        return "Váy"
    return detect_product_type(product_name) or ""


def _canonical_product_type(value: object) -> str | None:
    key = _key(value)

    mapping = {
        "ao thun": "Áo thun",
        "ao so mi": "Áo sơ mi",
        "vay": "Váy",
        "dam": "Váy",
        "chan vay": "Chân váy",
        "quan jeans": "Quần jeans",
        "quan jean": "Quần jeans",
        "ao khoac": "Áo khoác",
        "blazer": "Áo khoác",
        "set bo": "Set bộ",
    }

    return mapping.get(key)


def _infer_gender(product_name: str, product_type: str) -> str:
    name_key = _key(product_name)
    if (
        " nu" in f" {name_key} "
        or product_type == "Váy"
        or "dam" in name_key
        or "chan vay" in name_key
    ):
        return "nữ"
    if " nam" in f" {name_key} ":
        return "nam"
    return ""


def _infer_use_cases(product_name: str, product_type: str) -> list[str]:
    name_key = _key(product_name)
    use_cases: list[str] = []
    if "vay den dang a" in name_key:
        use_cases.extend(["đi làm", "đi chơi"])
    if (
        "cong so" in name_key
        or "blazer" in name_key
        or "so mi" in name_key
        or product_type in {"Áo sơ mi", "Áo khoác"}
    ):
        use_cases.append("đi làm")
    if "vay hoa" in name_key:
        use_cases.append("đi chơi")
    return list(dict.fromkeys(use_cases))


def _infer_styles(product_name: str, product_type: str) -> list[str]:
    name_key = _key(product_name)
    styles: list[str] = []
    if "cong so" in name_key or product_type in {"Áo sơ mi", "Áo khoác"}:
        styles.append("công sở")
    if product_type == "Váy":
        styles.append("nữ tính")
    if "den" in name_key or product_type in {"Áo sơ mi", "Áo khoác"}:
        styles.append("basic")
    return list(dict.fromkeys(styles))


def _is_sellable_inventory_row(row: dict[str, Any]) -> bool:
    status = _clean_text(row.get("status")).lower()
    listing_status = _clean_text(row.get("listing_status")).lower()
    return (
        status == "active"
        and listing_status in SELLABLE_LISTING_STATUSES
        and _safe_int(row.get("quantity")) > 0
        and _database_price(row) is not None
    )


def _database_search_text(product: dict[str, Any]) -> str:
    columns = [
        "product_name",
        "category_name",
        "product_type",
        "brand",
        "tags",
        "short_description",
        "color",
        "size",
    ]
    return re.sub(
        r"\s+",
        " ",
        " ".join(_clean_text(product.get(column)) for column in columns),
    ).strip()


def _load_database_inventory(shop_id: str | None) -> pd.DataFrame:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(DATABASE_INVENTORY_SQL, {"shop_id": shop_id})
            rows = cursor.fetchall()
    return pd.DataFrame(rows)


def _prepare_database_catalog(
    inventory: pd.DataFrame,
    shop_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    product_map: dict[str, dict[str, Any]] = {}
    variants: list[dict[str, Any]] = []

    for row in inventory.to_dict("records"):
        product_name = _clean_text(row.get("product_name"))
        if not product_name:
            continue

        resolved_shop_id = _clean_text(row.get("shop_id")) or shop_id
        product_id = _stable_logical_product_id(
            resolved_shop_id, row.get("product_id"), product_name
        )
        inferred_product_type = _infer_product_type(product_name)
        product_type = (
            _canonical_product_type(inferred_product_type) or inferred_product_type
        )
        gender = _infer_gender(product_name, product_type)
        use_cases = _infer_use_cases(product_name, product_type)
        styles = _infer_styles(product_name, product_type)
        color, size = _parse_variant(row.get("variant"))
        price = _database_price(row)
        raw_quantity = _safe_int(row.get("quantity"))
        sellable = _is_sellable_inventory_row(row)
        available_qty = raw_quantity if sellable else 0
        variant_status = "in_stock" if available_qty > 0 else "out_of_stock"

        tag_values = [
            product_type,
            gender,
            *(use_cases or []),
            *(styles or []),
            color or "",
            size or "",
        ]
        tags = "|".join(dict.fromkeys(value for value in tag_values if value))

        variant_payload = {
            "variant_id": _clean_text(row.get("id")) or _clean_text(row.get("sku")),
            "product_id": product_id,
            "shop_id": resolved_shop_id,
            "sku": _clean_text(row.get("sku")),
            "variant": _clean_text(row.get("variant")),
            "product_name": product_name,
            "color": color,
            "size": size,
            "quantity": available_qty,
            "raw_quantity": raw_quantity,
            "stock_qty": available_qty,
            "available_qty": available_qty,
            "price_vnd": price or 0,
            "selling_price": _valid_price(row.get("selling_price")) or 0,
            "effective_price_vnd": price or 0,
            "sale_price_vnd": _valid_price(row.get("sale_price_vnd")) or 0,
            "sale_price": _valid_price(row.get("sale_price")) or 0,
            "status": variant_status,
            "inventory_status": _clean_text(row.get("status")),
            "listing_status": _clean_text(row.get("listing_status")),
            "sellable": sellable,
            "source": "database",
        }
        variants.append(variant_payload)

        product = product_map.get(product_id)
        if product is None:
            product = {
                "product_id": product_id,
                "shop_id": resolved_shop_id,
                "product_name": product_name,
                "category_code": "Fashion",
                "category_name": "Thời trang",
                "product_type": product_type,
                "brand": "",
                "tags": tags,
                "short_description": "",
                "price_vnd": price or 0,
                "selling_price": _valid_price(row.get("selling_price")) or 0,
                "sale_price_vnd": _valid_price(row.get("sale_price_vnd")) or 0,
                "sale_price": _valid_price(row.get("sale_price")) or 0,
                "effective_price_vnd": price or 0,
                "stock_total": 0,
                "available_qty": 0,
                "quantity": 0,
                "sku": "",
                "color": "",
                "size": "",
                "listing_status": _clean_text(row.get("listing_status")),
                "inventory_status": _clean_text(row.get("status")),
                "status": "active",
                "sellable": False,
                "source": "database",
                "gender": gender,
                "use_case": "|".join(use_cases),
                "style": "|".join(styles),
                "rating": 0.0,
                "sold_30d": 0,
            }
            product_map[product_id] = product

        if tags:
            existing_tags = [
                tag for tag in str(product.get("tags") or "").split("|") if tag
            ]
            product["tags"] = "|".join(
                dict.fromkeys([*existing_tags, *tags.split("|")])
            )

        current_price = _valid_price(product.get("effective_price_vnd"))
        if price is not None and (
            current_price is None or (sellable and price < current_price)
        ):
            product["price_vnd"] = price
            product["selling_price"] = _valid_price(row.get("selling_price")) or price
            product["effective_price_vnd"] = price

        if sellable:
            product["stock_total"] = (
                _safe_int(product.get("stock_total")) + available_qty
            )
            product["available_qty"] = (
                _safe_int(product.get("available_qty")) + available_qty
            )
            product["quantity"] = _safe_int(product.get("quantity")) + available_qty
            product["sellable"] = True
            product["listing_status"] = _clean_text(row.get("listing_status"))
            if not product.get("sku"):
                product["sku"] = variant_payload["sku"]
                product["color"] = color or ""
                product["size"] = size or ""

    products = list(product_map.values())
    for product in products:
        current_product_type = product.get("product_type")

    product["product_type"] = (
        _canonical_product_type(current_product_type)
        or _infer_product_type(str(product.get("product_name") or ""))
        or str(current_product_type or "")
    )

    if not product.get("short_description"):
        detail_parts = [
            str(product.get("product_type") or "").lower(),
            str(product.get("use_case") or "").replace("|", "/"),
            str(product.get("style") or "").replace("|", "/"),
        ]
        product["short_description"] = ", ".join(part for part in detail_parts if part)
        product["search_text"] = _database_search_text(product)

    products_df = pd.DataFrame(products)
    variants_df = pd.DataFrame(variants)
    metadata = {
        "database_row_count": len(inventory),
        "logical_product_count": len(products_df),
        "variant_count": len(variants_df),
        "sellable_variant_count": int(
            sum(1 for variant in variants if variant.get("sellable"))
        ),
    }
    return products_df, variants_df, metadata


def _prepare_products(products: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()

    if "product_type" in products.columns:
        products["product_type"] = products["product_type"].apply(
            lambda value: _canonical_product_type(value) or _clean_text(value)
        )

    numeric_columns = [
        "price_vnd",
        "sale_price_vnd",
        "rating",
        "sold_30d",
        "stock_total",
    ]
    for column in numeric_columns:
        if column in products.columns:
            products[column] = pd.to_numeric(products[column], errors="coerce").fillna(
                0
            )

    products["effective_price_vnd"] = products["sale_price_vnd"].where(
        products["sale_price_vnd"] > 0, products["price_vnd"]
    )
    products["tags_text"] = _as_text(products["tags"]).str.replace(
        "|", " ", regex=False
    )

    search_columns = [
        "product_name",
        "category_name",
        "product_type",
        "brand",
        "tags_text",
        "short_description",
    ]
    products["search_text"] = (
        products[search_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return products


def _prepare_variants(variants: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    variants = variants.copy()
    if "stock_qty" in variants.columns:
        variants["stock_qty"] = pd.to_numeric(
            variants["stock_qty"], errors="coerce"
        ).fillna(0)

    if not inventory.empty and {"variant_id", "available_qty"}.issubset(
        inventory.columns
    ):
        inventory = inventory.copy()
        inventory["available_qty"] = pd.to_numeric(
            inventory["available_qty"], errors="coerce"
        ).fillna(0)
        available = inventory.groupby("variant_id", as_index=False)[
            "available_qty"
        ].sum()
        variants = variants.merge(available, on="variant_id", how="left")
        variants["available_qty"] = variants["available_qty"].fillna(
            variants["stock_qty"]
        )
    else:
        variants["available_qty"] = variants.get("stock_qty", 0)

    return variants


def load_data(
    csv_dir: Path = CSV_DIR, include_product_csv: bool | None = None
) -> DataStore:
    source = catalog_source()
    shop_id = active_shop_id()
    if include_product_csv is None:
        include_product_csv = include_simulated_data()

    metadata = {
        "database_row_count": 0,
        "logical_product_count": 0,
        "variant_count": 0,
        "sellable_variant_count": 0,
    }

    if source == "database":
        inventory = _load_database_inventory(shop_id)
        products, variants, metadata = _prepare_database_catalog(inventory, shop_id)
        if include_product_csv:
            csv_products = _prepare_products(
                _read_csv(csv_dir, PRODUCT_FILES["products"])
            )
            csv_inventory = _read_csv(csv_dir, PRODUCT_FILES["inventory"])
            csv_variants = _prepare_variants(
                _read_csv(csv_dir, PRODUCT_FILES["variants"]), csv_inventory
            )
            products = pd.concat([products, csv_products], ignore_index=True)
            variants = pd.concat([variants, csv_variants], ignore_index=True)
    elif include_product_csv:
        products = _read_csv(csv_dir, PRODUCT_FILES["products"])
        variants = _read_csv(csv_dir, PRODUCT_FILES["variants"])
        inventory = _read_csv(csv_dir, PRODUCT_FILES["inventory"])
        products = _prepare_products(products)
        variants = _prepare_variants(variants, inventory)
        metadata["logical_product_count"] = len(products)
        metadata["variant_count"] = len(variants)
        metadata["sellable_variant_count"] = len(
            variants[variants["available_qty"].fillna(0).astype(float) > 0]
        )
    else:
        products = pd.DataFrame()
        variants = pd.DataFrame()
        inventory = pd.DataFrame()

    synonyms = _read_csv(csv_dir, SUPPORT_FILES["synonyms"])
    policies = _read_csv(csv_dir, SUPPORT_FILES["policies"])

    return DataStore(
        products=products,
        variants=variants,
        inventory=inventory,
        synonyms=synonyms,
        policies=policies,
        source=source,
        active_shop_id=shop_id,
        database_row_count=metadata["database_row_count"],
        logical_product_count=metadata["logical_product_count"],
        variant_count=metadata["variant_count"],
        sellable_variant_count=metadata["sellable_variant_count"],
    )
