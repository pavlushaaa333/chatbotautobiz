from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import product_data_source
from app.data_loader import DataStore, load_data
from app.size_recommender import load_size_chart

RAG_SUPPORTED_CATEGORIES = {"Fashion", "Accessories"}
SHOP_SCOPE_ITEMS = [
    "áo thun",
    "áo polo",
    "áo sơ mi",
    "áo khoác",
    "quần jeans",
    "quần kaki",
    "quần short",
    "váy/đầm",
    "đồ công sở",
    "phụ kiện thời trang",
]
SHOP_OUT_OF_SCOPE_ITEMS = [
    "bánh mì",
    "nước cam",
    "đồ ăn",
    "đồ uống",
    "trà sữa",
    "salad",
    "mỹ phẩm",
    "serum",
    "sữa rửa mặt",
    "mẹ và bé",
    "bỉm",
    "bình sữa",
    "giày sneaker",
    "giày dép nếu không có trong catalog",
    "đồ thể thao chuyên dụng nếu không có trong catalog",
]


@dataclass(frozen=True, slots=True)
class RAGDocument:
    id: str
    content: str
    metadata: dict[str, Any]


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _int_value(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _metadata(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _fashion_products(products: pd.DataFrame) -> pd.DataFrame:
    if products.empty or "category_code" not in products.columns:
        return products.iloc[0:0].copy()
    return products[
        products["category_code"].fillna("").astype(str).isin(RAG_SUPPORTED_CATEGORIES)
    ].copy()


def _product_document(row: dict[str, Any]) -> RAGDocument:
    product_id = _clean_value(row.get("product_id"))
    product_name = _clean_value(row.get("product_name"))
    product_type = _clean_value(row.get("product_type"))
    tags = _clean_value(row.get("tags")).replace("|", ", ")
    content = "\n".join(
        part
        for part in [
            f"Tên sản phẩm: {product_name}",
            f"Mã sản phẩm: {product_id}",
            f"Danh mục: {product_type or _clean_value(row.get('category_name'))}",
            f"Thương hiệu: {_clean_value(row.get('brand'))}",
            f"Giá gốc: {_int_value(row.get('price_vnd'))}",
            f"Giá sau giảm: {_int_value(row.get('effective_price_vnd') or row.get('sale_price_vnd'))}",
            f"Mô tả: {_clean_value(row.get('short_description'))}",
            f"Phong cách/từ khóa: {tags}" if tags else "",
        ]
        if part
    )
    return RAGDocument(
        id=f"product:{product_id}",
        content=content,
        metadata=_metadata(
            {
                "doc_type": "product",
                "source": f"product:{product_id}",
                "product_id": product_id,
                "product_name": product_name,
                "category_code": _clean_value(row.get("category_code")),
                "product_type": product_type,
            }
        ),
    )


def _variant_stock(row: dict[str, Any]) -> int:
    if "available_qty" in row:
        return _int_value(row.get("available_qty"))
    return _int_value(row.get("stock_qty"))


def _variant_document(
    row: dict[str, Any], product_by_id: dict[str, dict[str, Any]]
) -> RAGDocument:
    product_id = _clean_value(row.get("product_id"))
    product = product_by_id.get(product_id, {})
    product_name = _clean_value(product.get("product_name"))
    product_type = _clean_value(product.get("product_type"))
    sku = _clean_value(row.get("sku"))
    stock = _variant_stock(row)
    status = "Còn hàng" if stock > 0 else "Hết hàng"
    content = "\n".join(
        [
            f"Sản phẩm: {product_name}",
            f"Mã sản phẩm: {product_id}",
            f"SKU: {sku}",
            f"Size: {_clean_value(row.get('size'))}",
            f"Màu: {_clean_value(row.get('color'))}",
            f"Tồn kho: {stock}",
            f"Trạng thái: {status}",
            f"Giá sau giảm: {_int_value(product.get('effective_price_vnd') or product.get('sale_price_vnd'))}",
        ]
    )
    return RAGDocument(
        id=f"variant:{sku or _clean_value(row.get('variant_id'))}",
        content=content,
        metadata=_metadata(
            {
                "doc_type": "variant",
                "source": f"variant:{sku}",
                "product_id": product_id,
                "product_name": product_name,
                "sku": sku,
                "size": _clean_value(row.get("size")),
                "color": _clean_value(row.get("color")),
                "stock": stock,
                "category_code": _clean_value(product.get("category_code")),
                "product_type": product_type,
            }
        ),
    )


def _policy_documents(policies: pd.DataFrame) -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    for row in policies.to_dict("records") if not policies.empty else []:
        policy_key = _clean_value(row.get("policy_key"))
        policy_value = _clean_value(row.get("policy_value"))
        if not policy_key or not policy_value:
            continue
        documents.append(
            RAGDocument(
                id=f"policy:{policy_key}",
                content=f"Chính sách {policy_key}: {policy_value}",
                metadata=_metadata(
                    {
                        "doc_type": "policy",
                        "source": f"policy:{policy_key}",
                        "policy_type": policy_key,
                    }
                ),
            )
        )

    defaults = {
        "size_advice": "Shop có thể tư vấn size dựa trên nhu cầu, form dáng và size khách đang tìm.",
        "returns": "Chính sách đổi trả cần dựa trên chính sách shop đang cung cấp; không tự bịa điều kiện nếu dữ liệu chưa có.",
        "fashion_scope": (
            "AutoBiz Fashion chỉ tư vấn quần áo và phụ kiện thời trang trong phạm vi catalog hiện tại."
        ),
    }
    existing = {doc.metadata.get("policy_type") for doc in documents}
    for key, value in defaults.items():
        if key in existing:
            continue
        documents.append(
            RAGDocument(
                id=f"policy:{key}",
                content=value,
                metadata=_metadata(
                    {
                        "doc_type": "policy",
                        "source": f"policy:{key}",
                        "policy_type": key,
                    }
                ),
            )
        )
    return documents


def _size_chart_documents() -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    for row in load_size_chart():
        product_type = _clean_value(row.get("product_type"))
        gender = _clean_value(row.get("gender"))
        size = _clean_value(row.get("size"))
        if not product_type or not size:
            continue
        content = (
            f"Bảng size {product_type} {gender} size {size}: "
            f"chiều cao {row.get('height_min_cm')}-{row.get('height_max_cm')}cm, "
            f"cân nặng {row.get('weight_min_kg')}-{row.get('weight_max_kg')}kg. "
            f"Ghi chú form: {_clean_value(row.get('fit_note'))}."
        )
        documents.append(
            RAGDocument(
                id=f"size_chart:{product_type}:{gender}:{size}",
                content=content,
                metadata=_metadata(
                    {
                        "doc_type": "size_chart",
                        "source": f"size_chart:{product_type}:{gender}:{size}",
                        "product_type": product_type,
                        "gender": gender,
                        "size": size,
                    }
                ),
            )
        )
    return documents


def _catalog_scope_document() -> RAGDocument:
    content = (
        "AutoBiz Fashion là shop quần áo và phụ kiện thời trang.\n"
        "Shop bán:\n"
        + "\n".join(f"- {item}" for item in SHOP_SCOPE_ITEMS)
        + "\n\nShop không bán:\n"
        + "\n".join(f"- {item}" for item in SHOP_OUT_OF_SCOPE_ITEMS)
        + "\n\nVí dụ shop không bán bánh mì, nước cam, mỹ phẩm, serum, sữa rửa mặt, bỉm, bình sữa, giày sneaker hoặc quần áo đá bóng chuyên dụng.\n"
        "Nếu khách hỏi ngoài phạm vi, không search/gợi ý sai domain; hãy nói rõ shop không bán mặt hàng đó."
    )
    return RAGDocument(
        id="catalog_scope:autobiz_fashion",
        content=content,
        metadata=_metadata(
            {
                "doc_type": "catalog_scope",
                "source": "catalog_scope:autobiz_fashion",
            }
        ),
    )


def build_rag_documents(data_store: DataStore | None = None) -> list[RAGDocument]:
    store = data_store or load_data()
    products = (
        _fashion_products(store.products)
        if product_data_source() == "csv"
        else store.products.iloc[0:0].copy()
    )
    product_rows = products.to_dict("records")
    product_by_id = {_clean_value(row.get("product_id")): row for row in product_rows}

    documents: list[RAGDocument] = []
    documents.extend(_product_document(row) for row in product_rows)

    if not store.variants.empty and product_by_id:
        variants = store.variants[
            store.variants["product_id"].fillna("").astype(str).isin(product_by_id)
        ].copy()
        documents.extend(
            _variant_document(row, product_by_id) for row in variants.to_dict("records")
        )

    documents.extend(_policy_documents(store.policies))
    documents.extend(_size_chart_documents())
    documents.append(_catalog_scope_document())
    return documents
