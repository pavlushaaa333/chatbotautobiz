from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog_gate import CatalogDomainGate
from app.normalizer import (
    PRODUCT_TYPES_BY_CATEGORY,
    contains_phrase,
    normalize_text,
    product_type_to_category,
    strip_accents,
)
from app.product_repository import (
    ProductRepository,
    ProductRepositoryConfigurationError,
    ProductRepositoryError,
)


SALE_PRICE_FIELDS = ("sale_price", "sale_price_vnd")
REGULAR_PRICE_FIELDS = ("selling_price", "effective_price_vnd", "price_vnd", "price")
PRICE_FIELDS = (*SALE_PRICE_FIELDS, *REGULAR_PRICE_FIELDS)
SELLABLE_LISTING_STATUSES = {"active", "low_stock"}
ACTIVE_STATUSES = {"active", "in_stock"}
FASHION_INTENT_PREFIXES = {
    "áo": ("Áo",),
    "quần": ("Quần",),
    "váy": ("Váy", "Chân váy"),
}
SUPPORTED_CATALOG_CATEGORIES = {"Fashion", "Accessories"}
HEALTHY_LUNCH_TYPES = {"Combo ăn trưa", "Salad", "Cơm văn phòng", "Nước ép"}
SWEET_SNACK_TYPES = {"Trà sữa", "Bánh ngọt", "Đồ ăn vặt"}
PRODUCT_RESOLVE_STOPWORDS = {
    "ao",
    "so",
    "mi",
    "thun",
    "mau",
    "size",
    "co",
    "con",
    "khong",
    "shop",
    "mua",
    "muon",
    "tim",
    "xem",
    "can",
    "cho",
    "mau",
    "nam",
    "nu",
    "unisex",
    "di",
    "lam",
    "cong",
    "so",
    "trang",
    "den",
    "xanh",
    "do",
}


def _contains_phrase_exact(text: str, phrase: str) -> bool:
    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(phrase_norm)}(?!\w)", text_norm))


def _valid_price(value: Any) -> int | None:
    try:
        price = int(float(value))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def resolve_product_price(product: dict[str, Any]) -> int | None:
    for field in PRICE_FIELDS:
        price = _valid_price(product.get(field))
        if price is not None:
            return price
    return None


def _price_matches_criteria(price: int | None, criteria: dict[str, Any]) -> bool:
    budget_min = criteria.get("budget_min_vnd")
    budget_max = criteria.get("budget_max_vnd")
    price_direction = criteria.get("price_direction")
    if budget_min is None and budget_max is None:
        return True
    if price is None or pd.isna(price):
        return False
    price = int(price)

    if budget_min is not None:
        min_price = int(budget_min)
        if price_direction == "above":
            if price <= min_price:
                return False
        elif price < min_price:
            return False

    if budget_max is not None and price > int(budget_max):
        return False

    return True


def _status_key(value: Any) -> str:
    return normalize_text(str(value or "")).lower()


def _product_type_key(value: Any) -> str:
    return " ".join(strip_accents(normalize_text(str(value or ""))).casefold().split())


def _product_type_key_set(values: list[Any]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        key = _product_type_key(value)
        if key:
            keys.add(key)
    return keys


def _safe_stock(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _is_sellable_product(product: dict[str, Any]) -> bool:
    status = _status_key(product.get("inventory_status") or product.get("status"))
    if status and status not in ACTIVE_STATUSES:
        return False

    listing_status = _status_key(product.get("listing_status"))
    if listing_status and listing_status not in SELLABLE_LISTING_STATUSES:
        return False

    if resolve_product_price(product) is None:
        return False

    stock = _safe_stock(
        product.get("stock_total", product.get("available_qty", product.get("quantity")))
    )
    return stock > 0


def _is_sellable_variant(variant: dict[str, Any], stock: int) -> bool:
    if stock <= 0:
        return False

    status = _status_key(variant.get("inventory_status") or variant.get("status"))
    if status and status not in ACTIVE_STATUSES:
        return False

    listing_status = _status_key(variant.get("listing_status"))
    if listing_status and listing_status not in SELLABLE_LISTING_STATUSES:
        return False

    has_price_field = any(field in variant for field in PRICE_FIELDS)
    if has_price_field and resolve_product_price(variant) is None:
        return False

    return True


def _resolved_price_series(products: pd.DataFrame) -> pd.Series:
    values = [resolve_product_price(row) for row in products.to_dict("records")]
    return pd.Series(values, index=products.index, dtype="Int64")


class ProductSearchEngine:
    def __init__(self, products: pd.DataFrame, variants: pd.DataFrame):
        products = products[
            products["category_code"].fillna("").astype(str).isin(SUPPORTED_CATALOG_CATEGORIES)
        ].reset_index(drop=True).copy()
        self.products = products
        self.variants = variants[
            variants["product_id"].fillna("").astype(str).isin(set(products["product_id"].astype(str)))
        ].copy()
        self.catalog_gate = CatalogDomainGate(self.products, self.variants)
        self.variant_text_by_product = self._build_variant_text_by_product()
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products["search_text"].fillna(""))

    def _build_variant_text_by_product(self) -> dict[str, str]:
        if self.variants.empty or "product_id" not in self.variants.columns:
            return {}

        value_columns = [column for column in ["size", "color"] if column in self.variants.columns]
        if not value_columns:
            return {}

        variant_text: dict[str, str] = {}
        for product_id, group in self.variants.groupby("product_id"):
            values: list[str] = []
            for column in value_columns:
                values.extend(str(value) for value in group[column].dropna().unique() if str(value).strip())
            variant_text[str(product_id)] = " ".join(values)
        return variant_text

    def search(self, criteria: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        if criteria.get("catalog_coverage") in {"not_supported", "unsupported"}:
            return []
        if (criteria.get("catalog_gate") or {}).get("status") == "unsupported":
            return []

        candidates = self._filter_products(criteria)
        if candidates.empty:
            return []

        semantic_scores = self._semantic_scores(criteria, candidates)
        ranked = []
        for row in candidates.itertuples(index=False):
            row_dict = row._asdict()
            score, reasons = self._score_product(row_dict, criteria, semantic_scores)
            result = self._serialize_product(row_dict)
            result["score"] = round(score, 3)
            result["matched_reasons"] = reasons
            ranked.append(result)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def products_by_exact_type(self, product_type: str | None) -> list[dict[str, Any]]:
        requested_key = _product_type_key(product_type)
        if not requested_key or "product_type" not in self.products.columns:
            return []

        products = self.products.copy()
        if "status" in products.columns:
            products = products[products["status"].fillna("").astype(str).str.lower().eq("active")]
        product_type_keys = products["product_type"].fillna("").astype(str).map(_product_type_key)
        products = products[product_type_keys == requested_key]
        return [self._serialize_product(row._asdict()) for row in products.itertuples(index=False)]

    def check_availability(self, criteria: dict[str, Any]) -> dict[str, Any]:
        product = self._find_product_by_requested_name(criteria.get("requested_product_name"))
        if product is None:
            return {
                "status": "product_not_found",
                "product": None,
                "variant": None,
                "alternatives": [],
            }

        product_id = str(product["product_id"])
        product_variants = self.variants[
            self.variants["product_id"].fillna("").astype(str) == product_id
        ].copy()
        matched_variant = self._find_variant(product_variants, criteria.get("size"), criteria.get("color"))
        product_payload = self._serialize_product(product)

        if matched_variant is not None:
            variant_payload = self._serialize_variant(matched_variant)
            if variant_payload["stock"] > 0:
                return {
                    "status": "in_stock",
                    "product": product_payload,
                    "variant": variant_payload,
                    "alternatives": [],
                }

            return {
                "status": "out_of_stock",
                "product": product_payload,
                "variant": variant_payload,
                "alternatives": self._alternative_variants(product_payload, product_variants, matched_variant),
            }

        return {
            "status": "variant_not_found",
            "product": product_payload,
            "variant": None,
            "alternatives": self._alternative_variants(product_payload, product_variants, None),
        }

    def check_variant_availability(self, criteria: dict[str, Any]) -> dict[str, Any]:
        product = self._resolve_product_for_criteria(criteria)
        if product is None:
            return {
                "status": "product_not_found",
                "product": None,
                "variant": None,
                "alternatives": [],
            }

        product_id = str(product["product_id"])
        product_variants = self.variants[
            self.variants["product_id"].fillna("").astype(str) == product_id
        ].copy()
        product_payload = self._serialize_product(product)
        size = criteria.get("size")
        color = criteria.get("color")

        if size and color:
            matched_variant = self._find_variant(product_variants, size, color)
            if matched_variant is not None:
                variant_payload = self._serialize_variant(matched_variant)
                if variant_payload["stock"] > 0:
                    return {
                        "status": "in_stock",
                        "product": product_payload,
                        "variant": variant_payload,
                        "alternatives": [],
                    }
                return {
                    "status": "out_of_stock",
                    "product": product_payload,
                    "variant": variant_payload,
                    "alternatives": self._alternative_variants(product_payload, product_variants, matched_variant),
                }

            return {
                "status": "variant_not_found",
                "product": product_payload,
                "variant": None,
                "alternatives": self._alternative_variants(product_payload, product_variants, None),
            }

        filtered = product_variants.copy()
        if size:
            size_norm = strip_accents(normalize_text(str(size)))
            filtered = filtered[
                filtered["size"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                == size_norm
            ]
        if color:
            color_norm = strip_accents(normalize_text(str(color)))
            filtered = filtered[
                filtered["color"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                == color_norm
            ]

        if filtered.empty:
            return {
                "status": "variant_not_found",
                "product": product_payload,
                "variant": None,
                "alternatives": self._alternative_variants(product_payload, product_variants, None),
            }

        matched_variants = filtered.to_dict("records")
        options = [
            {**product_payload, **self._serialize_variant(variant)}
            for variant in matched_variants
            if self._serialize_variant(variant)["stock"] > 0
        ]
        if not options:
            matched_variant = matched_variants[0]
            return {
                "status": "out_of_stock",
                "product": product_payload,
                "variant": self._serialize_variant(matched_variant),
                "alternatives": self._alternative_variants(product_payload, product_variants, matched_variant),
            }

        if len(options) == 1:
            variant_payload = {
                key: options[0].get(key)
                for key in [
                    "variant_id",
                    "sku",
                    "size",
                    "color",
                    "stock",
                    "status",
                    "listing_status",
                    "inventory_status",
                ]
            }
            return {
                "status": "in_stock",
                "product": product_payload,
                "variant": variant_payload,
                "alternatives": [],
            }

        return {
            "status": "variant_options",
            "product": product_payload,
            "variant": None,
            "alternatives": options,
        }

    def product_by_id(self, product_id: str | None) -> dict[str, Any] | None:
        product = self._find_product_by_id(product_id)
        return self._serialize_product(product) if product is not None else None

    def resolve_product_from_text(
        self,
        text: str,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        product = self._resolve_product_row_from_text(text, criteria or {})
        return self._serialize_product(product) if product is not None else None

    def _filter_products(self, criteria: dict[str, Any]) -> pd.DataFrame:
        products = self.products.copy()
        products = products[products["status"].fillna("").str.lower().eq("active")]
        if not products.empty:
            products = products[
                products.apply(lambda row: _is_sellable_product(row.to_dict()), axis=1)
            ]

        exclude_ids = {str(product_id) for product_id in criteria.get("exclude_product_ids") or [] if product_id}
        if exclude_ids:
            products = products[~products["product_id"].fillna("").astype(str).isin(exclude_ids)]

        category = criteria.get("category_code")
        if category:
            products = products[products["category_code"].fillna("").str.lower() == str(category).lower()]

        products = self._apply_product_type_filter(products, criteria)

        if (
            criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
        ):
            prices = _resolved_price_series(products)
            products = products[
                prices.map(lambda price: _price_matches_criteria(price, criteria))
            ]

        if criteria.get("must_be_in_stock", False):
            products = products[products["stock_total"] > 0]

        products = self._apply_soft_gender_filter(products, criteria.get("gender"))

        variant_product_ids = self._matching_variant_product_ids(criteria)
        if variant_product_ids is not None:
            if variant_product_ids:
                variant_matched = products[products["product_id"].isin(variant_product_ids)]
                if not variant_matched.empty:
                    products = variant_matched
                else:
                    criteria["variant_filter_relaxed"] = True
            else:
                criteria["variant_filter_relaxed"] = True

        return products

    def _resolve_product_for_criteria(self, criteria: dict[str, Any]) -> dict[str, Any] | None:
        for field in ["selected_product_id", "product_id"]:
            product = self._find_product_by_id(criteria.get(field))
            if product is not None:
                return product

        product = self._find_product_by_requested_name(criteria.get("requested_product_name"))
        if product is not None:
            return product

        return self._resolve_product_row_from_text(
            str(criteria.get("raw_message") or criteria.get("normalized_message") or ""),
            criteria,
        )

    def _find_product_by_id(self, product_id: str | None) -> dict[str, Any] | None:
        if not product_id:
            return None
        product_id = str(product_id)
        active_products = self.products[self.products["status"].fillna("").str.lower().eq("active")]
        matched = active_products[active_products["product_id"].fillna("").astype(str) == product_id]
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()

    def _find_product_by_requested_name(self, requested_product_name: str | None) -> dict[str, Any] | None:
        if not requested_product_name:
            return None

        requested_norm = strip_accents(normalize_text(str(requested_product_name)))
        active_products = self.products[self.products["status"].fillna("").str.lower().eq("active")]
        for row in active_products.itertuples(index=False):
            product = row._asdict()
            product_norm = strip_accents(normalize_text(str(product.get("product_name") or "")))
            if product_norm == requested_norm:
                return product

        for row in active_products.itertuples(index=False):
            product = row._asdict()
            product_norm = strip_accents(normalize_text(str(product.get("product_name") or "")))
            if requested_norm in product_norm or product_norm in requested_norm:
                return product

        return None

    def _resolve_product_row_from_text(
        self,
        text: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any] | None:
        text_key = strip_accents(normalize_text(text))
        if not text_key:
            return None

        best: tuple[float, dict[str, Any]] | None = None
        active_products = self.products[self.products["status"].fillna("").str.lower().eq("active")]
        for row in active_products.itertuples(index=False):
            product = row._asdict()
            product_name = str(product.get("product_name") or "")
            product_name_key = strip_accents(normalize_text(product_name))
            product_type = str(product.get("product_type") or "")
            score = 0.0
            matched_name_token = False

            if product_name_key and (
                contains_phrase(text_key, product_name_key) or product_name_key in text_key
            ):
                score += 120
                matched_name_token = True

            name_tokens = [
                token
                for token in re.findall(r"\w+", product_name_key)
                if len(token) > 2 and token not in PRODUCT_RESOLVE_STOPWORDS
            ]
            for token in name_tokens:
                if contains_phrase(text_key, token):
                    score += 18 + min(len(token), 10)
                    matched_name_token = True

            if not matched_name_token:
                continue

            if criteria.get("product_type") and product_type == criteria.get("product_type"):
                score += 12
            if criteria.get("category_code") and product.get("category_code") == criteria.get("category_code"):
                score += 6

            tags_key = strip_accents(normalize_text(str(product.get("tags") or "").replace("|", " ")))
            for keyword in criteria.get("keywords") or []:
                keyword_key = strip_accents(normalize_text(str(keyword)))
                if keyword_key and contains_phrase(tags_key, keyword_key):
                    score += 3

            if best is None or score > best[0]:
                best = (score, product)

        if best is None or best[0] < 18:
            return None
        return best[1]

    def _find_variant(
        self,
        variants: pd.DataFrame,
        size: str | None,
        color: str | None,
    ) -> dict[str, Any] | None:
        if variants.empty:
            return None

        filtered = variants.copy()
        if size:
            size_norm = strip_accents(normalize_text(str(size)))
            filtered = filtered[
                filtered["size"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                == size_norm
            ]

        if color:
            color_norm = strip_accents(normalize_text(str(color)))
            filtered = filtered[
                filtered["color"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                == color_norm
            ]

        if filtered.empty:
            return None
        return filtered.iloc[0].to_dict()

    def _variant_stock(self, variant: dict[str, Any]) -> int:
        if "available_qty" in variant and not pd.isna(variant.get("available_qty")):
            stock = int(float(variant.get("available_qty") or 0))
        else:
            stock = int(float(variant.get("stock_qty") or 0))
        return stock if _is_sellable_variant(variant, stock) else 0

    def _serialize_variant(self, variant: dict[str, Any]) -> dict[str, Any]:
        stock = self._variant_stock(variant)
        status = str(variant.get("status") or "").strip() or ("in_stock" if stock > 0 else "out_of_stock")
        return {
            "variant_id": str(variant.get("variant_id") or ""),
            "sku": str(variant.get("sku") or ""),
            "size": str(variant.get("size") or ""),
            "color": str(variant.get("color") or ""),
            "stock": stock,
            "quantity": stock,
            "available_qty": stock,
            "listing_status": str(variant.get("listing_status") or ""),
            "inventory_status": str(variant.get("inventory_status") or ""),
            "source": str(variant.get("source") or ""),
            "status": "in_stock" if stock > 0 and status != "out_of_stock" else "out_of_stock",
        }

    def _alternative_variants(
        self,
        product: dict[str, Any],
        variants: pd.DataFrame,
        matched_variant: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if variants.empty:
            return []

        candidates = variants.copy()
        if matched_variant and matched_variant.get("size"):
            size_norm = strip_accents(normalize_text(str(matched_variant["size"])))
            candidates = candidates[
                candidates["size"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                == size_norm
            ]
            color_norm = strip_accents(normalize_text(str(matched_variant.get("color") or "")))
            candidates = candidates[
                candidates["color"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)))
                != color_norm
            ]

        alternatives: list[dict[str, Any]] = []
        for variant in candidates.to_dict("records"):
            variant_payload = self._serialize_variant(variant)
            if variant_payload["stock"] <= 0:
                continue
            alternatives.append(
                {
                    **product,
                    **variant_payload,
                }
            )
        return alternatives

    def _apply_product_type_filter(self, products: pd.DataFrame, criteria: dict[str, Any]) -> pd.DataFrame:
        product_type = criteria.get("product_type")
        apparel_intent = criteria.get("apparel_intent")
        product_types = products["product_type"].fillna("").astype(str)
        product_type_keys = product_types.map(_product_type_key)
        product_type_key = _product_type_key(product_type)
        product_type_value_keys = _product_type_key_set(criteria.get("product_types") or [])

        if product_type_key:
            return products[product_type_keys == product_type_key]

        if product_type_value_keys:
            return products[product_type_keys.isin(product_type_value_keys)]

        if criteria.get("category_code") == "Fashion":
            prefixes = FASHION_INTENT_PREFIXES.get(str(apparel_intent))
            if prefixes:
                return products[product_types.map(lambda value: value.startswith(prefixes))]

            return products

        return products

    def _apply_soft_gender_filter(self, products: pd.DataFrame, gender: str | None) -> pd.DataFrame:
        if not gender or products.empty or "tags" not in products.columns:
            return products

        tags = products["tags"].fillna("").astype(str).str.lower()
        matched = products[tags.map(lambda value: contains_phrase(value.replace("|", " "), gender))]
        return matched if not matched.empty else products

    def _matching_variant_product_ids(self, criteria: dict[str, Any]) -> set[str] | None:
        size = criteria.get("size")
        color = criteria.get("color")
        if not size and not color:
            return None

        variants = self.variants.copy()
        if "available_qty" in variants.columns:
            variants = variants[pd.to_numeric(variants["available_qty"], errors="coerce").fillna(0) > 0]
        if not variants.empty:
            variants = variants[
                variants.apply(
                    lambda row: _is_sellable_variant(
                        row.to_dict(),
                        _safe_stock(row.to_dict().get("available_qty", row.to_dict().get("stock_qty"))),
                    ),
                    axis=1,
                )
            ]

        if size:
            size_norm = strip_accents(normalize_text(str(size))).lower()
            variant_size = variants["size"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)).lower())
            variants = variants[variant_size.isin({size_norm, f"size {size_norm}"})]

        if color:
            color_norm = strip_accents(normalize_text(str(color))).lower()
            variant_color = variants["color"].fillna("").astype(str).map(lambda value: strip_accents(normalize_text(value)).lower())
            variants = variants[variant_color == color_norm]

        return set(variants["product_id"].astype(str).tolist())

    def _semantic_scores(self, criteria: dict[str, Any], candidates: pd.DataFrame) -> dict[str, float]:
        query = self._query_text(criteria)
        if not query.strip() or candidates.empty:
            return {}
        query_vector = self.vectorizer.transform([query])
        candidate_matrix = self.tfidf_matrix[candidates.index.to_list()]
        scores = cosine_similarity(query_vector, candidate_matrix).ravel()
        return {
            product_id: float(score)
            for product_id, score in zip(candidates["product_id"].astype(str), scores)
        }

    def _query_text(self, criteria: dict[str, Any]) -> str:
        parts = [
            criteria.get("normalized_message"),
            criteria.get("product_type"),
            " ".join(str(item) for item in criteria.get("product_types") or []),
            criteria.get("raw_product_type"),
            criteria.get("category_code"),
            criteria.get("color"),
            criteria.get("size"),
            criteria.get("gender"),
            criteria.get("style"),
            criteria.get("use_case"),
            criteria.get("skin_type"),
            criteria.get("concern"),
            criteria.get("age_group"),
            criteria.get("apparel_intent"),
        ]
        parts.extend(criteria.get("keywords") or [])
        return " ".join(str(part) for part in parts if part)

    def _score_product(
        self,
        product: dict[str, Any],
        criteria: dict[str, Any],
        semantic_scores: dict[str, float],
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []

        if criteria.get("category_code") and product.get("category_code") == criteria["category_code"]:
            score += 30
            reasons.append("đúng nhóm sản phẩm")

        product_type = str(product.get("product_type") or "")
        product_type_key = _product_type_key(product_type)
        requested_product_type_key = _product_type_key(criteria.get("product_type"))
        product_type_value_keys = _product_type_key_set(criteria.get("product_types") or [])
        if requested_product_type_key and product_type_key == requested_product_type_key:
            score += 45
            reasons.append("đúng loại sản phẩm")
            if product_type_value_keys:
                score += 8
                reasons.append("đúng loại ưu tiên")
        elif not requested_product_type_key and product_type_value_keys and product_type_key in product_type_value_keys:
            score += 45
            reasons.append("đúng loại sản phẩm")

        apparel_intent = criteria.get("apparel_intent")
        prefixes = FASHION_INTENT_PREFIXES.get(str(apparel_intent))
        if not requested_product_type_key and not product_type_value_keys and prefixes and product_type.startswith(prefixes):
            score += 35
            reasons.append(f"đúng nhóm {apparel_intent}")

        budget = criteria.get("budget_max_vnd")
        price = resolve_product_price(product)
        if budget and price is not None and price <= int(budget):
            score += 15
            reasons.append("giá trong ngân sách")

        if product.get("stock_total", 0) > 0:
            score += 20
            reasons.append("còn hàng")

        if criteria.get("variant_filter_relaxed") and (criteria.get("size") or criteria.get("color")):
            reasons.append("cùng loại sản phẩm, chưa khớp chính xác màu/size đã hỏi")

        searchable = normalize_text(
            " ".join(
                str(product.get(column, ""))
                for column in ["product_name", "tags", "short_description", "search_text"]
            )
            + " "
            + self.variant_text_by_product.get(str(product.get("product_id")), "")
        )
        searchable_no_accents = strip_accents(searchable)
        color = normalize_text(str(criteria.get("color") or ""))
        if self._has_healthy_lunch_intent(criteria):
            if product_type in HEALTHY_LUNCH_TYPES:
                score += 35
                reasons.append("phù hợp bữa trưa healthy")
            elif product_type in SWEET_SNACK_TYPES and not contains_phrase(searchable, "healthy"):
                score -= 35

        for keyword in criteria.get("keywords") or []:
            keyword_norm = normalize_text(str(keyword))
            if color and keyword_norm == color:
                if _contains_phrase_exact(searchable, keyword_norm):
                    score += 18
                    reasons.append(f"có màu {keyword}")
                continue
            if keyword_norm and (
                contains_phrase(searchable, keyword_norm)
                or contains_phrase(searchable_no_accents, strip_accents(keyword_norm))
            ):
                score += 10
                reasons.append(f"khớp từ khóa '{keyword}'")

        gender = criteria.get("gender")
        if gender and contains_phrase(searchable, str(gender)):
            score += 12
            reasons.append(f"phù hợp giới tính {gender}")

        if float(product.get("rating", 0) or 0) >= 4.5:
            score += 5
            reasons.append("rating tốt")

        if int(product.get("sold_30d", 0) or 0) >= 100:
            score += 5
            reasons.append("bán chạy 30 ngày")

        semantic_score = semantic_scores.get(str(product["product_id"]), 0.0)
        if semantic_score > 0:
            score += semantic_score * 30
            reasons.append("nội dung mô tả gần với nhu cầu")

        return score, reasons

    def _has_healthy_lunch_intent(self, criteria: dict[str, Any]) -> bool:
        if criteria.get("category_code") != "Food":
            return False
        query = normalize_text(
            " ".join(
                str(part)
                for part in [criteria.get("normalized_message"), *(criteria.get("keywords") or [])]
                if part
            )
        )
        return any(
            contains_phrase(query, phrase)
            for phrase in ["healthy", "ít calo", "ăn trưa", "buổi trưa", "trưa"]
        )

    def _serialize_product(self, product: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "product_id",
            "product_name",
            "category_code",
            "category_name",
            "product_type",
            "brand",
            "tags",
            "short_description",
            "shop_id",
            "sku",
            "color",
            "size",
            "quantity",
            "available_qty",
            "price_vnd",
            "sale_price_vnd",
            "sale_price",
            "selling_price",
            "effective_price_vnd",
            "stock_total",
            "listing_status",
            "inventory_status",
            "source",
            "gender",
            "use_case",
            "style",
            "rating",
            "sold_30d",
        ]
        result: dict[str, Any] = {}
        for field in fields:
            value = product.get(field)
            if pd.isna(value):
                result[field] = None
            elif field in {
                "price_vnd",
                "sale_price_vnd",
                "sale_price",
                "selling_price",
                "effective_price_vnd",
                "stock_total",
                "quantity",
                "available_qty",
                "sold_30d",
            }:
                result[field] = int(value)
            elif field == "rating":
                result[field] = float(value)
            else:
                result[field] = str(value)
        price = resolve_product_price(product)
        if price is not None:
            result["effective_price_vnd"] = price
        return result


class StaticCatalogDimensions:
    def available_product_types(self, category_code: str | None = None) -> list[str]:
        if category_code:
            return list(PRODUCT_TYPES_BY_CATEGORY.get(category_code, []))
        values: list[str] = []
        for supported_category in sorted(SUPPORTED_CATALOG_CATEGORIES):
            values.extend(PRODUCT_TYPES_BY_CATEGORY.get(supported_category, []))
        return list(dict.fromkeys(values))

    def summary(self) -> dict[str, Any]:
        return {
            "category": sorted(SUPPORTED_CATALOG_CATEGORIES),
            "product_type": self.available_product_types(),
            "tags": [],
            "material": [],
            "gender": [],
            "style": [],
            "color": [],
            "size": [],
        }


class StaticCatalogGate:
    def __init__(self) -> None:
        self.dimensions = StaticCatalogDimensions()

    def apply(self, criteria: dict[str, Any]) -> dict[str, Any]:
        gated = dict(criteria)
        category_code = gated.get("category_code")
        product_type = gated.get("product_type")

        if not category_code and product_type:
            category_code = product_type_to_category(str(product_type))
            gated["category_code"] = category_code

        gated["catalog_dimensions"] = self.dimensions.summary()
        gated["catalog_available_product_types"] = self.dimensions.available_product_types()

        if (
            category_code
            and category_code not in SUPPORTED_CATALOG_CATEGORIES
            and gated.get("intent") not in {"policy_question", "negative_feedback"}
        ):
            requested = product_type or gated.get("requested_product_group") or "sản phẩm này"
            gated.update(
                {
                    "intent": "out_of_scope_request",
                    "category_code": None,
                    "product_type": None,
                    "product_types": [],
                    "catalog_coverage": "unsupported",
                    "catalog_gate": {
                        "status": "unsupported",
                        "category_code": None,
                        "product_types": [],
                        "unsupported_terms": [requested],
                        "requested_product_concepts": [requested],
                        "matched_chunks": [],
                    },
                    "requested_product_group": requested,
                    "out_of_scope_items": [requested],
                    "apparel_intent": None,
                    "keywords": [],
                    "need_clarification": False,
                    "clarification_questions": [],
                }
            )
            return gated

        if gated.get("intent") == "general_buying_intent" and not gated.get("category_code"):
            gated["category_code"] = "Fashion"

        gated["catalog_gate"] = {
            "status": "supported",
            "category_code": gated.get("category_code"),
            "product_types": gated.get("product_types") or [],
            "unsupported_terms": [],
            "requested_product_concepts": [],
            "matched_chunks": [],
        }
        gated["catalog_coverage"] = "supported"
        gated.setdefault("out_of_scope_items", [])
        return gated

    def canonical_product_type(self, text: str) -> str | None:
        text_key = normalize_text(text)
        for product_type in self.dimensions.available_product_types():
            if contains_phrase(text_key, product_type):
                return product_type
        return None


class PostgresProductSearchEngine:
    def __init__(self, repository: ProductRepository | None = None):
        self.repository = repository or ProductRepository()
        self.catalog_gate = StaticCatalogGate()
        self.last_error: str | None = None

    def search(self, criteria: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        self.last_error = None
        if criteria.get("catalog_coverage") in {"not_supported", "unsupported"}:
            return []
        if (criteria.get("catalog_gate") or {}).get("status") == "unsupported":
            return []

        try:
            rows = self.repository.search_products(
                keyword=self._keyword(criteria),
                product_type=criteria.get("product_type"),
                category_code=criteria.get("category_code"),
                color=criteria.get("color"),
                size=criteria.get("size"),
                max_price=criteria.get("budget_max_vnd"),
                must_be_in_stock=criteria.get("must_be_in_stock"),
                shop_id=criteria.get("shop_id"),
                limit=(
                    max(top_k, 50)
                    if criteria.get("budget_min_vnd") is not None
                    else top_k
                ),
            )
        except ProductRepositoryError as exc:
            self.last_error = str(exc)
            return []

        results: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            product = self._product_payload(row, criteria)
            price = resolve_product_price(product)
            if not _price_matches_criteria(price, criteria):
                continue
            product["score"] = round(100.0 - index, 3)
            product["matched_reasons"] = ["dữ liệu sản phẩm từ PostgreSQL"]
            results.append(product)
            if len(results) >= top_k:
                break
        return results

    def check_availability(self, criteria: dict[str, Any]) -> dict[str, Any]:
        return self.check_variant_availability(criteria)

    def check_variant_availability(self, criteria: dict[str, Any]) -> dict[str, Any]:
        self.last_error = None
        try:
            product = self._resolve_product_for_criteria(criteria)
            if product is None:
                return self._availability("product_not_found")

            variants = self._variants_for_product(product, criteria)
            selected = self._select_variant(
                variants or [product],
                size=criteria.get("size"),
                color=criteria.get("color"),
            )
            product_payload = self._product_payload(product, criteria)

            if selected is None:
                return self._availability(
                    "variant_not_found",
                    product=product_payload,
                    alternatives=self._in_stock_alternatives(product_payload, variants),
                )

            variant_payload = self._variant_payload(selected)
            if int(variant_payload.get("stock") or 0) > 0:
                return self._availability(
                    "in_stock",
                    product=product_payload,
                    variant=variant_payload,
                )

            return self._availability(
                "out_of_stock",
                product=product_payload,
                variant=variant_payload,
                alternatives=self._in_stock_alternatives(product_payload, variants, selected),
            )
        except ProductRepositoryError as exc:
            self.last_error = str(exc)
            return self._availability("database_error", error=str(exc))

    def product_by_id(self, product_id: str | None) -> dict[str, Any] | None:
        self.last_error = None
        if not product_id:
            return None
        try:
            product = self.repository.get_product_by_id(str(product_id))
        except ProductRepositoryError as exc:
            self.last_error = str(exc)
            return None
        return self._product_payload(product, {}) if product else None

    def resolve_product_from_text(
        self,
        text: str,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        criteria = dict(criteria or {})
        criteria["raw_message"] = text
        products = self.search(criteria, top_k=1)
        return products[0] if products else None

    def products_by_exact_type(self, product_type: str | None) -> list[dict[str, Any]]:
        requested_key = _product_type_key(product_type)
        if not requested_key:
            return []

        self.last_error = None
        try:
            rows = self.repository.search_products(
                keyword=None,
                product_type=product_type,
                category_code=None,
                color=None,
                size=None,
                max_price=None,
                must_be_in_stock=False,
                shop_id=None,
                limit=50,
            )
        except ProductRepositoryError as exc:
            self.last_error = str(exc)
            return []

        products: list[dict[str, Any]] = []
        for row in rows:
            product = self._product_payload(row, {"product_type": product_type})
            if _product_type_key(product.get("product_type")) == requested_key:
                products.append(product)
        return products

    def _resolve_product_for_criteria(self, criteria: dict[str, Any]) -> dict[str, Any] | None:
        sku = criteria.get("sku") or criteria.get("selected_variant_sku")
        if sku:
            return self.repository.get_product_by_sku(str(sku), shop_id=criteria.get("shop_id"))

        for field in ["selected_product_id", "product_id"]:
            product_id = criteria.get(field)
            if product_id:
                return self.repository.get_product_by_id(
                    str(product_id),
                    shop_id=criteria.get("shop_id"),
                )

        rows = self.repository.search_products(
            keyword=self._keyword(criteria),
            product_type=criteria.get("product_type"),
            category_code=criteria.get("category_code"),
            color=criteria.get("color"),
            size=criteria.get("size"),
            max_price=criteria.get("budget_max_vnd"),
            must_be_in_stock=False,
            shop_id=criteria.get("shop_id"),
            limit=1,
        )
        return rows[0] if rows else None

    def _variants_for_product(
        self,
        product: dict[str, Any],
        criteria: dict[str, Any],
    ) -> list[dict[str, Any]]:
        product_id = product.get("product_id")
        if not product_id:
            return [product] if product.get("sku") else []
        try:
            variants = self.repository.get_available_variants(
                product_id=str(product_id),
                shop_id=criteria.get("shop_id"),
                limit=50,
            )
        except ProductRepositoryConfigurationError:
            if product.get("sku"):
                return [product]
            raise
        return variants or ([product] if product.get("sku") else [])

    def _select_variant(
        self,
        variants: list[dict[str, Any]],
        *,
        size: str | None,
        color: str | None,
    ) -> dict[str, Any] | None:
        candidates = variants
        if size:
            size_key = self._key(size)
            candidates = [
                variant for variant in candidates if self._key(variant.get("size")) == size_key
            ]
        if color:
            color_key = self._key(color)
            candidates = [
                variant for variant in candidates if self._key(variant.get("color")) == color_key
            ]
        if not candidates:
            return None
        return candidates[0]

    def _in_stock_alternatives(
        self,
        product: dict[str, Any],
        variants: list[dict[str, Any]],
        selected: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        alternatives: list[dict[str, Any]] = []
        selected_sku = str((selected or {}).get("sku") or "")
        for variant in variants:
            payload = self._variant_payload(variant)
            if int(payload.get("stock") or 0) <= 0:
                continue
            if selected_sku and payload.get("sku") == selected_sku:
                continue
            alternatives.append({**product, **payload})
        return alternatives

    def _product_payload(self, product: dict[str, Any] | None, criteria: dict[str, Any]) -> dict[str, Any]:
        product = dict(product or {})
        product_id = product.get("product_id") or product.get("sku") or ""
        price = resolve_product_price(product)
        qty = int(product.get("stock_total", product.get("available_qty") or 0) or 0)
        product["product_id"] = str(product_id)
        product["product_name"] = str(product.get("product_name") or "")
        product["effective_price_vnd"] = price
        product["sale_price_vnd"] = (
            _valid_price(product.get("sale_price_vnd"))
            or _valid_price(product.get("sale_price"))
            or price
        )
        product["price_vnd"] = (
            _valid_price(product.get("price_vnd"))
            or _valid_price(product.get("selling_price"))
            or _valid_price(product.get("price"))
            or price
        )
        product["stock_total"] = qty
        product["category_code"] = product.get("category_code") or criteria.get("category_code")
        product["product_type"] = product.get("product_type") or criteria.get("product_type")
        return product

    def _variant_payload(self, variant: dict[str, Any]) -> dict[str, Any]:
        stock = int(variant.get("stock", variant.get("available_qty") or 0) or 0)
        return {
            "variant_id": str(variant.get("variant_id") or variant.get("sku") or ""),
            "sku": str(variant.get("sku") or ""),
            "size": variant.get("size"),
            "color": variant.get("color"),
            "stock": stock,
            "status": "in_stock" if stock > 0 else "out_of_stock",
        }

    def _availability(
        self,
        status: str,
        *,
        product: dict[str, Any] | None = None,
        variant: dict[str, Any] | None = None,
        alternatives: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "product": product,
            "variant": variant,
            "alternatives": alternatives or [],
        }
        if error:
            payload["error"] = error
        return payload

    def _keyword(self, criteria: dict[str, Any]) -> str | None:
        for field in ["requested_product_name", "raw_message", "normalized_message"]:
            value = str(criteria.get(field) or "").strip()
            if value:
                return value
        keywords = [str(item) for item in criteria.get("keywords") or [] if str(item).strip()]
        return " ".join(keywords) if keywords else None

    def _key(self, value: Any) -> str:
        return strip_accents(normalize_text(str(value or ""))).lower()
