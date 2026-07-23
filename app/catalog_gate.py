from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.normalizer import (
    CATEGORY_KEYWORDS,
    COLORS,
    STYLES,
    contains_phrase,
    detect_category,
    detect_product_type,
    detect_product_types,
    normalize_text,
    product_type_to_category,
    strip_accents,
)


REQUEST_ANCHORS = [
    "có bán",
    "có",
    "còn",
    "mua",
    "bán",
    "cần",
    "tìm",
    "gợi ý",
    "tư vấn",
    "recommend",
    "thêm",
]
REQUEST_SPLIT_PATTERN = re.compile(r"\b(?:và thêm|hoặc|và|thêm)\b")
REQUEST_CUTOFF_PATTERN = re.compile(
    r"\b(?:shop có|shop co|không|khong|ko|ạ|nhé|nhe|giá|gia|dưới|duoi|trên|tren|"
    r"size|màu|mau|dành cho|danh cho|cho|để|de|còn|con|k)\b"
)
GENERIC_FILLER_PHRASES = [
    "shop",
    "mình",
    "minh",
    "tôi",
    "toi",
    "bạn",
    "ban",
    "mấy",
    "may",
    "mua",
    "bán",
    "sản phẩm",
    "san pham",
    "nhóm sản phẩm",
    "nhom san pham",
    "một cái",
    "mot cai",
    "một",
    "mot",
    "cái",
    "cai",
]
GENERIC_CATALOG_QUERY_CHUNKS = {
    "gi",
    "nao",
    "cai nao",
    "co cai nao",
    "nhung gi",
    "san pham gi",
    "nhung san pham gi",
    "san pham nao",
    "co san pham nao",
    "nhung san pham nao",
    "ban gi",
    "co gi",
    "mat hang gi",
    "nhung mat hang gi",
    "mat hang nao",
    "mon nao",
    "co mon nao",
    "nhung mat hang nao",
    "loai hang gi",
    "nhung loai hang gi",
    "loai hang nao",
    "nhung loai hang nao",
    "nhom san pham gi",
    "nhung nhom san pham gi",
    "nhom san pham nao",
    "nhung nhom san pham nao",
    "mau nao",
    "co mau nao",
    "nhung mau nao",
    "loai nao",
    "nhung loai nao",
}
ATTRIBUTE_STOPWORDS = {
    "ao",
    "ban",
    "basic",
    "best",
    "cai",
    "can",
    "classic",
    "co",
    "con",
    "daily",
    "de",
    "dep",
    "di",
    "do",
    "hang",
    "he",
    "khong",
    "mac",
    "mau",
    "minh",
    "moi",
    "mot",
    "nao",
    "nhe",
    "nha",
    "o",
    "plus",
    "san",
    "seller",
    "shop",
    "size",
    "style",
    "toi",
    "xin",
}
FASHION_GENERIC_TERMS = {"áo", "quần áo", "đồ", "outfit", "trang phục", "thời trang"}
ACCESSORY_GENERIC_TERMS = {"phụ kiện"}
GENDER_TERMS = {"nam", "nữ", "unisex", "phi giới tính"}
SUPPORTED_FALLBACK_PRODUCT_TYPES = {
    "áo ba lỗ": "Áo thun",
    "ba lỗ": "Áo thun",
}
MATERIAL_HINTS = {
    "cotton",
    "denim",
    "jeans",
    "jean",
    "kaki",
    "linen",
    "lụa",
    "lua",
    "oxford",
}
SPECIFIC_PRODUCT_CONCEPT_ALIASES: list[tuple[str, list[str]]] = [
    (
        "bộ quần áo đá bóng",
        [
            "bộ quần áo đá bóng",
            "bộ đồ đá bóng",
            "quần áo đá bóng",
            "đồ đá bóng",
            "đồ bóng đá",
            "bộ đồ bóng đá",
            "bộ bóng đá",
            "football kit",
        ],
    ),
    ("áo bóng đá", ["áo bóng đá"]),
    ("áo đấu", ["áo đấu", "jersey"]),
    ("đồ thể thao", ["đồ thể thao", "đồ thể thao chuyên dụng", "quần áo thể thao"]),
    ("đồng phục", ["đồng phục"]),
    ("giày sneaker", ["giày sneaker", "sneaker", "giày dép", "giày"]),
    ("túi xách", ["túi xách"]),
    ("áo khoác da biker", ["áo khoác da biker", "áo da biker", "biker jacket"]),
]
TEAM_CONTEXT_PATTERN = re.compile(
    r"\b(?P<prefix>đội tuyển|doi tuyen|câu lạc bộ|cau lac bo|clb)\s+"
    r"(?P<name>.+?)(?=$|\s+(?:k|không|khong|ko|ạ|nhé|nhe|size|màu|mau|có|co))"
)
TEAM_NAME_ALIASES: list[tuple[str, list[str]]] = [
    ("bồ đào nha", ["bồ đào nha", "portugal"]),
    ("mu", ["mu", "man utd", "man united", "manchester united"]),
    ("real madrid", ["real madrid"]),
]
CONTEXTUAL_INTENTS = {
    "policy_question",
    "negative_feedback",
    "reject_current_product",
    "request_alternative_product",
    "request_similar_product",
    "variant_availability_check",
}


@dataclass(slots=True)
class CatalogDimensions:
    category_codes: set[str] = field(default_factory=set)
    category_names_by_code: dict[str, str] = field(default_factory=dict)
    category_code_by_term: dict[str, str] = field(default_factory=dict)
    product_type_by_term: dict[str, str] = field(default_factory=dict)
    product_types_by_category: dict[str, set[str]] = field(default_factory=dict)
    product_name_terms: set[str] = field(default_factory=set)
    tag_terms: set[str] = field(default_factory=set)
    description_terms: set[str] = field(default_factory=set)
    material_terms: set[str] = field(default_factory=set)
    gender_terms: set[str] = field(default_factory=set)
    style_terms: set[str] = field(default_factory=set)
    color_terms: set[str] = field(default_factory=set)
    size_terms: set[str] = field(default_factory=set)
    generic_terms_by_category: dict[str, set[str]] = field(default_factory=dict)

    @property
    def product_types(self) -> set[str]:
        return set(self.product_type_by_term.values())

    @property
    def attribute_terms(self) -> set[str]:
        return (
            self.tag_terms
            | self.material_terms
            | self.gender_terms
            | self.style_terms
            | self.color_terms
            | self.size_terms
            | self.product_name_terms
        )

    @property
    def searchable_terms(self) -> set[str]:
        return (
            set(self.category_code_by_term)
            | set(self.product_type_by_term)
            | self.product_name_terms
            | self.description_terms
            | self.tag_terms
            | self.material_terms
            | self.gender_terms
            | self.style_terms
            | self.color_terms
            | self.size_terms
            | {term for terms in self.generic_terms_by_category.values() for term in terms}
        )

    @property
    def intent_tag_terms(self) -> set[str]:
        return self.tag_terms - self.color_terms - self.gender_terms - self.style_terms - self.material_terms

    def available_product_types(self, category_code: str | None = None) -> list[str]:
        if category_code:
            values = self.product_types_by_category.get(category_code, set())
        else:
            values = self.product_types
        return sorted(values)

    def summary(self) -> dict[str, Any]:
        return {
            "category": sorted(self.category_codes),
            "product_type": self.available_product_types(),
            "tags": sorted(self.tag_terms),
            "material": sorted(self.material_terms),
            "gender": sorted(self.gender_terms),
            "style": sorted(self.style_terms),
            "color": sorted(self.color_terms),
            "size": sorted(self.size_terms),
        }


def _key(text: str | None) -> str:
    return strip_accents(normalize_text(text)).lower()


def _display(text: str | None) -> str:
    return normalize_text(text)


def _term_spans(text_key: str, term_key: str) -> list[tuple[int, int]]:
    if not text_key or not term_key:
        return []
    pattern = rf"(?<!\w){re.escape(term_key)}(?!\w)"
    return [(match.start(), match.end()) for match in re.finditer(pattern, text_key)]


def _add_term(terms: set[str], value: str | None) -> None:
    term = _key(value)
    if term:
        terms.add(term)


def _split_tag_text(value: str | None) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _product_name_keywords(name: str | None) -> set[str]:
    text = _key(name)
    if not text:
        return set()

    tokens = [token for token in re.findall(r"\w+", text) if len(token) > 1]
    terms = {text}
    for length in range(1, min(4, len(tokens)) + 1):
        for index in range(0, len(tokens) - length + 1):
            terms.add(" ".join(tokens[index : index + length]))
    return terms


def build_catalog_dimensions(products: pd.DataFrame, variants: pd.DataFrame) -> CatalogDimensions:
    dimensions = CatalogDimensions()
    if products.empty:
        return dimensions

    active_products = products[products["status"].fillna("").astype(str).str.lower().eq("active")]
    for row in active_products.to_dict("records"):
        category_code = str(row.get("category_code") or "").strip()
        category_name = str(row.get("category_name") or "").strip()
        product_type = str(row.get("product_type") or "").strip()

        if category_code:
            dimensions.category_codes.add(category_code)
            if category_name:
                dimensions.category_names_by_code[category_code] = category_name
                dimensions.category_code_by_term[_key(category_name)] = category_code
            dimensions.category_code_by_term[_key(category_code)] = category_code

        if product_type:
            dimensions.product_type_by_term[_key(product_type)] = product_type
            dimensions.product_types_by_category.setdefault(category_code, set()).add(product_type)
            first_word = normalize_text(product_type).split(maxsplit=1)[0]
            if first_word:
                dimensions.generic_terms_by_category.setdefault(category_code, set()).add(_key(first_word))

        for term in _product_name_keywords(str(row.get("product_name") or "")):
            dimensions.product_name_terms.add(term)

        for tag in _split_tag_text(row.get("tags")):
            tag_key = _key(tag)
            if not tag_key:
                continue
            dimensions.tag_terms.add(tag_key)
            if tag_key in {_key(value) for value in GENDER_TERMS}:
                dimensions.gender_terms.add(tag_key)
            if tag_key in {_key(value) for value in STYLES}:
                dimensions.style_terms.add(tag_key)
            if tag_key in {_key(value) for value in COLORS}:
                dimensions.color_terms.add(tag_key)
            if tag_key in MATERIAL_HINTS:
                dimensions.material_terms.add(tag_key)

        for term in _product_name_keywords(str(row.get("short_description") or "")):
            dimensions.description_terms.add(term)
            if term in MATERIAL_HINTS:
                dimensions.material_terms.add(term)

    if not variants.empty:
        for color in variants.get("color", pd.Series(dtype=str)).dropna().unique():
            _add_term(dimensions.color_terms, str(color))
        for size in variants.get("size", pd.Series(dtype=str)).dropna().unique():
            _add_term(dimensions.size_terms, str(size))

    if "Fashion" in dimensions.category_codes:
        dimensions.generic_terms_by_category.setdefault("Fashion", set()).update(
            _key(term) for term in FASHION_GENERIC_TERMS
        )
    if "Accessories" in dimensions.category_codes:
        dimensions.generic_terms_by_category.setdefault("Accessories", set()).update(
            _key(term) for term in ACCESSORY_GENERIC_TERMS
        )

    return dimensions


class CatalogDomainGate:
    def __init__(self, products: pd.DataFrame, variants: pd.DataFrame):
        self.dimensions = build_catalog_dimensions(products, variants)

    def apply(self, criteria: dict[str, Any]) -> dict[str, Any]:
        gated = dict(criteria)
        text = gated.get("normalized_message") or normalize_text(gated.get("raw_message"))
        text = normalize_text(str(text))

        evaluation = self._evaluate(gated, text)
        gated["catalog_gate"] = evaluation
        gated["catalog_dimensions"] = self.dimensions.summary()
        gated["catalog_available_product_types"] = self.dimensions.available_product_types()

        if evaluation["status"] == "unsupported":
            unsupported_terms = evaluation["unsupported_terms"] or ["sản phẩm này"]
            gated.update(
                {
                    "intent": "out_of_scope_request",
                    "category_code": None,
                    "product_type": None,
                    "product_types": [],
                    "catalog_coverage": "unsupported",
                    "response_mode": None,
                    "requested_product_group": unsupported_terms[0],
                    "out_of_scope_items": unsupported_terms,
                    "raw_product_type": None,
                    "color": None,
                    "size": None,
                    "gender": None,
                    "style": None,
                    "use_case": None,
                    "skin_type": None,
                    "concern": None,
                    "age_group": None,
                    "apparel_intent": None,
                    "keywords": [],
                    "need_clarification": False,
                    "clarification_questions": [],
                }
            )
            return gated

        closest_fallback = self._closest_supported_fallback(evaluation)
        if closest_fallback:
            requested_product_group = closest_fallback.get("term") or closest_fallback.get("chunk") or "sản phẩm này"
            suggested_product_type = closest_fallback.get("product_type")
            gated.update(
                {
                    "intent": "unsupported_product_request",
                    "category_code": None,
                    "product_type": None,
                    "product_types": [],
                    "catalog_coverage": "closest_alternative_available",
                    "response_mode": "offer_closest_product_type",
                    "requested_product_group": requested_product_group,
                    "suggested_product_type": suggested_product_type,
                    "out_of_scope_items": [requested_product_group],
                    "raw_product_type": None,
                    "need_clarification": True,
                    "clarification_questions": [],
                }
            )
            return gated

        supported_category = evaluation.get("category_code")
        supported_types = evaluation.get("product_types") or []
        if supported_category:
            gated["category_code"] = supported_category
        if supported_types:
            gated["product_types"] = supported_types
            gated["product_type"] = supported_types[0]

        if evaluation["unsupported_terms"]:
            gated["catalog_coverage"] = "partially_supported"
            gated["out_of_scope_items"] = evaluation["unsupported_terms"]
            if gated.get("intent") == "product_availability_check":
                gated["intent"] = "mixed_product_request"
        else:
            gated["catalog_coverage"] = "supported"
            gated["out_of_scope_items"] = []

        if not gated.get("category_code") and gated.get("product_type"):
            gated["category_code"] = self._category_for_product_type(gated["product_type"])

        return gated

    def canonical_product_type(self, text: str) -> str | None:
        text_key = _key(text)
        if text_key in self.dimensions.product_type_by_term:
            return self.dimensions.product_type_by_term[text_key]

        for term, product_type in self.dimensions.product_type_by_term.items():
            if term and contains_phrase(text_key, term):
                return product_type
        return None

    def _evaluate(self, criteria: dict[str, Any], text: str) -> dict[str, Any]:
        if criteria.get("intent") in CONTEXTUAL_INTENTS and not (
            criteria.get("intent") == "variant_availability_check"
            and (criteria.get("product_type") or criteria.get("category_code"))
        ):
            return {
                "status": "supported",
                "category_code": criteria.get("category_code"),
                "product_types": criteria.get("product_types") or [],
                "unsupported_terms": [],
                "requested_product_concepts": [],
                "matched_chunks": [],
            }

        if criteria.get("intent") == "general_buying_intent" or criteria.get("response_mode") == "ask_product_type":
            return {
                "status": "supported",
                "category_code": criteria.get("category_code") or "Fashion",
                "product_types": [],
                "unsupported_terms": [],
                "requested_product_concepts": [],
                "matched_chunks": [],
            }

        if criteria.get("response_mode") == "catalog_overview" or criteria.get("intent") == "catalog_overview":
            return {
                "status": "supported",
                "category_code": criteria.get("category_code"),
                "product_types": [],
                "unsupported_terms": [],
                "requested_product_concepts": [],
                "matched_chunks": [],
            }

        parsed_category = criteria.get("category_code") or detect_category(text)
        detected_types = self._prefer_specific_product_types(
            self._detected_product_types(criteria, text),
            text,
        )
        supported_types = [item for item in detected_types if self._is_catalog_product_type(item)]
        unsupported_types = [item for item in detected_types if item and not self._is_catalog_product_type(item)]
        direct_supported_types = self._catalog_product_types_in_text(text)
        for product_type in direct_supported_types:
            if product_type not in supported_types:
                supported_types.append(product_type)
        supported_types = self._prefer_specific_product_types(supported_types, text)

        specific_concepts = self._extract_specific_product_concepts(text)
        unsupported_specific_concepts = [
            concept for concept in specific_concepts if not self._concept_matches_catalog(concept)
        ]
        chunks = self._extract_product_chunks(text)
        chunk_results = [self._classify_chunk(chunk) for chunk in chunks]
        chunk_supported = [result for result in chunk_results if result["supported"]]
        chunk_unsupported = [result["term"] for result in chunk_results if result["unsupported"]]
        for result in chunk_supported:
            fallback_product_type = result.get("product_type")
            if fallback_product_type and fallback_product_type not in supported_types:
                supported_types.append(fallback_product_type)

        category_supported = parsed_category in self.dimensions.category_codes if parsed_category else False
        category_unsupported = bool(parsed_category and parsed_category not in self.dimensions.category_codes)
        generic_category = self._generic_category_in_text(text)
        supported_category = (
            self._category_for_product_type(supported_types[0])
            if supported_types
            else parsed_category
            if category_supported
            else generic_category
        )

        supported_evidence = bool(
            supported_types
            or chunk_supported
            or self._requested_product_exists(criteria.get("requested_product_name"))
            or (not unsupported_specific_concepts and (category_supported or generic_category))
        )
        unsupported_terms = self._dedupe_terms(
            [
                *unsupported_specific_concepts,
                *chunk_unsupported,
                *(unsupported_types if not chunk_unsupported else []),
                self._category_label(parsed_category, text)
                if category_unsupported and not chunk_unsupported
                else None,
            ]
        )

        if unsupported_specific_concepts and not chunk_supported and not self._requested_product_exists(criteria.get("requested_product_name")):
            return self._result("unsupported", unsupported_terms or unsupported_specific_concepts)

        if unsupported_types and not supported_types:
            return self._result("unsupported", unsupported_terms or unsupported_types)

        if category_unsupported and not supported_types and not self._requested_product_exists(criteria.get("requested_product_name")):
            return self._result("unsupported", unsupported_terms or [self._category_label(parsed_category, text)])

        if unsupported_types and not supported_types and not supported_evidence:
            return self._result("unsupported", unsupported_terms or unsupported_types)

        if chunks and chunk_results[0]["unsupported"] and not chunk_results[0]["supported"]:
            return self._result("unsupported", unsupported_terms or [chunk_results[0]["term"]])

        if unsupported_terms and not supported_evidence:
            return self._result("unsupported", unsupported_terms)

        if not supported_evidence and criteria.get("catalog_coverage") in {"not_supported", "unsupported"}:
            return self._result("unsupported", unsupported_terms or criteria.get("out_of_scope_items") or ["sản phẩm này"])

        return {
            "status": "supported",
            "category_code": supported_category,
            "product_types": supported_types,
            "unsupported_terms": unsupported_terms,
            "requested_product_concepts": specific_concepts,
            "matched_chunks": chunk_results,
        }

    def _result(self, status: str, unsupported_terms: list[str] | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "category_code": None,
            "product_types": [],
            "unsupported_terms": self._dedupe_terms(unsupported_terms or []),
            "requested_product_concepts": self._dedupe_terms(unsupported_terms or []),
            "matched_chunks": [],
        }

    def _detected_product_types(self, criteria: dict[str, Any], text: str) -> list[str]:
        values: list[str] = []
        for value in criteria.get("product_types") or []:
            if value and value not in values:
                values.append(str(value))
        if criteria.get("product_type") and criteria["product_type"] not in values:
            values.append(str(criteria["product_type"]))
        for value in detect_product_types(text):
            if value not in values:
                values.append(value)
        detected = detect_product_type(text)
        if detected and detected not in values:
            values.append(detected)
        return values

    def _catalog_product_types_in_text(self, text: str) -> list[str]:
        matches: list[str] = []
        text_key = _key(text)
        for term, product_type in self.dimensions.product_type_by_term.items():
            if term and _term_spans(text_key, term) and product_type not in matches:
                matches.append(product_type)
        return self._prefer_specific_product_types(matches, text)

    def _prefer_specific_product_types(self, values: list[str], text: str) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            canonical = self.dimensions.product_type_by_term.get(_key(value)) or value
            key = _key(canonical)
            if not key or key in seen:
                continue
            deduped.append(canonical)
            seen.add(key)

        text_key = _key(text)
        spans_by_key = {
            _key(product_type): _term_spans(text_key, _key(product_type))
            for product_type in deduped
        }
        all_spans = [
            (key, start, end)
            for key, spans in spans_by_key.items()
            for start, end in spans
        ]
        if not all_spans:
            return deduped

        covered_short_terms: set[str] = set()
        for key, spans in spans_by_key.items():
            if not spans:
                continue
            has_independent_occurrence = any(
                not any(
                    other_key != key
                    and other_start <= start
                    and end <= other_end
                    and (other_end - other_start) > (end - start)
                    for other_key, other_start, other_end in all_spans
                )
                for start, end in spans
            )
            if not has_independent_occurrence:
                covered_short_terms.add(key)

        return [product_type for product_type in deduped if _key(product_type) not in covered_short_terms]

    def _extract_product_chunks(self, text: str) -> list[str]:
        chunks: list[str] = []
        normalized = normalize_text(text)
        for anchor in REQUEST_ANCHORS:
            anchor_key = normalize_text(anchor)
            pattern = rf"(?:^|\s){re.escape(anchor_key)}\s+(.+)"
            for match in re.finditer(pattern, normalized):
                tail = match.group(1).strip()
                cutoff = REQUEST_CUTOFF_PATTERN.search(tail)
                if cutoff:
                    tail = tail[: cutoff.start()].strip()
                for part in REQUEST_SPLIT_PATTERN.split(tail):
                    chunk = self._clean_chunk(part)
                    if chunk and chunk not in chunks:
                        chunks.append(chunk)
        return chunks

    def _clean_chunk(self, chunk: str) -> str:
        cleaned = normalize_text(chunk)
        if self._is_generic_catalog_query_chunk(cleaned):
            return ""

        changed = True
        while changed:
            changed = False
            for phrase in GENERIC_FILLER_PHRASES:
                phrase_norm = normalize_text(phrase)
                if cleaned == phrase_norm:
                    return ""
                if cleaned.startswith(f"{phrase_norm} "):
                    cleaned = cleaned[len(phrase_norm) :].strip()
                    changed = True
                    if self._is_generic_catalog_query_chunk(cleaned):
                        return ""
        if self._is_generic_catalog_query_chunk(cleaned):
            return ""
        return cleaned

    def _is_generic_catalog_query_chunk(self, chunk: str) -> bool:
        chunk_key = _key(chunk)
        return not chunk_key or chunk_key in GENERIC_CATALOG_QUERY_CHUNKS

    def _classify_chunk(self, chunk: str) -> dict[str, Any]:
        if not chunk:
            return {"chunk": chunk, "term": chunk, "supported": False, "unsupported": False}

        fallback_match = self._fallback_product_type_match(chunk)
        if fallback_match:
            requested_term, fallback_product_type = fallback_match
            return {
                "chunk": chunk,
                "term": requested_term,
                "supported": True,
                "unsupported": False,
                "category_code": self._category_for_product_type(fallback_product_type),
                "product_type": fallback_product_type,
                "fallback_reason": "closest_supported_product_type",
            }

        specific_concepts = self._extract_specific_product_concepts(chunk)
        unsupported_concepts = [
            concept for concept in specific_concepts if not self._concept_matches_catalog(concept)
        ]
        if unsupported_concepts:
            return {
                "chunk": chunk,
                "term": unsupported_concepts[0],
                "supported": False,
                "unsupported": True,
                "requested_product_concepts": specific_concepts,
            }

        if self._has_supported_catalog_term(chunk):
            return {"chunk": chunk, "term": chunk, "supported": True, "unsupported": False}

        generic_category = self._generic_category_in_text(chunk)
        if generic_category:
            residual = self._residual_product_words(chunk)
            unsupported = bool(residual)
            return {
                "chunk": chunk,
                "term": chunk,
                "supported": not unsupported,
                "unsupported": unsupported,
                "residual": residual,
                "category_code": generic_category,
            }

        return {"chunk": chunk, "term": chunk, "supported": False, "unsupported": True}

    def _fallback_product_type(self, text: str) -> str | None:
        fallback_match = self._fallback_product_type_match(text)
        if fallback_match:
            return fallback_match[1]
        return None

    def _fallback_product_type_match(self, text: str) -> tuple[str, str] | None:
        for alias, product_type in SUPPORTED_FALLBACK_PRODUCT_TYPES.items():
            if contains_phrase(text, alias) and self._is_catalog_product_type(product_type):
                return _display(alias), product_type
        return None

    def _closest_supported_fallback(self, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        for match in evaluation.get("matched_chunks") or []:
            if match.get("fallback_reason") == "closest_supported_product_type":
                return match
        return None

    def _extract_specific_product_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        text_norm = normalize_text(text)
        for canonical, aliases in SPECIFIC_PRODUCT_CONCEPT_ALIASES:
            for alias in aliases:
                if contains_phrase(text_norm, alias):
                    concepts.append(canonical if contains_phrase(text_norm, canonical) else alias)
                    break

        for match in TEAM_CONTEXT_PATTERN.finditer(text_norm):
            prefix = normalize_text(match.group("prefix"))
            name = normalize_text(match.group("name"))
            name = re.sub(
                r"\s+(?:của|cua|cho|size|màu|mau|k|không|khong|ko|ạ|nhé|nhe).*$",
                "",
                name,
            ).strip()
            if name:
                concepts.append(f"{prefix} {name}")

        for canonical, aliases in TEAM_NAME_ALIASES:
            if any(contains_phrase(text_norm, alias) for alias in aliases):
                if not any(contains_phrase(concept, canonical) for concept in concepts):
                    concepts.append(canonical)

        return self._dedupe_terms(concepts)

    def _concept_matches_catalog(self, concept: str) -> bool:
        concept_key = _key(concept)
        if not concept_key:
            return False

        product_name_support_terms = {
            term for term in self.dimensions.product_name_terms if " " in term and len(term) >= 5
        }
        description_support_terms = {
            term for term in self.dimensions.description_terms if " " in term and len(term) >= 5
        }
        catalog_terms = (
            set(self.dimensions.product_type_by_term)
            | product_name_support_terms
            | description_support_terms
            | self.dimensions.tag_terms
            | self.dimensions.style_terms
            | self.dimensions.material_terms
        )
        return any(
            concept_key == term
            or contains_phrase(concept_key, term)
            or contains_phrase(term, concept_key)
            for term in catalog_terms
            if term
        )

    def _has_supported_catalog_term(self, text: str) -> bool:
        text_key = _key(text)
        if text_key in self.dimensions.product_type_by_term:
            return True
        if text_key in self.dimensions.category_code_by_term:
            return True
        product_name_support_terms = {
            term for term in self.dimensions.product_name_terms if " " in term and len(term) >= 5
        }
        description_support_terms = {
            term for term in self.dimensions.description_terms if " " in term and len(term) >= 5
        }
        return any(
            contains_phrase(text_key, term)
            for term in set(self.dimensions.product_type_by_term)
            | set(self.dimensions.category_code_by_term)
            | product_name_support_terms
            | description_support_terms
            | self.dimensions.intent_tag_terms
        )

    def _generic_category_in_text(self, text: str) -> str | None:
        for category_code, terms in self.dimensions.generic_terms_by_category.items():
            if any(contains_phrase(text, term) for term in terms):
                return category_code
        return None

    def _residual_product_words(self, chunk: str) -> list[str]:
        residual = _key(chunk)
        removable_terms = sorted(
            self.dimensions.searchable_terms
            | ATTRIBUTE_STOPWORDS
            | {term for terms in self.dimensions.generic_terms_by_category.values() for term in terms},
            key=len,
            reverse=True,
        )
        for term in removable_terms:
            if not term:
                continue
            residual = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", residual)
        tokens = [token for token in re.findall(r"\w+", residual) if token and token not in ATTRIBUTE_STOPWORDS]
        return tokens

    def _is_catalog_product_type(self, product_type: str | None) -> bool:
        return bool(product_type and _key(product_type) in self.dimensions.product_type_by_term)

    def _category_for_product_type(self, product_type: str | None) -> str | None:
        if not product_type:
            return None
        for category_code, product_types in self.dimensions.product_types_by_category.items():
            if product_type in product_types:
                return category_code
        return product_type_to_category(product_type)

    def _requested_product_exists(self, requested_product_name: str | None) -> bool:
        if not requested_product_name:
            return False
        requested = _key(requested_product_name)
        return requested in self.dimensions.product_name_terms

    def _category_label(self, category_code: str | None, text: str | None = None) -> str | None:
        if not category_code:
            return None
        if text and category_code in CATEGORY_KEYWORDS:
            for keyword in CATEGORY_KEYWORDS[category_code]:
                if contains_phrase(text, keyword):
                    return keyword
        return self.dimensions.category_names_by_code.get(category_code) or str(category_code)

    def _dedupe_terms(self, values: list[str | None]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            text = _display(value)
            key = _key(text)
            if not key or key in seen:
                continue
            result.append(text)
            seen.add(key)
        return result
