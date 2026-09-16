from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.normalizer import (
    AGE_GROUPS,
    CONCERNS,
    SKIN_TYPES,
    STYLES,
    apply_synonyms,
    contains_phrase,
    detect_category,
    detect_product_type,
    detect_product_types,
    extract_all_matches,
    extract_budget,
    extract_color,
    extract_first_match,
    extract_price_constraint,
    extract_size,
    normalize_color,
    normalize_text,
    product_type_to_category,
    strip_accents,
)

FASHION_SIZE_TYPES = {
    "Áo thun",
    "Áo sơ mi",
    "Váy",
    "Quần jeans",
    "Áo khoác",
    "Chân váy",
    "Set bộ",
}
IN_STOCK_PHRASES = [
    "còn hàng",
    "còn không",
    "còn ko",
    "còn k",
    "có không",
    "có ko",
    "có k",
    "có sẵn",
    "sẵn hàng",
    "giao hôm nay",
    "còn size",
    "còn màu",
]
NEGATIVE_STOCK_PHRASES = ["không cần còn hàng", "khong can con hang"]
GENDER_ALIASES = {
    "unisex": ["unisex", "phi giới tính"],
    "nam": ["dành cho nam", "cho nam", "bạn nam", "nam"],
    "nữ": ["dành cho nữ", "cho nữ", "bạn nữ", "nữ"],
}
USE_CASE_ALIASES = {
    "mặc ở nhà": ["mặc ở nhà", "ở nhà"],
    "đi làm": ["đi làm", "công sở"],
    "đi chơi": ["đi chơi"],
    "đi học": ["đi học"],
    "dự tiệc": ["dự tiệc", "tiệc"],
    "tập gym": ["tập gym", "gym"],
    "thoải mái": ["thoải mái"],
    "quà tặng": ["quà tặng", "quà sinh nhật"],
}
CATALOG_BROWSING_PATTERNS = [
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?ban\s+nhung\s+.+\s+nhu\s+nao\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+nhung\s+(?:loai|mau|san pham)?\s*.+\s+(?:nao|gi)\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?dang\s+ban\s+mau\s+.+\s+nao\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+nhung\s+san\s+pham\s+nao\s+thuoc\s+nhom\s+.+\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+.+\s+(?:khong|ko|k)\b"
    ),
]
CATALOG_OVERVIEW_PATTERNS = [
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?ban\s+(?:nhung\s+)?san\s+pham\s+(?:gi|nao)\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+(?:nhung\s+)?san\s+pham\s+(?:gi|nao)\b"
    ),
    re.compile(r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?ban\s+gi\b"),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?dang\s+ban\s+gi\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+ban\s+gi\s+(?:khong|ko|k)\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+(?:nhung\s+)?mat\s+hang\s+(?:gi|nao)\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+(?:nhung\s+)?nhom\s+san\s+pham\s+nao\b"
    ),
    re.compile(
        r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?ban\s+(?:nhung\s+)?loai\s+hang\s+nao\b"
    ),
    re.compile(r"\b(?:shop|ben minh|ben shop|cua hang)?\s*(?:minh\s+)?co\s+gi\b"),
]
GENERAL_BUYING_INTENT_PATTERNS = [
    re.compile(
        r"\b(?:toi|tui|minh|em|anh|chi)?\s*(?:muon|can)\s+(?:mua|xem)\s+(?:do|quan ao|hang|san pham)\b"
    ),
    re.compile(r"\b(?:toi|tui|minh|em|anh|chi)?\s*can\s+mua\s+hang\b"),
    re.compile(
        r"\btu\s+van(?:\s+cho)?\s+(?:toi|tui|minh|em|anh|chi)?\s*mua\s+(?:do|quan ao|hang)\b"
    ),
    re.compile(
        r"\btu\s+van\s+(?:do|quan ao|hang|san pham)\s+(?:cho\s+)?(?:toi|tui|minh|em|anh|chi)\b"
    ),
    re.compile(
        r"\b(?:toi|tui|minh|em|anh|chi)?\s*muon\s+duoc\s+tu\s+van\s+(?:do|quan ao|hang|san pham)\b"
    ),
    re.compile(r"\bshop\s+ban\s+gi\s+(?:vay|the|a|ha)\b"),
    re.compile(r"\bshop\s+ban\s+nhung\s+gi\b"),
    re.compile(r"\bco\s+(?:san\s+pham|hang|do)\s+gi\s+(?:khong|ko|k)\b"),
]
UNKNOWN_PRODUCT_NEED_PATTERNS = [
    re.compile(
        r"\b(?:toi|tui|minh|em|anh|chi)?\s*(?:chua|khong)\s+biet\s+(?:mua|chon|lay)\s+(?:gi|do\s+gi|san\s+pham\s+gi)\b"
    ),
    re.compile(
        r"\b(?:toi|tui|minh|em|anh|chi)?\s*phan\s+van\s+(?:chua\s+)?biet\s+(?:mua|chon)\s+(?:gi|do\s+gi)\b"
    ),
]
PURCHASE_INTENT_PATTERNS = [
    re.compile(r"\b(?:toi|tui|minh|em|anh|chi)?\s*(?:lay|chot)\b"),
    re.compile(r"\bdat\s+(?:giup|cho|mau|don|hang)\b"),
    re.compile(r"\b(?:dong y mua|chot don|dat giup|mua cai nay|mua mau nay)\b"),
    re.compile(
        r"\b(?:toi|tui|minh|em|anh|chi)?\s*muon\s+mua\s+(?:\d+|mot|hai|ba|bon|tu|nam)\s+(?:cai|chiec|san pham|ao)\b"
    ),
    re.compile(
        r"\b(?:lay|cho\s+(?:toi|tui|minh|em|anh|chi)?)\s+(?:\d+|mot|hai|ba|bon|tu|nam)\s+(?:cai|chiec|san pham|ao)\b"
    ),
]
SIZE_RECOMMENDATION_PHRASES = [
    "tư vấn size",
    "chọn size",
    "size nào",
    "size gì",
    "mặc size",
    "nên mặc",
    "mặc vừa",
    "mặc rộng",
    "mặc ôm",
]
SPECIFIC_SEARCH_PHRASES = [
    "muốn tìm",
    "muốn mua",
    "cần",
    "gợi ý",
    "tư vấn",
    "recommend",
    "dưới",
    "trên",
    "không quá",
    "màu",
    "size",
    "mặc",
    "đi chơi",
    "đi làm",
    "công sở",
    "mùa hè",
    "đẹp",
    "best seller",
]
POLICY_PATTERNS: list[tuple[str, list[str]]] = [
    ("cod", ["cod", "thanh toán khi nhận hàng", "trả tiền khi nhận hàng"]),
    (
        "free_shipping",
        ["freeship", "free ship", "miễn phí ship", "miễn phí vận chuyển"],
    ),
    ("size_exchange", ["đổi size", "đổi kích cỡ", "đổi hàng", "đổi trả"]),
    ("inspection", ["kiểm hàng", "đồng kiểm", "xem hàng trước"]),
    ("delivery_time", ["giao hàng mất bao lâu", "bao lâu nhận", "thời gian giao hàng"]),
]
REJECT_CURRENT_PRODUCT_PHRASES = [
    "mẫu này không hợp",
    "không hợp",
    "tôi không thích mẫu này",
    "mình không thích mẫu này",
    "không thích mẫu này",
]
ALTERNATIVE_PRODUCT_PHRASES = [
    "cho tôi mẫu khác",
    "cho mình mẫu khác",
    "mẫu khác",
    "đổi sản phẩm khác",
    "đổi mẫu khác",
    "sản phẩm khác",
]
SIMILAR_PRODUCT_PHRASES = [
    "mẫu tương tự",
    "sản phẩm tương tự",
    "kiểu tương tự",
]
NEGATIVE_FEEDBACK_PHRASES = [
    "xấu quá",
    "đắt quá",
    "tư vấn kỳ",
    "tư vấn lạ",
    "chắc chật",
    "size này chật",
]


def _detect_in_stock_intent(text: str) -> bool:
    if any(contains_phrase(text, phrase) for phrase in NEGATIVE_STOCK_PHRASES):
        return False
    text_key = strip_accents(normalize_text(text))
    if re.search(r"\b(?:con|co)\b.+\b(?:khong|ko|k)\b", text_key):
        return True
    return any(contains_phrase(text, phrase) for phrase in IN_STOCK_PHRASES)


def _keywords(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            text = str(candidate).strip()
            key = normalize_text(text)
            if text and key not in seen:
                result.append(text)
                seen.add(key)
    return result


def _detect_requested_product(text: str) -> dict[str, str] | None:
    return None


def _detect_demo_color(text: str) -> str | None:
    return None


def _detect_catalog_browsing(text: str) -> bool:
    text_no_accents = strip_accents(normalize_text(text))
    return any(pattern.search(text_no_accents) for pattern in CATALOG_BROWSING_PATTERNS)


def _detect_catalog_overview(text: str) -> bool:
    text_no_accents = strip_accents(normalize_text(text))
    return any(pattern.search(text_no_accents) for pattern in CATALOG_OVERVIEW_PATTERNS)


def _detect_general_buying_intent(text: str) -> bool:
    text_no_accents = strip_accents(normalize_text(text))
    return any(
        pattern.search(text_no_accents) for pattern in GENERAL_BUYING_INTENT_PATTERNS
    )


def _detect_unknown_product_need(text: str) -> bool:
    text_no_accents = strip_accents(normalize_text(text))
    return any(
        pattern.search(text_no_accents) for pattern in UNKNOWN_PRODUCT_NEED_PATTERNS
    )


def _detect_purchase_intent(text: str) -> bool:
    text_no_accents = strip_accents(normalize_text(text))
    if any(pattern.search(text_no_accents) for pattern in PURCHASE_INTENT_PATTERNS):
        return True
    return bool(
        re.search(
            r"\b(?:mau|cai|san pham)\s+(?:nay|dau tien|thu hai|cuoi)\b",
            text_no_accents,
        )
        and re.search(r"\b(?:lay|chot|mua|dat)\b", text_no_accents)
    )


def _text_key(text: str) -> str:
    return strip_accents(normalize_text(text))


def _detect_policy_key(text: str) -> str | None:
    for policy_key, phrases in POLICY_PATTERNS:
        if any(contains_phrase(text, phrase) for phrase in phrases):
            return policy_key
    return None


def _detect_reject_current_product(text: str) -> bool:
    return any(
        contains_phrase(text, phrase) for phrase in REJECT_CURRENT_PRODUCT_PHRASES
    )


def _detect_similar_product_request(text: str) -> bool:
    return any(contains_phrase(text, phrase) for phrase in SIMILAR_PRODUCT_PHRASES)


def _detect_alternative_product_request(text: str) -> bool:
    if re.search(r"(?<!\w)màu\s+khác(?!\w)", normalize_text(text)):
        return False
    if _detect_similar_product_request(text):
        return True
    return any(contains_phrase(text, phrase) for phrase in ALTERNATIVE_PRODUCT_PHRASES)


def _detect_negative_feedback(text: str) -> str | None:
    key = _text_key(text)
    if any(contains_phrase(text, phrase) for phrase in NEGATIVE_FEEDBACK_PHRASES):
        if "dat" in key or "gia" in key:
            return "price"
        if "size" in key or "chat" in key or "rong" in key:
            return "size"
        if "tu van" in key:
            return "advice"
        return "style"
    return None


def _detect_price_direction(text: str) -> str | None:
    key = _text_key(text)
    if re.search(r"\b(?:re hon|gia tot hon|mem hon|thap hon)\b", key):
        return "lower"
    if re.search(r"\b(?:cao cap hon|chat lieu tot hon|tot hon|xin hon)\b", key):
        return "higher"
    return None


def _detect_style_change(text: str) -> str | None:
    key = _text_key(text)
    if "lich su hon" in key or "cong so hon" in key or "trang trong hon" in key:
        return "công sở"
    if "tre trung hon" in key:
        return "trẻ trung"
    if "thoai mai hon" in key:
        return "thoải mái"
    return None


def _detect_variant_followup(
    text: str, *, product_type: str | None, color: str | None, size: str | None
) -> bool:
    key = _text_key(text)
    if product_type and not key.startswith("mau "):
        return False
    if "mau khac" in key or "size khac" in key:
        return True
    if re.search(r"\b(?:co|con)\s+(?:mau|size)\b", key) and not product_type:
        return True
    if re.search(r"\b(?:co|con)\s+(?:mau|size)\b", key) and "mau nay" in key:
        return True
    if re.search(r"\b(?:mau|size)\s+.+\b(?:co|con)\b", key) and "mau" in key:
        return True
    if "mau" in key and color and not product_type:
        return True
    if "size" in key and size and not product_type:
        return True
    if "mau" in key and "mau" in key.split()[:2]:
        return True
    return False


def _extract_height_cm(text: str) -> int | None:
    normalized = strip_accents(normalize_text(text))
    match = re.search(r"\b(\d)\s*m\s*(\d{1,2})\b", normalized)
    if match:
        meters = int(match.group(1))
        centimeters = int(match.group(2))
        if centimeters < 10:
            centimeters *= 10
        return meters * 100 + centimeters

    match = re.search(r"\b(\d{2,3})\s*cm\b", normalized)
    if match:
        return int(match.group(1))

    match = re.search(r"\bcao\s+(\d{2,3})\b", normalized)
    if match:
        value = int(match.group(1))
        if 120 <= value <= 220:
            return value
    return None


def _extract_weight_kg(text: str) -> int | None:
    normalized = strip_accents(normalize_text(text))
    match = re.search(r"\b(\d{2,3})(?:[.,]\d+)?\s*(?:kg|ki\s*lo|ky|kilo)\b", normalized)
    if match:
        value = int(match.group(1))
        if 30 <= value <= 180:
            return value

    match = re.search(r"\b(?:nang|can\s+nang)\s+(\d{2,3})\b", normalized)
    if match:
        value = int(match.group(1))
        if 30 <= value <= 180:
            return value
    return None


def _detect_fit_preference(text: str) -> str | None:
    if any(
        contains_phrase(text, phrase) for phrase in ["mặc rộng", "rộng", "thoải mái"]
    ):
        return "rộng"
    if any(contains_phrase(text, phrase) for phrase in ["mặc ôm", "ôm body", "ôm"]):
        return "ôm"
    if any(contains_phrase(text, phrase) for phrase in ["vừa người", "vừa"]):
        return "vừa"
    return None


def _detect_size_recommendation_intent(
    text: str,
    *,
    height_cm: int | None,
    weight_kg: int | None,
) -> bool:
    normalized = normalize_text(text)
    if height_cm and weight_kg:
        return True
    if any(
        contains_phrase(normalized, phrase) for phrase in SIZE_RECOMMENDATION_PHRASES
    ):
        return True
    if weight_kg and contains_phrase(normalized, "mặc") and extract_size(normalized):
        return True
    return False


def _has_specific_search_need(
    text: str,
    *,
    requested_product: dict[str, str] | None,
    budget_min_vnd: int | None,
    budget_max_vnd: int | None,
    color: str | None,
    size: str | None,
    gender: str | None,
    style: str | None,
    use_case: str | None,
) -> bool:
    if (
        requested_product
        or budget_min_vnd
        or budget_max_vnd
        or color
        or size
        or gender
        or style
        or use_case
    ):
        return True
    return any(contains_phrase(text, phrase) for phrase in SPECIFIC_SEARCH_PHRASES)


def _detect_gender(text: str) -> str | None:
    for gender, aliases in GENDER_ALIASES.items():
        if any(contains_phrase(text, alias) for alias in aliases):
            return gender
    return None


def _detect_use_case(text: str) -> str | None:
    text_no_accents = text
    if re.search(r"(?<!\w)mặc\b.*\bở nhà(?!\w)", text_no_accents):
        return "mặc ở nhà"
    for use_case, aliases in USE_CASE_ALIASES.items():
        if any(contains_phrase(text, alias) for alias in aliases):
            return use_case
    return None


def _detect_apparel_intent(text: str, product_type: str | None) -> str | None:
    if contains_phrase(text, "áo") or (product_type or "").startswith("Áo"):
        return "áo"
    if contains_phrase(text, "quần") or (product_type or "").startswith("Quần"):
        return "quần"
    if contains_phrase(text, "váy") or product_type in {"Váy", "Chân váy"}:
        return "váy"
    if contains_phrase(text, "túi") or product_type == "Túi tote":
        return "túi"
    return None


def _detect_concern(text: str, category_code: str | None) -> str | None:
    if category_code != "Cosmetics":
        return None

    for concern in CONCERNS:
        if concern == "nám":
            if re.search(r"(?<!\w)nám(?!\w)", normalize_text(text)):
                return "nám"
            continue
        if contains_phrase(text, concern):
            return concern
    return None


def parse_customer_message(
    message: str, synonyms: pd.DataFrame | dict[str, str] | None = None
) -> dict[str, Any]:
    normalized_message = apply_synonyms(normalize_text(message), synonyms)

    requested_product = _detect_requested_product(normalized_message)
    out_of_scope_items: list[str] = []
    requested_product_group = None
    raw_product_type = None
    raw_type_keywords: list[str] = []
    detected_product_types = detect_product_types(normalized_message)
    product_types = list(dict.fromkeys(detected_product_types))
    product_type = (
        product_types[0] if product_types else detect_product_type(normalized_message)
    )
    detected_category = detect_category(normalized_message)
    product_category = product_type_to_category(product_type)
    category_code = (
        detected_category
        if detected_category == "Accessories"
        else product_category or detected_category
    )
    catalog_coverage = "supported"

    if requested_product:
        category_code = "Fashion"
        product_type = requested_product["product_type"]
        product_types = [product_type]
        catalog_coverage = "supported"
        requested_product_group = None
        raw_product_type = None
        raw_type_keywords = []

    if (
        category_code == "Accessories"
        and product_type
        and product_category != "Accessories"
    ):
        product_type = None
        product_types = [
            item
            for item in product_types
            if product_type_to_category(item) == "Accessories"
        ]

    price_constraint = extract_price_constraint(normalized_message)
    budget_min_vnd = price_constraint.get("budget_min_vnd")
    budget_max_vnd = price_constraint.get("budget_max_vnd")
    if budget_max_vnd is None and budget_min_vnd is None:
        budget_max_vnd = extract_budget(normalized_message)
    color = extract_color(normalized_message)
    if requested_product:
        color = normalize_color(_detect_demo_color(normalized_message)) or color
    size = extract_size(normalized_message)
    style_matches = extract_all_matches(normalized_message, STYLES)
    style = style_matches[0] if style_matches else None
    skin_type = extract_first_match(normalized_message, SKIN_TYPES)
    concern = _detect_concern(normalized_message, category_code)
    age_group = extract_first_match(normalized_message, AGE_GROUPS)
    must_be_in_stock = _detect_in_stock_intent(normalized_message)
    gender = _detect_gender(normalized_message)
    use_case = _detect_use_case(normalized_message)
    height_cm = _extract_height_cm(normalized_message)
    weight_kg = _extract_weight_kg(normalized_message)
    fit_preference = _detect_fit_preference(normalized_message)
    policy_key = _detect_policy_key(normalized_message)
    rejects_current_product = _detect_reject_current_product(normalized_message)
    alternative_product_requested = _detect_alternative_product_request(
        normalized_message
    )
    similar_product_requested = _detect_similar_product_request(normalized_message)
    negative_feedback_reason = _detect_negative_feedback(normalized_message)
    price_direction = price_constraint.get("price_direction") or _detect_price_direction(
        normalized_message
    )
    has_alternative_price_direction = price_direction in {"lower", "higher"}
    style_change = _detect_style_change(normalized_message)
    if style_change and not style:
        style = style_change
    variant_followup = _detect_variant_followup(
        normalized_message,
        product_type=product_type,
        color=color,
        size=size,
    )
    apparel_intent = _detect_apparel_intent(normalized_message, product_type)
    if catalog_coverage == "not_supported":
        apparel_intent = None

    if (
        not skin_type
        and category_code == "Cosmetics"
        and contains_phrase(normalized_message, "dầu")
    ):
        skin_type = "da dầu"

    keyword_values = _keywords(
        color,
        size,
        gender,
        raw_type_keywords,
        style_matches,
        use_case,
        skin_type,
        concern,
        age_group,
        product_types,
    )
    for keyword in ["healthy", "ít calo"]:
        if (
            contains_phrase(normalized_message, keyword)
            and keyword not in keyword_values
        ):
            keyword_values.append(keyword)
    if (
        category_code == "Food"
        and any(
            contains_phrase(normalized_message, phrase)
            for phrase in ["trưa", "buổi trưa", "ăn trưa"]
        )
        and "trưa" not in keyword_values
    ):
        keyword_values.append("trưa")
    if category_code == "Accessories":
        if any(
            contains_phrase(normalized_message, phrase)
            for phrase in ["quà", "sinh nhật", "quà tặng", "quà sinh nhật"]
        ):
            if "quà tặng" not in keyword_values:
                keyword_values.append("quà tặng")
        if (
            contains_phrase(normalized_message, "phụ kiện")
            and "phụ kiện" not in keyword_values
        ):
            keyword_values.append("phụ kiện")

    response_mode = None
    is_unknown_product_need = (
        _detect_unknown_product_need(normalized_message)
        and not requested_product
        and not product_type
        and budget_min_vnd is None
        and budget_max_vnd is None
        and color is None
        and size is None
        and gender is None
        and style is None
        and use_case is None
    )
    is_general_buying_intent = (
        _detect_general_buying_intent(normalized_message)
        and not requested_product
        and not product_type
        and budget_min_vnd is None
        and budget_max_vnd is None
        and color is None
        and size is None
        and gender is None
        and style is None
        and use_case is None
    )
    is_purchase_intent = _detect_purchase_intent(normalized_message)
    is_size_recommendation = _detect_size_recommendation_intent(
        normalized_message,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )

    has_explicit_product_search = bool(
        product_type
        and any(
            contains_phrase(normalized_message, phrase)
            for phrase in [
                "tôi muốn tìm",
                "mình muốn tìm",
                "muốn tìm",
                "tôi muốn mua",
                "mình muốn mua",
                "muốn mua",
                "tìm cho tôi",
                "tìm giúp",
                "gợi ý",
                "cho xem",
            ]
        )
    )

    if has_explicit_product_search and height_cm is None and weight_kg is None:
        is_size_recommendation = False

    if policy_key:
        response_mode = "policy_question"
    elif is_purchase_intent:
        response_mode = "purchase_intent"
    elif negative_feedback_reason:
        response_mode = "negative_feedback"
    elif has_alternative_price_direction and price_direction == "lower":
        response_mode = "cheaper_product"
    elif has_alternative_price_direction and price_direction == "higher":
        response_mode = "premium_product"
    elif style_change:
        response_mode = "style_change"
    elif rejects_current_product and not (
        alternative_product_requested or similar_product_requested
    ):
        response_mode = "reject_current_product"
    elif similar_product_requested:
        response_mode = "similar_product"
    elif alternative_product_requested or rejects_current_product:
        response_mode = "alternative_product"
    elif variant_followup:
        response_mode = "variant_availability"
    elif is_size_recommendation:
        response_mode = "size_recommendation"
    elif is_unknown_product_need:
        response_mode = "ask_use_case"
    elif is_general_buying_intent:
        response_mode = "ask_product_type"
    elif _detect_catalog_overview(normalized_message):
        response_mode = "catalog_overview"
    elif _detect_catalog_browsing(normalized_message) and not _has_specific_search_need(
        normalized_message,
        requested_product=requested_product,
        budget_min_vnd=budget_min_vnd,
        budget_max_vnd=budget_max_vnd,
        color=color,
        size=size,
        gender=gender,
        style=style,
        use_case=use_case,
    ):
        response_mode = "catalog_browsing"

    clarification_questions: list[str] = []
    context_response_modes = {
        "policy_question",
        "negative_feedback",
        "cheaper_product",
        "premium_product",
        "style_change",
        "variant_availability",
        "reject_current_product",
        "similar_product",
        "alternative_product",
    }
    if is_size_recommendation and not product_type and fit_preference is None:
        clarification_questions.append(
            "Bạn đang muốn chọn size cho áo thun, áo sơ mi hay sản phẩm nào ạ?"
        )
    elif is_unknown_product_need:
        clarification_questions.append(
            "Bạn đang cần đồ để đi làm, đi chơi, đi học hay mặc hằng ngày ạ?"
        )
    elif is_general_buying_intent:
        clarification_questions.append("Bạn đang muốn tìm sản phẩm nào ạ?")
    elif (
        response_mode not in context_response_modes
        and not out_of_scope_items
        and not category_code
        and not product_type
        and response_mode != "catalog_overview"
    ):
        clarification_questions.append(
            "Bạn muốn tìm nhóm sản phẩm nào trong catalog hiện tại của shop?"
        )

    if (
        not requested_product
        and not is_size_recommendation
        and response_mode
        not in {"catalog_browsing", "catalog_overview", "ask_product_type"}
        and catalog_coverage != "not_supported"
        and category_code == "Fashion"
        and product_type in FASHION_SIZE_TYPES
        and not size
    ):
        clarification_questions.append("Bạn muốn tìm size nào?")

    if (
        not requested_product
        and not is_size_recommendation
        and response_mode
        not in {"catalog_browsing", "catalog_overview", "ask_product_type"}
        and catalog_coverage != "not_supported"
        and category_code == "Fashion"
        and budget_min_vnd is None
        and budget_max_vnd is None
    ):
        clarification_questions.append("Ngân sách tối đa của bạn khoảng bao nhiêu?")

    intent = "product_search"
    if policy_key:
        intent = "policy_question"
    elif is_purchase_intent:
        intent = "purchase_intent"
    elif negative_feedback_reason:
        intent = "negative_feedback"
    elif rejects_current_product and not (
        has_alternative_price_direction
        or style_change
        or alternative_product_requested
        or similar_product_requested
    ):
        intent = "reject_current_product"
    elif (
        has_alternative_price_direction
        or style_change
        or alternative_product_requested
        or rejects_current_product
    ):
        intent = (
            "request_similar_product"
            if similar_product_requested
            else "request_alternative_product"
        )
    elif variant_followup:
        intent = "variant_availability_check"
    elif is_size_recommendation:
        intent = "size_recommendation"
        if not product_type and fit_preference is None:
            response_mode = "ask_product_for_size"
        else:
            response_mode = "size_recommendation"
    elif is_unknown_product_need:
        intent = "general_buying_intent"
    elif is_general_buying_intent:
        intent = "general_buying_intent"
    elif response_mode == "catalog_overview":
        intent = "catalog_overview"
    if requested_product:
        intent = "product_availability_check"

    return {
        "raw_message": message,
        "normalized_message": normalized_message,
        "intent": intent,
        "purchase_intent": is_purchase_intent,
        "category_code": category_code,
        "product_type": product_type,
        "product_types": product_types,
        "raw_product_type": raw_product_type,
        "response_mode": response_mode,
        "requested_product_name": (
            requested_product["requested_product_name"] if requested_product else None
        ),
        "requested_product_group": requested_product_group,
        "out_of_scope_items": out_of_scope_items,
        "catalog_coverage": catalog_coverage,
        "budget_min_vnd": budget_min_vnd,
        "budget_max_vnd": budget_max_vnd,
        "must_be_in_stock": must_be_in_stock,
        "color": color,
        "size": size,
        "gender": gender,
        "style": style,
        "use_case": use_case,
        "skin_type": skin_type,
        "concern": concern,
        "age_group": age_group,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "fit_preference": fit_preference,
        "policy_key": policy_key,
        "reject_current_product": rejects_current_product,
        "negative_feedback_reason": negative_feedback_reason,
        "price_direction": price_direction,
        "style_change": style_change,
        "variant_followup": variant_followup,
        "apparel_intent": apparel_intent,
        "keywords": keyword_values,
        "need_clarification": bool(clarification_questions),
        "clarification_questions": clarification_questions,
    }
