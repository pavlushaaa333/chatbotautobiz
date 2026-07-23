from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Iterable

import pandas as pd

ACCESSORY_PRIORITY_KEYWORDS = [
    "phụ kiện",
    "quà tặng",
    "quà sinh nhật",
    "vintage",
    "cute",
]


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Fashion": [
        "thời trang",
        "áo",
        "quần",
        "váy",
        "chân váy",
        "set bộ",
        "đi làm",
        "công sở",
        "outfit",
    ],
    "Cosmetics": [
        "mỹ phẩm",
        "da",
        "mụn",
        "serum",
        "son",
        "kem chống nắng",
        "sữa rửa mặt",
        "toner",
        "tẩy trang",
        "kem dưỡng",
    ],
    "MotherBaby": [
        "mẹ và bé",
        "em bé",
        "bé",
        "bỉm",
        "sữa bột",
        "bình sữa",
        "khăn ướt",
        "đồ chơi",
        "ăn dặm",
    ],
    "Food": [
        "đồ ăn",
        "ăn trưa",
        "healthy",
        "ít calo",
        "salad",
        "cơm",
        "trà sữa",
        "bánh",
        "nước ép",
        "combo",
    ],
    "Accessories": [
        "phụ kiện",
        "quà",
        "quà tặng",
        "quà sinh nhật",
        "sinh nhật",
        "túi",
        "ví",
        "ốp điện thoại",
        "kẹp tóc",
        "mũ",
        "dây chuyền",
        "khuyên tai",
    ],
}

PRODUCT_TYPES_BY_CATEGORY: dict[str, list[str]] = {
    "Fashion": [
        "Áo thun",
        "Áo sơ mi",
        "Váy",
        "Quần jeans",
        "Áo khoác",
        "Chân váy",
        "Set bộ",
    ],
    "Cosmetics": [
        "Son môi",
        "Kem chống nắng",
        "Sữa rửa mặt",
        "Toner",
        "Serum",
        "Kem dưỡng",
        "Tẩy trang",
    ],
    "MotherBaby": [
        "Bỉm",
        "Sữa bột",
        "Bình sữa",
        "Khăn ướt",
        "Đồ chơi",
        "Quần áo trẻ em",
        "Ghế ăn dặm",
    ],
    "Food": [
        "Bánh ngọt",
        "Cơm văn phòng",
        "Trà sữa",
        "Salad",
        "Đồ ăn vặt",
        "Nước ép",
        "Combo ăn trưa",
    ],
    "Accessories": [
        "Túi tote",
        "Ví",
        "Ốp điện thoại",
        "Kẹp tóc",
        "Mũ",
        "Dây chuyền",
        "Khuyên tai",
    ],
}

PRODUCT_TYPE_ALIASES: dict[str, list[str]] = {
    "Áo thun": ["áo thun", "áo phông", "áo ba lỗ", "ba lỗ", "t-shirt", "tshirt"],
    "Áo sơ mi": ["áo sơ mi", "sơ mi", "so mi", "áo đi làm"],
    "Váy": ["váy", "đầm", "váy dáng a", "váy chữ a"],
    "Quần jeans": ["quần jeans", "quần jean", "jeans"],
    "Áo khoác": ["áo khoác", "blazer", "jacket"],
    "Chân váy": ["chân váy"],
    "Set bộ": ["set bộ", "đồ bộ"],
    "Son môi": ["son môi", "son"],
    "Kem chống nắng": ["kem chống nắng", "chống nắng"],
    "Sữa rửa mặt": ["sữa rửa mặt", "srm", "rửa mặt"],
    "Toner": ["toner", "nước cân bằng"],
    "Serum": ["serum"],
    "Kem dưỡng": ["kem dưỡng", "dưỡng ẩm"],
    "Tẩy trang": ["tẩy trang"],
    "Bỉm": ["bỉm", "tã"],
    "Sữa bột": ["sữa bột"],
    "Bình sữa": ["bình sữa"],
    "Khăn ướt": ["khăn ướt"],
    "Đồ chơi": ["đồ chơi"],
    "Quần áo trẻ em": ["quần áo trẻ em", "đồ trẻ em", "quần áo bé"],
    "Ghế ăn dặm": ["ghế ăn dặm", "ăn dặm"],
    "Bánh ngọt": ["bánh ngọt", "bánh"],
    "Cơm văn phòng": ["cơm văn phòng", "cơm trưa"],
    "Trà sữa": ["trà sữa"],
    "Salad": ["salad"],
    "Đồ ăn vặt": ["đồ ăn vặt", "snack", "ăn vặt"],
    "Nước ép": ["nước ép"],
    "Combo ăn trưa": ["combo ăn trưa", "ăn trưa", "combo trưa"],
    "Túi tote": ["túi tote", "tote", "túi"],
    "Ví": ["ví", "bóp"],
    "Ốp điện thoại": ["ốp điện thoại", "ốp lưng", "case điện thoại"],
    "Kẹp tóc": ["kẹp tóc"],
    "Mũ": ["mũ", "nón"],
    "Dây chuyền": ["dây chuyền", "vòng cổ"],
    "Khuyên tai": ["khuyên tai", "bông tai"],
    "set bộ": ["set đồ", "bộ đồ"],
}

COLOR_ALIASES = {
    "đỏ": "đỏ đô",
    "đỏ đô": "đỏ đô",
    "trắng": "trắng",
    "xanh nhạt": "xanh nhạt",
    "xanh navy": "xanh navy",
    "đen": "đen",
    "be": "be",
    "xám": "xám",
}

COLORS = [
    "sọc xanh trắng",
    "xanh nhạt",
    "xanh navy",
    "trắng",
    "đen",
    "xám",
    "đỏ đô",
    "đỏ",
    "be",
    "nâu",
    "xanh pastel",
    "xanh",
    "hồng đào",
    "hồng",
    "đỏ đất",
    "cam cháy",
    "vàng",
    "nude",
    "bạc",
    "trong suốt",
]
STYLES = [
    "công sở",
    "đi làm",
    "vintage",
    "cute",
    "thoải mái",
    "basic",
    "tối giản",
    "trẻ trung",
    "sang trọng",
    "nữ tính",
    "thể thao",
    "healthy",
    "ít calo",
]
SKIN_TYPES = ["da dầu", "da khô", "da nhạy cảm", "da hỗn hợp", "da thường", "da mụn"]
CONCERNS = ["mụn", "mụn ẩn", "thâm", "nám", "lão hóa", "khô da", "dầu", "lỗ chân lông"]
AGE_GROUPS = ["0-6 tháng", "6-12 tháng", "1-2 tuổi", "2-3 tuổi", "3-5 tuổi", "sơ sinh"]


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return (
        "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        .replace("đ", "d")
        .replace("Đ", "D")
    )


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[“”\"']", " ", text)
    text = re.sub(r"[,;:!?()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _synonym_items(
    synonyms: pd.DataFrame | Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    if synonyms is None:
        return []
    if isinstance(synonyms, pd.DataFrame):
        if not {"raw_term", "normalized_term"}.issubset(synonyms.columns):
            return []
        rows = (
            synonyms[["raw_term", "normalized_term"]].dropna().itertuples(index=False)
        )
        items = [
            (str(raw).lower().strip(), str(normalized).lower().strip())
            for raw, normalized in rows
        ]
    else:
        items = [
            (str(raw).lower().strip(), str(normalized).lower().strip())
            for raw, normalized in synonyms.items()
        ]
    return sorted(
        (item for item in items if item[0] and item[1]),
        key=lambda item: len(item[0]),
        reverse=True,
    )


def apply_synonyms(text: str, synonyms: pd.DataFrame | Mapping[str, str] | None) -> str:
    normalized = normalize_text(text)
    for raw_term, normalized_term in _synonym_items(synonyms):
        pattern = rf"(?<!\w){re.escape(raw_term)}(?!\w)"
        normalized = re.sub(pattern, normalized_term, normalized)
    return normalize_text(normalized)


def contains_phrase(text: str, phrase: str) -> bool:
    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False

    pattern = rf"(?<!\w){re.escape(phrase_norm)}(?!\w)"
    if re.search(pattern, text_norm):
        return True

    text_no_accents = strip_accents(text_norm)
    phrase_no_accents = strip_accents(phrase_norm)
    pattern_no_accents = rf"(?<!\w){re.escape(phrase_no_accents)}(?!\w)"
    return bool(re.search(pattern_no_accents, text_no_accents))


def normalize_color(color: str | None) -> str | None:
    if not color:
        return None

    normalized = normalize_text(color)
    color_key = strip_accents(normalized)
    for alias, canonical in COLOR_ALIASES.items():
        alias_key = strip_accents(normalize_text(alias))
        if color_key == alias_key:
            return canonical
    return normalized


def _contains_phrase(text: str, phrase: str) -> bool:
    return contains_phrase(text, phrase)


def detect_category(text: str) -> str | None:
    if any(_contains_phrase(text, keyword) for keyword in ACCESSORY_PRIORITY_KEYWORDS):
        return "Accessories"

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if _contains_phrase(text, keyword))
        if score:
            scores[category] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def product_type_to_category(product_type: str | None) -> str | None:
    if not product_type:
        return None
    for category, product_types in PRODUCT_TYPES_BY_CATEGORY.items():
        if product_type in product_types:
            return category
    return None


def detect_product_type(text: str) -> str | None:
    best_match: tuple[int, str] | None = None
    for product_type, aliases in PRODUCT_TYPE_ALIASES.items():
        for alias in aliases:
            if _contains_phrase(text, alias):
                candidate = (len(alias), product_type)
                if best_match is None or candidate[0] > best_match[0]:
                    best_match = candidate
    return best_match[1] if best_match else None


def detect_product_types(text: str) -> list[str]:
    matched_candidates: list[tuple[int, str, str]] = []

    for product_type, aliases in PRODUCT_TYPE_ALIASES.items():
        for alias in aliases:
            if _contains_phrase(text, alias):
                matched_candidates.append(
                    (
                        len(normalize_text(alias)),
                        product_type,
                        alias,
                    )
                )

    if not matched_candidates:
        return []

    # Ưu tiên alias dài hơn, ví dụ "chân váy" trước "váy"
    matched_candidates.sort(key=lambda item: item[0], reverse=True)

    results: list[str] = []
    matched_aliases: list[str] = []

    for _, product_type, alias in matched_candidates:
        alias_norm = normalize_text(alias)

        # Nếu alias ngắn nằm trong alias dài đã match thì bỏ qua
        if any(
            alias_norm != previous and contains_phrase(previous, alias_norm)
            for previous in matched_aliases
        ):
            continue

        if product_type not in results:
            results.append(product_type)
            matched_aliases.append(alias_norm)

    return results


def _parse_number(raw: str) -> float:
    compact = raw.strip().replace(" ", "")
    if "," in compact and "." not in compact:
        compact = compact.replace(",", ".")
    compact = re.sub(r"(?<=\d)[.,](?=\d{3}(\D|$))", "", compact)
    return float(compact)


def _to_vnd(raw_number: str, unit: str | None) -> int:
    value = _parse_number(raw_number)
    unit_norm = strip_accents((unit or "").lower())
    if unit_norm in {"k", "nghin", "ngan"}:
        value *= 1_000
    elif unit_norm in {"tr", "trieu", "m"}:
        value *= 1_000_000
    elif value < 10_000:
        value *= 1_000
    return int(round(value))


def extract_budget(text: str) -> int | None:
    normalized = strip_accents(normalize_text(text))
    patterns = [
        r"(?:duoi|toi da|khong qua|nho hon|<=?)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|tr|trieu|m)?",
        r"(?:tam|khoang|co tam|gia tam|budget)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|tr|trieu|m)?",
        r"(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|tr|trieu|m)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return _to_vnd(
                match.group(1), match.group(2) if len(match.groups()) > 1 else None
            )
    return None


def extract_price_constraint(text: str) -> dict[str, int | str | None]:
    normalized = strip_accents(normalize_text(text))
    number_unit = r"(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|tr|trieu|m)?"

    between_patterns = [
        rf"\b(?:tu\s+)?{number_unit}\s*(?:den|toi|-)\s*{number_unit}\b",
    ]
    for pattern in between_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {
                "budget_min_vnd": _to_vnd(match.group(1), match.group(2)),
                "budget_max_vnd": _to_vnd(match.group(3), match.group(4)),
                "price_direction": "between",
            }

    below_patterns = [
        rf"\b(?:duoi|khong qua|toi da|nho hon|re hon|<=?)\s*{number_unit}\b",
    ]
    for pattern in below_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {
                "budget_min_vnd": None,
                "budget_max_vnd": _to_vnd(match.group(1), match.group(2)),
                "price_direction": "below",
            }

    at_least_patterns = [
        rf"\b(?:tu\s+)?{number_unit}\s*(?:tro len|tro di)\b",
        rf"\b(?:it nhat|toi thieu)\s*{number_unit}\b",
    ]
    for pattern in at_least_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {
                "budget_min_vnd": _to_vnd(match.group(1), match.group(2)),
                "budget_max_vnd": None,
                "price_direction": "at_least",
            }

    above_patterns = [
        rf"\b(?:gia\s+tu\s+tren|tu\s+tren|tren|cao hon|lon hon|hon)\s*{number_unit}\b",
    ]
    for pattern in above_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {
                "budget_min_vnd": _to_vnd(match.group(1), match.group(2)),
                "budget_max_vnd": None,
                "price_direction": "above",
            }

    return {
        "budget_min_vnd": None,
        "budget_max_vnd": None,
        "price_direction": None,
    }


def extract_first_match(text: str, values: Iterable[str]) -> str | None:
    best_match: tuple[int, str] | None = None
    for value in values:
        if _contains_phrase(text, value):
            candidate = (len(value), value)
            if best_match is None or candidate[0] > best_match[0]:
                best_match = candidate
    return best_match[1] if best_match else None


def extract_color(text: str) -> str | None:
    matched = extract_first_match(text, COLORS)
    if not matched:
        return None

    if normalize_text(matched) == "đỏ":
        text_norm = normalize_text(text)
        text_key = strip_accents(text_norm)
        has_explicit_red = bool(re.search(r"(?<!\w)đỏ(?!\w)", text_norm))
        has_unaccented_color_context = bool(re.search(r"\bmau\s+do\b", text_key))
        if not has_explicit_red and not has_unaccented_color_context:
            return None

    return normalize_color(matched)


def extract_all_matches(text: str, values: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for value in values:
        if _contains_phrase(text, value) and value not in matches:
            matches.append(value)
    return matches


FREESIZE_ALIASES = (
    "size freesize",
    "free size",
    "size free",
    "freesize",
    "free",
    "fs",
)


def extract_size(text: str) -> str | None:
    normalized_key = strip_accents(normalize_text(text)).lower()
    for alias in FREESIZE_ALIASES:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_key):
            return "Freesize"

    normalized = normalized_key.upper()
    match = re.search(r"\bSIZE\s*(XL|L|M|S)\b", normalized)
    if match:
        return match.group(1)
    match = re.search(r"\b(XL|L|M|S)\b", normalized)
    return match.group(1) if match else None
