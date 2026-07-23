from __future__ import annotations

from typing import Any

from app.order_draft_service import build_summary
from app.product_repository import SAFE_PRODUCT_DATA_ERROR_REPLY

DATABASE_PRODUCT_RULES = [
    "PRODUCT_DATA_FROM_DATABASE là nguồn sự thật duy nhất cho tên sản phẩm, SKU, giá, màu, size, phân loại, trạng thái và tồn kho.",
    "RAG_CONTEXT chỉ dùng để hỗ trợ hiểu nhu cầu, chính sách, cách tư vấn, từ đồng nghĩa và thông tin mô tả ổn định.",
    "Không được suy đoán giá hoặc tồn kho.",
    "Không được lấy giá hoặc tồn kho từ RAG_CONTEXT.",
    "Không được giới thiệu sản phẩm không có trong PRODUCT_DATA_FROM_DATABASE.",
]


def build_model_context(
    *,
    user_message: str,
    product_data_from_database: list[dict[str, Any]],
    rag_context: list[dict[str, Any]],
    rules: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "USER_MESSAGE": user_message,
        "PRODUCT_DATA_FROM_DATABASE": product_data_from_database,
        "RAG_CONTEXT": rag_context,
        "RULES": rules or DATABASE_PRODUCT_RULES,
    }


def build_model_context_text(context: dict[str, Any]) -> str:
    sections = []
    for key in ["USER_MESSAGE", "PRODUCT_DATA_FROM_DATABASE", "RAG_CONTEXT", "RULES"]:
        sections.append(f"{key}\n{context.get(key)}")
    return "\n\n".join(sections)


OUT_OF_SCOPE_COLORS = {
    "trắng",
    "đen",
    "đỏ",
    "đỏ đô",
    "xanh",
    "xanh navy",
    "xanh nhạt",
    "be",
    "xám",
}
CATALOG_STYLE_TAG_EXCLUSIONS = {
    "áo",
    "áo thun",
    "áo sơ mi",
    "sơ mi",
    "quần",
    "quần jeans",
    "jeans",
    "váy",
    "đầm",
    "chân váy",
    "set bộ",
    "đồ bộ",
    "túi",
    "túi tote",
    "mũ",
    "nón",
    "khuyên tai",
    "bông tai",
    "phụ kiện",
    "nam",
    "nữ",
    "nu",
    "unisex",
    "trắng",
    "đen",
    "đỏ",
    "đỏ đô",
    "xanh",
    "xanh navy",
    "xanh nhạt",
    "sọc xanh trắng",
    "be",
    "xám",
}
GENERIC_STYLE_TAGS = {"đẹp"}
PRODUCT_TYPE_DISPLAY_ORDER = [
    "Áo thun",
    "Áo sơ mi",
    "Váy",
    "Set bộ",
    "Mũ",
    "Khuyên tai",
]
ORDER_STATIC_REPLIES = {
    "ask_order_product": "Dạ, bạn muốn mua mẫu sản phẩm nào ạ? Bạn có thể gửi tên mẫu hoặc chọn trong các sản phẩm mình vừa tư vấn.",
    "ask_order_color": "Dạ, bạn muốn lấy màu nào ạ?",
    "ask_order_size": "Dạ, bạn muốn lấy size nào ạ?",
    "ask_order_color_and_size": "Dạ, bạn muốn lấy màu nào và size nào ạ?",
    "ask_order_quantity": "Bạn muốn lấy bao nhiêu cái ạ?",
    "ask_customer_name": "Bạn cho mình xin tên người nhận ạ?",
    "ask_customer_phone": "Bạn cho mình xin số điện thoại nhận hàng ạ?",
    "invalid_customer_phone": "Số điện thoại này chưa đúng định dạng. Bạn kiểm tra và gửi lại giúp mình nhé.",
    "ask_customer_address": "Bạn cho mình xin địa chỉ giao hàng đầy đủ ạ?",
    "ask_payment_method": "Hiện shop hỗ trợ thanh toán COD. Bạn muốn chọn COD chứ ạ?",
    "order_draft_cancelled": "Dạ, mình đã hủy đơn nháp này. Bạn có thể tiếp tục xem sản phẩm khác bất cứ lúc nào ạ.",
    "order_draft_submission_failed": "Mình chưa thể gửi đơn nháp sang shop lúc này. Thông tin của bạn vẫn được giữ trong cuộc trò chuyện. Bạn có thể nhắn “Gửi lại” để thử lại ạ.",
}


def money(vnd: int | float | None) -> str:
    if vnd is None:
        return "chưa rõ giá"
    return f"{int(vnd):,}".replace(",", ".") + "đ"


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _available_product_types(criteria: dict[str, Any]) -> list[str]:
    values = (
        criteria.get("catalog_available_product_types")
        or (criteria.get("catalog_dimensions") or {}).get("product_type")
        or []
    )
    return [str(value) for value in values if str(value).strip()]


def _ordered_product_types(criteria: dict[str, Any]) -> list[str]:
    items = _available_product_types(criteria)
    priority = {value: index for index, value in enumerate(PRODUCT_TYPE_DISPLAY_ORDER)}
    return sorted(items, key=lambda item: (priority.get(item, len(priority)), item))


def _scope_sentence(criteria: dict[str, Any]) -> str:
    items = [item.lower() for item in _available_product_types(criteria)]
    if not items:
        return "các sản phẩm đang có trong catalog"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" và {items[-1]}"


def _scope_bullets(criteria: dict[str, Any]) -> str:
    items = _ordered_product_types(criteria)
    if not items:
        return "- Các sản phẩm đang có trong catalog"
    return "\n".join(f"- {item}" for item in items)


def _product_type_examples(criteria: dict[str, Any], limit: int = 2) -> str:
    product_types = [item.lower() for item in _ordered_product_types(criteria)]
    preferred = [item for item in product_types if item in {"áo thun", "áo sơ mi"}]
    examples = list(dict.fromkeys([*preferred, *product_types]))[:limit]
    if not examples:
        return "áo thun, áo sơ mi"
    return ", ".join(examples)


def _join_items(items: list[str]) -> str:
    if not items:
        return "mặt hàng đó"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" hoặc {items[-1]}"


def _base_out_of_scope_item(item: str) -> str:
    words = str(item).split()
    if words and words[-1].lower() in OUT_OF_SCOPE_COLORS:
        return " ".join(words[:-1])
    if len(words) >= 2 and " ".join(words[-2:]).lower() in OUT_OF_SCOPE_COLORS:
        return " ".join(words[:-2])
    return str(item)


def _catalog_product_label(criteria: dict[str, Any]) -> str:
    product_type = criteria.get("product_type")
    if product_type:
        return str(product_type).lower()
    product_types = [
        str(item) for item in criteria.get("product_types") or [] if str(item).strip()
    ]
    if len(product_types) == 1:
        return product_types[0].lower()
    apparel_intent = criteria.get("apparel_intent")
    if apparel_intent:
        return str(apparel_intent).lower()
    return "sản phẩm"


def _style_tags(product: dict[str, Any]) -> list[str]:
    raw_tags = str(product.get("tags") or "")
    tags: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, raw_tag in enumerate(raw_tags.split("|")):
        tag = raw_tag.strip()
        key = tag.lower()
        if (
            not tag
            or key in seen
            or key in CATALOG_STYLE_TAG_EXCLUSIONS
            or key == "nan"
        ):
            continue
        tags.append((index, tag))
        seen.add(key)

    tags.sort(key=lambda item: (item[1].lower() in GENERIC_STYLE_TAGS, item[0]))
    return [tag for _, tag in tags[:2]]


def _catalog_browsing_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        products,
        key=lambda product: (
            _safe_int(product.get("stock_total")) > 0,
            _safe_float(product.get("rating")),
            _safe_int(product.get("sold_30d")),
            -_safe_int(product.get("effective_price_vnd")),
        ),
        reverse=True,
    )


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if not text or text.lower() == "nan" else text


def _criteria_detail_parts(criteria: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    if criteria.get("gender"):
        parts.append(str(criteria["gender"]))
    if criteria.get("color"):
        parts.append(f"màu {str(criteria['color']).lower()}")
    if criteria.get("size"):
        parts.append(f"size {criteria['size']}")
    if criteria.get("budget_max_vnd"):
        parts.append(f"dưới {money(criteria.get('budget_max_vnd'))}")
    return parts


def _short_description(product: dict[str, Any]) -> str:
    description = _clean_text(product.get("short_description"))
    if len(description) <= 120:
        return description
    return description[:117].rstrip() + "..."


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _product_note(product: dict[str, Any], criteria: dict[str, Any]) -> str:
    details: list[str] = []
    if not criteria.get("variant_filter_relaxed"):
        if criteria.get("color"):
            details.append(f"Màu {str(criteria['color']).lower()}")
        if criteria.get("size"):
            details.append(f"size {criteria['size']}")

    description = _short_description(product)
    if description:
        details.append(
            _lower_first(description.rstrip("."))
            if details
            else description.rstrip(".")
        )
    else:
        style_tags = _style_tags(product)
        if style_tags:
            details.append(f"Phong cách {'/'.join(style_tags)}")

    stock = _safe_int(product.get("stock_total"))
    details.append(f"hiện còn {stock} sản phẩm" if stock > 0 else "hiện tạm hết hàng")
    sentence = ", ".join(details).strip()
    if not sentence:
        return ""
    return sentence[0].upper() + sentence[1:] + "."


def _build_product_search_reply(
    criteria: dict[str, Any], products: list[dict[str, Any]]
) -> str:
    product_type = criteria.get("product_type")
    product_label = (
        f"một vài mẫu {str(product_type).lower()}"
        if product_type
        else "một vài sản phẩm"
    )
    use_case = _clean_text(criteria.get("use_case"))
    detail_parts = _criteria_detail_parts(criteria)

    if criteria.get("response_mode") == "cheaper_product":
        intro = f"Dạ, shop lọc được {product_label} có giá thấp hơn mẫu hiện tại:"
    elif criteria.get("response_mode") == "premium_product":
        intro = f"Dạ, shop gợi ý {product_label} chất liệu/đánh giá tốt hơn; giá có thể cao hơn mẫu trước:"
    elif criteria.get("response_mode") == "style_change":
        intro = f"Dạ, shop đổi sang phong cách {str(criteria.get('style') or '').lower()} và giữ các tiêu chí chính cho bạn:"
    elif criteria.get("response_mode") in {
        "alternative_product",
        "similar_product",
        "reject_current_product",
    }:
        intro = (
            f"Dạ, shop đổi sang {product_label} khác và đã loại mẫu bạn vừa từ chối:"
        )
    elif use_case:
        intro = f"Dạ, với nhu cầu {use_case}, shop gợi ý cho bạn {product_label}:"
    elif detail_parts:
        intro = f"Dạ, shop tìm được {product_label} phù hợp với nhu cầu {', '.join(detail_parts)}:"
    else:
        intro = f"Dạ, shop tìm được {product_label} phù hợp cho bạn:"

    lines = [intro, ""]
    for index, product in enumerate(products[:5], start=1):
        lines.append(
            f"{index}. {product.get('product_name')} - {money(product.get('effective_price_vnd'))}"
        )
        note = _product_note(product, criteria)
        if note:
            lines.append(f"   {note}")
        lines.append("")

    lines.append("Bạn muốn mình lọc tiếp theo size, màu hoặc ngân sách không ạ?")
    return "\n".join(lines).strip()


def _build_catalog_browsing_reply(
    criteria: dict[str, Any], products: list[dict[str, Any]]
) -> str:
    product_label = _catalog_product_label(criteria)
    if not products:
        return (
            f"Dạ hiện tại shop chưa có mẫu {product_label} đang hiển thị trong catalog ạ.\n"
            "Bạn muốn mình lọc thêm theo size, màu hoặc ngân sách không ạ?"
        )

    lines = [f"Dạ shop mình hiện có một số mẫu {product_label} như:"]
    for index, product in enumerate(_catalog_browsing_products(products), start=1):
        stock_status = (
            "còn hàng" if _safe_int(product.get("stock_total")) > 0 else "hết hàng"
        )
        style_tags = _style_tags(product)
        style_clause = f", phong cách {'/'.join(style_tags)}" if style_tags else ""
        lines.append(
            f"{index}. {product.get('product_name')} - "
            f"{money(product.get('effective_price_vnd'))}, {stock_status}{style_clause}."
        )
    lines.append("Bạn muốn mình lọc thêm theo size, màu hoặc ngân sách không ạ?")
    return "\n".join(lines)


def _build_catalog_overview_reply(criteria: dict[str, Any]) -> str:
    product_types = _ordered_product_types(criteria)
    if not product_types:
        product_types = ["Các sản phẩm đang có trong catalog"]

    lines = ["Dạ shop mình hiện đang bán các nhóm sản phẩm như:"]
    lines.extend(f"- {product_type}" for product_type in product_types)
    lines.extend(
        [
            "",
            "Bạn muốn mình gợi ý theo nhu cầu nào, ví dụ đi chơi, đi làm, "
            "mặc hằng ngày, theo size hoặc theo ngân sách không ạ?",
        ]
    )
    return "\n".join(lines)


def _build_ask_product_type_reply(criteria: dict[str, Any]) -> str:
    product_types = _ordered_product_types(criteria)
    if not product_types:
        product_types = ["Các sản phẩm đang có trong catalog"]

    lines = [
        "Xin chào bạn, bạn đang muốn tìm sản phẩm nào ạ?",
        "",
        "Hiện AutoBiz Fashion có các nhóm như:",
    ]
    lines.extend(f"- {product_type}" for product_type in product_types)
    lines.extend(
        [
            "",
            "Bạn muốn mình tư vấn nhóm nào trước ạ?",
        ]
    )
    return "\n".join(lines)


def _build_ask_use_case_reply() -> str:
    return "Bạn đang cần đồ để đi làm, đi chơi, đi học hay mặc hằng ngày ạ?"


def _build_collect_product_preferences_reply(criteria: dict[str, Any]) -> str:
    product_type = str(criteria.get("product_type") or "sản phẩm này").lower()
    missing = set(criteria.get("missing_fields") or [])
    if not missing:
        return "Dạ, bạn muốn mình lọc thêm theo màu hoặc size không ạ?"

    if missing == {"gender", "use_case", "budget_max_vnd"}:
        return (
            f"Dạ, bạn muốn tìm {product_type} cho nam/nữ/unisex, "
            "mặc đi làm hay đi chơi, và ngân sách khoảng bao nhiêu ạ?"
        )

    parts: list[str] = []
    if "gender" in missing:
        parts.append("cho nam/nữ/unisex")
    if "use_case" in missing:
        parts.append("mặc đi làm, đi chơi, đi học hay mặc hằng ngày")
    if "budget_max_vnd" in missing:
        parts.append("ngân sách khoảng bao nhiêu")

    if len(parts) == 1:
        return f"Dạ, {parts[0]} ạ?"
    return "Dạ, bạn cho mình thêm " + ", ".join(parts[:-1]) + f" và {parts[-1]} ạ?"


def _format_height(height_cm: int | None) -> str:
    if height_cm is None:
        return "chiều cao của bạn"
    if height_cm >= 100:
        return f"{height_cm // 100}m{height_cm % 100:02d}"
    return f"{height_cm}cm"


def _size_product_label(criteria: dict[str, Any]) -> str:
    product_type = str(criteria.get("product_type") or "sản phẩm này").lower()
    gender = str(criteria.get("gender") or "").strip().lower()
    if gender:
        return f"{product_type} {gender}"
    return product_type


def _build_size_recommendation_reply(criteria: dict[str, Any]) -> str:
    recommendation = criteria.get("size_recommendation") or {}
    recommended_size = recommendation.get("recommended_size")
    alternative_size = recommendation.get("alternative_size")
    fit_notes = recommendation.get("fit_notes") or {}
    product_label = _size_product_label(criteria)

    if not recommended_size:
        return (
            f"Mình chưa có đủ bảng size structured cho {product_label} để tư vấn chắc chắn ạ. "
            "Bạn cho mình thêm số đo vòng ngực/vòng eo hoặc chọn nhóm sản phẩm khác trong catalog nhé."
        )

    height_text = _format_height(criteria.get("height_cm"))
    weight = criteria.get("weight_kg")
    fit_preference = criteria.get("fit_preference") or recommendation.get(
        "fit_preference"
    )
    fit_aliases = {
        "slim": "slim",
        "ôm": "slim",
        "regular": "regular",
        "vừa": "regular",
        "loose": "loose",
        "rộng": "loose",
    }
    fit_mode = fit_aliases.get(str(fit_preference or "").strip().lower())
    fit_labels = {
        "slim": "mặc ôm",
        "regular": "mặc vừa người",
        "loose": "mặc rộng",
    }
    fit_label = fit_labels.get(fit_mode)
    boundary_case = bool(
        recommendation.get("boundary_case") or recommendation.get("is_between_sizes")
    )
    size_rank = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}

    if boundary_case and alternative_size and not fit_mode:
        recommended_note = str(fit_notes.get(recommended_size) or "vừa người").lower()
        recommended_clause = (
            "mặc vừa người" if "vừa" in recommended_note else recommended_note
        )
        alternative_is_larger = size_rank.get(
            str(alternative_size), 99
        ) > size_rank.get(str(recommended_size), 99)
        alternative_clause = (
            "Nếu thích thoải mái hơn hoặc sản phẩm có form ôm"
            if alternative_is_larger
            else "Nếu thích mặc gọn hoặc ôm hơn"
        )
        lines = [
            f"Với chiều cao {height_text} và cân nặng {weight}kg, bạn có thể chọn size {recommended_size} nếu thích {recommended_clause} cho {product_label} ạ.",
            f"{alternative_clause}, bạn có thể cân nhắc size {alternative_size}.",
        ]
    elif fit_label:
        lines = [
            f"Với chiều cao {height_text}, cân nặng {weight}kg và sở thích {fit_label}, bạn nên chọn size {recommended_size} cho {product_label} ạ.",
        ]
    else:
        lines = [
            f"Với chiều cao {height_text} và cân nặng {weight}kg, bạn có thể tham khảo size {recommended_size} cho {product_label} ạ.",
        ]

    if boundary_case and alternative_size:
        boundary_sizes = sorted(
            [str(recommended_size), str(alternative_size)],
            key=lambda size: size_rank.get(size.upper(), 99),
        )
        lines.extend(
            [
                "",
                f"Bạn đang nằm ở khoảng giữa size {boundary_sizes[0]} và {boundary_sizes[1]}.",
            ]
        )

    if boundary_case and not fit_mode:
        lines.extend(
            [
                "",
                "Bạn thích mặc ôm, vừa người hay rộng để mình chốt size sát hơn ạ?",
            ]
        )
    elif fit_mode == "loose":
        lines.extend(
            [
                "",
                f"Vì bạn thích mặc rộng, size {recommended_size} là lựa chọn phù hợp hơn.",
            ]
        )
    elif fit_mode == "slim":
        lines.extend(
            [
                "",
                f"Vì bạn thích mặc ôm, size {recommended_size} là lựa chọn phù hợp hơn.",
            ]
        )
    elif fit_mode == "regular":
        lines.extend(
            [
                "",
                f"Vì bạn thích mặc vừa người, size {recommended_size} là lựa chọn phù hợp hơn.",
            ]
        )

    lines.append("")
    note = fit_notes.get(recommended_size)
    if note:
        lines.append(f"- Size {recommended_size}: {str(note).lower()}")
    else:
        lines.append(
            f"- Size {recommended_size}: khoảng phù hợp nhất theo bảng size hiện tại"
        )

    if alternative_size:
        alternative_note = fit_notes.get(alternative_size)
        if alternative_note:
            lines.append(f"- Size {alternative_size}: {str(alternative_note).lower()}")
        else:
            lines.append(
                f"- Size {alternative_size}: phù hợp nếu bạn thích mặc rộng hoặc thoải mái hơn"
            )

    return "\n".join(lines)


def _product_info_lines(
    product: dict[str, Any], variant: dict[str, Any], *, sale_label: bool = True
) -> list[str]:
    price_label = "Giá"
    return [
        f"{product.get('product_name') or 'Sản phẩm'}:",
        f"- Mã sản phẩm: {product.get('product_id')}",
        f"- SKU: {variant.get('sku')}",
        f"- Size: {variant.get('size')}",
        f"- Màu: {variant.get('color')}",
        f"- {price_label}: {money(product.get('effective_price_vnd'))}",
        f"- Tồn kho: {variant.get('stock')} sản phẩm",
    ]


def _alternative_color_labels(alternatives: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for alternative in alternatives:
        color = str(alternative.get("color") or "").strip()
        key = color.lower()
        if color and key not in seen:
            labels.append(color)
            seen.add(key)
    return labels


def _build_availability_reply(
    criteria: dict[str, Any], products: list[dict[str, Any]]
) -> str:
    availability = criteria.get("availability") or {}
    product = availability.get("product") or {}
    variant = availability.get("variant") or {}
    status = availability.get("status")
    if status == "database_error":
        return SAFE_PRODUCT_DATA_ERROR_REPLY

    requested_name = (
        criteria.get("requested_product_name")
        or product.get("product_name")
        or "sản phẩm này"
    )
    size = variant.get("size") or criteria.get("size")
    color = variant.get("color") or criteria.get("color")

    if status == "in_stock":
        raw_message = str(criteria.get("raw_message") or "").lower()

        asks_stock_quantity = any(
            phrase in raw_message
            for phrase in [
                "còn bao nhiêu",
                "còn mấy",
                "tồn bao nhiêu",
                "tồn kho bao nhiêu",
                "còn bao nhiêu cái",
            ]
        )

        if asks_stock_quantity:
            name = product.get("product_name") or requested_name
            stock = variant.get("stock")

            if stock is None:
                for key in ("available_qty", "stock_total", "quantity"):
                    value = product.get(key)
                    if value is not None:
                        stock = value
                        break

            stock = _safe_int(stock)

            details = []

            if color:
                details.append(f"màu {str(color).lower()}")

            if size:
                details.append(f"size {size}")

            variant_text = ", ".join(details)

            if variant_text:
                return f"Dạ, mẫu {name} {variant_text} hiện còn " f"{stock} sản phẩm ạ."

            return f"Dạ, mẫu {name} hiện còn {stock} sản phẩm ạ."
        if criteria.get("intent") == "mixed_product_request":
            lines = [
                f"Dạ shop có {requested_name} size {size} màu {str(color).lower()} ạ.",
                "",
                *_product_info_lines(product, variant, sale_label=False),
            ]
            out_items = criteria.get("out_of_scope_items") or []
            if out_items:
                base_items = [_base_out_of_scope_item(item) for item in out_items]
                lines.extend(
                    [
                        "",
                        f"Tuy nhiên, hiện tại shop không bán {_join_items(base_items)}. "
                        "Shop hiện tập trung vào các nhóm sản phẩm như "
                        f"{_scope_sentence(criteria)}.",
                        "Với áo sơ mi denim, bạn có thể phối cùng quần jeans hoặc quần kaki nếu muốn theo phong cách casual.",
                    ]
                )
            return "\n".join(lines)

        lines = [
            "Dạ còn hàng ạ.",
            "",
            *_product_info_lines(product, variant, sale_label=True),
            "",
            "Shop có hỗ trợ COD và freeship cho đơn từ 499.000đ.",
        ]
        return "\n".join(lines)

    if status == "out_of_stock":
        requested_parts = [f"Dạ, mẫu {requested_name}"]
        if criteria.get("size") and size:
            requested_parts.append(f"size {size}")
        if color:
            requested_parts.append(f"màu {str(color).lower()}")
        lines = [
            f"{' '.join(requested_parts)} hiện đang hết hàng ạ.",
            "",
        ]
        alternatives = availability.get("alternatives") or []
        if alternatives:
            color_labels = _alternative_color_labels(alternatives)
            if color_labels:
                lines.append("Shop hiện còn:")
                for color_label in color_labels:
                    lines.append(f"- Màu {color_label.lower()}")
            else:
                lines.append("Shop còn một vài lựa chọn khác cùng mẫu.")
        else:
            lines.append("Shop chưa có màu thay thế cùng size ở thời điểm này.")
        return "\n".join(lines)

    if status == "variant_not_found":
        lines = [
            f"Dạ shop có mẫu {requested_name}, nhưng hiện chưa có đúng variant bạn hỏi ạ.",
        ]
        alternatives = availability.get("alternatives") or []
        if alternatives:
            lines.append("")
            color_labels = _alternative_color_labels(alternatives)
            if criteria.get("color") and color_labels:
                lines.append("Shop hiện còn:")
                for color_label in color_labels:
                    lines.append(f"- Màu {color_label.lower()}")
            else:
                lines.append("Một vài lựa chọn gần nhất đang có:")
                lines.append("")
                for index, alternative in enumerate(alternatives, start=1):
                    lines.append(
                        f"{index}. Size {alternative.get('size')}, màu {alternative.get('color')} - "
                        f"còn {alternative.get('stock')} sản phẩm - "
                        f"{money(alternative.get('effective_price_vnd'))}"
                    )
        return "\n".join(lines)

    if status == "variant_options":
        alternatives = availability.get("alternatives") or []
        lines = [f"Dạ mẫu {requested_name} đang có các variant sau ạ:", ""]
        for index, alternative in enumerate(alternatives, start=1):
            lines.append(
                f"{index}. SKU {alternative.get('sku')} - "
                f"size {alternative.get('size')}, màu {alternative.get('color')} - "
                f"{money(alternative.get('effective_price_vnd'))} - "
                f"còn {alternative.get('stock')} sản phẩm"
            )
        if not criteria.get("size") or not criteria.get("color"):
            lines.extend(["", "Bạn muốn mình giữ size/màu nào cho mẫu này ạ?"])
        return "\n".join(lines)

    return f"Dạ mình chưa tìm thấy sản phẩm {requested_name} trong danh sách hiện tại của shop."


def _build_policy_reply(criteria: dict[str, Any]) -> str:
    policy = criteria.get("policy") or {}
    value = _clean_text(policy.get("policy_value"))
    if value:
        return f"Dạ, theo chính sách hiện có của shop: {value}."

    label = _clean_text(policy.get("policy_label")) or "chính sách này"
    return f"Dạ hiện dữ liệu chính sách của shop chưa có thông tin về {label}, nên mình chưa thể khẳng định ạ."


def _build_negative_feedback_reply(criteria: dict[str, Any]) -> str:
    reason = criteria.get("negative_feedback_reason")
    if reason == "price":
        return "Dạ mình xin lỗi, mẫu này có vẻ chưa hợp ngân sách của bạn. Mình có thể đổi sang mẫu rẻ hơn và vẫn giữ nhu cầu ban đầu."
    if reason == "size":
        return "Dạ mình xin lỗi, phần size có thể chưa đúng cảm giác mặc bạn muốn. Mình sẽ giữ sản phẩm hiện tại và có thể kiểm tra size khác hoặc tư vấn lại theo kiểu ôm/vừa/rộng."
    if reason == "advice":
        return "Dạ mình xin lỗi vì phần tư vấn chưa đúng ý bạn. Mình sẽ giữ lại các tiêu chí trước đó và điều chỉnh theo lý do bạn muốn đổi."
    return "Dạ mình xin lỗi, mẫu này chưa hợp gu của bạn. Mình có thể đổi sang mẫu khác cùng nhu cầu và không lặp lại mẫu vừa rồi."


def _build_out_of_scope_reply(criteria: dict[str, Any]) -> str:
    items = criteria.get("out_of_scope_items") or []
    item_text = _join_items(items)
    return (
        "Dạ hiện tại shop không có sản phẩm này ạ.\n\n"
        f"Shop hiện chưa bán {item_text}. "
        "Shop hiện đang bán các nhóm:\n\n"
        f"{_scope_bullets(criteria)}\n\n"
        "Bạn muốn xem nhóm nào trong các nhóm trên ạ?"
    )


def _build_closest_product_type_offer_reply(criteria: dict[str, Any]) -> str:
    requested = criteria.get("requested_product_group") or "sản phẩm này"
    suggested = criteria.get("suggested_product_type") or "sản phẩm tương tự"

    return (
        f"Dạ, hiện shop chưa có {requested} trong catalog ạ.\n\n"
        f"Shop đang có một số mẫu {str(suggested).lower()} "
        "có phong cách khá gần với nhu cầu của bạn. "
        "Bạn có muốn mình giới thiệu các mẫu đó không ạ?"
    )


def _build_closest_product_type_decline_reply(criteria: dict[str, Any]) -> str:
    requested = criteria.get("requested_product_group") or "sản phẩm này"
    return (
        f"Dạ, mình chưa giới thiệu mẫu thay thế cho {requested} nhé. "
        "Bạn có thể nhắn nhóm sản phẩm khác trong catalog để mình tìm tiếp ạ."
    )


def _build_nearest_over_budget_reply(criteria: dict[str, Any]) -> str:
    nearest_products = criteria.get("nearest_over_budget_products") or []
    product = nearest_products[0] if nearest_products else None
    budget_text = money(criteria.get("budget_max_vnd"))
    if not product:
        return (
            f"Hiện shop chưa có mẫu khớp hoàn toàn trong ngân sách {budget_text}. "
            "Bạn có thể tăng ngân sách hoặc đổi màu/size giúp mình nhé."
        )

    return (
        f"Hiện shop chưa có mẫu khớp hoàn toàn trong ngân sách {budget_text}.\n\n"
        f"Mẫu gần nhất là {product.get('product_name')} - {money(product.get('effective_price_vnd'))}. "
        "Bạn có muốn mình giới thiệu mẫu này không ạ?"
    )


def _build_product_type_out_of_stock_reply(criteria: dict[str, Any]) -> str:
    product_type = str(criteria.get("product_type") or "sản phẩm này").lower()
    color = criteria.get("color")
    size = criteria.get("size")
    out_of_stock_products = criteria.get("out_of_stock_products") or []

    product_name = product_type
    if out_of_stock_products:
        product_name = out_of_stock_products[0].get("product_name") or product_type

    details = []
    if color:
        details.append(f"màu {str(color).lower()}")
    if size:
        details.append(f"size {size}")

    detail_text = " ".join(details)

    if detail_text:
        return (
            f"Dạ shop có mẫu {product_name} {detail_text}, "
            "nhưng hiện mẫu này đang tạm hết hàng ạ.\n\n"
            "Bạn có muốn mình gợi ý các sản phẩm khác đang còn hàng không?"
        )

    return (
        f"Dạ shop có nhóm {product_type}, nhưng hiện các mẫu thuộc nhóm này "
        "đang tạm hết hàng ạ.\n\n"
        "Bạn có muốn mình gợi ý các sản phẩm khác đang còn hàng không?"
    )


def build_reply(criteria: dict[str, Any], products: list[dict[str, Any]]) -> str:
    response_mode = criteria.get("response_mode")
    if criteria.get("product_data_error") or response_mode == "product_data_error":
        return SAFE_PRODUCT_DATA_ERROR_REPLY

    if response_mode == "show_order_draft_summary" and criteria.get("order_draft"):
        return build_summary(criteria["order_draft"])
    if response_mode in ORDER_STATIC_REPLIES:
        return ORDER_STATIC_REPLIES[str(response_mode)]

    if criteria.get("response_mode") == "offer_closest_product_type":
        return _build_closest_product_type_offer_reply(criteria)

    if criteria.get("response_mode") == "decline_closest_product_type":
        return _build_closest_product_type_decline_reply(criteria)

    if criteria.get("intent") == "policy_question":
        return _build_policy_reply(criteria)

    if criteria.get("intent") == "negative_feedback":
        return _build_negative_feedback_reply(criteria)

    if (
        criteria.get("intent") == "variant_availability_check"
        and criteria.get("response_mode") == "ask_product_for_variant"
    ):
        return "Bạn muốn mình kiểm tra màu/size này cho mẫu sản phẩm nào ạ?"

    if criteria.get("intent") in {
        "product_availability_check",
        "mixed_product_request",
        "variant_availability_check",
    }:
        return _build_availability_reply(criteria, products)

    if criteria.get("intent") == "size_recommendation":
        if criteria.get("response_mode") == "ask_product_for_size":
            examples = _product_type_examples(criteria)
            return f"Bạn đang muốn chọn size cho {examples} hay sản phẩm nào ạ?"
        if criteria.get("response_mode") == "ask_size_profile":
            return "Bạn cho mình chiều cao, cân nặng và nếu có thì sở thích mặc ôm/vừa/rộng để mình tư vấn size chính xác hơn ạ."
        return _build_size_recommendation_reply(criteria)

    if criteria.get("intent") == "out_of_scope_request":
        return _build_out_of_scope_reply(criteria)

    if criteria.get("response_mode") == "ask_product_type":
        return _build_ask_product_type_reply(criteria)

    if criteria.get("response_mode") == "ask_use_case":
        return _build_ask_use_case_reply()

    if criteria.get("response_mode") in {"no_pending_context", "choose_product_type"}:
        available_text = _scope_sentence(criteria)
        return (
            "Bạn muốn mình gợi ý sản phẩm thuộc nhóm nào ạ? "
            f"Hiện shop có các nhóm như {available_text}.\n\n"
            "Bạn chỉ cần nhắn tên nhóm sản phẩm trong các nhóm này."
        )

    if criteria.get("response_mode") == "collect_product_preferences":
        return _build_collect_product_preferences_reply(criteria)

    if criteria.get("response_mode") == "product_type_out_of_stock":
        return _build_product_type_out_of_stock_reply(criteria)

    if criteria.get("response_mode") == "invalid_product_type_choice":
        available_text = _scope_sentence(criteria)
        return (
            "Dạ hiện shop chưa có nhóm sản phẩm đó ạ. "
            f"Shop đang có: {available_text}. "
            "Bạn muốn mình gợi ý nhóm nào trong các nhóm này ạ?"
        )

    if criteria.get("response_mode") == "catalog_overview":
        return _build_catalog_overview_reply(criteria)

    if criteria.get("catalog_coverage") in {"not_supported", "unsupported"}:
        return _build_out_of_scope_reply(criteria)

    if criteria.get("response_mode") == "catalog_browsing":
        return _build_catalog_browsing_reply(criteria, products)

    can_search = bool(criteria.get("category_code") or criteria.get("product_type"))

    if not products and criteria.get("need_clarification") and not can_search:
        questions = criteria.get("clarification_questions") or [
            "Bạn cho mình thêm thông tin để tư vấn chính xác hơn nhé?"
        ]
        return "Mình cần thêm một chút thông tin:\n" + "\n".join(
            f"- {question}" for question in questions
        )

    if not products:
        if criteria.get("budget_max_vnd"):
            return _build_nearest_over_budget_reply(criteria)
        return "Mình chưa tìm thấy sản phẩm phù hợp với các điều kiện này. Bạn có thể nới ngân sách hoặc đổi màu/size giúp mình nhé."

    return _build_product_search_reply(criteria, products)
