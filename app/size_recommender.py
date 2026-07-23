from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.normalizer import normalize_text, strip_accents

SIZE_CHART_PATH = PROJECT_ROOT / "data" / "size_chart.csv"
SIZE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
HEIGHT_BOUNDARY_MARGIN_CM = 2
WEIGHT_BOUNDARY_MARGIN_KG = 1


def _key(value: Any) -> str:
    return strip_accents(normalize_text(str(value or "")))


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _size_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    size = str(row.get("size") or "").upper()
    return SIZE_ORDER.get(size, 99), size


@lru_cache(maxsize=1)
def load_size_chart(path: str | Path = SIZE_CHART_PATH) -> list[dict[str, Any]]:
    chart_path = Path(path)
    if not chart_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with chart_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            for field in [
                "height_min_cm",
                "height_max_cm",
                "weight_min_kg",
                "weight_max_kg",
            ]:
                row[field] = _to_int(row.get(field))
            row["product_type"] = str(row.get("product_type") or "").strip()
            row["gender"] = str(row.get("gender") or "").strip().lower()
            row["size"] = str(row.get("size") or "").strip().upper()
            row["fit_note"] = str(row.get("fit_note") or "").strip()
            if row["product_type"] and row["size"]:
                rows.append(row)
    return rows


def _row_matches_profile(row: dict[str, Any], height_cm: int, weight_kg: int) -> bool:
    height_min = row.get("height_min_cm")
    height_max = row.get("height_max_cm")
    weight_min = row.get("weight_min_kg")
    weight_max = row.get("weight_max_kg")
    return (
        height_min is not None
        and height_max is not None
        and weight_min is not None
        and weight_max is not None
        and height_min <= height_cm <= height_max
        and weight_min <= weight_kg <= weight_max
    )


def _distance_from_row(row: dict[str, Any], height_cm: int, weight_kg: int) -> float:
    height_min = int(row.get("height_min_cm") or 0)
    height_max = int(row.get("height_max_cm") or 0)
    weight_min = int(row.get("weight_min_kg") or 0)
    weight_max = int(row.get("weight_max_kg") or 0)
    height_center = (height_min + height_max) / 2
    weight_center = (weight_min + weight_max) / 2
    height_span = max(1, height_max - height_min)
    weight_span = max(1, weight_max - weight_min)
    return (
        abs(height_cm - height_center) / height_span
        + abs(weight_kg - weight_center) / weight_span
    )


def _outside_distance_from_row(
    row: dict[str, Any], height_cm: int, weight_kg: int
) -> float:
    height_min = int(row.get("height_min_cm") or 0)
    height_max = int(row.get("height_max_cm") or 0)
    weight_min = int(row.get("weight_min_kg") or 0)
    weight_max = int(row.get("weight_max_kg") or 0)
    height_span = max(1, height_max - height_min)
    weight_span = max(1, weight_max - weight_min)
    height_distance = 0
    weight_distance = 0
    if height_cm < height_min:
        height_distance = height_min - height_cm
    elif height_cm > height_max:
        height_distance = height_cm - height_max
    if weight_kg < weight_min:
        weight_distance = weight_min - weight_kg
    elif weight_kg > weight_max:
        weight_distance = weight_kg - weight_max
    return height_distance / height_span + weight_distance / weight_span


def _near_boundary(row: dict[str, Any], height_cm: int, weight_kg: int) -> bool:
    return bool(_boundary_directions(row, height_cm, weight_kg))


def _boundary_directions(
    row: dict[str, Any], height_cm: int, weight_kg: int
) -> list[tuple[str, int, float]]:
    checks = [
        (
            "height",
            -1,
            abs(height_cm - int(row.get("height_min_cm") or height_cm)),
            HEIGHT_BOUNDARY_MARGIN_CM,
        ),
        (
            "height",
            1,
            abs(height_cm - int(row.get("height_max_cm") or height_cm)),
            HEIGHT_BOUNDARY_MARGIN_CM,
        ),
        (
            "weight",
            -1,
            abs(weight_kg - int(row.get("weight_min_kg") or weight_kg)),
            WEIGHT_BOUNDARY_MARGIN_KG,
        ),
        (
            "weight",
            1,
            abs(weight_kg - int(row.get("weight_max_kg") or weight_kg)),
            WEIGHT_BOUNDARY_MARGIN_KG,
        ),
    ]
    directions: list[tuple[str, int, float]] = []
    for dimension, direction, distance, margin in checks:
        if distance <= margin:
            directions.append((dimension, direction, float(distance)))
    return sorted(directions, key=lambda item: item[2])


def _height_matches(row: dict[str, Any], height_cm: int) -> bool:
    height_min = row.get("height_min_cm")
    height_max = row.get("height_max_cm")
    return (
        height_min is not None
        and height_max is not None
        and height_min <= height_cm <= height_max
    )


def _weight_matches(row: dict[str, Any], weight_kg: int) -> bool:
    weight_min = row.get("weight_min_kg")
    weight_max = row.get("weight_max_kg")
    return (
        weight_min is not None
        and weight_max is not None
        and weight_min <= weight_kg <= weight_max
    )


def _fit_mode(fit_preference: str | None) -> str | None:
    fit_key = _key(fit_preference)
    if fit_key in {"slim", "om", "gon", "body", "mac om", "om body"}:
        return "slim"
    if fit_key in {"loose", "rong", "thoai mai", "oversize", "oversized", "mac rong"}:
        return "loose"
    if fit_key in {"regular", "vua", "vua nguoi", "mac vua"}:
        return "regular"
    return None


def _select_recommended(
    rows: list[dict[str, Any]],
    height_cm: int,
    weight_kg: int,
    fit_mode: str | None,
) -> dict[str, Any]:
    if fit_mode == "slim":
        return sorted(rows, key=_size_sort_key)[0]
    if fit_mode == "loose":
        return sorted(rows, key=_size_sort_key)[-1]
    return min(
        rows,
        key=lambda row: (
            _distance_from_row(row, height_cm, weight_kg),
            _size_sort_key(row),
        ),
    )


def _nearest_alternative(
    rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    height_cm: int,
    weight_kg: int,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("size") != recommended.get("size")]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _outside_distance_from_row(row, height_cm, weight_kg),
            _distance_from_row(row, height_cm, weight_kg),
            abs(
                SIZE_ORDER.get(str(row.get("size") or ""), 99)
                - SIZE_ORDER.get(str(recommended.get("size") or ""), 99)
            ),
        ),
    )


def _adjacent_row(
    rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    direction: int,
) -> dict[str, Any] | None:
    sorted_rows = sorted(rows, key=_size_sort_key)
    recommended_index = sorted_rows.index(recommended)
    candidate_index = recommended_index + direction
    if 0 <= candidate_index < len(sorted_rows):
        return sorted_rows[candidate_index]
    return None


def _boundary_alternative(
    rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    height_cm: int,
    weight_kg: int,
) -> dict[str, Any] | None:
    for _, direction, _ in _boundary_directions(recommended, height_cm, weight_kg):
        alternative = _adjacent_row(rows, recommended, direction)
        if alternative is not None:
            return alternative
    return None


def _apply_fit_preference_to_boundary(
    recommended: dict[str, Any],
    alternative: dict[str, Any] | None,
    fit_mode: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Điều chỉnh size chính theo sở thích mặc khi khách nằm ở vùng biên.

    - slim: ưu tiên size nhỏ hơn
    - loose: ưu tiên size lớn hơn
    - regular/None: giữ kết quả được tính theo độ gần tâm
    """
    if alternative is None or fit_mode not in {"slim", "loose"}:
        return recommended, alternative

    recommended_rank = SIZE_ORDER.get(
        str(recommended.get("size") or "").upper(),
        99,
    )
    alternative_rank = SIZE_ORDER.get(
        str(alternative.get("size") or "").upper(),
        99,
    )

    if fit_mode == "loose" and alternative_rank > recommended_rank:
        return alternative, recommended

    if fit_mode == "slim" and alternative_rank < recommended_rank:
        return alternative, recommended

    return recommended, alternative


def _fit_notes(*rows: dict[str, Any] | None) -> dict[str, str]:
    return {
        str(row.get("size")): str(row.get("fit_note") or "")
        for row in rows
        if row and row.get("size")
    }


def _filter_rows(product_type: str, gender: str | None) -> list[dict[str, Any]]:
    product_key = _key(product_type)
    rows = [
        row for row in load_size_chart() if _key(row.get("product_type")) == product_key
    ]
    if not rows:
        return []

    if gender:
        gender_key = _key(gender)
        gender_rows = [row for row in rows if _key(row.get("gender")) == gender_key]
        if gender_rows:
            return gender_rows
    return rows


def recommend_size(
    product_type: str,
    gender: str | None,
    height_cm: int,
    weight_kg: int,
    fit_preference: str | None = None,
) -> dict[str, Any]:
    rows = sorted(_filter_rows(product_type, gender), key=_size_sort_key)
    if not rows:
        return {
            "recommended_size": None,
            "alternative_size": None,
            "boundary_case": False,
            "confidence": "low",
            "reason": "Chưa có bảng size phù hợp cho nhóm sản phẩm này.",
            "fit_notes": {},
            "data_source": str(SIZE_CHART_PATH),
            "is_between_sizes": False,
        }

    matching_rows = [
        row for row in rows if _row_matches_profile(row, height_cm, weight_kg)
    ]
    fit_mode = _fit_mode(fit_preference)
    if matching_rows:
        matching_rows = sorted(matching_rows, key=_size_sort_key)

        recommended = _select_recommended(matching_rows, height_cm, weight_kg, fit_mode)
        alternative = _nearest_alternative(
            matching_rows, recommended, height_cm, weight_kg
        )
        if alternative is None:
            alternative = _boundary_alternative(rows, recommended, height_cm, weight_kg)
        boundary_case = len(matching_rows) > 1 or _near_boundary(
            recommended, height_cm, weight_kg
        )
        if boundary_case:
            recommended, alternative = _apply_fit_preference_to_boundary(
                recommended,
                alternative,
                fit_mode,
            )

        recommended_size = str(recommended.get("size") or "")
        alternative_size = str(alternative.get("size") or "") if alternative else None

        if fit_mode == "loose" and alternative:
            reason = (
                f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
                f"nằm gần vùng giữa size "
                f"{min(recommended_size, alternative_size, key=lambda size: SIZE_ORDER.get(size, 99))} "
                f"và "
                f"{max(recommended_size, alternative_size, key=lambda size: SIZE_ORDER.get(size, 99))}. "
                f"Vì bạn thích mặc rộng, size {recommended_size} "
                f"sẽ thoải mái hơn."
            )
        elif fit_mode == "slim" and alternative:
            reason = (
                f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
                f"nằm gần vùng giữa size "
                f"{min(recommended_size, alternative_size, key=lambda size: SIZE_ORDER.get(size, 99))} "
                f"và "
                f"{max(recommended_size, alternative_size, key=lambda size: SIZE_ORDER.get(size, 99))}. "
                f"Vì bạn thích mặc ôm, size {recommended_size} "
                f"sẽ gọn người hơn."
            )
        elif len(matching_rows) > 1 and alternative:
            reason = (
                f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
                f"nằm trong vùng giao nhau giữa size "
                f"{recommended_size} và {alternative_size}."
            )
        elif boundary_case and alternative:
            reason = (
                f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
                f"nằm gần biên size "
                f"{recommended_size}/{alternative_size}."
            )
        else:
            reason = (
                f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
                f"nằm rõ trong khoảng size {recommended_size}."
            )

        return {
            "recommended_size": recommended_size,
            "alternative_size": alternative_size,
            "boundary_case": boundary_case,
            "confidence": "medium" if boundary_case else "high",
            "reason": reason,
            "fit_preference": fit_mode,
            "fit_notes": _fit_notes(recommended, alternative),
            "data_source": str(SIZE_CHART_PATH),
            "is_between_sizes": boundary_case,
        }

    height_rows = [row for row in rows if _height_matches(row, height_cm)]
    weight_rows = [row for row in rows if _weight_matches(row, weight_kg)]
    closest = min(
        rows,
        key=lambda row: (
            _outside_distance_from_row(row, height_cm, weight_kg),
            _distance_from_row(row, height_cm, weight_kg),
            _size_sort_key(row),
        ),
    )
    partial_candidates = [
        row
        for row in [*height_rows, *weight_rows]
        if row.get("size") != closest.get("size")
    ]
    alternative = _nearest_alternative(
        partial_candidates or rows, closest, height_cm, weight_kg
    )
    boundary_case = bool(height_rows or weight_rows or alternative)

    fit_mode = _fit_mode(fit_preference)

    if boundary_case:
        closest, alternative = _apply_fit_preference_to_boundary(
            closest,
            alternative,
            fit_mode,
        )
    if fit_mode == "loose" and alternative:
        reason = (
            f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
            f"chưa cùng rơi trọn vào một size. "
            f"Vì bạn thích mặc rộng, size {closest.get('size')} "
            f"là lựa chọn thoải mái hơn."
        )
    elif fit_mode == "slim" and alternative:
        reason = (
            f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
            f"chưa cùng rơi trọn vào một size. "
            f"Vì bạn thích mặc ôm, size {closest.get('size')} "
            f"là lựa chọn gọn hơn."
        )
    elif height_rows and weight_rows:
        reason = (
            f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
            f"không cùng rơi trọn một size; "
            f"size {closest.get('size')} là khoảng gần nhất theo bảng size."
        )
    else:
        reason = (
            f"Chiều cao {height_cm}cm và cân nặng {weight_kg}kg "
            f"chưa khớp hoàn toàn bảng size; "
            f"size {closest.get('size')} là khoảng gần nhất."
        )

    return {
        "recommended_size": closest.get("size"),
        "alternative_size": (
            alternative.get("size")
            if alternative
            else None
        ),
        "boundary_case": boundary_case,
        "confidence": "low",
        "reason": reason,
        "fit_preference": fit_mode,
        "fit_notes": _fit_notes(closest, alternative),
        "data_source": str(SIZE_CHART_PATH),
        "is_between_sizes": boundary_case,
    }
