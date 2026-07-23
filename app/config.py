from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_FOLDER_NAME = "autobiz_product_search_simulated_dataset"


AUTOBIZ_CATALOG_SOURCE = (
    os.getenv(
        "AUTOBIZ_CATALOG_SOURCE",
        "database",
    )
    .strip()
    .lower()
)

AUTOBIZ_ACTIVE_SHOP_ID = os.getenv(
    "AUTOBIZ_ACTIVE_SHOP_ID",
    "",
).strip()

AUTOBIZ_INCLUDE_SIMULATED_DATA = os.getenv(
    "AUTOBIZ_INCLUDE_SIMULATED_DATA", "false"
).strip().lower() in {"1", "true", "yes"}


def _candidate_data_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.getenv("AUTOBIZ_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT / DATASET_FOLDER_NAME,
            PROJECT_ROOT / "data" / DATASET_FOLDER_NAME,
            PROJECT_ROOT.parent / DATASET_FOLDER_NAME,
            PROJECT_ROOT.parent / "data" / DATASET_FOLDER_NAME,
            Path.cwd() / DATASET_FOLDER_NAME,
            Path.cwd() / "data" / DATASET_FOLDER_NAME,
        ]
    )
    return candidates


def find_data_dir() -> Path:
    for candidate in _candidate_data_dirs():
        if (candidate / "data_raw" / "csv").is_dir():
            return candidate.resolve()

    checked = "\n".join(f"- {path}" for path in _candidate_data_dirs())
    raise RuntimeError(
        "Không tìm thấy dataset AutoBiz. Hãy đặt biến môi trường "
        "AUTOBIZ_DATA_DIR trỏ tới thư mục autobiz_product_search_simulated_dataset "
        "hoặc đặt dataset trong thư mục data/ của project.\n"
        f"Các đường dẫn đã kiểm tra:\n{checked}"
    )


DATA_DIR = find_data_dir()
CSV_DIR = DATA_DIR / "data_raw" / "csv"

PRODUCT_DATA_SOURCE_ENV = "PRODUCT_DATA_SOURCE"
AUTOBIZ_CATALOG_SOURCE_ENV = "AUTOBIZ_CATALOG_SOURCE"
AUTOBIZ_ACTIVE_SHOP_ID_ENV = "AUTOBIZ_ACTIVE_SHOP_ID"
AUTOBIZ_INCLUDE_SIMULATED_DATA_ENV = "AUTOBIZ_INCLUDE_SIMULATED_DATA"

SUPPORTED_PRODUCT_DATA_SOURCES = {"postgres", "csv"}
SUPPORTED_CATALOG_SOURCES = {"database", "csv"}


def product_data_source() -> str:
    catalog_value = os.getenv(AUTOBIZ_CATALOG_SOURCE_ENV, "").strip().lower()
    if catalog_value == "database":
        return "postgres"
    if catalog_value == "csv":
        return "csv"

    value = os.getenv(PRODUCT_DATA_SOURCE_ENV, "postgres").strip().lower()
    if value not in SUPPORTED_PRODUCT_DATA_SOURCES:
        return "postgres"
    return value


PRODUCT_DATA_SOURCE = product_data_source()


def catalog_source() -> str:
    value = os.getenv(AUTOBIZ_CATALOG_SOURCE_ENV, "").strip().lower()
    if value in SUPPORTED_CATALOG_SOURCES:
        return value
    return "csv" if product_data_source() == "csv" else "database"


def active_shop_id() -> str | None:
    value = (
        os.getenv(AUTOBIZ_ACTIVE_SHOP_ID_ENV, "").strip()
        or os.getenv("SHOP_ID", "").strip()
    )
    return value or None


def include_simulated_data() -> bool:
    value = os.getenv(AUTOBIZ_INCLUDE_SIMULATED_DATA_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return catalog_source() == "csv"


N8N_ORDER_WEBHOOK_URL = os.getenv("N8N_ORDER_WEBHOOK_URL", "").strip()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


N8N_ORDER_WEBHOOK_TIMEOUT_SECONDS = _int_env(
    "N8N_ORDER_WEBHOOK_TIMEOUT_SECONDS",
    10,
)
