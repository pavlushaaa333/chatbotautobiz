from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    """Raised when PostgreSQL settings are incomplete or unsupported."""


class DatabaseConnectionError(RuntimeError):
    """Raised when a PostgreSQL connection cannot be opened safely."""


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str = "prefer"
    connect_timeout_seconds: int = 5

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        missing = [
            name
            for name in ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise DatabaseConfigurationError(
                "Missing PostgreSQL environment variables: " + ", ".join(missing)
            )

        return cls(
            host=os.environ["DB_HOST"].strip(),
            port=_int_env("DB_PORT", 6543),
            dbname=os.environ["DB_NAME"].strip(),
            user=os.environ["DB_USER"].strip(),
            password=os.environ["DB_PASSWORD"],
            sslmode=os.getenv("DB_SSLMODE", "prefer").strip() or "prefer",
            connect_timeout_seconds=_int_env("DB_CONNECT_TIMEOUT_SECONDS", 5),
        )

    def safe_label(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} sslmode={self.sslmode}"
        )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _import_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:
        raise DatabaseConfigurationError(
            "PostgreSQL support requires psycopg 3. Install project dependencies first."
        ) from exc


@contextmanager
def get_connection(settings: PostgresSettings | None = None) -> Iterator[Any]:
    resolved = settings or PostgresSettings.from_env()
    psycopg, dict_row = _import_psycopg()
    connection = None
    try:
        connection = psycopg.connect(
            host=resolved.host,
            port=resolved.port,
            dbname=resolved.dbname,
            user=resolved.user,
            password=resolved.password,
            sslmode=resolved.sslmode,
            connect_timeout=resolved.connect_timeout_seconds,
            row_factory=dict_row,
        )
        yield connection
    except DatabaseConfigurationError:
        raise
    except Exception as exc:
        raise DatabaseConnectionError(
            f"Cannot connect to PostgreSQL ({resolved.safe_label()}): {exc.__class__.__name__}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def test_connection() -> None:
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS result")
                result = cursor.fetchone()

                print("Kết nối Supabase PostgreSQL thành công.")
                print(f"SELECT 1 trả về: {result}")

                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                    """)

                tables = cursor.fetchall()

                print("\nCác bảng trong schema public:")

                if not tables:
                    print("- Không tìm thấy bảng nào.")
                    return

                for table in tables:
                    if isinstance(table, dict):
                        print(f"- {table['table_name']}")
                    else:
                        print(f"- {table[0]}")

    except Exception as error:
        print(
            "Kết nối Supabase PostgreSQL thất bại: " f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    test_connection()
