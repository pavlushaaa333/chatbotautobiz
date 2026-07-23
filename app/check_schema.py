from app.database import get_connection


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    table_name,
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name IN ('products', 'inventory')
                ORDER BY table_name, ordinal_position;
                """)

            print("=== COLUMNS ===")
            for row in cur.fetchall():
                print(row)

            cur.execute("SELECT * FROM public.products LIMIT 5;")
            print("\n=== PRODUCTS ===")
            for row in cur.fetchall():
                print(row)

            cur.execute("SELECT * FROM public.inventory LIMIT 5;")
            print("\n=== INVENTORY ===")
            for row in cur.fetchall():
                print(row)


if __name__ == "__main__":
    main()
