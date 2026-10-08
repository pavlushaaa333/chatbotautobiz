"""Offline config checks and explicit Customer Bot process entrypoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def configuration_errors(values, *, telegram=False):
    errors = []
    if values.get("AUTOBIZ_CUSTOMER_INGRESS_ENABLED", "").strip().lower() != "true":
        errors.append("AUTOBIZ_CUSTOMER_INGRESS_ENABLED must be true after Core provisioning")
    account = values.get("AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE", "").strip()
    if not account or account.startswith("<"):
        errors.append("AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE must match the Core installation")
    secret = values.get("N8N_ORDER_WEBHOOK_SECRET", "").strip()
    if len(secret.encode()) < 32 or secret.startswith("<"):
        errors.append("N8N_ORDER_WEBHOOK_SECRET requires the active Core account credential (32+ bytes)")
    url = urlsplit(values.get("N8N_ORDER_WEBHOOK_URL", "").strip())
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path != "/webhook/v1/customer/order-intents"):
        errors.append("N8N_ORDER_WEBHOOK_URL requires HTTPS and /webhook/v1/customer/order-intents")
    if not values.get("TELEGRAM_BOT_TOKEN", "").strip() or values.get("TELEGRAM_BOT_TOKEN", "").startswith("<"):
        errors.append("TELEGRAM_BOT_TOKEN must be the Customer Bot token")
    ledger = values.get("AUTOBIZ_CUSTOMER_RECEIPT_DB", "").strip()
    if not ledger or ledger.startswith("<") or not Path(ledger).is_absolute():
        errors.append("AUTOBIZ_CUSTOMER_RECEIPT_DB must be an absolute persistent-volume path shared by both processes")
    if values.get("AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED", "false").lower() != "false":
        errors.append("AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED must remain false")
    if telegram:
        try:
            UUID(values.get("AUTOBIZ_ACTIVE_SHOP_ID", ""))
        except (ValueError, TypeError, AttributeError):
            errors.append("AUTOBIZ_ACTIVE_SHOP_ID requires the Core shop UUID for this instance")
        if values.get("AUTOBIZ_CATALOG_SOURCE", "database").lower() not in {"database", "csv"}:
            errors.append("AUTOBIZ_CATALOG_SOURCE must be database or csv")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "api", "telegram"])
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("env file does not exist")
    # Explicit per-instance file avoids accidentally mixing another shop's env.
    load_dotenv(args.env_file.resolve(), override=True)
    os.chdir(ROOT)
    errors = configuration_errors(os.environ, telegram=args.command in {"check", "telegram"})
    if errors:
        for error in errors:
            print("CONFIG_REQUIRED: " + error)
        return 2
    print("Local configuration checks passed; Core provisioning, reachability and credentials are NOT verified.")
    if args.command == "api":
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, workers=1)
    elif args.command == "telegram":
        from app.telegram_bot import main as run_telegram
        run_telegram()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
