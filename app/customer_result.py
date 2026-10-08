"""Authenticated results, bound to accepted Core receipts, with durable replay fences."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import UUID

CONTRACT = "autobiz.customer-order-result.v1"


class AuthenticationUnavailable(RuntimeError):
    pass


def _db():
    path = Path(
        os.getenv("AUTOBIZ_CUSTOMER_RECEIPT_DB", "data/customer-receipts.sqlite3")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS receipts (account TEXT NOT NULL, draft TEXT NOT NULL, "
        "conversation TEXT NOT NULL, external_reference TEXT NOT NULL, "
        "PRIMARY KEY(account,draft))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results (account TEXT NOT NULL, result_id TEXT NOT NULL, "
        "digest TEXT NOT NULL, state TEXT NOT NULL, PRIMARY KEY(account,result_id))"
    )
    return connection


def remember_receipt(receipt, intent):
    # Caller must first validate DRAFT_CREATED and every receipt UUID.
    expected = (
        intent["external_account_reference"],
        receipt["draft_order_id"],
        intent["customer_conversation_id"],
        intent["external_order_reference"],
    )
    db = _db()
    try:
        with db:
            db.execute("INSERT OR IGNORE INTO receipts VALUES (?,?,?,?)", expected)
            stored = db.execute(
                "SELECT account,draft,conversation,external_reference FROM receipts "
                "WHERE account=? AND draft=?",
                expected[:2],
            ).fetchone()
            if stored != expected:
                raise ValueError("Core receipt conflicts with stored customer request")
    finally:
        db.close()


def verify_result(envelope, *, now=None):
    if not isinstance(envelope, dict) or set(envelope) != {"result", "customer_auth"}:
        raise ValueError("bounded result envelope required")
    result, proof = envelope["result"], envelope["customer_auth"]
    required = {
        "contract",
        "external_account_reference",
        "result_id",
        "draft_order_id",
        "customer_conversation_id",
        "result_kind",
        "order_id",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result["contract"] != CONTRACT
    ):
        raise ValueError("bounded customer result required")
    account = os.getenv("AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE", "").strip()
    secret = os.getenv("N8N_ORDER_WEBHOOK_SECRET", "").strip().encode()
    if not account or len(secret) < 32:
        raise AuthenticationUnavailable("customer result authentication is unavailable")
    if result["external_account_reference"] != account:
        raise ValueError("customer result account denied")
    for key in ("result_id", "draft_order_id"):
        if not isinstance(result[key], str):
            raise ValueError("result identifiers must be UUID strings")
        UUID(result[key])
    if result["order_id"] is not None:
        UUID(result["order_id"])
    if result["result_kind"] not in {
        "OWNER_APPROVED_ORDER",
        "OWNER_REJECTED_ORDER",
        "OWNER_ORDER_REVALIDATION_REQUIRED",
    }:
        raise ValueError("unsupported customer outcome")
    if (result["result_kind"] == "OWNER_APPROVED_ORDER") != (
        result["order_id"] is not None
    ):
        raise ValueError("customer outcome does not match official order identity")
    if not isinstance(proof, dict) or set(proof) != {"created", "signature"}:
        raise ValueError("customer proof required")
    created = proof["created"]
    if (
        type(created) is not int
        or abs((time.time() if now is None else now) - created) > 300
    ):
        raise ValueError("expired customer proof")
    raw = json.dumps(
        result,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if len(raw) > 4096 or not isinstance(proof["signature"], str):
        raise ValueError("invalid customer proof")
    material = CONTRACT.encode() + b"\n" + str(created).encode() + b"\n" + raw
    expected = hmac.new(secret, material, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, proof["signature"]):
        raise ValueError("invalid customer proof")
    return result, hashlib.sha256(raw).hexdigest()


def deliver_result(envelope, send, *, now=None):
    result, digest = verify_result(envelope, now=now)
    account, result_id = result["external_account_reference"], result["result_id"]
    db = _db()
    try:
        db.execute("BEGIN IMMEDIATE")
        receipt = db.execute(
            "SELECT conversation FROM receipts WHERE account=? AND draft=?",
            (account, result["draft_order_id"]),
        ).fetchone()
        if receipt is None or receipt[0] != result["customer_conversation_id"]:
            raise ValueError("result does not match an accepted Core draft receipt")
        previous = db.execute(
            "SELECT digest,state FROM results WHERE account=? AND result_id=?",
            (account, result_id),
        ).fetchone()
        if previous is not None:
            if previous[0] != digest:
                raise ValueError("conflicting customer result replay")
            if previous[1] != "DELIVERED":
                raise RuntimeError(
                    "customer delivery outcome is unknown; reconciliation required"
                )
            return {"status": "DELIVERED", "result_id": result_id, "duplicate": True}
        conversation = receipt[0]
        chat_id = conversation.removeprefix("telegram_")
        if not chat_id.isdigit() or int(chat_id) <= 0:
            raise ValueError("private customer Telegram destination required")
        db.execute(
            "INSERT INTO results VALUES (?,?,?,'SENDING')", (account, result_id, digest)
        )
        db.commit()  # Durable before provider call; unknown outcomes must never auto-resend.
        label = {
            "OWNER_APPROVED_ORDER": "Đơn hàng đã được shop xác nhận.",
            "OWNER_REJECTED_ORDER": "Đơn hàng đã bị shop từ chối.",
            "OWNER_ORDER_REVALIDATION_REQUIRED": "Đơn hàng cần shop kiểm tra lại giá hoặc tồn kho.",
        }[result["result_kind"]]
        send(
            chat_id,
            f"{label}\nMã đơn: {result['order_id'] or result['draft_order_id']}",
        )
        db.execute(
            "UPDATE results SET state='DELIVERED' WHERE account=? AND result_id=?",
            (account, result_id),
        )
        db.commit()
        return {"status": "DELIVERED", "result_id": result_id, "duplicate": False}
    finally:
        db.close()
