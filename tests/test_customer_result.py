import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from app.customer_result import CONTRACT, deliver_result, remember_receipt

SECRET = "synthetic-result-test-account-secret-32-bytes"
NOW = 1791388800


class CustomerResultTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).parents[1] / ".test-data"
        test_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=test_root)
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict(
            os.environ,
            {
                "AUTOBIZ_CUSTOMER_RECEIPT_DB": self.directory.name
                + "/receipts.sqlite3",
                "AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE": "account-a",
                "N8N_ORDER_WEBHOOK_SECRET": SECRET,
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.result = {
            "contract": CONTRACT,
            "external_account_reference": "account-a",
            "result_id": str(uuid4()),
            "draft_order_id": str(uuid4()),
            "customer_conversation_id": "telegram_10001",
            "result_kind": "OWNER_APPROVED_ORDER",
            "order_id": str(uuid4()),
        }
        remember_receipt(
            {"draft_order_id": self.result["draft_order_id"]},
            {
                "external_account_reference": "account-a",
                "customer_conversation_id": "telegram_10001",
                "external_order_reference": "external-test-order",
            },
        )

    def envelope(self, result=None, created=NOW):
        result = result or self.result
        raw = json.dumps(
            result, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        material = CONTRACT.encode() + b"\n" + str(created).encode() + b"\n" + raw
        return {
            "result": result,
            "customer_auth": {
                "created": created,
                "signature": hmac.new(
                    SECRET.encode(), material, hashlib.sha256
                ).hexdigest(),
            },
        }

    def test_authenticated_duplicate_and_restart_send_once(self):
        send = Mock()
        first = deliver_result(self.envelope(), send, now=NOW)
        duplicate = deliver_result(self.envelope(created=NOW + 1), send, now=NOW + 1)
        self.assertEqual(first["status"], "DELIVERED")
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.args[0], "10001")

    def test_tamper_expiry_wrong_account_and_unknown_draft_denied(self):
        send = Mock()
        envelope = self.envelope()
        envelope["result"]["customer_conversation_id"] = "telegram_99999"
        with self.assertRaises(ValueError):
            deliver_result(envelope, send, now=NOW)
        with self.assertRaises(ValueError):
            deliver_result(self.envelope(created=NOW - 301), send, now=NOW)
        for key, value in (
            ("external_account_reference", "account-b"),
            ("draft_order_id", str(uuid4())),
            ("customer_conversation_id", "telegram_99999"),
        ):
            result = {**self.result, key: value}
            with self.assertRaises(ValueError):
                deliver_result(self.envelope(result), send, now=NOW)
        send.assert_not_called()

    def test_ambiguous_provider_outcome_does_not_auto_resend(self):
        send = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            deliver_result(self.envelope(), send, now=NOW)
        with self.assertRaises(RuntimeError):
            deliver_result(self.envelope(created=NOW + 1), send, now=NOW + 1)
        self.assertEqual(send.call_count, 1)

    def test_conflicting_replay_denied(self):
        send = Mock()
        deliver_result(self.envelope(), send, now=NOW)
        result = {
            **self.result,
            "result_kind": "OWNER_REJECTED_ORDER",
            "order_id": None,
        }
        with self.assertRaises(ValueError):
            deliver_result(self.envelope(result), send, now=NOW)
        self.assertEqual(send.call_count, 1)

    def test_http_callback_acknowledges_only_verified_delivery(self):
        from app import main
        from fastapi import HTTPException

        class Request:
            async def body(inner):
                return json.dumps(inner.value).encode()

            async def json(inner):
                return inner.value

        request = Request()
        request.value = self.envelope()
        with (
            patch("app.customer_result.time.time", return_value=NOW),
            patch.object(main, "_send_telegram_message") as send,
        ):
            self.assertEqual(
                asyncio.run(main.customer_order_result(request))["status"], "DELIVERED"
            )
            self.assertTrue(
                asyncio.run(main.customer_order_result(request))["duplicate"]
            )
            self.assertEqual(send.call_count, 1)
            request.value = {}
            with self.assertRaises(HTTPException) as denied:
                asyncio.run(main.customer_order_result(request))
            self.assertEqual(denied.exception.status_code, 401)
            request.value = self.envelope()
            with patch.dict(os.environ, {"N8N_ORDER_WEBHOOK_SECRET": ""}):
                with self.assertRaises(HTTPException) as unavailable:
                    asyncio.run(main.customer_order_result(request))
                self.assertEqual(unavailable.exception.status_code, 503)
        with patch.dict(os.environ, {"AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED": "false"}):
            with self.assertRaises(HTTPException) as legacy:
                main.order_status_webhook(
                    main.OrderStatusPayload(
                        event="order_approved",
                        draft_order_id="test",
                        customer_chat_id="12345",
                    )
                )
            self.assertEqual(legacy.exception.status_code, 410)

    def test_telegram_negative_receipt_is_not_success(self):
        from app import main
        from fastapi import HTTPException

        response = Mock()
        response.json.return_value = {"ok": False}
        with (
            patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "synthetic-test-token"}),
            patch.object(main.requests, "post", return_value=response),
        ):
            with self.assertRaises(HTTPException) as denied:
                main._send_telegram_message("12345", "Test")
            self.assertEqual(denied.exception.status_code, 502)
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "synthetic-test-token"}), patch.object(
            main.requests, "post", side_effect=main.requests.Timeout("synthetic-private-url")
        ):
            with self.assertRaises(HTTPException) as failed:
                main._send_telegram_message("12345", "Test")
            self.assertEqual(failed.exception.status_code, 502)
            self.assertTrue(failed.exception.__suppress_context__)
            self.assertNotIn("synthetic-private-url", failed.exception.detail)


if __name__ == "__main__":
    unittest.main()
