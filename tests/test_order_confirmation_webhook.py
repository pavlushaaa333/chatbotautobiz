from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import requests

import app.order_draft_service as order_draft_service
import app.service as service_module
from app.order_draft_service import (
    ORDER_STATE_AWAITING_CONFIRMATION,
    ORDER_STATE_PENDING_SHOP_APPROVAL,
    SubmissionResult,
    blank_order_draft,
    build_n8n_payload,
    is_order_confirmation_message,
    recalculate_totals,
    submit_draft_order_to_n8n,
)
from app.service import ChatbotService, CONVERSATION_STORE


class FakeRagService:
    def status(self) -> dict:
        return {"ready": False}


class FakeSearchEngine:
    def __init__(self) -> None:
        self.calls = 0

    def check_variant_availability(self, criteria: dict) -> dict:
        self.calls += 1
        return {
            "status": "in_stock",
            "product": {
                "product_id": "product-1",
                "product_name": "Ao thun basic cotton",
                "effective_price_vnd": 149000,
            },
            "variant": {
                "sku": "M05-TEE-BASIC-WHT-S",
                "color": "Trang",
                "size": "S",
                "stock": 81,
                "status": "active",
            },
        }


class FakeResponse:
    def __init__(self, status_code: int, *, should_raise: bool = False) -> None:
        self.status_code = status_code
        self.should_raise = should_raise

    def raise_for_status(self) -> None:
        if self.should_raise:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def valid_draft(draft_id: str = "DRAFT-TEST-0001") -> dict:
    draft = blank_order_draft(draft_id)
    draft.update(
        {
            "shop_id": "shop-1",
            "customer_chat_id": "chat-1",
            "status": ORDER_STATE_AWAITING_CONFIRMATION,
            "payment_method": "cod",
        }
    )
    draft["customer"].update(
        {
            "name": "Nguyen Van A",
            "phone": "0901234567",
            "address": "1 Nguyen Trai, Quan 1",
            "note": "Giao gio hanh chinh",
        }
    )
    draft["items"][0].update(
        {
            "product_id": "product-1",
            "product_name": "Ao thun basic cotton",
            "sku": "M05-TEE-BASIC-WHT-S",
            "color": "Trang",
            "size": "S",
            "quantity": 1,
            "unit_price": 149000,
            "variant_stock": 81,
        }
    )
    recalculate_totals(draft)
    return draft


def fake_service() -> ChatbotService:
    service = ChatbotService.__new__(ChatbotService)
    service.product_data_source = "csv"
    service.search_engine = FakeSearchEngine()
    service.rag_service = FakeRagService()
    return service


class OrderConfirmationWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        CONVERSATION_STORE.clear()

    def test_confirmation_submits_once_and_duplicate_does_not_post_again(self) -> None:
        service = fake_service()
        draft = valid_draft()
        context = {"order_draft": draft, "pending_action": "confirm_order_draft"}
        submit_result = SubmissionResult(
            ok=True,
            status_code=202,
            payload={"draft_order_id": draft["draft_order_id"]},
        )

        with patch.object(service_module, "submit_draft_order_to_n8n", return_value=submit_result) as submit:
            result = service._submit_order_draft("chat-1", context, "xac nhan")
            duplicate = service._submit_order_draft("chat-1", context, "xac nhan")

        self.assertEqual(submit.call_count, 1)
        self.assertEqual(draft["status"], ORDER_STATE_PENDING_SHOP_APPROVAL)
        self.assertTrue(draft["submitted_to_n8n"])
        self.assertFalse(draft["submission_in_progress"])
        self.assertIn("gửi đơn nháp sang shop", result["reply"])
        self.assertIn("đang chờ duyệt", duplicate["reply"])

    def test_confirmation_without_draft_returns_wrong_state(self) -> None:
        service = fake_service()
        service.parse = lambda message: {"raw_message": message}

        with patch.object(service_module, "submit_draft_order_to_n8n") as submit:
            result = service._handle_order_message("xac nhan", "chat-empty")

        submit.assert_not_called()
        self.assertIsNotNone(result)
        self.assertIsNone(result["pending_action"])
        self.assertIn("chưa có đơn nháp nào đang chờ xác nhận", result["reply"])

    def test_negative_confirmation_text_does_not_submit(self) -> None:
        service = fake_service()
        draft = valid_draft()
        context = {"order_draft": draft, "pending_action": "confirm_order_draft"}

        self.assertFalse(is_order_confirmation_message("khong xac nhan"))
        self.assertFalse(is_order_confirmation_message("xac nhan giup minh con hang khong"))
        with patch.object(service_module, "submit_draft_order_to_n8n") as submit:
            result = service._submit_order_draft("chat-1", context, "khong xac nhan")

        submit.assert_not_called()
        self.assertEqual(draft["status"], ORDER_STATE_AWAITING_CONFIRMATION)
        self.assertIn("chưa có đơn nháp nào đang chờ xác nhận", result["reply"])

    def test_missing_required_data_does_not_submit(self) -> None:
        service = fake_service()
        draft = valid_draft()
        draft["customer"]["phone"] = None
        context = {"order_draft": draft, "pending_action": "confirm_order_draft"}

        with patch.object(service_module, "submit_draft_order_to_n8n") as submit:
            result = service._submit_order_draft("chat-1", context, "xac nhan")

        submit.assert_not_called()
        self.assertEqual(service.search_engine.calls, 0)
        self.assertIn("còn thiếu: phone", result["reply"])

    def test_retry_after_webhook_failure_keeps_draft_id_and_posts_again(self) -> None:
        service = fake_service()
        draft = valid_draft("DRAFT-RETRY-0001")
        context = {"order_draft": draft, "pending_action": "confirm_order_draft"}
        first_result = SubmissionResult(
            ok=False,
            error_type="http_error",
            payload={"draft_order_id": draft["draft_order_id"]},
        )
        second_result = SubmissionResult(
            ok=True,
            status_code=200,
            payload={"draft_order_id": draft["draft_order_id"]},
        )

        with patch.object(
            service_module,
            "submit_draft_order_to_n8n",
            side_effect=[first_result, second_result],
        ) as submit:
            failed = service._submit_order_draft("chat-1", context, "xac nhan")
            retried = service._submit_order_draft("chat-1", context, "gui lai")

        self.assertEqual(submit.call_count, 2)
        self.assertEqual(draft["draft_order_id"], "DRAFT-RETRY-0001")
        self.assertIn("Gửi lại", failed["reply"])
        self.assertIn("gửi đơn nháp sang shop", retried["reply"])
        self.assertEqual(draft["status"], ORDER_STATE_PENDING_SHOP_APPROVAL)

    def test_payload_contains_required_fields(self) -> None:
        draft = valid_draft()
        payload = build_n8n_payload(draft, conversation_id="chat-1")

        self.assertEqual(payload["event"], "draft_order_confirmed")
        self.assertEqual(payload["draft_order_id"], draft["draft_order_id"])
        self.assertEqual(payload["shop_id"], "shop-1")
        self.assertEqual(payload["customer_chat_id"], "chat-1")
        self.assertEqual(payload["customer_name"], "Nguyen Van A")
        self.assertEqual(payload["phone"], "0901234567")
        self.assertEqual(payload["address"], "1 Nguyen Trai, Quan 1")
        self.assertEqual(payload["items"][0]["line_total"], 149000)
        self.assertEqual(payload["subtotal"], 149000)
        self.assertIsNone(payload["shipping_fee"])
        self.assertEqual(payload["total"], 149000)
        self.assertEqual(payload["payment_method"], "cod")
        self.assertEqual(payload["note"], "Giao gio hanh chinh")
        self.assertEqual(payload["source"], "telegram_customer_bot")

    def test_webhook_submit_accepts_200_201_202(self) -> None:
        draft = valid_draft()

        for status_code in [200, 201, 202]:
            with self.subTest(status_code=status_code):
                with patch.dict(
                    os.environ,
                    {
                        "N8N_DRAFT_ORDER_WEBHOOK_URL": "https://n8n.example.test/webhook",
                        "N8N_ORDER_WEBHOOK_URL": "",
                    },
                ):
                    with patch.object(order_draft_service.requests, "post", return_value=FakeResponse(status_code)) as post:
                        result = submit_draft_order_to_n8n(draft)

                self.assertTrue(result.ok)
                self.assertEqual(result.status_code, status_code)
                post.assert_called_once()
                self.assertEqual(post.call_args.kwargs["json"]["draft_order_id"], draft["draft_order_id"])

    def test_webhook_submit_prefers_existing_url_env(self) -> None:
        draft = valid_draft()

        with patch.dict(
            os.environ,
            {
                "N8N_ORDER_WEBHOOK_URL": "https://n8n.example.test/existing",
                "N8N_DRAFT_ORDER_WEBHOOK_URL": "https://n8n.example.test/draft",
            },
        ):
            with patch.object(order_draft_service.requests, "post", return_value=FakeResponse(200)) as post:
                result = submit_draft_order_to_n8n(draft)

        self.assertTrue(result.ok)
        self.assertEqual(post.call_args.args[0], "https://n8n.example.test/existing")

    def test_webhook_submit_reports_missing_url_http_error_and_timeout(self) -> None:
        draft = valid_draft()

        with patch.dict(
            os.environ,
            {
                "N8N_DRAFT_ORDER_WEBHOOK_URL": "",
                "N8N_ORDER_WEBHOOK_URL": "",
            },
        ):
            missing_url = submit_draft_order_to_n8n(draft)
        self.assertFalse(missing_url.ok)
        self.assertEqual(missing_url.error_type, "missing_webhook_url")

        with patch.dict(
            os.environ,
            {
                "N8N_DRAFT_ORDER_WEBHOOK_URL": "https://n8n.example.test/webhook",
                "N8N_ORDER_WEBHOOK_URL": "",
            },
        ):
            with patch.object(
                order_draft_service.requests,
                "post",
                return_value=FakeResponse(500, should_raise=True),
            ):
                http_error = submit_draft_order_to_n8n(draft)
        self.assertFalse(http_error.ok)
        self.assertEqual(http_error.status_code, 500)
        self.assertEqual(http_error.error_type, "http_error")

        with patch.dict(
            os.environ,
            {
                "N8N_DRAFT_ORDER_WEBHOOK_URL": "https://n8n.example.test/webhook",
                "N8N_ORDER_WEBHOOK_URL": "",
            },
        ):
            with patch.object(
                order_draft_service.requests,
                "post",
                Mock(side_effect=requests.Timeout("timeout")),
            ):
                timeout = submit_draft_order_to_n8n(draft)
        self.assertFalse(timeout.ok)
        self.assertEqual(timeout.error_type, "timeout")


if __name__ == "__main__":
    unittest.main()
