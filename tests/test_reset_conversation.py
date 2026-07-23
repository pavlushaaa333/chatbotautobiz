from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ["PRODUCT_DATA_SOURCE"] = "csv"

import app.service as service_module  # noqa: E402
from app.order_draft_service import (  # noqa: E402
    ORDER_STATE_AWAITING_CONFIRMATION,
    ORDER_STATE_PENDING_SHOP_APPROVAL,
    ORDER_STATE_SUBMISSION_FAILED,
    ORDER_STATE_SUBMITTING,
    blank_order_draft,
    recalculate_totals,
)
from app.service import CONVERSATION_STORE, ChatbotService  # noqa: E402


class FakeRagService:
    def status(self) -> dict:
        return {"ready": False}


class FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeUpdate:
    def __init__(self, chat_id: int) -> None:
        self.effective_chat = SimpleNamespace(id=chat_id)
        self.message = FakeMessage()


class FakeTelegramService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset_conversation(self, conversation_id: str) -> dict[str, bool]:
        self.calls.append(conversation_id)
        return {
            "conversation_cleared": True,
            "order_draft_cleared": True,
        }


def valid_draft(status: str = ORDER_STATE_AWAITING_CONFIRMATION) -> dict:
    draft = blank_order_draft("DRAFT-RESET-0001")
    draft.update(
        {
            "shop_id": "shop-1",
            "customer_chat_id": "chat-A",
            "status": status,
            "payment_method": "cod",
            "submitted_to_n8n": status == ORDER_STATE_PENDING_SHOP_APPROVAL,
            "submission_in_progress": status == ORDER_STATE_SUBMITTING,
            "last_submission_failed": status == ORDER_STATE_SUBMISSION_FAILED,
        }
    )
    draft["customer"].update(
        {
            "name": "Nguyen Van A",
            "phone": "0901234567",
            "address": "1 Nguyen Trai",
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


def reset_only_service() -> ChatbotService:
    service = ChatbotService.__new__(ChatbotService)
    service.product_data_source = "csv"
    service.rag_service = FakeRagService()
    service.parse = lambda message: {"raw_message": message}
    return service


def import_telegram_bot_or_skip(test_case: unittest.TestCase):
    try:
        import app.telegram_bot as telegram_bot
    except ModuleNotFoundError as exc:
        if exc.name == "telegram":
            test_case.skipTest("python-telegram-bot is not installed in this Python environment.")
        raise
    return telegram_bot


class ResetConversationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        CONVERSATION_STORE.clear()

    def test_reset_when_draft_exists_clears_state(self) -> None:
        service = reset_only_service()
        CONVERSATION_STORE["chat-A"] = {
            "pending_action": "confirm_order_draft",
            "selected_product_id": "product-1",
            "last_product_ids": ["product-1"],
            "last_product_type": "Ao thun",
            "height_cm": 170,
            "weight_kg": 65,
            "fit_preference": "regular",
            "order_draft": valid_draft(),
        }

        result = service.reset_conversation("chat-A")

        self.assertEqual(
            result,
            {
                "conversation_cleared": True,
                "order_draft_cleared": True,
            },
        )
        self.assertIsNone(CONVERSATION_STORE.get("chat-A", {}).get("pending_action"))
        self.assertIsNone(CONVERSATION_STORE.get("chat-A", {}).get("order_draft"))

    def test_reset_without_draft_does_not_error(self) -> None:
        service = reset_only_service()
        CONVERSATION_STORE["chat-empty"] = {
            "pending_action": "choose_product_type",
            "selected_product_id": "product-2",
        }

        result = service.reset_conversation("chat-empty")

        self.assertEqual(
            result,
            {
                "conversation_cleared": True,
                "order_draft_cleared": False,
            },
        )
        self.assertNotIn("chat-empty", CONVERSATION_STORE)

    def test_reset_isolated_to_one_chat(self) -> None:
        service = reset_only_service()
        CONVERSATION_STORE["chat-A"] = {
            "pending_action": "confirm_order_draft",
            "order_draft": valid_draft(),
        }
        CONVERSATION_STORE["chat-B"] = {
            "pending_action": "confirm_order_draft",
            "order_draft": valid_draft(),
        }

        service.reset_conversation("chat-A")

        self.assertNotIn("chat-A", CONVERSATION_STORE)
        self.assertIn("chat-B", CONVERSATION_STORE)
        self.assertIsNotNone(CONVERSATION_STORE["chat-B"].get("order_draft"))

    async def test_reset_command_uses_telegram_conversation_id_and_replies_success(self) -> None:
        telegram_bot = import_telegram_bot_or_skip(self)

        fake_service = FakeTelegramService()
        update = FakeUpdate(12345)

        with patch.object(telegram_bot, "bot", fake_service):
            await telegram_bot.reset_command(update, None)

        self.assertEqual(fake_service.calls, ["telegram_12345"])
        self.assertEqual(
            update.message.replies,
            [
                "Đã xóa phiên tư vấn và đơn nháp hiện tại.\n\n"
                "Bạn muốn tìm sản phẩm gì ạ?"
            ],
        )

    async def test_start_resets_before_greeting(self) -> None:
        telegram_bot = import_telegram_bot_or_skip(self)

        fake_service = FakeTelegramService()
        update = FakeUpdate(67890)

        with patch.object(telegram_bot, "bot", fake_service):
            await telegram_bot.start_command(update, None)

        self.assertEqual(fake_service.calls, ["telegram_67890"])
        self.assertEqual(
            update.message.replies,
            [
                "Xin chào! Tôi là trợ lý tư vấn AutoBiz.\n"
                "Bạn đang muốn tìm sản phẩm gì?"
            ],
        )

    def test_retry_after_reset_from_submission_failure_has_no_effect(self) -> None:
        service = reset_only_service()
        CONVERSATION_STORE["chat-failed"] = {
            "pending_action": "confirm_order_draft",
            "order_draft": valid_draft(ORDER_STATE_SUBMISSION_FAILED),
        }

        service.reset_conversation("chat-failed")

        with patch.object(service_module, "submit_draft_order_to_n8n") as submit:
            result = service._handle_order_message("gui lai", "chat-failed")

        submit.assert_not_called()
        self.assertIsNotNone(result)
        self.assertIsNone(result["pending_action"])
        self.assertIsNone(result["order_draft"])
        self.assertNotIn("Xác nhận", result["reply"])
        self.assertNotIn("Gửi lại", result["reply"])


if __name__ == "__main__":
    unittest.main()
