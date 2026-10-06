from __future__ import annotations
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
from uuid import UUID

from app.customer_intent import build_customer_intent, validate_receipt
from app.order_draft_service import (
    new_draft_id,
    submit_draft_order_to_n8n,
    set_item_variant,
    clear_item_variant,
    set_item_product,
)
from app.search_engine import PostgresProductSearchEngine
from tests.test_order_confirmation_webhook import valid_draft

ENV = {
    "AUTOBIZ_CUSTOMER_INGRESS_ENABLED": "true",
    "AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE": "customer-account",
    "N8N_ORDER_WEBHOOK_URL": "https://n8n.example.invalid/webhook/customer-order-intents",
    "N8N_ORDER_WEBHOOK_SECRET": "synthetic-customer-test-credential-at-least-32-bytes",
}


def receipt(ref="DRAFT-TEST-0001"):
    return {
        "status": "DRAFT_CREATED",
        "draft_status": "draft",
        "draft_order_id": "11111111-1111-4111-8111-111111111111",
        "run_id": "22222222-2222-4222-8222-222222222222",
        "event_id": "33333333-3333-4333-8333-333333333333",
        "external_order_reference": ref,
    }


class CustomerIntentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_only_bounded_intent_leaves_bot(self):
        draft = valid_draft()
        draft["items"][0]["variant"] = "exact-catalog-variant"
        payload = build_customer_intent(draft, "telegram_1001")
        self.assertEqual(
            payload["items"][0],
            {
                "sku": "M05-TEE-BASIC-WHT-S",
                "variant": "exact-catalog-variant",
                "quantity": 1,
            },
        )
        for key in (
            "shop_id",
            "owner_id",
            "total",
            "stock",
            "created_at",
            "raw_message",
        ):
            self.assertNotIn(key, payload)
        self.assertEqual(payload["customer"]["name"], "Nguyen Van A")

    def test_exact_catalog_variant_survives_search_selection_and_intent(self):
        engine = PostgresProductSearchEngine.__new__(PostgresProductSearchEngine)
        selected = engine._variant_payload(
            {
                "sku": "M05-TEE-BASIC-WHT-S",
                "variant": "Trắng / S",
                "size": "S",
                "color": "Trắng",
                "stock": 5,
            }
        )
        draft = valid_draft()
        set_item_variant(draft["items"][0], selected)
        intent = build_customer_intent(draft, "telegram_1001")
        self.assertEqual(intent["items"][0]["variant"], "Trắng / S")

    def test_switching_product_or_variant_clears_catalog_variant(self):
        draft = valid_draft()
        item = draft["items"][0]
        item["variant"] = "stale-variant"
        clear_item_variant(item)
        self.assertIsNone(item["variant"])
        item["variant"] = "stale-variant"
        set_item_product(item, {"product_id": "other-product"})
        self.assertIsNone(item["variant"])

    def test_retry_and_reloaded_draft_keep_identity_and_payload(self):
        draft = valid_draft()
        first = build_customer_intent(draft, "telegram_1001")
        second = build_customer_intent(deepcopy(draft), "telegram_1001")
        self.assertEqual(first, second)
        self.assertEqual(first["provider_event_id"], first["external_order_reference"])

    def test_new_draft_identity_does_not_reset_each_process(self):
        self.assertNotEqual(new_draft_id(), new_draft_id())
        UUID(new_draft_id().rsplit("-", 1)[1])

    def test_missing_account_or_secret_fails_without_network(self):
        with (
            patch.dict(os.environ, {"AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE": ""}),
            patch("app.order_draft_service.requests.post") as post,
        ):
            self.assertFalse(submit_draft_order_to_n8n(valid_draft()).ok)
            post.assert_not_called()
        with (
            patch.dict(
                os.environ,
                {
                    "N8N_ORDER_WEBHOOK_SECRET": "",
                    "N8N_DRAFT_ORDER_WEBHOOK_SECRET": "",
                    "AUTOBIZ_WEBHOOK_SECRET": "",
                },
            ),
            patch("app.order_draft_service.requests.post") as post,
        ):
            self.assertFalse(submit_draft_order_to_n8n(valid_draft()).ok)
            post.assert_not_called()

    def test_transport_202_alone_is_not_draft_success(self):
        response = Mock(status_code=202)
        response.json.return_value = {"status": "ACCEPTED", "event_id": "event-only"}
        with patch("app.order_draft_service.requests.post", return_value=response):
            result = submit_draft_order_to_n8n(valid_draft())
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "invalid_core_receipt")

    def test_valid_core_receipt_is_success(self):
        response = Mock(status_code=202)
        response.json.return_value = receipt()
        with patch(
            "app.order_draft_service.requests.post", return_value=response
        ) as post:
            result = submit_draft_order_to_n8n(valid_draft())
        self.assertTrue(result.ok)
        self.assertNotIn("shop_id", post.call_args.kwargs["json"])

    def test_wrong_reference_or_malformed_receipt_is_rejected(self):
        payload = build_customer_intent(valid_draft(), "telegram_1001")
        for value in (receipt("another-order"), {}, {"status": "DRAFT_CREATED"}):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                validate_receipt(value, payload)

    def test_invalid_quantity_and_control_characters_are_rejected(self):
        for change in (
            lambda d: d["items"][0].update(quantity=True),
            lambda d: d["customer"].update(address="address\nforged"),
        ):
            draft = valid_draft()
            change(draft)
            with self.assertRaises(ValueError):
                build_customer_intent(draft, "telegram_1001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
