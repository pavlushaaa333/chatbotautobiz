import unittest
from pathlib import Path

from app.integration_cli import configuration_errors


class IntegrationConfigurationTests(unittest.TestCase):
    def configuration(self):
        return {"AUTOBIZ_CUSTOMER_INGRESS_ENABLED":"true",
                "AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE":"synthetic-account",
                "N8N_ORDER_WEBHOOK_SECRET":"synthetic-account-credential-32-bytes",
                "N8N_ORDER_WEBHOOK_URL":"https://example.invalid/webhook/v1/customer/order-intents",
                "TELEGRAM_BOT_TOKEN":"synthetic-customer-token",
                "AUTOBIZ_CUSTOMER_RECEIPT_DB":str(Path.cwd()/"work/shop-a/receipts.sqlite3"),
                "AUTOBIZ_ACTIVE_SHOP_ID":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}

    def test_valid_bounded_configuration(self):
        self.assertEqual(configuration_errors(self.configuration(), telegram=True), [])

    def test_missing_identity_and_auth_never_echo_values(self):
        values = self.configuration()
        values.update(AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE="",N8N_ORDER_WEBHOOK_SECRET="private-short-value")
        errors = configuration_errors(values)
        self.assertEqual(len(errors), 2)
        self.assertNotIn("private-short-value", str(errors))

    def test_wrong_endpoint_legacy_and_relative_ledger_denied(self):
        values = self.configuration()
        values.update(N8N_ORDER_WEBHOOK_URL="https://example.invalid/owner-messages",
                      AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED="true",AUTOBIZ_CUSTOMER_RECEIPT_DB="data/ledger.sqlite3")
        self.assertEqual(len(configuration_errors(values)), 3)

    def test_missing_shop_uuid_denied_for_telegram(self):
        values = self.configuration()
        values["AUTOBIZ_ACTIVE_SHOP_ID"] = ""
        self.assertEqual(len(configuration_errors(values, telegram=True)), 1)


if __name__ == "__main__":
    unittest.main()
