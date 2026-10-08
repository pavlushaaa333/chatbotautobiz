# AutoBiz component identities — confirmed by owner, 2026-10-08

- Customer Bot: `D:/AutoBiz_AI_Chatbot`. Customer consultation and draft intake; Telegram entrypoint `app/telegram_bot.py`, backend/callback HTTP app `app/main.py`.
- Owner Bot / main AutoBiz system: `D:/autobiz-main`. Contains Python Core, database and n8n owner integration. Owner ingress: `backend/src/autobiz/owner_surface_ingress.py`, route `/v1/integration/owner-messages`.
- `D:/AutoBiz_Core_Integration` is a partial candidate source snapshot from `minh2509/autobiz`, SHA `20d41f7368c759ba846ca91930f357bed24bae2a`; it is not a separate owner/customer bot or the authoritative main folder. Earlier Core changes and local tests were performed there.

The earlier candidate Core changes have NOT been transferred to `D:/autobiz-main`. Its customer_order_ingress.py and customer_order_runtime.py were absent at this verification. Before connecting or rolling out, reconcile the tested candidate patch with the actual main repository/deployed revision and validate that target.

Target flow: AutoBiz_AI_Chatbot customer consultation -> n8n customer adapter -> AutoBiz main Core -> owner approval through Owner Bot -> same Core -> n8n customer result adapter -> AutoBiz_AI_Chatbot customer callback.

CUSTOMER_BOT installation/account refers to the customer application registered in the main Core. Owner Telegram bindings are separate authority. Customer callback URL belongs to the HTTP backend of AutoBiz_AI_Chatbot, not to the Owner/Core service or a Telegram t.me link.

Local folder identity does not prove which process/domain is deployed. Telegram polling alone does not expose app/main.py over HTTPS. Keep service/domain mapping unverified until deployment metadata is read.

No deployment, restart, workflow activation, credential modification or source transfer to autobiz-main is authorized by this identity correction.
