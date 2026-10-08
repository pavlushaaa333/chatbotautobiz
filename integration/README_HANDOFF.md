# Bàn giao Customer Bot cho hệ thống AutoBiz hiện có

Customer Bot là folder `AutoBiz_AI_Chatbot`; Owner Bot/Core là `autobiz-main`. Bot chưa deploy. Gói này chuẩn bị source/config, không tạo installation, cấp secret, publish workflow hoặc thay service đang chạy.

## Người nhận cần điền gì

Sao chép `integration/customer-bot.env.example` thành file env riêng ngoài Git, ví dụ `.env.shop-a`. Template giữ ingress tắt cho tới khi Core sẵn sàng.

| Biến | Người tích hợp lấy/cấp |
|---|---|
| `AUTOBIZ_CUSTOMER_ACCOUNT_REFERENCE` | `external_account_reference` của CUSTOMER_BOT installation trong Core, đúng shop và owner actor. Không dùng Telegram token làm account reference. |
| `AUTOBIZ_ACTIVE_SHOP_ID` | Shop UUID của catalog instance này, phải cùng shop với installation. |
| `N8N_ORDER_WEBHOOK_SECRET` | Active account credential thông qua luồng provisioning Core/secret manager được duyệt; tối thiểu 32 bytes. Dùng cùng credential cho ingress và callback. |
| `TELEGRAM_BOT_TOKEN` | Token của **Customer Bot**, tách với Owner Bot. Không ghi token trong chat/Git. |
| `AUTOBIZ_CUSTOMER_RECEIPT_DB` | Đường dẫn tuyệt đối tới SQLite ledger trên persistent volume, dùng chung cho hai process Customer Bot. Mỗi shop dùng ledger riêng. |
| `AUTOBIZ_DATA_DIR`, `DB_*` | Resource/catalog configuration hiện có cho đúng shop; không có catalog data hoặc DB credentials trong gói bàn giao. Consultation hiện vẫn yêu cầu resource directory có `data_raw/csv`, kể cả khi catalog dùng PostgreSQL. |

`installation_id` do Core quản lý; bot không cần thêm một biến installation ID để gửi đơn. Bot gửi account reference, Core tra installation để quyết định shop/actor. Không tự tạo UUID hoặc sửa account nhằm vượt qua Core checks.

## Cách chạy sau khi đã cấu hình

Từ root project, dùng Python environment đã cài `requirements.txt`:

```powershell
python -m app.integration_cli check --env-file .env.shop-a
python -m app.integration_cli api --env-file .env.shop-a --host 127.0.0.1 --port 8000
```

Ở process khác, cùng root và cùng env file:

```powershell
python -m app.integration_cli telegram --env-file .env.shop-a
```

`check` chỉ kiểm tra cấu hình local, không gọi n8n/Core/Telegram/DB, không in giá trị secret. Thiếu gate/account/credential/token/absolute ledger/shop UUID sẽ báo tên trường cần cấu hình. Không dùng kết quả check làm bằng chứng installation đã được cấp hoặc endpoint truy cập được.

API callback được tách khỏi consultation/RAG: `/health` trả status đơn giản và `/webhooks/customer-order-results` không cần khởi tạo catalog. Các route chat/search vẫn cần catalog/model resources. Telegram process vẫn chạy polling; chỉ **một** polling process cho mỗi token. API chạy một worker trong bản bàn giao.

Muốn n8n trên Railway gọi callback thì người tích hợp phải deploy API Customer Bot hoặc cung cấp HTTPS endpoint được duyệt. Loopback trên máy local không truy cập được từ Railway. Chỉ bind `0.0.0.0` khi host/container/network đã được người vận hành quyết định; dùng HTTPS reverse proxy/service domain.

## Contract nối vào n8n/Core

- Customer → n8n: `POST https://n8n-production-8b39b.up.railway.app/webhook/v1/customer/order-intents`.
- Payload: `autobiz.customer-order-intent.v1`, envelope `{intent, customer_auth}`; HMAC-SHA256 trên contract + timestamp + canonical sorted UTF-8 JSON. Giữ đúng SKU/variant, không dùng giá/tổng tiền của bot làm authority.
- Bot chỉ chấp nhận receipt `DRAFT_CREATED` có draft/run/event UUID và external reference khớp. HTTP 202 ACCEPTED đơn thuần không đủ.
- Core → n8n → Customer: callback `POST <Customer Bot HTTPS base>/webhooks/customer-order-results`, contract `autobiz.customer-order-result.v1`, envelope `{result, customer_auth}`. Bot kiểm tra account/HMAC/time/draft receipt/conversation trước khi gửi Telegram.
- Callback trả `DELIVERED`, `result_id`, `duplicate`; n8n chuyển thành signed ACTION_RESULT về Core. Duplicate không gửi Telegram thêm. Ledger SENDING/unknown outcome yêu cầu reconciliation, không tự retry provider send.
- Legacy `/webhooks/order-status` mặc định tắt; giữ `AUTOBIZ_LEGACY_ORDER_STATUS_ENABLED=false`.

Người nhận cần cấu hình n8n `AUTOBIZ_CUSTOMER_BOT_CALLBACK_URL=<HTTPS base>/webhooks/customer-order-results`. Hai JSON trong `integration/workflows` là candidate, giữ inactive; workflow ingress live đang là draft. Không execute/import/publish production chỉ vì nhận gói này.

## Multi-shop

Hiện tại **một instance = một shop/account/catalog/Customer Telegram token**. Chạy instance riêng, env riêng và persistent ledger riêng cho Shop A/B. Không chạy hai polling process dùng chung token. Một Customer Bot token phục vụ nhiều shop cần cơ chế chọn shop, lookup account theo tenant và cô lập conversation/catalog; bản này chưa triển khai kiểu đó.

Core phải xác minh shop và owner membership, installation ACTIVE, `customer_surface=telegram`, `customer_ingress_principal` đúng bridge principal và active credential generation. Owner Telegram binding là authority riêng, không lấy ID/chat/token Customer Bot làm Owner binding.

## Việc còn lại bên hệ thống chính

1. Port/review candidate Core vào đúng full repository `autobiz-main` và commit deploy thực tế. Core candidate trước đó ở `AutoBiz_Core_Integration`; chưa chuyển vào `autobiz-main`.
2. Provision CUSTOMER_BOT installation/account/credential đúng shop và kiểm tra Owner PRIVATE ACTIVE binding.
3. Deploy Customer API + Telegram polling, persistent volume và resource/catalog đúng shop theo quyền được duyệt.
4. Core bật customer gate/TTL và worker lanes runtime/surface-delivery/customer-result sau khi review; callback workflow dùng shared signed bridges có sẵn.
5. Sau khi endpoint/credential ready, người vận hành duyệt publish đúng workflow Customer ingress/result; kiểm tra retention của cả shared workflows.
6. Test synthetic: khách chốt → draft (chưa trừ kho) → chủ approve/reject → đúng khách nhận kết quả; test duplicate, sai shop, hết hạn, giá/tồn kho thay đổi và restart ledger.

Không dùng 15 staged Railway changes có sẵn như một gói deploy. Project tên autobiz-staging nhưng environment đã đọc được là production.

## Test bàn giao

```powershell
$env:AUTOBIZ_CUSTOMER_RECEIPT_DB = "$PWD/work/test-receipts.sqlite3"
python -m unittest tests.test_integration_configuration tests.test_customer_result tests.test_customer_intent tests.test_order_confirmation_webhook
```

Tests dùng synthetic credentials và mock provider/HTTP. Chưa là live E2E. Bot conversation store còn in-memory; SQLite receipt ledger không phải toàn bộ conversation recovery.

Gói không chứa `.env`, token/secret, DB/catalog export, model cache, vectorstore, venv hoặc dữ liệu khách. Nếu hệ thống nhận chưa có resource/catalog, cần cấp riêng bằng kênh được phép.
