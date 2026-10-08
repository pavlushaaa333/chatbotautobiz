# AutoBiz Customer Chatbot

Bot tư vấn cho khách và gửi yêu cầu đặt hàng vào hệ thống chủ shop AutoBiz hiện có.

Luồng tích hợp: Customer Bot → n8n → AutoBiz Core → Owner Bot duyệt/từ chối → n8n → Customer Bot.

## Bàn giao cho người tích hợp

Đọc [integration/README_HANDOFF.md](integration/README_HANDOFF.md) để cấu hình và chạy. Copy [integration/customer-bot.env.example](integration/customer-bot.env.example) thành env riêng ngoài Git. Cấu hình mẫu dùng contract/webhook của n8n hiện có; feature gate mặc định tắt.

Người vận hành cần điền URL callback HTTPS của Customer Bot trên n8n và kiểm tra URL ingress n8n trong env. Đồng thời phải cung cấp **đúng shop/account của CUSTOMER_BOT installation đã cấp**, account credential, token Customer Bot, catalog và persistent ledger path. Không dùng `local_fixture` hoặc token Owner Bot thay thế.

Workflow nội bộ và patch Core được bàn giao riêng, không phát hành trong repo công khai này. Người vận hành phải review phần Core còn thiếu trong hệ thống chủ shop. Push source này không tự cấp installation, deploy service hay publish n8n.

Một instance hiện phục vụ một shop/account/catalog. Secret không được commit. Core vẫn quyết định giá/tồn kho, quyền owner và tạo order sau approval; bot không tự tạo official order.

## Kiểm tra offline

```powershell
python -m unittest tests.test_integration_configuration tests.test_customer_result tests.test_customer_intent tests.test_order_confirmation_webhook
```

Test dùng synthetic credentials và mock HTTP/Telegram; không chứng minh live E2E. Hướng dẫn chạy API callback và Telegram polling có trong tài liệu bàn giao.
