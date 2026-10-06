# Bàn giao Customer Bot -> n8n -> Core: Gate 1 local

Ngày: 2026-10-06. Trạng thái: bản triển khai để review; chưa kết nối live, chưa deploy/merge, chưa đạt toàn bộ workstream Customer Bot -> Owner approval.

## Đã làm

- Bot tại D:\AutoBiz_AI_Chatbot, branch codex/customer-n8n-intake: bounded intent, HMAC theo account, UUID draft reference, kiểm tra receipt Core, giữ nguyên catalog variant từ search tới draft. Feature flag mặc định tắt.
- Core tại D:\AutoBiz_Core_Integration, branch codex/customer-order-ingress: endpoint candidate /v1/integration/customer-order-intents, trusted installation + membership, active Doppler credential generation, RFC9421, durable nonce, reuse Phase 9 / Phase 5H, atomic draft + queued runtime, replay/conflict/multi-shop fences.
- n8n: workflow mới AUTOBIZ_ADAPTER_Telegram_Customer_Order_Intent, ID kckJmHWzoP9REHL2. Sáu node, gọi shared signing bridge hiện có. Đã lưu bản nháp, chưa Publish/Execute.
- URL: https://n8n-production-8b39b.up.railway.app/workflow/kckJmHWzoP9REHL2
- Cài đặt workflow mới: không lưu payload execution thành công/lỗi/manual/progress. Cần kiểm tra chính sách lưu của shared signing workflow trước khi chạy dữ liệu khách thật.

## Bằng chứng

- 61 kiểm thử Core pass trong lần chạy cuối cho ingress, CustomerSurfaceService, OrderAutomationService, IntegrationRuntime và factory gate/TTL.
- 19 kiểm thử bot pass bằng mock HTTP; không gửi Telegram hoặc request n8n thật.
- 14 kiểm tra Code node pass bằng Node local; không phải chứng cứ n8n live.
- 3 cross-component checks: actual Bot builder/HMAC -> local workflow Code -> reference RFC9421 -> Core TestClient -> isolated PostgreSQL. Unicode proof, retry và tamper đều đúng. Không chạy n8n/Telegram network.
- Lint các file thay đổi pass. Có một cảnh báo deprecation Starlette/httpx trong test harness.
- Kiểm tra: canonical Core price; +1 draft và +0 orders; inventory unchanged; same request/new app same draft/run; concurrent aliases one draft/run; conflicting shipping rejected; bad quantity/SKU rollback; unauthorized installation/principal/account/membership denied; bad/expired HMAC denied; signed-body tamper and nonce replay denied; HTTP 202 ACCEPTED alone rejected by Bot; exact catalog variant retained.
- Kết quả không chứng minh owner nhận pending, approve/reject, official order/stock mutation, committed callback hoặc chatbot restart conversation recovery.

## Môi trường và nguồn

Core là partial source snapshot 109 file từ minh2509/autobiz, staging/s3-shop-c tại SHA 20d41f7368c759ba846ca91930f357bed24bae2a. Local baseline commit 5de8cbc8a9e2e303f2191c071550b79736176d54 chỉ là commit snapshot, không phải SHA upstream. SOURCE_SNAPSHOT.json ghi rõ giới hạn; không giả định đây là full authenticated clone.

Test PostgreSQL riêng: loopback 127.0.0.1:55479, user autobiz_test, database autobiz_p4bc_20261006120000_ce000001. Chỉ DB disposable này được ghi. Test guards kiểm tra host/database/user trước khi truncate fixtures.

Dùng business schema và migrations tới 0030, ngoại trừ 0004_knowledge_rag vì Windows PostgreSQL chưa có pgvector. Đây là focused order proof, không phải full migration/RAG/release proof.

Graphify CLI chưa có và graph GitHub vượt giới hạn fetch nội dung; chưa query/rebuild graph. Ghi rõ đây là phần closeout còn thiếu, không nâng roadmap phase. Canonical contracts có độ ưu tiên cao hơn graph navigation.

.env của bot không thay đổi. Việc xóa .env.example đã có từ trước, không đưa vào thay đổi của task. Landing page, stable owner/ERP workflows và Control Plane không chỉnh sửa.

## Checklist để bật thật

- [ ] Review candidate contract và ingress diff.
- [ ] Xác định Core API staging đang chạy từ repo/branch/service nào; triển khai candidate từ repo thật sau review.
- [ ] Cấp CUSTOMER_BOT installation/account cho đúng shop và actor có membership; cấu hình customer_surface=telegram, customer_ingress_principal đúng bridge principal.
- [ ] Provision active opaque credential generation bằng luồng Core/Control Plane được duyệt; cấu hình bí mật qua secret manager, không gửi token/secret trong chat.
- [ ] Enable Core gate + TTL rõ ràng; xác nhận route ready với staging target đúng.
- [ ] Verify no customer payload persistence on shared signing bridge.
- [ ] Publish workflow Customer mới sau khi Core route/installation ready.
- [ ] Cấu hình bot flag/account/webhook/installation secret; khởi động lại bot.
- [ ] Chạy Gate 1 với dữ liệu synthetic: receipt DRAFT_CREATED; 1 draft; 0 orders; inventory unchanged; replay same IDs.
- [ ] Review Gate 1 live evidence trước Gate 2.
- [ ] Nối worker/customer pending -> Owner approval delivery. Existing owner Telegram transport đã có, nhưng customer dispatcher chưa nối.
- [ ] Nối approve/reject CustomerOrderConversion và prove exactly-once official order/stock/audit/outbox.
- [ ] Chốt CUSTOMER_CALLBACK_DELIVERY_AUTH_CONTRACT; đóng authentication/dedup gap của /webhooks/order-status trước callback live.
- [ ] Prove Shop A/B isolation, wrong/expired/stale owner decision, forged pending ID, reject terminal, duplicate/retry/restart safety.

Hiện cần thông tin installation/account Customer Bot và service triển khai Core staging. Chỉ cung cấp tên/mã định danh; không gửi secret.
