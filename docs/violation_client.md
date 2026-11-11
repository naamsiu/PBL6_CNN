# Tích hợp gửi vi phạm từ Pi

Chỉ sửa app/api_client.py và app/monitor.py; thêm tài liệu này và tests/test_violation_api.py. Giữ nguyên model, notebook, cấu hình và các file khác.

## Cách chạy

Điền WEB_SERVER_URL và DEVICE_API_KEY trong .env như trước. Server phải có đủ 6 loại sự kiện trong docs/violation_codes.md. Chạy chế độ xác thực bình thường để nhận session_id; chế độ local không xác thực không gửi vi phạm. Không gọi heartbeat vì chưa có endpoint.

Luồng: mở phiên thành công -> landmark + temporal engine phát hiện một trong 6 sự kiện đủ điều kiện -> lấy frame camera gốc (không overlay) -> JPEG -> lưu sự kiện xuống ổ đĩa -> worker riêng POST /api/v1/device/violations/ -> xóa file chờ chỉ khi nhận 201 VIOLATION_RECORDED hoặc 200 EVENT_ALREADY_RECORDED. Response server có code ở cấp ngoài cùng như hợp đồng đã cung cấp.

Request có đúng event_id, session_id, violation_code, detected_at, confidence và file image (image/jpeg). requests tự tạo boundary và Content-Length; kiểm tra cả giới hạn JPEG 2 MiB, 1920x1080 và tổng multipart 2 MiB + 16 KiB. detected_at là UTC có múi giờ tại thời điểm xếp sự kiện, không thay bằng thời gian gửi lại. Cần đồng bộ giờ Pi với server.

## Sự kiện và confidence

Bản này gửi 6 mã: DROWSINESS, DISTRACTED_LOOK_AWAY, DISTRACTED_LOOK_DOWN, YAWNING, HEAD_NOD, FACE_NOT_VISIBLE. Xem bảng điều kiện và trạng thái không gửi trong violation_codes.md; violation_codes.json chứa danh mục để đối chiếu bên Django. Cần khai báo đủ cả 6 mã trước khi chạy.

Mỗi mã có chống trùng theo đợt và cooldown 30 giây độc lập. Nhiều hành vi đồng thời có thể tạo nhiều bản ghi khác mã. Sau khi khởi động lại, trạng thái đợt/cooldown bắt đầu lại; sự kiện đã lưu vẫn giữ UUID cũ nhưng không hợp nhất với đợt mới.

confidence = 0.0 cho mọi cảnh báo do không có xác suất đã hiệu chuẩn theo quy ước chưa có ước lượng. Không dùng điểm này để tự duyệt khi chưa thiết kế/hiệu chỉnh xác suất buồn ngủ.

## Hàng đợi và lỗi

Mặc định lưu tại violation_outbox trong thư mục dự án; có thể đặt VIOLATION_OUTBOX_DIR. File JSON chứa metadata và JPEG base64, không chứa token. Lưu atomic trước khi đánh dấu đợt đã gửi; lỗi ổ đĩa được báo và thử xếp lại sau 5 giây nếu cảnh báo còn tồn tại. Việc encode/lưu ảnh chạy một lần trên luồng camera; HTTP chạy nền bằng client riêng.

Timeout, lỗi mạng, HTTP 5xx/408/429 thử lại sau ít nhất 5 giây, tôn trọng Retry-After dạng số giây. UUID, metadata và bytes JPEG giữ nguyên qua retry và khởi động lại. JSON lỗi server không hợp lệ không làm mất ảnh. HTTP 2xx không có mã xác nhận mong đợi cũng giữ và thử lại.

4xx khác hoặc chuyển hướng: giữ file với blocked=true và last_error, không tự thử lại. Sau khi quản trị xử lý token/cấu hình hoặc nguyên nhân lỗi, có thể dừng chương trình, sửa blocked=false và next_attempt=0 để thử lại nguyên sự kiện; không đổi event_id/metadata/ảnh. Sự kiện quá 24 giờ sẽ bị server từ chối; cần kiểm tra và lưu trữ thủ công, không tự đổi detected_at. Khi đổi base URL, các sự kiện của URL cũ không gửi sang URL mới. Giữ nguyên thiết bị/API key tương ứng khi khôi phục hàng đợi. Chỉ chạy một tiến trình trên mỗi thư mục outbox. Không đưa thư mục outbox vào gói chia sẻ mã nguồn.

Khi đóng ứng dụng, worker ngừng nhận lượt gửi mới và đợi request hiện tại kết thúc trước khi đóng client; các sự kiện còn lại giữ trên đĩa cho lần chạy sau. Không tự xóa file lỗi/hết hạn, vì vậy cần theo dõi dung lượng ổ đĩa khi mất mạng dài ngày.

## Kiểm thử

python -m pytest tests/test_violation_api.py tests/test_monitor.py -q

26 tests passed trong môi trường sửa: multipart thực sự đã prepare, ảnh resize, retry sau restart, timeout/503, phản hồi không xác nhận, giữ lỗi 400, confidence không hợp lệ, chống trùng theo đợt/cooldown và pipeline monitor có sẵn. HTTP được giả lập; chưa kiểm thử với Django/Cloudinary thật hoặc camera Pi thật.
