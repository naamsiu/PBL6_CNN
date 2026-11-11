# Bản sửa: bỏ EYE_UNCERTAIN / eye boxes

- Không có symbol `eye_curtain` trong mã nguồn ZIP cũ. Trạng thái `EYE_UNCERTAIN`, alert `EYE:UNRELIABLE`, khung mắt và lớp banner tương ứng là các thành phần gây nhiễu được loại bỏ.
- Không còn vẽ 2 eye boxes. Landmark dots và các chỉ số chi tiết được tắt theo mặc định (`SHOW_LANDMARKS=0`, `SHOW_EYE_DEBUG=0`).
- EAR từ 98 điểm WFLW, dùng median 3 khoảng mí mắt / khoảng cách hai khóe, với open baseline riêng từng mắt sau 12 mẫu mở mắt rõ ràng.
- Pose yaw/pitch/roll vượt ngưỡng, điểm mắt ngoài frame/face, mắt quá nhỏ hoặc hình học không hợp lệ => trạng thái mắt UNKNOWN, không cộng closed time, không báo drowsy từ các khung này.
- Báo buồn ngủ do nhắm mắt trước đó không tiếp tục hiển thị hoặc tái phát khi chuỗi đo mắt bị gián đoạn. Lần nhắm sau cần đạt lại ngưỡng thời gian.
- Pose mất tập trung / ngáp / gục dùng bộ đếm và điều kiện riêng; không dùng khung mắt để suy đoán.
- Không thay đổi endpoint API, request auth, mã vi phạm và outbox; vẫn giữ FACE_NOT_VISIBLE khi không phát hiện bất kỳ mặt nào đủ lâu.
- Thêm `--headless` cho `monitor` để tránh imshow/destroyAllWindows trên OpenCV headless.

**Giới hạn:** Chỉ có tọa độ landmark, không có head/eye occlusion classifier. Vì vậy không thể phân biệt chắc chắn mọi trường hợp kính đen, che mắt hay landmark đoán sai. Các ngưỡng EAR cần được hiệu chỉnh bằng video thực tế.

**Kiểm thử:** 48 unit tests pass (kiểm tra logic). Môi trường đóng gói không có onnxruntime, do đó các phép thử import dùng stub đơn giản; chưa chạy NanoDet/Landmark ONNX thật và chưa đo độ chính xác/FPS trên Pi.
