# Bảng mã sự kiện gửi về Django

Đăng ký trước cả 6 mã dưới đây trong danh mục loại vi phạm của Django. Tên hiển thị có thể đổi, code phải khớp chính xác, kể cả chữ hoa. File violation_codes.json cùng thư mục là dữ liệu tham khảo để nhập danh mục, không phải Django fixture (chưa có schema model backend). Pi không tự tạo loại vi phạm trên server.

| violation_code | Tên hiển thị | Điều kiện theo cấu hình hiện tại |
|---|---|---|
| DROWSINESS | Buồn ngủ | Nhắm mắt ≥ 1,5 s; hoặc nhắm mắt ≥ 0,7 s kèm gật đầu; hoặc PERCLOS đạt ngưỡng; hoặc ngáp lặp lại. |
| DISTRACTED_LOOK_AWAY | Quay đầu mất tập trung | Góc yaw lệch > 35° và bộ đếm nhìn lệch đạt ≥ 2 s. |
| DISTRACTED_LOOK_DOWN | Cúi nhìn xuống mất tập trung | Góc cúi > 30° và bộ đếm nhìn lệch đạt ≥ 2 s. |
| YAWNING | Ngáp | Miệng mở vượt ngưỡng động sau hiệu chỉnh, liên tục ≥ 0,7 s. |
| HEAD_NOD | Gật/cúi đầu | Cúi so với mốc > 20° liên tục ≥ 0,45 s; có fallback dịch chuyển mặt khi không có pose. |
| FACE_NOT_VISIBLE | Không thấy khuôn mặt | Camera vẫn có ảnh, detector chạy thành công nhưng không thấy mặt liên tục ≥ 2 s; khoảng cách quan sát không quá 0,5 s. |

PERCLOS hiện dùng cửa sổ 30 s, cần ít nhất 10 s quan sát hợp lệ và độ phủ ≥ 60%; tỷ lệ nhắm mắt ≥ 20% kích hoạt cảnh báo theo engine. Ngáp lặp lại là ít nhất 2 lần trong 60 s. Đổi các biến .env tương ứng sẽ đổi ngưỡng hành vi; timeout FACE_NOT_VISIBLE 2 s và cooldown theo mã 30 s hiện nằm trong monitor.py.

Bộ đếm nhìn lệch của engine dùng chung cho quay ngang và cúi xuống: nếu chuyển hướng trong cùng đợt nhìn lệch, mã gửi dựa trên away_reason hiện tại, không yêu cầu mỗi hướng tích lũy lại đủ 2 s. Mức nghiêm trọng sau 5 s vẫn dùng cùng mã; Pi không gửi severity, server tiếp tục tạo MEDIUM/PENDING.

HEAD_NOD có thể phản ánh cúi đầu kéo dài, không khẳng định người đó ngủ. FACE_NOT_VISIBLE chỉ phản ánh mất quan sát; ngáp đơn lẻ cũng không chứng minh vi phạm. Tất cả cần admin xem ảnh và đánh giá.

## Quy tắc gửi

- Từng mã có UUID, trạng thái đã ghi nhận và cooldown 30 s độc lập. Hành vi kéo dài không tạo thêm sự kiện sau mỗi 30 s.
- DROWSINESS chỉ mở lại đợt sau AWAKE. Mã khác mở lại sau ít nhất 0,4 s quan sát hợp lệ không còn hành vi. Mất dữ liệu không tính là phục hồi. Sau khi mở lại, vẫn phải hết cooldown mới xếp sự kiện mới.
- Nhiều hành vi cùng lúc có thể tạo nhiều bản ghi, dùng cùng frame nhưng mỗi mã có UUID riêng. Ví dụ HEAD_NOD và DROWSINESS; YAWNING và DROWSINESS ở lần ngáp lặp lại. Đây là các mã khác nhau, không phải request retry trùng.
- Chỉ FACE_NOT_VISIBLE được tạo từ frame không có mặt. Lỗi camera/landmark/detector không dùng cờ cảnh báo cũ để tạo vi phạm mới, không gửi ảnh đen hoặc ảnh cũ.
- Với DROWSINESS do mắt và quan sát mắt đáng tin cậy: confidence=min(P_closed_left, P_closed_right). Các trường hợp khác gửi 0.0 theo quy ước chưa có ước lượng, không phải xác suất hành vi bằng 0. Các luật không cung cấp xác suất đã hiệu chỉnh.
- Metadata, ảnh, UUID và thời gian giữ nguyên khi gửi lại. Retry và hàng đợi dùng api_client.py như bản trước.

## Trạng thái/lỗi hiện có nhưng KHÔNG gửi qua API vi phạm

| Mã/trạng thái nội bộ | Ý nghĩa | Xử lý hiện tại |
|---|---|---|
| CAMERA_ERROR | Không đọc được camera | Hiển thị lỗi cục bộ; không có ảnh thật để gửi |
| FACE_DETECTOR_ERROR | Lỗi suy luận detector khuôn mặt | Hiển thị lỗi cục bộ; không coi như NO_FACE |
| LANDMARK_ERROR | Lỗi suy luận landmark | Hiển thị trạng thái thiếu quan sát |
| LANDMARK_ERROR | Lỗi suy luận landmark mắt | Ghi log, mắt UNKNOWN; các hành vi từ landmark hợp lệ vẫn có thể được gửi |
| MONITORING | Mắt không đủ điều kiện đo EAR (pose/landmark/vùng nhìn) | Bỏ qua mẫu mắt, không cảnh báo và không gửi vi phạm |
| EYES_CLOSED / BLINK_OR_CLOSING | Nhắm mắt chưa đủ điều kiện DROWSINESS | Theo dõi tiếp, không gửi sự kiện riêng |
| AWAKE / MOUTH_OPEN | Hai mắt OPEN, miệng bình thường / miệng vừa mở nhưng chưa thành ngáp | Không gửi vi phạm riêng |

Django chưa có endpoint heartbeat/lỗi thiết bị theo hợp đồng được cung cấp. Các lỗi kỹ thuật ở bảng này chưa truyền về server; cần API riêng nếu muốn quản trị theo dõi sức khỏe thiết bị. Giữ nguyên không đoán route hay thêm trường vào request violations.

## Kiểm thử

Các bài kiểm thử mô phỏng vi phạm và bộ đếm thời gian nằm trong `tests/test_violation_api.py`,
`tests/test_temporal.py`, `tests/test_eye_gate_logic.py`; các bài kiểm thử model ONNX
thật cần cài `onnxruntime` và `onnx`. HTTP được giả lập, chưa gọi server Django thật.
