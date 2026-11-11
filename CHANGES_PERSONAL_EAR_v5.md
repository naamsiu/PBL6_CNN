# V5 — Personalized EAR, distance-aware quality, two-state eyes

## Mục tiêu

Tránh `UNKNOWN` kéo dài khi EAR nằm giữa hai ngưỡng của V4. Giữ nguyên NanoDet, landmark 98 điểm (+pose), camera, xác thực, API và mã vi phạm. **Không có CNN mắt, không vẽ box mắt.**

## Hiệu chuẩn 3 giây (mỗi phiên / bấm R)

Yêu cầu tài xế nhìn thẳng, mắt mở; thu tối thiểu `EAR_CALIBRATION_MIN_SAMPLES=15` frame hợp lệ trong ít nhất 3 giây. Loại bottom 40% EAR trong thống kê để hạn chế chớp mắt ngắn. Dữ liệu cho từng mắt:

- **Open EAR**: trung vị EAR nhóm mẫu mở, không phụ thuộc ngưỡng mở toàn dân.
- **EAR variability**: 1.4826 x Median Absolute Deviation (MAD), sàn 0.002.
- **Eye width + face bbox**: chiều rộng mắt và chiều cao box NanoDet ở lúc nhìn thẳng.
- **Landmark stability**: trung vị RMS độ lệch 6 tọa độ ngang điểm mí mắt trong hệ trục đã chuẩn hóa theo góc, chiều rộng và dịch chuyển. *Không dùng độ cao mí mắt làm thước đo jitter vì chớp mắt thật sẽ làm thay đổi độ cao đó.*

Lưu vào RAM của phiên. Không cập nhật online để tránh drift. Hiệu chuẩn không được coi là có thể xác nhận mắt mở thực tế một cách độc lập: nó dựa vào chỉ dẫn người dùng.

## Ngưỡng V5 và quyết định

`close = min(0.68 * baseline, baseline - 2.5 * variability)`

`open = max(min(0.74 * baseline, baseline - 1.5 * variability), close + max(0.004, 0.035 * baseline))`

*Các hệ số này chỉ là giá trị thử nghiệm, không phải các ngưỡng kiểm định an toàn.*

Từng mắt: median EAR của tối đa 3 quan sát hợp lệ gần nhất, xác nhận 2 frame liên tiếp vượt ngưỡng để chuyển trạng thái OPEN/CLOSED. Nếu EAR nằm ở khoảng giữa, **giữ trạng thái cũ**, không chuyển UNKNOWN. Sau khi calibration thành công, trạng thái khởi tạo OPEN (theo yêu cầu mắt mở lúc calibration).

Mất đo, pose quá lớn, mắt nhỏ hoặc geometry không ổn định => UNKNOWN cho quan sát đó. Không lấy mẫu kém chất lượng để cộng giây nhắm mắt hay PERCLOS. Mất track lâu sẽ bỏ trạng thái CLOSED cũ khi bắt đầu đo lại.

## Quan hệ hai mắt / cảnh báo

- Hai mắt CLOSED hợp lệ: `CLOSED` tổng, được đo thời gian nhắm.
- Có ít nhất một mắt OPEN hợp lệ: `OPEN` tổng, không thể là *cả hai cùng nhắm* (có thể ghi `PARTIAL_EYE_OPEN`).
- Không có mắt OPEN và thiếu bằng chứng hai mắt CLOSED: `UNKNOWN` tổng, không phát cảnh báo mắt.
- Không hiện banner mắt không rõ, không có eye rectangles. Nhắm mắt dài, PERCLOS và các cảnh báo miệng/pose giữ nguyên API.

## Nhận biết gần/xa

`face_scale = current_box_height / baseline_box_height`; `expected_width = baseline_eye_width * face_scale`. Kiểm tra số pixel tối thiểu, bề rộng thực so với dự kiến, độ ổn định qua 3 frame và độ lệch hình học ngang so với calibration. Khi xa, cho phép thêm lượng jitter ngang nhỏ do sai số lượng tử nhưng không tạo chi tiết mới từ zoom số. **EAR dimensionless không được nhân theo face_scale.**

## Các khóa .env

```env
EAR_CALIBRATION_SECONDS=3.0
EAR_CALIBRATION_MIN_SAMPLES=15
EAR_PERSONAL_CLOSED_RATIO=0.68
EAR_PERSONAL_OPEN_RATIO=0.74
EAR_CONFIRM_FRAMES=2
EAR_FILTER_FRAMES=3
EAR_STABILITY_FRAMES=3
EAR_SHAPE_MIN_TOLERANCE=0.14
EAR_MIN_WIDTH_PX=8
EAR_EXPECTED_WIDTH_MIN_RATIO=0.55
SHOW_EYE_DEBUG=0
EYE_LOG=0
```

Bật `SHOW_EYE_DEBUG=1` hoặc `EYE_LOG=1` khi kiểm thử. Nếu tốc độ thấp (<10 FPS), việc xác nhận 2 frame + median 3 có thể làm nhắm/mở được xác nhận chậm hơn; đối chiếu timestamps trong CSV.

## Kiểm thử và giới hạn

Chạy `python -m pytest -q` sau khi cài `requirements-dev.txt`. Test logic bao gồm mắt hí, mắt lệch nhau, noise EAR, hình học jitter, mặt xa, mất pose, chuyển trạng thái và temporal alarms. Các test có thể sử dụng dữ liệu mô phỏng / giả lập thư viện import nếu chưa có ONNX Runtime; **chưa thay thế thử nghiệm camera/ONNX trên Raspberry Pi 5**. Không tự động kết luận mắt bị che chỉ từ vị trí landmark.
