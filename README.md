# PBL6 — Face detector + Landmark V5 Personal EAR (không CNN mắt)

## Kiến trúc

NanoDet phát hiện mặt → Landmark 98 điểm + pose → EAR hình học cho hai mắt, MAR cho miệng, yaw/pitch cho tư thế → bộ đếm thời gian và PERCLOS → cảnh báo + API gốc.

Không còn model CNN mắt, crop mắt CNN, hoặc xác suất CNN. Các cảnh báo giữ mã API cũ: `DROWSINESS`, `YAWNING`, `HEAD_NOD`, `DISTRACTED_LOOK_AWAY`, `DISTRACTED_LOOK_DOWN`, `FACE_NOT_VISIBLE`. Trường `confidence` của API giữ nguyên nhưng gửi `0.0` vì EAR là chỉ số hình học, không phải xác suất được hiệu chuẩn.

## QUAN TRỌNG: Thiếu trọng số ONNX landmark mới

File notebook `train-landmark-wflw-lapa (4)(1).ipynb` chỉ chứa mã huấn luyện và export, **không chứa ONNX đã huấn luyện**. Cần export model thực tế từ notebook và chép vào:

`models/landmark_v4_2_pi_256_f3f4_2conv_pose3.onnx`

Notebook export `landmarks98 [1,98,2]` (chuẩn hóa crop), `pose_deg [1,3]` (yaw, pitch, roll), RGB / 255, crop của notebook 1.25 (runtime hiện cấu hình 1.33 để nới cằm), input 256×256. Không tự ý dùng ONNX landmark 128 cũ thay thế. NanoDet cũ vẫn được giữ vì `code.zip` chỉ chứa benchmark/download dataset, không có model detector ONNX mới.

## Chạy

```bash
pip install -r requirements.txt
python cli.py model-info
python cli.py monitor --skip-auth
python cli.py monitor
```

Thiết lập `.env`: `LANDMARK_MODEL`, `EAR_CALIBRATION_SECONDS`, `EAR_CALIBRATION_MIN_SAMPLES`, `EAR_PERSONAL_CLOSED_RATIO`, `EAR_PERSONAL_OPEN_RATIO`, `EAR_MIN_WIDTH_PX`, `EAR_CONFIRM_FRAMES`, `EAR_FILTER_FRAMES`, `EAR_MAX_YAW`, `EAR_MAX_PITCH` và các tham số thời gian/cảnh báo. Giá trị EAR mặc định 0.16/0.22 là **điểm khởi đầu**, cần hiệu chỉnh bằng video thực tế mắt mở, chớp, nhắm, nghiêng đầu và điều kiện sáng khác nhau. Landmark V4 có tăng cường dữ liệu mắt trong huấn luyện nhưng chưa có kết quả thử nghiệm runtime để khẳng định chính xác nhắm mắt/ngáp/gục.

API và logic session/auth/violation vẫn giữ nguyên. `R` đặt lại baseline mắt (mắt mở), miệng và tư thế; không cần CNN hoặc hiệu chuẩn mắt nhắm. Xem `CHANGES_PERSONAL_EAR_v5.md` cho thuật toán V5. `EYE_LOG=1` ghi log trạng thái và thời gian xử lý hình học.

## V5: EAR cá nhân hóa + bộ nhớ OPEN/CLOSED (không eye curtain / eye boxes)

- Mặc định **không vẽ** hai box mắt, landmark, hoặc overlay debug. Có thể bật `SHOW_EYE_DEBUG=1` để xem EAR/threshold/baseline, hoặc `SHOW_LANDMARKS=1` để vẽ điểm.
- **3 giây đầu phiên, nhìn thẳng và mở mắt**. Thu baseline riêng cho mỗi mắt: EAR mở điển hình, MAD của EAR (độ dao động), chiều rộng mắt cùng chiều cao bbox mặt, và độ ổn định vị trí ngang của landmark được chuẩn hóa theo chiều rộng mắt. Không yêu cầu EAR mở > 0.20 (mắt hí vẫn hiệu chuẩn được khi hình học hợp lệ).
- Ngưỡng đóng/mở động riêng cho mỗi mắt: mặc định `0.68×baseline` và `0.74×baseline`, có giảm ngưỡng đóng khi EAR mở trong lúc hiệu chuẩn quá nhiễu. Median EAR 3 frame + yêu cầu 2 phép đo liên tiếp để chuyển trạng thái. **EAR ở giữa hai ngưỡng giữ trạng thái OPEN/CLOSED trước đó; không phải UNKNOWN.**
- `UNKNOWN` chỉ dành cho frame mắt không đo được, bị che, quá ít pixel, pose vượt giới hạn, hoặc landmark thiếu ổn định. Một mắt OPEN đáng tin cho phép kết luận không phải *hai mắt cùng nhắm*. Chỉ khi cả **hai mắt CLOSED đáng tin** mới tính nhắm mắt vào bộ đếm cảnh báo.
- Bbox NanoDet cung cấp **tỷ lệ mặt gần/xa** (so chiều cao hiện tại với baseline) để đánh giá chiều rộng mắt dự kiến và chất lượng. **Không nhân EAR/threshold EAR trực tiếp với khoảng cách.** Frame mắt quá nhỏ bị bỏ qua thay vì đoán mò.
- Mất mặt/mất pose/xoay quá ngưỡng không cộng thời gian nhắm, không phát `EYE_UNCERTAIN`, không gửi cảnh báo mắt cũ. Pose/ngáp/gục và API gốc vẫn độc lập. Đo thời gian thực bằng đồng hồ monotonic, không đếm frame giả định FPS.
- EAR baseline được giữ nguyên trong phiên, không tự cập nhật để tránh trôi ngưỡng khi buồn ngủ. Bấm `R` hoặc bắt đầu phiên mới sẽ xóa toàn bộ baseline của phiên trước.
- `EYE_LOG=1` lưu CSV có baseline EAR, MAD, eye width, ổn định landmark, EAR lọc, ngưỡng, độ lệch và trạng thái từng mắt; dùng để tinh chỉnh trên video thực tế.

**Chạy thử camera (trên Raspberry Pi hoặc PC có giao diện):**

```bash
python cli.py monitor --skip-auth --backend opencv --camera 0 --width 1280 --height 720
```

**Qua SSH / không GUI:** thêm `--headless`. Cần `models/landmark_v4_2_pi_256_f3f4_2conv_pose3.onnx` đúng đường dẫn trong `.env`.

**Giới hạn:** nếu ảnh mắt chỉ còn vài pixel hoặc bị mất nét thật, độ tin cậy EAR không thể được bảo đảm bằng bộ lọc. Baseline mở mắt giả định người dùng thực sự nhìn thẳng và mở mắt trong lúc hiệu chuẩn. Đây là logic thử nghiệm, chưa đánh giá precision/recall trên video và ONNX thực tế.

## Face far / bounding box extends below chin

See `CHANGES_FAR_CAMERA.md` for zoom, crop, resolution and pixel-gating settings. The visible box expansion does not alter EAR calculations, pose or API.
