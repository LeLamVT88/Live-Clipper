# Phát hiện và cắt graphic lineup từ video

Pipeline nhận trực tiếp video, không cần transcript, ASR hoặc text LLM. Mặc
định quét 600 giây đầu; có thể đổi khoảng quét hoặc chọn toàn bộ video.

## Luồng xử lý

1. FFprobe đọc duration và xác nhận video stream.
2. PySceneDetect tìm hard cut, dissolve và gentle overlay transition.
3. Mỗi scene ngắn dùng frame giữa. Scene dài được chia thành visual unit tối
   đa 3 giây và lấy frame giữa từng unit. Cách này phủ dày hơn ba mốc
   25%/50%/75% khi scene rất dài.
4. `PP-OCRv6_tiny_det` đo text layout trên mọi frame đại diện.
   `PP-OCRv6_tiny_rec` đọc mọi frame có đủ text/layout; không dùng một top-k
   chung có thể làm mất một vùng thời gian.
5. Full-roster card được nhận qua formation, số áo, danh sách tên và các từ
   như `LINEUP`, `STARTING XI`, `SUBSTITUTES`. Chuỗi player-by-player được nhận
   qua ít nhất ba frame gần nhau có layout tương thích, số áo/tên và nội dung
   thay đổi.
6. Mỗi candidate được scan lại ở mật độ tối đa 0,5 giây/frame trong vùng đệm
   12 giây. Dense pass tinh chỉnh hai biên. Nếu dense OCR không xác nhận lại,
   candidate coarse vẫn được giữ và kết quả được gắn cờ review.
7. Evidence được snap ra ngoài tới scene cut gần nhất, rồi thêm safety padding
   4 giây trước và 6 giây sau. Hai lineup liên tiếp không được phép overlap.

## Cài đặt

Main environment chứa PySceneDetect và công cụ export; PaddleOCR chạy trong
Python 3.11 riêng vì Paddle chưa hỗ trợ Python 3.14:

```bash
.venv/bin/pip install -r requirements.txt
python3.11 -m venv .venv-ocr
.venv-ocr/bin/pip install -r requirements.txt
```

Model mặc định là `PP-OCRv6_tiny_det` và `PP-OCRv6_tiny_rec`, chạy CPU và cache
trong `.cache/paddlex`.

## Phát hiện lineup

```bash
.venv/bin/python src/lineup/detect_from_video.py \
  --source-video data/raw_data/<giai>/<video>.mp4 \
  --overwrite
```

Quét toàn bộ nguồn nếu pre-match dài hơn 10 phút:

```bash
.venv/bin/python src/lineup/detect_from_video.py \
  --source-video data/raw_data/<giai>/<video>.mp4 \
  --scan-end full \
  --overwrite
```

Các option chính:

- `--scan-start`, `--scan-end`: nhận giây hoặc `HH:MM:SS`.
- `--expected-lineups`: `1`, `2`, hoặc `auto`; mặc định `2`.
- `--coarse-unit-seconds`: khoảng đại diện tối đa; mặc định 3 giây.
- `--dense-unit-seconds`: mật độ scan candidate; mặc định 0,5 giây.
- `--dense-padding`: vùng scan thêm quanh candidate; mặc định 12 giây.
- `--padding-before`, `--padding-after`: phần dư an toàn của clip.
- `--scene-threshold`, `--scene-min-length-seconds`,
  `--scene-snap-radius`: cấu hình PySceneDetect.

Output mặc định:

```text
outputs/<giai>/<video>/predictions/lineup/
├── lineup_segments.csv
├── visual_detection_raw_response.json
└── detection_metadata.json
```

Nếu dự kiến hai lineup nhưng chỉ thấy một, CLI vẫn ghi candidate đã tìm thấy,
đặt trạng thái `incomplete` và trả exit code 2 để không báo thành công giả.

## Export clip

```bash
.venv/bin/python src/lineup/export_clips.py \
  --segments-csv outputs/<giai>/<video>/predictions/lineup/lineup_segments.csv \
  --overwrite
```

Clip được ghi vào `outputs/<giai>/<video>/clips/lineup`.
