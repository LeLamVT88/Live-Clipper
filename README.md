# Live Clipper - Football Lineup Detection

Project dùng MobileNetV3-Small ở `0,5 FPS` để phát hiện đoạn đội hình trong
video bóng đá. Sau đó project chỉ tách các đoạn lineup ở `2 FPS` và dùng
PaddleOCR để đọc tên, số áo.

## Cấu trúc thư mục

```text
live-clipper/
├── data/
│   ├── raw_videos/        # video gốc
│   ├── frames/            # frame 0,5 FPS cho MobileNet
│   ├── ocr_frames/        # frame 2 FPS chỉ trong các đoạn lineup
│   ├── ocr_samples/       # frame và ground truth để phát triển OCR
│   ├── processed/         # metadata và dataset CSV
│   └── ground_truth.csv   # khoảng lineup của Đội 1 và Đội 2
├── outputs/
│   ├── predictions/       # checkpoint, metrics và kết quả dự đoán
│   └── clips/             # clip lineup được cắt từ video gốc
├── src/
│   ├── lineup/            # phát hiện và xuất clip lineup
│   │   ├── utils.py
│   │   ├── extract_frames.py
│   │   ├── build_frame_labels.py
│   │   ├── build_dataset_index.py
│   │   ├── train_mobilenet.py
│   │   ├── predict_mobilenet.py
│   │   ├── aggregate.py
│   │   ├── export_clips.py
│   │   └── evaluate_segments.py
│   └── ocr/               # tách frame lineup và đọc tên, số áo
│       ├── ocr_smoke_test.py
│       ├── run_lineup_ocr.py
│       └── resolve_lineup.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Cài đặt

PaddlePaddle trên macOS hiện cần Python 3.9-3.13, vì vậy project dùng chung
Python 3.11 cho cả MobileNet và OCR:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy kiểm tra trên frame lineup mẫu:

```bash
.venv/bin/python src/ocr/ocr_smoke_test.py
```

Script dùng hai model CPU nhẹ `PP-OCRv6_small_det` và
`PP-OCRv6_small_rec`. Lần chạy đầu cần mạng để tải model vào
`.cache/paddlex/`; các lần sau dùng cache nội bộ và có thể chạy offline. Việc
tải model không phải là lấy roster hay thông tin cầu thủ từ API.

## Chuẩn bị ground truth

Đặt video dùng để train vào `data/raw_videos/`, ví dụ:

```text
data/raw_videos/match1.mp4
data/raw_videos/match2.mp4
```

Điền `data/ground_truth.csv` theo format:

```csv
video,Đội 1,Đội 2
match1.mp4,1.00-2.00,2.56-3.56
match2.mp4,3.12-3.40,4.05-4.35
```

Mỗi video chỉ có một hàng. Mỗi ô đội dùng format `START-END`. Dạng ngắn
`MM.SS` (ví dụ `1.00` là 1 phút) và dạng đầy đủ `HH:MM:SS` đều được chấp
nhận. Khoảng giữa `Đội 1` và `Đội 2` tự nhận `label = 0`.

## Train MobileNetV3

### 1. Tách frame

Mặc định script lấy `0,5 frame/giây`, tức một frame mỗi hai giây, trong 10
phút đầu của tất cả video:

```bash
python src/lineup/extract_frames.py
```

Có thể chọn một hoặc nhiều video cụ thể:

```bash
python src/lineup/extract_frames.py --video "match1.mp4"
python src/lineup/extract_frames.py --video "match1.mp4" --video "match2.mp4"
```

Mặc định `--duration` là `00:10:00`. Có thể truyền `--duration` hoặc `--end`
để chọn khoảng khác.

Frame và metadata được lưu tại:

```text
data/frames/<video_name>/frame_000001.jpg
data/processed/<video_name>/extracted_frames.csv
data/processed/extracted_frames.csv
```

### 2. Gán nhãn frame

```bash
python src/lineup/build_frame_labels.py --only-ground-truth-videos
```

Kết quả riêng từng video:

```text
data/processed/<video_name>/frame_labels.csv
```

File có các cột:

```csv
video,frame_path,timestamp,timestamp_seconds,label
```

`label = 1` nếu timestamp nằm trong một trong hai đoạn lineup, ngược lại là
`0`. Các khoảng dùng quy ước `[start, end)`: tính `start` và không tính `end`.

Nên giữ `--only-ground-truth-videos` khi tạo dữ liệu train để video chưa có
ground truth không bị coi nhầm là toàn bộ `label = 0`.

### 3. Tạo dataset train/validation/test

```bash
python src/lineup/build_dataset_index.py --verify-files
```

Kết quả:

```text
data/processed/all_frame_labels.csv
```

Dataset được chia theo video, không chia ngẫu nhiên từng frame, nên một video
chỉ xuất hiện trong đúng một tập. Mặc định validation và test cùng chiếm khoảng
15% số video. Có thể chỉ định video giữ lại:

```bash
python src/lineup/build_dataset_index.py \
  --val-video "match_val" \
  --test-video "match_test" \
  --verify-files
```

Giá trị truyền vào là `video_id`, tức tên thư mục con trong `data/processed/`.

### 4. Train MobileNetV3-Small

```bash
python src/lineup/train_mobilenet.py
```

Model sử dụng pretrained ImageNet và train theo hai giai đoạn:

1. Đóng băng feature extractor và train classifier.
2. Mở các block cuối để fine-tune.

Validation được dùng để chọn checkpoint, threshold và cửa sổ temporal
smoothing. Test chỉ được đánh giá sau khi các lựa chọn này hoàn tất.

Kết quả:

```text
outputs/predictions/mobilenet_v3_small_lineup.pt
outputs/predictions/mobilenet_v3_small_metrics.csv
outputs/predictions/mobilenet_v3_small_history.csv
outputs/predictions/mobilenet_v3_small_predictions.csv
```

Có thể chọn thiết bị thủ công:

```bash
python src/lineup/train_mobilenet.py --device mps
python src/lineup/train_mobilenet.py --device cuda
python src/lineup/train_mobilenet.py --device cpu
```

Mặc định `--device auto` ưu tiên CUDA, sau đó MPS, cuối cùng CPU.

## Dự đoán video mới

### 1. Tách frame của video

Đặt video mới vào `data/raw_videos/`, sau đó chạy:

```bash
python src/lineup/extract_frames.py --video "new_match.mp4"
```

Lần extract này cập nhật `data/processed/extracted_frames.csv`, là input mặc
định của script inference.

### 2. Chạy MobileNetV3 inference

```bash
python src/lineup/predict_mobilenet.py
```

Script tự đọc từ checkpoint:

- kích thước ảnh;
- raw threshold;
- smoothing window;
- threshold sau smoothing.

Kết quả được lưu tại:

```text
outputs/predictions/mobilenet_v3_small_inference.csv
```

Các cột kết quả quan trọng:

```text
score             # xác suất lineup thô của từng frame
raw_pred_label    # nhãn dùng raw threshold
smoothed_score    # score sau temporal smoothing
pred_label        # nhãn cuối dùng threshold đã chọn trên validation
```

Có thể truyền input hoặc output khác:

```bash
python src/lineup/predict_mobilenet.py \
  --input-csv data/processed/new_match/extracted_frames.csv \
  --output-csv outputs/predictions/new_match_predictions.csv
```

### 3. Gom prediction thành đoạn lineup

```bash
python src/lineup/aggregate.py \
  --merge-gap-seconds 6 \
  --min-duration-seconds 8
```

Mặc định `aggregate.py` dùng `pred_label`, tức nhãn đã áp dụng smoothing và
threshold lưu trong checkpoint. Kết quả:

```text
outputs/predictions/lineup_segments.csv
```

Nếu muốn thử threshold khác, script sẽ ưu tiên `smoothed_score`:

```bash
python src/lineup/aggregate.py --threshold 0.6
```

### 4. Tách đoạn lineup ở 2 FPS và chạy OCR

Script đọc `lineup_segments.csv`, quay lại video gốc và chỉ tách frame trong
các khoảng lineup. MobileNet vẫn chạy ở `0,5 FPS`; `2 FPS` chỉ dùng cho OCR:

```bash
.venv/bin/python src/ocr/run_lineup_ocr.py
```

Có thể chọn file segment cụ thể:

```bash
.venv/bin/python src/ocr/run_lineup_ocr.py \
  --segments-csv outputs/predictions/premier_match_01_segments.csv
```

Mặc định script dùng `2 FPS`, model CPU nhẹ và ngưỡng OCR `0.80`. Kết quả:

```text
data/ocr_frames/<video>/segment_01/*.jpg
outputs/predictions/ocr_frames.csv
outputs/predictions/ocr_raw_detections.csv
```

`ocr_frames.csv` chứa timestamp của từng frame. `ocr_raw_detections.csv` chứa
text, confidence, bounding box và tọa độ tâm chuẩn hóa để bước sau ghép tên với
số áo xuyên nhiều frame.

Để chỉ kiểm tra việc tách frame mà chưa chạy OCR:

```bash
.venv/bin/python src/ocr/run_lineup_ocr.py --extract-only
```

### 5. Gộp đa frame và ghép tên với số áo

Sau khi có `ocr_raw_detections.csv`, chạy:

```bash
.venv/bin/python src/ocr/resolve_lineup.py
```

Resolver không dùng roster hoặc API. Script tự:

1. Đọc dòng dạng danh sách, kể cả khi OCR gộp thành
   `99 DONNARUMMA`.
2. Tìm các frame hiển thị sơ đồ đội hình dựa trên số lượng và vị trí số áo.
3. Gom số áo và tên theo cùng một slot xuyên nhiều frame.
4. Nếu số áo quá nhỏ hoặc bị dính theo cột, chỉ OCR lại các ô số trên tối đa
   ba frame đại diện. Không OCR lại toàn bộ clip.
5. Loại vùng danh sách dự bị để số áo dự bị không bị ghép nhầm vào đội hình.
6. Tách nhiều đội hình nếu hai đội nằm trong cùng một segment.
7. Dùng đồng thuận đa frame để sửa biến thể OCR và ghép tên đầy đủ.

Kết quả:

```text
outputs/predictions/resolved_lineups.csv
outputs/predictions/resolved_lineups_diagnostics.csv
```

Các cột chính:

```csv
lineup_index,resolution_method,shirt_number,formation_label,player_name,pair_confidence
1,formation,1,Alisson,Alisson Becker,0.997292
1,formation,17,Jones,Curtis Jones,0.999953
```

`resolution_method` cho biết kết quả đến từ `table`, `table+local_ocr`,
`formation` hay `formation+local_ocr`. File diagnostics có một dòng cho mỗi
segment với trạng thái `resolved`/`unresolved` và nguyên nhân. Resolver chỉ
xuất lineup khi tìm đủ số cầu thủ yêu cầu; nó không tự đoán cho đủ 11.

Nếu chỉ muốn kiểm tra kết quả OCR toàn frame và tắt lượt OCR cục bộ:

```bash
.venv/bin/python src/ocr/resolve_lineup.py --disable-local-ocr
```

Lượt OCR cục bộ là cần thiết với các kiểu đồ họa có số rất nhỏ trên áo hoặc
cột số sát nhau. Nó vẫn chạy hoàn toàn offline sau khi model đã được cache.

### 6. Xuất các đoạn lineup thành clip MP4

Mỗi dòng trong `lineup_segments.csv` được xuất thành một file MP4 riêng:

```bash
python src/lineup/export_clips.py
```

Mặc định script đọc video nguồn từ `data/raw_videos/` và lưu clip vào
`outputs/clips/`. Điểm cắt được tái mã hóa bằng H.264/AAC để bám chính xác mốc
thời gian. Có thể chọn file segment khác, ví dụ:

```bash
python src/lineup/export_clips.py \
  --segments-csv outputs/predictions/premier_match_01_segments.csv
```

Nếu ưu tiên tốc độ và chấp nhận điểm cắt có thể lệch theo keyframe:

```bash
python src/lineup/export_clips.py --copy-codecs
```

Script không ghi đè clip đã có. Truyền `--overwrite` khi muốn thay thế chúng.
CSV đầu vào cần có cột `video` và cặp `start_seconds`/`end_seconds` hoặc
`start`/`end`.

### 7. Đánh giá theo đoạn trên tập test

Đánh giá xem model có tìm đủ hai đoạn lineup, lệch mốc bao nhiêu và có nhận
nhầm khoảng giới thiệu trọng tài/bắt tay hay không:

```bash
python src/lineup/evaluate_segments.py
```

Script mặc định dùng `pred_label` của các dòng `split=test`, ghép frame thành
khoảng nửa mở `[start, end)` và tính detection tại temporal IoU từ `0.5`. Ba
file kết quả được lưu tại:

```text
outputs/predictions/mobilenet_v3_small_segment_matches.csv
outputs/predictions/mobilenet_v3_small_segment_metrics_by_video.csv
outputs/predictions/mobilenet_v3_small_segment_metrics.csv
```

Các chỉ số chính gồm segment precision/recall/F1, temporal IoU, sai số mốc
đầu/cuối và tỷ lệ khoảng trọng tài/bắt tay vẫn được giữ là label `0`.

Để đánh giá cùng rule hậu xử lý sẽ dùng khi vận hành:

```bash
python src/lineup/evaluate_segments.py \
  --merge-gap-seconds 6 \
  --min-duration-seconds 8
```

Mốc bỏ đoạn ngắn `8` giây được chọn từ validation hiện tại. Cần hiệu chỉnh lại
trên validation nếu dataset thay đổi, rồi mới dùng test để báo cáo kết quả cuối
cùng.

## Ghi chú

- Bước phát hiện đoạn lineup sử dụng MobileNetV3-Small ở `0,5 FPS`.
  PaddleOCR được tách thành môi trường riêng và chỉ chạy ở `2 FPS` trong những
  đoạn đã phát hiện.
- `resolve_lineup.py` ghép kết quả theo thời gian và tọa độ, không tra cứu
  roster trên mạng.
- Resolver hiện hỗ trợ danh sách số-tên cùng hàng, dòng OCR gộp
  `số + tên`, sơ đồ số-tên cùng slot và sơ đồ được hiện dần qua nhiều frame.
  Kiểu đồ họa chỉ có số trên một mini-pitch nhưng tên hiện riêng ở vùng khác
  cần thêm bộ theo dõi slot đang được highlight; diagnostics sẽ giữ các đoạn
  này ở trạng thái `unresolved` thay vì ghép đoán.
- Temporal smoothing là centered mean, phù hợp xử lý video offline vì sử dụng
  cả frame trước và frame sau.
- Tên frame không chứa timestamp thật. `build_frame_labels.py` luôn dùng
  metadata do `extract_frames.py` tạo.
- Nếu thiếu video, frame, cột CSV hoặc checkpoint, script sẽ báo lỗi rõ ràng.
