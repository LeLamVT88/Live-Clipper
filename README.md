# Live Clipper - Football Lineup Detection

Project dùng MobileNetV3-Small để phát hiện các frame hiển thị đội hình trong video
bóng đá, sau đó gom các frame được dự đoán thành đoạn thời gian lineup.

## Cấu trúc thư mục

```text
live-clipper/
├── data/
│   ├── raw_videos/        # video gốc
│   ├── frames/            # frame tách từ video
│   ├── processed/         # metadata và dataset CSV
│   └── ground_truth.csv   # khoảng lineup của Đội 1 và Đội 2
├── outputs/
│   ├── clips/             # clip ground truth để kiểm tra
│   └── predictions/       # checkpoint, metrics và kết quả dự đoán
├── src/
│   ├── utils.py
│   ├── extract_frames.py
│   ├── build_frame_labels.py
│   ├── build_dataset_index.py
│   ├── train_mobilenet.py
│   ├── predict_mobilenet.py
│   ├── aggregate.py
│   └── cut_clip.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Cài đặt

Tạo môi trường Python và cài package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`cut_clip.py` cần FFmpeg có sẵn trong `PATH`:

```bash
ffmpeg -version
```

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

Mặc định script lấy 1 frame/giây trong 10 phút đầu của tất cả video:

```bash
python src/extract_frames.py --fps 1
```

Để lấy 2 giây một frame:

```bash
python src/extract_frames.py --fps 0.5
```

Có thể chọn một hoặc nhiều video cụ thể:

```bash
python src/extract_frames.py --video "match1.mp4" --fps 1
python src/extract_frames.py --video "match1.mp4" --video "match2.mp4" --fps 1
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
python src/build_frame_labels.py --only-ground-truth-videos
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
python src/build_dataset_index.py --verify-files
```

Kết quả:

```text
data/processed/all_frame_labels.csv
```

Dataset được chia theo video, không chia ngẫu nhiên từng frame, nên một video
chỉ xuất hiện trong đúng một tập. Mặc định validation và test cùng chiếm khoảng
15% số video. Có thể chỉ định video giữ lại:

```bash
python src/build_dataset_index.py \
  --val-video "match_val" \
  --test-video "match_test" \
  --verify-files
```

Giá trị truyền vào là `video_id`, tức tên thư mục con trong `data/processed/`.

### 4. Train MobileNetV3-Small

```bash
python src/train_mobilenet.py
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
python src/train_mobilenet.py --device mps
python src/train_mobilenet.py --device cuda
python src/train_mobilenet.py --device cpu
```

Mặc định `--device auto` ưu tiên CUDA, sau đó MPS, cuối cùng CPU.

## Dự đoán video mới

### 1. Tách frame của video

Đặt video mới vào `data/raw_videos/`, sau đó chạy:

```bash
python src/extract_frames.py --video "new_match.mp4" --fps 1
```

Lần extract này cập nhật `data/processed/extracted_frames.csv`, là input mặc
định của script inference.

### 2. Chạy MobileNetV3 inference

```bash
python src/predict_mobilenet.py
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
python src/predict_mobilenet.py \
  --input-csv data/processed/new_match/extracted_frames.csv \
  --output-csv outputs/predictions/new_match_predictions.csv
```

### 3. Gom prediction thành đoạn lineup

```bash
python src/aggregate.py \
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
python src/aggregate.py --threshold 0.6
```

### 4. Đánh giá theo đoạn trên tập test

Đánh giá xem model có tìm đủ hai đoạn lineup, lệch mốc bao nhiêu và có nhận
nhầm khoảng giới thiệu trọng tài/bắt tay hay không:

```bash
python src/evaluate_segments.py
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
python src/evaluate_segments.py \
  --merge-gap-seconds 6 \
  --min-duration-seconds 8
```

Mốc bỏ đoạn ngắn `8` giây được chọn từ validation hiện tại. Cần hiệu chỉnh lại
trên validation nếu dataset thay đổi, rồi mới dùng test để báo cáo kết quả cuối
cùng.

## Kiểm tra ground truth bằng clip

Cắt hai đoạn lineup thật của mỗi video:

```bash
python src/cut_clip.py
```

Clip được lưu trong `outputs/clips/`. Mặc định FFmpeg dùng stream copy để cắt
nhanh. Dùng re-encode nếu cần điểm cắt chính xác hơn:

```bash
python src/cut_clip.py --reencode
```

## Ghi chú

- Pipeline hiện chỉ sử dụng MobileNetV3-Small; không còn baseline logistic
  regression hoặc bước OCR.
- Temporal smoothing là centered mean, phù hợp xử lý video offline vì sử dụng
  cả frame trước và frame sau.
- Tên frame không chứa timestamp thật. `build_frame_labels.py` luôn dùng
  metadata do `extract_frames.py` tạo.
- Nếu thiếu video, frame, cột CSV hoặc checkpoint, script sẽ báo lỗi rõ ràng.
