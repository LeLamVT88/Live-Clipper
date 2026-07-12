# Live Clipper - Football Lineup Detection

Project này chuẩn bị dataset frame-level từ video bóng đá để dùng cho bước train/detect đoạn lineup sau này.

## Cấu trúc thư mục

```text
live-clipper/
├── data/
│   ├── raw_videos/        # chứa video gốc
│   ├── frames/            # frame tách từ video
│   ├── crops/             # vùng lineup đã crop
│   ├── processed/         # file csv đã xử lý
│   └── ground_truth.csv   # start/end lineup thật
├── outputs/
│   ├── clips/             # clip lineup đã cắt
│   ├── predictions/       # kết quả model dự đoán
│   └── logs/
├── src/
│   ├── utils.py
│   ├── extract_frames.py
│   ├── build_frame_labels.py
│   ├── detect_lineup.py
│   ├── aggregate.py
│   ├── localize_crop.py
│   ├── ocr_check.py
│   └── cut_clip.py
├── .env
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

`clip_video.py` cần FFmpeg có sẵn trong `PATH`. Kiểm tra bằng:

```bash
ffmpeg -version
```

## Chuẩn bị ground truth

Đặt video gốc vào `data/raw_videos/`, ví dụ:

```text
data/raw_videos/match1.mp4
data/raw_videos/match2.mp4
```

Điền `data/ground_truth.csv` theo format:

```csv
video,start,end
match1.mp4,00:03:12,00:04:05
match2.mp4,00:02:40,00:03:25
```

`start` và `end` dùng format `HH:MM:SS`. Script cũng chấp nhận `HH:MM:SS.mmm`.

## Thứ tự chạy

1. Tách frame từ video, mặc định 1 frame/giây:

```bash
python src/extract_frames.py --fps 1
```

Nếu máy của bạn không có alias `python`, dùng `python3` thay thế trong các lệnh bên dưới.

Nếu chỉ muốn lấy frame trong 1 giờ đầu video, ví dụ 2 giây lấy 1 frame:

```bash
python src/extract_frames.py --fps 0.5 --duration 01:00:00
```

Nếu chỉ muốn tách một video cụ thể trong `data/raw_videos/`, truyền đúng tên file:

```bash
python src/extract_frames.py --video "match1.mp4" --fps 0.5 --duration 01:00:00
```

Có thể truyền `--video` nhiều lần để tách vài video đã chọn.

Script sẽ lưu frame vào:

```text
data/frames/<video_name>/frame_000001.jpg
```

Đồng thời tạo metadata riêng cho từng video:

```text
data/processed/<video_name>/extracted_frames.csv
```

Script vẫn ghi thêm `data/processed/extracted_frames.csv` như file tổng hợp của lần extract gần nhất.

2. Gán nhãn cho từng frame dựa trên `data/ground_truth.csv`:

```bash
python src/build_frame_labels.py
```

Nếu trong `data/raw_videos/` có video chưa điền ground truth, tránh đưa chúng vào dataset train bằng:

```bash
python src/build_frame_labels.py --only-ground-truth-videos
```

Kết quả riêng từng video được lưu tại:

```text
data/processed/<video_name>/frame_labels.csv
```

Script vẫn ghi thêm `data/processed/frame_labels.csv` như file tổng hợp của lần build label gần nhất.

File này có các cột:

```csv
video,frame_path,timestamp,timestamp_seconds,label
```

`label = 1` nếu timestamp của frame nằm trong đoạn lineup thật, ngược lại `label = 0`.

3. Cắt clip lineup thật để kiểm tra ground truth:

```bash
python src/cut_clip.py
```

Clip được lưu vào `outputs/clips/`.

Nếu muốn cắt chính xác hơn thay vì stream copy nhanh, có thể dùng:

```bash
python src/cut_clip.py --reencode
```

## Các script cho bước sau

Các file dưới đây đã được tạo sẵn theo cấu trúc pipeline, chưa dùng YOLO/OCR thật:

```bash
python src/detect_lineup.py
python src/aggregate.py
python src/localize_crop.py --x 0 --y 0 --w 640 --h 120 --only-lineup
python src/ocr_check.py
```

## Ghi chú

- Bước này chưa dùng OCR, LLM, YOLO hay model detection.
- Nếu thiếu video, sai timestamp, hoặc thiếu cột CSV, script sẽ báo lỗi rõ ràng.
- `build_frame_labels.py` dùng metadata từ `extract_frames.py`, vì tên frame `frame_000001.jpg` không tự chứa timestamp.
