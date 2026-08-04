# Live Clipper - Football Lineup Detection

Project dùng MobileNetV3-Small ở `0,5 FPS` để phát hiện đoạn đội hình trong
video bóng đá. Trong mỗi segment, project tách frame ở `2 FPS`, dùng scout OCR
ở `0,5 FPS` để tìm bảng lineup hoàn chỉnh rồi OCR thích ứng 3 frame, 7 frame
hoặc toàn segment sau khi đã bỏ vùng `SUBSTITUTES`.

Entrypoint vận hành là `src/run_full_pipeline.py`. Script tự chạy
tuần tự MobileNet, tách hai segment, xuất clip và OCR.

## Cấu trúc thư mục

```text
live-clipper/
├── data/
│   ├── raw_videos/       # video dùng để train/evaluate
│   ├── data_test/        # video mới, có thể chia theo giải đấu
│   │   ├── premier/
│   │   ├── seriesA/
│   │   ├── uefa/
│   │   └── worldcup/
│   ├── frames/           # frame phục vụ train cũ
│   ├── processed/        # metadata, label và dataset train
│   └── ground_truth.csv
├── outputs/
│   ├── predictions/
│   │   └── mobilenet/     # checkpoint và kết quả train/evaluate dùng chung
│   └── runs/
│       └── <video>/       # toàn bộ output của riêng một video
│           ├── clips/
│           ├── frames/
│           └── predictions/
│               ├── mobilenet/
│               └── ocr/
├── src/
│   ├── run_full_pipeline.py # entrypoint đầy đủ cho video mới
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
│       ├── run_pipeline.py # entrypoint nội bộ của riêng bước OCR
│       ├── config.py       # cấu hình và tham số CLI
│       ├── workflow.py     # điều phối ba tầng fallback
│       ├── attempts.py     # chạy OCR/resolve một lần thử
│       ├── frames.py       # đọc segment và tách frame
│       ├── ocr_engine.py   # adapter PaddleOCR
│       ├── schema.py       # định dạng cột CSV dùng chung
│       ├── selector.py     # chấm điểm và chọn frame lineup
│       ├── selection_io.py # crop frame và ghi diagnostics
│       └── resolver/
│           ├── common.py
│           ├── layout.py
│           ├── table.py
│           ├── table_refinement.py
│           ├── formation.py
│           ├── formation_refinement.py
│           ├── player_names.py
│           ├── local_models.py
│           ├── quality.py
│           └── pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Cài đặt

Pipeline dùng `.venv` cho tách frame, MobileNet và xuất clip; riêng OCR dùng
`.venv-ocr` để tránh xung đột phiên bản PaddlePaddle. Python 3.11 tương thích
với cả hai nhóm thư viện:

```bash
python3.11 -m venv .venv
python3.11 -m venv .venv-ocr
.venv/bin/python -m pip install -r requirements.txt
.venv-ocr/bin/python -m pip install -r requirements.txt
```

Chạy kiểm tra trên frame lineup mẫu:

```bash
.venv-ocr/bin/python src/ocr/ocr_smoke_test.py
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
outputs/predictions/mobilenet/mobilenet_v3_small_lineup.pt
outputs/predictions/mobilenet/mobilenet_v3_small_metrics.csv
outputs/predictions/mobilenet/mobilenet_v3_small_history.csv
outputs/predictions/mobilenet/mobilenet_v3_small_predictions.csv
```

Có thể chọn thiết bị thủ công:

```bash
python src/lineup/train_mobilenet.py --device mps
python src/lineup/train_mobilenet.py --device cuda
python src/lineup/train_mobilenet.py --device cpu
```

Mặc định `--device auto` ưu tiên CUDA, sau đó MPS, cuối cùng CPU.

## Chạy pipeline cho video mới

Đặt video vào một thư mục con của `data/data_test/`, ví dụ:

```text
data/data_test/uefa/uefa_match_01.mp4
data/data_test/worldcup/worldcup_match_01.mp4
```

Chạy một video bằng tên file; pipeline tự tìm trong các thư mục con:

```bash
.venv/bin/python src/run_full_pipeline.py --video "uefa_match_01.mp4"
```

Chạy nhiều video trong cùng một lệnh bằng cách lặp lại `--video`:

```bash
.venv/bin/python src/run_full_pipeline.py \
  --video "worldcup_match_01.mp4" \
  --video "worldcup_match_02.mp4"
```

Nếu bỏ `--video`, pipeline xử lý tuần tự toàn bộ video trong `data/data_test/`.
Mặc định mỗi video được đọc trong 10 phút đầu. Có thể chọn khoảng khác:

```bash
.venv/bin/python src/run_full_pipeline.py \
  --video "uefa_match_01.mp4" \
  --start 00:01:00 \
  --end 00:08:00
```

Pipeline chạy năm bước:

1. Tách frame `0,5 FPS` cho MobileNet.
2. Chạy MobileNetV3-Small với threshold và smoothing lưu trong checkpoint.
3. Gom prediction và bắt buộc hai segment/video. Nếu MobileNet nối hai màn
   hình lineup thành một đoạn, hậu xử lý cắt tại đáy score cục bộ hợp lệ.
4. Xuất mỗi segment thành một clip MP4 bằng FFmpeg.
5. Dùng `.venv-ocr` để chạy scout OCR, OCR thích ứng và resolver.

Nếu một bước thất bại, pipeline dừng tại bước đó. Exit code `2` nghĩa là OCR
đã chạy xong nhưng còn segment không vượt quality gate.

### Cách OCR xử lý mỗi segment

Mặc định pipeline:

1. Scout OCR toàn frame ở `0,5 FPS`.
2. Nhận diện layout sơ đồ hoặc danh sách số-tên.
3. Nếu có `SUBSTITUTES`, tự xác định bảng nằm bên trái/phải và crop phía đội
   hình đối diện.
4. Tầng 1 OCR cửa sổ ba frame quanh scout frame tốt nhất rồi resolve.
5. Quality gate chỉ chấp nhận kết quả đủ 11 người, 11 số áo khác nhau, tên
   không rỗng/không chứa token giao diện và mọi cặp số-tên đạt confidence
   tối thiểu `0.80`.
6. Segment không đạt được thử lại ở tầng 2 với cửa sổ bảy frame quanh cùng
   scout frame. Ba frame cũ được tái sử dụng, nên chỉ bốn frame mới phải chạy
   OCR.
7. Nếu vẫn không đạt, tầng 3 OCR toàn bộ frame `2 FPS` của riêng segment đó.
   Segment scout không tìm được bảng lineup cũng đi thẳng tới tầng này.

Pipeline dùng model CPU nhẹ và ngưỡng OCR `0.80`.

### Output theo từng video

Mọi file sinh ra khi vận hành được gom theo tên video, không ghi vào
`outputs/clips/` hoặc thư mục prediction dùng chung:

Ví dụ `worldcup_match_01.mp4` sẽ tạo thư mục
`outputs/runs/worldcup_match_01/`:

```text
outputs/runs/<video>/
├── clips/
│   ├── <video>_lineup_01_<start>_to_<end>.mp4
│   └── <video>_lineup_02_<start>_to_<end>.mp4
├── frames/
│   ├── mobilenet/<video>/*.jpg
│   ├── ocr/<video>/segment_01/*.jpg
│   └── ocr_selected/<video>/segment_01/*.jpg
└── predictions/
    ├── mobilenet/
    │   ├── extracted_frames.csv
    │   ├── mobilenet_v3_small_inference.csv
    │   └── lineup_segments.csv
    └── ocr/
        ├── ocr_frames.csv
        ├── ocr_scout_detections.csv
        ├── ocr_selected_frames.csv
        ├── ocr_frame_selection_diagnostics.csv
        ├── ocr_raw_detections.csv
        ├── pipeline_attempts.csv
        ├── resolved_lineups.csv
        └── resolved_lineups_diagnostics.csv
```

Ba file MobileNet lần lượt chứa metadata frame, score dự đoán từng frame và
hai khoảng lineup cuối cùng. Trong thư mục OCR, `ocr_frames.csv` chứa toàn bộ
timestamp `2 FPS`; `ocr_frame_selection_diagnostics.csv` ghi layout, vùng crop,
frame được chọn và trạng thái fallback. `ocr_raw_detections.csv` chứa kết quả
OCR của tầng cuối được dùng cho từng segment. `pipeline_attempts.csv` ghi số
frame, số frame OCR mới, trạng thái resolver và kết quả quality gate ở mỗi
tầng. `resolved_lineups.csv` là kết quả cuối và chỉ chứa lineup đã vượt quality
gate; file diagnostics bên cạnh giải thích segment đã resolve được hay chưa.

### Cách resolver ghép tên với số áo

`src/ocr/run_pipeline.py` được pipeline chính gọi nội bộ và tự chạy resolver;
không còn entrypoint resolve riêng.
Resolver không dùng roster hoặc API và tự:

1. Đọc dòng dạng danh sách, kể cả khi OCR gộp thành
   `99 DONNARUMMA`.
2. Tìm các frame hiển thị sơ đồ đội hình dựa trên các cặp số áo/tên đã đọc
   được; các tên còn lại trong cùng vùng sơ đồ được dùng làm anchor để OCR lại
   số ngay phía trên.
3. Nếu frame đồng thời có bảng `SUBSTITUTES` và sơ đồ xuất phát, tự xác định
   bảng dự bị nằm bên trái hay bên phải, loại đúng phía đó và chỉ xử lý phía
   sơ đồ.
4. Gom số áo và tên theo cùng một slot xuyên nhiều frame.
5. Nếu số áo quá nhỏ hoặc bị dính theo cột, chỉ OCR lại các ô số trên tối đa
   ba frame đại diện. Mỗi ô được nhận dạng trên ảnh màu, grayscale, CLAHE,
   Otsu và ảnh đảo màu; kết quả được chọn bằng đồng thuận thay vì tin một lần
   đọc duy nhất.
6. Loại vùng danh sách dự bị để số áo dự bị không bị ghép nhầm vào đội hình.
7. Tách nhiều đội hình nếu hai đội nằm trong cùng một segment.
8. Dùng đồng thuận đa frame để sửa biến thể OCR và ghép tên đầy đủ; loại các
   token giao diện như `TEAM FORMATION`.
9. Chỉ xuất lineup khi đủ 11 cầu thủ và 11 số áo là duy nhất. Nếu còn số trùng,
   diagnostics giữ segment ở trạng thái `unresolved`.

Kết quả:

```text
outputs/runs/<video>/predictions/ocr/resolved_lineups.csv
outputs/runs/<video>/predictions/ocr/resolved_lineups_diagnostics.csv
```

Các cột chính:

```csv
lineup_index,resolution_method,shirt_number,formation_label,player_name,pair_confidence
1,formation,23,F. MENDY,F. MENDY,0.997292
```

`resolution_method` cho biết kết quả đến từ `table`, `table+local_ocr`,
`formation` hay `formation+local_ocr`. File diagnostics có một dòng cho mỗi
segment với trạng thái `resolved`/`unresolved` và nguyên nhân. Resolver chỉ
xuất lineup khi tìm đủ số cầu thủ yêu cầu; nó không tự đoán cho đủ 11.

Lượt OCR cục bộ là cần thiết với các kiểu đồ họa có số rất nhỏ trên áo hoặc
cột số sát nhau. Nó vẫn chạy hoàn toàn offline sau khi model đã được cache.

Pipeline chính tự xuất clip ở bước 4 và ghi đè clip cùng tên khi chạy lại.
Điểm cắt được tái mã hóa bằng H.264/AAC để bám chính xác mốc thời gian.

## Đánh giá theo đoạn trên tập test

Đánh giá xem model có tìm đủ hai đoạn lineup, lệch mốc bao nhiêu và có nhận
nhầm khoảng giới thiệu trọng tài/bắt tay hay không:

```bash
python src/lineup/evaluate_segments.py
```

Script mặc định dùng `pred_label` của các dòng `split=test`, ghép frame thành
khoảng nửa mở `[start, end)` và tính detection tại temporal IoU từ `0.5`. Ba
file kết quả được lưu tại:

```text
outputs/predictions/mobilenet/mobilenet_v3_small_segment_matches.csv
outputs/predictions/mobilenet/mobilenet_v3_small_segment_metrics_by_video.csv
outputs/predictions/mobilenet/mobilenet_v3_small_segment_metrics.csv
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
  Trong segment đã phát hiện, scout OCR chạy ở `0,5 FPS`; OCR chi tiết thử lần
  lượt 3 frame, 7 frame rồi toàn segment `2 FPS` khi quality gate yêu cầu.
- `src/run_full_pipeline.py` là entrypoint vận hành duy nhất.
  `src/ocr/run_pipeline.py`, workflow, OCR và resolver là module nội bộ, không
  cần gọi riêng.
- Output vận hành luôn nằm trong `outputs/runs/<video>/`. Thư mục
  `outputs/clips/` cũ không còn được pipeline chính sử dụng.
- Hậu xử lý luôn yêu cầu đúng hai segment cho mỗi video; khi MobileNet nối hai
  màn hình lineup, đoạn được tách tại đáy score cục bộ hợp lệ.
- Resolver ghép kết quả theo thời gian và tọa độ, không tra cứu roster trên
  mạng.
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
