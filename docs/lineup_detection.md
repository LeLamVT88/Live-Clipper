# Phát hiện và cắt graphic lineup

Pipeline chỉ xử lý **600 giây đầu của video**. Video ngắn hơn thì dừng ở cuối
nguồn; video dài hơn cũng không có chế độ quét vượt quá mốc 600 giây.

## Luồng xử lý

1. Audio trong 600 giây đầu được chia tối đa 10 chunk, mỗi chunk 60 giây.
2. `qwen3-asr-flash` tạo một transcript thô cho từng chunk.
3. `qwen-plus` đọc toàn bộ transcript trong một request, trả tối đa hai khoảng
   lineup thô cùng exact text anchor. Timestamp này chỉ dùng để định vị vùng
   tìm kiếm; lệch nhiều giây vẫn chấp nhận được.
4. PySceneDetect tìm cut riêng trong từng vùng thô và bỏ qua khoảng trống giữa
   các candidate cách xa nhau. Mỗi scene dùng một frame đại diện; scene quá dài
   được chia thành visual unit tối đa 3 giây để không bỏ sót graphic overlay
   xuất hiện mà không tạo hard cut.
5. `PP-OCRv6_tiny_det` đi từ giữa khoảng thô sang hai phía và chỉ đo mật độ
   text. `PP-OCRv6_tiny_rec` chỉ chạy trên các unit có nhiều text và một số
   unit lân cận. OCR chỉ cần xác nhận cấu trúc lineup như formation, số áo,
   danh sách tên hoặc các từ `starting XI`/`substitutes`; không resolve chi
   tiết tên cầu thủ.
6. Các unit OCR dương tính tạo thành một graphic block. PySceneDetect căn chỉnh
   nhỏ hai đầu block vào cut gần nhất rồi timestamp này được dùng để xuất clip.

Luồng local giữ nguyên `team_name`, transcript evidence và reason từ Qwen, chỉ
thay biên thô bằng biên hình ảnh. Nhờ vậy graphic có thể xuất hiện trước khi
bình luận viên đọc, commentary có thể kết thúc trước/sau graphic, và sai số nội
suy timestamp trong chunk 60 giây không quyết định clip cuối.

Graphic ngắn dưới 8 giây vẫn được giữ để không làm mất kết quả, nhưng metadata
gắn `visual_lineup_boundary_may_be_incomplete`; đây thường là slideshow bị OCR
dừng ở slide đầu và cần nhánh nối slide trước khi coi là hoàn tất.

## Nhánh cứu hộ hiếm

Sparse scan 0–600 giây chỉ chạy khi:

- Qwen không trả đủ lineup dự kiến; hoặc
- OCR local không xác nhận được một candidate thô.

Đây **không phải full OCR 600 giây**. Tiny detector vẫn chỉ đọc một frame đại
diện mỗi scene/visual unit (tối đa khoảng 200 frame nếu video không có cut), sau
đó recognition chỉ đọc tối đa 40 frame được shortlist. Vì vậy nhánh này xử lý
được Qwen lệch hẳn hoặc lineup không có lời bình mà không OCR liên tục toàn bộ
18.000 frame của video 30 FPS.

Khi fallback chạy, OCR ưu tiên khớp team title với tên đội Qwen đã trả trước
khi xét khoảng cách thời gian. Event fallback trùng event local không được dùng
lại. Nếu không có transcript, OCR chỉ lấy team title ngắn trên graphic; khi
không đọc được title thì CSV mới dùng tên `Visual lineup N`. Cả hai trường hợp
đều gắn cờ review nhưng clip luôn nằm trong `clips/lineup`.

Các nguyên nhân thường làm candidate Qwen lệch hẳn gồm transcript bỏ sót/nhận
sai câu dẫn, một chunk chứa cả nghi thức và lineup, bình luận đọc đội hình trước
hoặc sau graphic, hoặc timestamp ký tự nội suy trong chunk không phản ánh nhịp
nói thật. PySceneDetect một mình cũng không thể biết scene nào là lineup; OCR là
tín hiệu semantic dùng để chọn đúng graphic, còn scene cut chỉ tinh chỉnh biên.

## Cài đặt

Qwen/PySceneDetect và PaddleOCR vẫn chạy trong hai môi trường Python riêng để
không xung đột dependency, nhưng cả hai đều cài từ cùng `requirements.txt`.
Các marker phiên bản trong file tự bỏ qua Paddle ở môi trường Python 3.14:

```bash
.venv/bin/pip install -r requirements.txt
python3.11 -m venv .venv-ocr
.venv-ocr/bin/pip install -r requirements.txt
```

Model mặc định là `PP-OCRv6_tiny_det` và `PP-OCRv6_tiny_rec`, chạy CPU và được
cache trong `.cache/paddlex`.

## Chạy pipeline

Tạo transcript 10 chunk x 60 giây:

```bash
.venv/bin/python src/transcript/transcribe_video.py \
  --video data/raw_data/<giai>/<video>.mp4
```

Phát hiện lineup và căn biên bằng OCR/PySceneDetect:

```bash
.venv/bin/python src/lineup/detect_from_transcript.py \
  --transcript outputs/<giai>/<video>/predictions/transcript/transcript.jsonl \
  --overwrite
```

`source_video` mặc định lấy từ `metadata.json` cạnh transcript. Có thể ghi đè
bằng `--source-video`. Nếu môi trường OCR nằm ở nơi khác, dùng `--ocr-python`
hoặc biến `LINEUP_OCR_PYTHON`. `--no-ocr` chỉ dành cho debug/so sánh, khi đó
CSV giữ timestamp thô của Qwen.

Các option PySceneDetect còn hữu ích:

- `--scene-threshold`: ngưỡng ContentDetector, mặc định 27.
- `--scene-min-length-seconds`: scene tối thiểu, mặc định 0,5 giây.
- `--scene-min-length-frames`: ghi đè bằng số frame cố định khi cần.
- `--scene-snap-radius`: khoảng tinh chỉnh cuối từ biên OCR đến cut, mặc định
  4 giây.

Các output nằm trong `predictions/lineup`:

- `lineup_segments.csv`: interval cuối dùng để export clip.
- `qwen_detection_raw_response.json`: Qwen response và chẩn đoán visual/OCR.
- `detection_metadata.json`: model, số request, fallback có chạy hay không,
  trạng thái completeness và lý do cần review.

Mỗi task trong `visual_refinement.local_ocr.tasks` hoặc
`visual_refinement.fallback_ocr.tasks` lưu thêm `selected_sample_seconds`,
`positive_sample_seconds` và mảng `units`. Mỗi unit ghi khoảng thời gian, frame
đại diện, mật độ text, trạng thái được chọn cho recognition và kết quả OCR nếu
có. Diagnostics chỉ lưu JSON, không lưu ảnh frame mặc định.
