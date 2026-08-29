# Lineup detection không băm nhỏ transcript

Luồng mặc định dùng transcript 60 giây đã tạo bởi `qwen3-asr-flash`, gọi model
text đúng một lần để tìm tối đa hai khoảng lineup thô, sau đó dùng
PySceneDetect để căn vào graphic thật. Không tách candidate thành audio 5–10
giây và không chạy ASR lần hai.

1. `qwen3-asr-flash` tạo transcript theo chunk 60 giây.
2. `qwen-plus` nhận toàn bộ transcript dưới dạng text trong một
   request và trả về exact start/end anchor cho mỗi đội.
3. Code xác minh anchor có thật trong evidence rồi nội suy vị trí ký tự bên
   trong chunk để tạo timestamp thô.
4. PySceneDetect tìm hard cut và thời điểm bắt đầu dissolve trong vùng lineup.
   Vùng scan được mở rộng quanh anchor để không bỏ mất graphic xuất hiện trước
   lời đọc hoặc còn hiển thị sau khi lời đọc đã kết thúc.
5. Logic thích ứng ưu tiên một scene ổn định dài 18–45 giây giữa hai cut. Nếu
   không có scene dài nhưng có nhiều cut dày đặc, hệ thống tự ghép toàn bộ
   slideshow và tách hai graphic liên tiếp theo biên giữa hai đội.
6. Cut đóng có thể đi trước hoặc sau phần bình luận tối đa 18 giây; vì vậy
   ACLE_01 có thể căn mốc Buriram thô `540–580` thành `541.967–568.833`.

Qwen Plus chỉ nhận text trong luồng này; không gửi frame, ảnh hoặc video cho
model. PySceneDetect là phần duy nhất đọc hình ảnh video. Detector gọi endpoint
OpenAI-compatible của DashScope Singapore ở non-thinking JSON mode.

Chạy detector:

```bash
.venv/bin/python src/lineup/detect_from_transcript.py \
  --transcript outputs/ACLE/ACLE_01/predictions/transcript/transcript.jsonl \
  --overwrite
```

Khi không truyền đường dẫn output, pipeline tự phản chiếu cấu trúc thư mục của
video gốc. Ví dụ `data/raw_data/ACLE/ACLE_01.mp4` tạo toàn bộ kết quả trong
`outputs/ACLE/ACLE_01/`: transcript ở `predictions/transcript`, kết quả
lineup ở `predictions/lineup`, và clip cuối ở `clips/lineup`. Các video nằm
ngoài `data/raw_data` dùng `outputs/<video-stem>`.

`source_video` mặc định được đọc từ `metadata.json` cạnh transcript. Có thể ghi
đè bằng `--source-video`. Model mặc định là
`qwen-plus`; có thể đổi bằng `--model`, `QWEN_LINEUP_MODEL` hoặc
`QWEN_TEXT_MODEL`. Endpoint có thể đổi bằng `--base-url` hoặc
`QWEN_LINEUP_BASE_URL`. Key ưu tiên `QWEN_LINEUP_API_KEY`, sau đó dùng
`DASHSCOPE_API_KEY`.

Các output nằm cạnh `lineup_segments.csv`:

- `qwen_detection_raw_response.json`: JSON và exact anchors từ model, kèm chẩn
  đoán scene refinement.
- `detection_metadata.json`: model, số chunk transcript, trạng thái complete,
  số request, token input/output, thời gian Qwen và xác nhận
  `second_pass_request_count=0`.

Các tùy chọn PyScene chính:

- `--scene-threshold`: ngưỡng ContentDetector, mặc định 27.
- `--scene-min-length-frames`: độ dài scene tối thiểu, mặc định 15 frame.
- `--scene-snap-radius`: bán kính tìm cut mở/gần biên, mặc định 4 giây.
- `--scene-slideshow`: ép cách ghép slideshow. Mặc định code đã tự phát hiện
  khi cả hai đội không có scene ổn định và vùng lineup có nhiều cut liên tiếp.
- `--no-scene-snap`: giữ timestamp thô từ transcript để so sánh.

Nếu model trả sai JSON, evidence ID không tồn tại hoặc anchor không được sao
chép nguyên văn, kết quả bị từ chối thay vì đoán timestamp.
