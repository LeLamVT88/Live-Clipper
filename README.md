# Live Clipper

Live Clipper là pipeline Python dùng để tự động tìm phần đồ họa **đội hình xuất phát** trong video một trận bóng đá, cắt phần video đó ra, sau đó đọc tên cầu thủ và số áo bằng OCR.

Kết quả chính của mỗi lần chạy gồm:

- các clip chứa phần giới thiệu đội hình;
- `players.json` chứa tên cầu thủ và số áo;
- `detection_result.json` chứa thời gian và độ tin cậy của các đoạn được phát hiện;
- các ảnh/CSV chẩn đoán phục vụ kiểm tra OCR (CSV chỉ được giữ khi dùng `--keep-diagnostics`).

Pipeline chạy cục bộ trên CPU, không cần API key. PaddleOCR sẽ tải model trong lần chạy đầu tiên nếu máy chưa có cache model.

## Luồng hoạt động

```text
Video trận đấu
      |
      v
Phát hiện cảnh (PySceneDetect)
      |
      v
Lấy mẫu frame theo độ dài từng cảnh
      |
      v
OCR sơ bộ + chấm điểm ngữ nghĩa/bố cục/chuyển động
      |
      v
Gộp các frame dương thành khoảng thời gian đội hình
      |
      v
Cắt clip bằng FFmpeg
      |
      v
OCR chi tiết theo 3 cấp: 3 frame -> 7 frame -> toàn clip 2 FPS
      |
      v
Ghép số áo với tên theo dạng bảng hoặc sơ đồ chiến thuật
      |
      v
Quality gate: đủ 2 đội x 11 cầu thủ, số áo duy nhất, tên hợp lệ
      |
      v
players.json
```

### 1. Tìm đoạn giới thiệu đội hình

Module `src/line_up/` thực hiện pha phát hiện:

1. `scene_detector.py` dùng `AdaptiveDetector` và `ContentDetector` của PySceneDetect để chia video theo các lần chuyển cảnh.
2. `sampler.py` lấy mẫu thích ứng. Cảnh ngắn lấy ít frame, cảnh dài lấy nhiều frame; nhờ vậy không phải OCR mọi frame của video.
3. `ocr.py` chạy PP-OCRv6 Tiny trên các frame mẫu.
4. `scorer.py` tìm các dấu hiệu như nhiều tên người, số áo, từ khóa `lineup`, `starting XI` hoặc sơ đồ `4-3-3`. Đồng thời nó loại các màn hình có đồng hồ trận đấu, VAR, trọng tài, bảng xếp hạng hoặc thống kê.
5. `multimodal.py` dùng độ ổn định giữa hai frame và bố cục các hộp chữ để giảm false positive yếu. Bằng chứng OCR mạnh vẫn được ưu tiên.
6. `interval_proposer.py` yêu cầu bằng chứng tồn tại qua nhiều frame, bám biên theo scene, gộp các đoạn liền nhau và chọn tối đa hai khoảng đội hình.

Nếu có một frame mạnh nhưng chưa đủ bằng chứng theo thời gian, pipeline lấy thêm các frame lân cận để thử phục hồi trước khi kết luận.

### 2. Cắt clip

`src/line_up/cli.py` dùng FFmpeg để cắt từng khoảng được phát hiện:

- thử `stream copy` trước để chạy nhanh và không giảm chất lượng;
- nếu điểm cắt không phù hợp keyframe, tự động fallback sang encode H.264/AAC.

Tên clip có dạng `<ten-video>_lineup_1.mp4`, `<ten-video>_lineup_2.mp4`.

### 3. Đọc tên và số áo

Module `src/mapping/` xử lý các clip đã cắt:

1. Trích frame ở 2 FPS.
2. Chạy một lượt “scout OCR” thưa ở 0.5 FPS để tìm frame có cấu trúc đội hình tốt nhất.
3. OCR chi tiết theo ba cấp để tiết kiệm thời gian:
   - cấp 1: 3 frame tốt nhất;
   - cấp 2: 7 frame nếu cấp 1 chưa đạt;
   - cấp 3: toàn bộ clip ở 2 FPS nếu vẫn chưa đạt.
4. Resolver hỗ trợ hai kiểu đồ họa phổ biến:
   - **table/list**: số áo và tên nằm cùng hàng hoặc cùng dòng;
   - **formation**: số áo và tên được bố trí trên sơ đồ chiến thuật.
5. Khi thiếu số áo, local OCR có thể crop vùng liên quan và đọc lại với model chi tiết hơn.
6. Kết quả chỉ qua quality gate khi mỗi đội có đủ 11 cầu thủ, số áo không trùng, tên không rỗng/không phải chữ giao diện và độ tin cậy của cặp tên–số đạt ngưỡng.

Nếu detector tạo một clip duy nhất chứa liên tiếp đội hình hai đội, mapper sẽ cố tìm hai lineup trong clip đó. Nếu có hai clip riêng, mỗi clip được xử lý theo fast path một đội rồi kết quả được chọn và đánh số lại ở cấp trận đấu.

## Yêu cầu hệ thống

- Python **3.11–3.13**. Khuyến nghị Python 3.11; PaddlePaddle/PaddleOCR hiện bị loại khỏi cài đặt trên Python 3.14 bởi `requirements.txt`.
- FFmpeg và `ffprobe` có trong `PATH`.
- Kết nối Internet ở lần chạy đầu để tải model PaddleOCR.
- CPU có thể chạy toàn bộ pipeline; không yêu cầu GPU.

Cài FFmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

## Cài đặt

Từ thư mục gốc của project:

```bash
python3.11 -m venv .venv-ocr
source .venv-ocr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Trên Windows, lệnh kích hoạt môi trường là:

```powershell
.venv-ocr\Scripts\Activate.ps1
```

Kiểm tra cài đặt:

```bash
python --version
ffmpeg -version
python run_lineup.py --help
```

> Project hiện không đọc các biến API trong `.env`; quá trình nhận diện dùng model PaddleOCR cục bộ.

## Cách chạy

### Chạy cơ bản

```bash
.venv-ocr/bin/python run_lineup.py "data/premier/premier_match_01.mp4"
```

Nếu video nằm dưới `data/<giai-dau>/`, thư mục đầu ra mặc định sẽ giữ cấu trúc tương ứng. Ví dụ lệnh trên ghi kết quả vào:

```text
outputs/premier/premier_match_01/
```

Nếu video nằm ngoài `data/`, kết quả mặc định là `outputs/<ten-video>/`.

### Chỉ quét phần đầu video

Hữu ích khi đồ họa đội hình luôn xuất hiện ở đầu chương trình:

```bash
.venv-ocr/bin/python run_lineup.py "data/match.mp4" --max-scan 600
```

`--max-scan 600` chỉ phân tích 600 giây đầu. Mặc định pipeline quét toàn bộ video.

### Chọn thư mục đầu ra

```bash
.venv-ocr/bin/python run_lineup.py "data/match.mp4" \
  --output-dir "outputs/demo-match"
```

### Giữ file chẩn đoán

```bash
.venv-ocr/bin/python run_lineup.py "data/match.mp4" \
  --keep-diagnostics
```

Tùy chọn này giữ các CSV trung gian như kết quả scout OCR, frame đã chọn, OCR thô, chẩn đoán resolver và lịch sử ba lần thử.

### Tắt OCR phục hồi cục bộ

```bash
.venv-ocr/bin/python run_lineup.py "data/match.mp4" \
  --disable-local-ocr
```

Chỉ nên dùng khi cần benchmark/debug. Việc tắt refinement có thể làm giảm khả năng đọc đủ số áo trên đồ họa khó.

### Toàn bộ tùy chọn

```text
python run_lineup.py VIDEO_PATH [--output-dir DIR] [--max-scan SECONDS]
                     [--disable-local-ocr] [--keep-diagnostics]
```

| Tham số | Ý nghĩa |
| --- | --- |
| `video_path` | Video trận đấu đầu vào. |
| `--output-dir` | Ghi đè thư mục kết quả mặc định. |
| `--max-scan N` | Chỉ quét `N` giây đầu; phải lớn hơn 0. |
| `--disable-local-ocr` | Không crop và OCR lại các vùng số áo khó. |
| `--keep-diagnostics` | Giữ các CSV trung gian để debug. |

Đường dẫn có khoảng trắng phải được đặt trong dấu ngoặc kép.

## Kết quả đầu ra

Với video `data/premier/premier_match_01.mp4`, cấu trúc điển hình là:

```text
outputs/premier/premier_match_01/
├── clips/
│   ├── premier_match_01_lineup_1.mp4
│   └── premier_match_01_lineup_2.mp4
├── mapping/
│   ├── frames/                  # frame trích ở 2 FPS
│   ├── selected_frames/         # crop/frame được chọn cho OCR chi tiết
│   └── *.csv                    # chỉ còn khi dùng --keep-diagnostics
├── detection_result.json
├── players.json
└── resolved_lineups.csv         # chỉ còn khi dùng --keep-diagnostics
```

Ví dụ `players.json`:

```json
[
  {
    "lineup_clip": "premier_match_01_lineup_1.mp4",
    "lineup_index": 1,
    "shirt_number": 9,
    "player_name": "PLAYER NAME"
  }
]
```

Ý nghĩa trường:

- `lineup_clip`: clip nguồn chứa đội hình;
- `lineup_index`: đội hình thứ nhất hoặc thứ hai trong trận;
- `shirt_number`: số áo OCR nhận được;
- `player_name`: tên cầu thủ đã được resolver chuẩn hóa từ nhiều frame.

`detection_result.json` chứa đường dẫn video, thời lượng đã quét, số scene, số frame đã OCR, thời gian xử lý, các khoảng lineup và thống kê tốc độ.

## Mã thoát

| Mã | Ý nghĩa |
| --- | --- |
| `0` | Pipeline hoàn tất và kết quả mapping vượt qua quality gate. |
| `1` | Lỗi đầu vào, FFmpeg, OCR, đọc/ghi file hoặc cấu hình không hợp lệ. |
| `2` | Không tìm thấy lineup hoặc kết quả OCR chưa đủ chất lượng/độ đầy đủ. Các artifact có thể vẫn được ghi để kiểm tra. |

## Cấu trúc mã nguồn

```text
run_lineup.py                 # entry point của pipeline đầy đủ
src/
├── line_up/                  # phát hiện khoảng thời gian có đội hình
│   ├── cli.py                # điều phối detect -> cắt clip -> mapping
│   ├── scene_detector.py     # chia scene
│   ├── sampler.py            # lấy mẫu thích ứng
│   ├── ocr.py                # OCR vòng phát hiện
│   ├── scorer.py             # chấm điểm và lọc false positive
│   ├── multimodal.py         # tín hiệu chuyển động và bố cục
│   └── interval_proposer.py  # tạo/gộp/chọn interval
└── mapping/                  # OCR chi tiết và ghép tên–số
    ├── pipeline.py           # workflow 3 cấp và quality gate
    ├── frames.py             # đọc clip, trích frame
    ├── selector.py           # scout và chọn frame tốt
    ├── engine.py             # PaddleOCR và chuẩn hóa detection
    └── resolver/             # resolver dạng table/formation, refinement
tests/                        # unit và integration tests
data/                         # video đầu vào, bị Git bỏ qua
outputs/                      # kết quả sinh ra, bị Git bỏ qua
```

Các ngưỡng của pha phát hiện nằm trong `src/line_up/config.py`. Các ngưỡng OCR/mapping nằm trong `src/mapping/config.py`.

## Chạy test

```bash
.venv-ocr/bin/python -m unittest discover -s tests -v
```

Phần lớn test dùng mock/dữ liệu tổng hợp nên không cần chạy OCR thật hoặc tải lại model.

## Lưu ý và xử lý lỗi

- **Chạy nhầm Python 3.14:** tạo lại virtual environment bằng Python 3.11–3.13; nếu không, PaddlePaddle và PaddleOCR sẽ không được cài từ `requirements.txt`.
- **`PaddleOCR is not installed`:** kích hoạt đúng `.venv-ocr` hoặc chạy trực tiếp bằng `.venv-ocr/bin/python`.
- **`ffmpeg`/`ffprobe` không tồn tại:** cài FFmpeg và đảm bảo lệnh có trong `PATH`.
- **Lần đầu chạy lâu:** PaddleOCR đang tải và khởi tạo model; các lần sau sẽ dùng cache trong `.cache/paddlex`.
- **Không tìm thấy lineup:** thử bỏ `--max-scan`, vì mặc định không thể biết chính xác thời điểm đài truyền hình chiếu đội hình.
- **Có clip nhưng `players.json` thiếu dữ liệu:** chạy lại với `--keep-diagnostics` và xem `mapping/pipeline_attempts.csv`, `mapping/resolved_lineups_diagnostics.csv` cùng các frame đã trích.
- **Kết quả khó đạt quality gate:** nguồn video mờ, chữ quá nhỏ, graphic chỉ xuất hiện rất ngắn hoặc số áo/tên không cùng lúc có thể khiến resolver không thu đủ 11 cặp tin cậy.

Video trong `data/`, kết quả trong `outputs/`, virtual environment, cache model và `.env` đều đã được `.gitignore` loại khỏi Git.
