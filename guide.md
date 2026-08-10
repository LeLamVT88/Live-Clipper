# Guide: Thiết kế Production Worker Service (Áp dụng cho module mới)

> Mục đích tài liệu này là **tách phần "production scaffolding"** ra khỏi logic nghiệp vụ.
> Repo hiện tại (`football-goal-detector`) là một ví dụ: logic nghiệp vụ là OCR phát hiện bàn thắng,
> nhưng **kiến trúc worker service** (nhận input từ Kafka, xử lý bằng thread pool, ghi kết quả vào MySQL)
> là phần tái sử dụng được cho mọi module xử lý video/ảnh/file theo kiểu batch.
>
> Nếu bạn đang làm một module khác (ví dụ: detect highlight, cắt clip theo nhu cầu, caption...),
> chỉ cần thay **lớp nghiệp vụ**, giữ nguyên phần khung worker bên dưới.

---

## 1. Tổng quan kiến trúc

```
                    ┌──────────────────────────────────────────────────────┐
                    │                     WORKER PROCESS                    │
                    │                                                       │
  Producer          │  ┌──────────────┐    ┌─────────────────────────────┐  │
  (Java service     │  │  Kafka       │    │  TaskRunner                 │  │
  gửi task)         │  │  Listener    │───▶│  (ThreadPoolExecutor)       │  │
                    │  │  (thread)    │    │  max_workers = N            │  │
                    │  └──────────────┘    │                             │  │
  KAFKA             │                      │  ┌────────────┐ ┌────────┐  │  │
  topic:            │                      │  │ Worker 1   │ │ Worker 2│  │  │
  ocr-play-clusters │                      │  │ (thread)   │ │ (thread)│  │  │
                    │                      │  └────────────┘ └────────┘  │  │
                    │                      │         │   │   │           │  │
                    │                      └─────────┼───┼───┼───────────┘  │
                    │                                ▼   ▼   ▼              │
                    │                       ┌────────────────────┐          │
                    │                       │   Business Logic    │          │
                    │                       │   (GoalDetector)     │          │
                    │                       └────────────────────┘          │
                    │                                │                       │
                    └────────────────────────────────┼───────────────────────┘
                                                     ▼
                                            ┌──────────────────┐
                                            │   MySQL          │
                                            │   (match_events) │
                                            └──────────────────┘
```

**Nguyên tắc cốt lõi:**

- Worker là **process chạy nền, không có endpoint HTTP**. Nó "nghe" việc từ Kafka, không ai gọi nó.
- Mỗi message Kafka = một **task** = một video cần xử lý.
- Task được xử lý **bất đồng bộ, song song** qua thread pool.
- Worker **không giữ state giữa các task** (stateless) → dễ scale ngang (chạy nhiều instance).
- Kết quả được **đẩy ra ngoài hệ thống** qua MySQL (không trả về qua Kafka).

---

## 2. Vòng đời process (`main.py`) — khung chính

Đây là file **điểm vào duy nhất**, không chứa logic nghiệp vụ, chỉ "bật" các thành phần và quản lý tắt mở.

```
┌────────────────────────────────────────────────────────────┐
│  main()                                                     │
│                                                             │
│  1. load_dotenv()          ← nạp config từ .env             │
│  2. db = DBAccess()        ← kiểm tra kết nối MySQL khi khởi động │
│  3. stop_event = Event()   ← cờ chung để tắt cả process     │
│  4. runner = TaskRunner()  ← tạo thread pool                │
│  5. start Kafka listener (daemon thread)                    │
│  6. đăng ký SIGTERM/SIGINT → graceful shutdown              │
│  7. main thread ngủ (wait) cho tới khi stop_event set       │
└────────────────────────────────────────────────────────────┘
```

### 2.1 Những điểm bắt buộc phải có

| Thành phần                      | Vai trò          | Chi tiết                                                                                               |
| --------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| `load_dotenv()`                 | Nạp config       | Gọi**trước khi import** mọi module dùng `os.environ`                                       |
| Khởi tạo sớm các client nặng | Fail-fast         | Nếu DB/Kafka không sẵn sàng, process phải**fail khi khởi động**, không fail giữa chừng |
| `threading.Event()`             | Cờ dừng chung   | Mọi thread đều kiểm tra cờ này để thoát sạch                                                  |
| `signal.signal(SIGTERM/SIGINT)` | Graceful shutdown | Cho phép task đang chạy**hoàn tất** rồi mới thoát                                         |
| Daemon thread                     | Thread nền       | Kafka listener là`daemon=True` để không chặn việc thoát process                                |

### 2.2 Graceful shutdown — mẫu chuẩn

```python
def shutdown_handler(signum, frame):
    stop_event.set()              # 1. báo các thread khác dừng nhận việc mới
    runner.shutdown(wait=True)    # 2. đợi task đang chạy xong (không hủy giữa chừng)
    kafka_thread.join(timeout=10) # 3. đợi thread listener thoát
    sys.exit(0)
```

> ⚠️ **Quan trọng:** `executor.shutdown(wait=True)` có nghĩa là **cho phép task đang chạy chạy hết**.
> Điều này tránh ghi dở dang vào DB. Nếu muốn hủy task giữa chừng thì dùng `wait=False` + cơ chế cancel riêng.

---

## 3. Input: Message Kafka

### 3.1 Format message

Listener nhận message dạng **JSON string**, hỗ trợ cả camelCase lẫn snake_case:

```json
{
  "filePath": "/data/videos/match_001.mp4",
  "startTime": 0.0,
  "endTime": 0.0
}
```

```python
file_path = payload.get('filePath') or payload.get('file_path')
start_time = payload.get('startTime') or payload.get('start_time', 0.0)
end_time = payload.get('endTime') or payload.get('end_time', 0.0)
```

> **Thiết kế:** luôn có **fallback key** (`file_path` thay vì `filePath`) và **default value** (0.0)
> để không crash khi producer thay đổi format nhẹ. Đây là pattern chống "contract break" giữa các team.

### 3.2 Cấu hình consumer

```python
conf = {
    'bootstrap.servers': ...,      # từ config/env
    'group.id': ...,               # consumer group → quyết định scale
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True,    # auto commit offset sau khi poll
}
```

**Các option quan trọng cần hiểu:**

- **`group.id`**: Nhiều instance dùng **cùng group.id** → task được chia đều giữa chúng (mỗi partition chỉ 1 instance xử lý). Đây là cách scale worker **ngang**.
- **`auto.offset.reset=earliest`**: Nếu consumer mới, đọc từ đầu (không bỏ sót task).
- **`enable.auto.commit=True`**: Tự động đánh dấu đã xử lý. **Lưu ý:** ở repo này commit là "poll-time commit", nên nếu process crash giữa chừng thì task đó có thể bị mất — cần cân nhắc at-least-once vs at-most-once. Với hệ thống này, DB insert dùng `ON DUPLICATE KEY UPDATE` để bù lại.

### 3.3 Auth tùy chọn (SASL)

Nếu môi trường cần xác thực, chỉ bật khi biến env `KAFKA_SASL_MECHANISM` tồn tại. **Password phải được mã hóa** (xem mục 6.3 AES).

### 3.4 Xử lý lỗi kết nối — retry với backoff

```python
retry_delay = 1.0
while not stop_event.is_set():
    try:
        consumer = self._init_consumer()   # thất bại → nhảy xuống except
        retry_delay = 1.0
    except Exception:
        if stop_event.wait(retry_delay):   # dùng wait để vừa ngủ vừa thoát được
            return
        retry_delay = min(retry_delay * 2, 30.0)   # exponential backoff
        continue
```

> **Pattern cần nhớ:**
>
> 1. **Exponential backoff** khi retry (1s → 2s → 4s → ... → max 30s).
> 2. Dùng `stop_event.wait(timeout)` thay vì `time.sleep()` → thread ngủ vẫn thoát được khi shutdown.
> 3. Khi gặp lỗi consumer nghiêm trọng → **đóng và tạo lại consumer**, không cố dùng object hỏng.

### 3.5 Luồng xử lý một message

```
1. Nhận message → decode utf-8
2. json.loads → nếu lỗi JSON → bỏ qua (log) → continue
3. Lấy file_path, start_time, end_time
4. Nếu thiếu file_path → log warning → bỏ qua
5. Tra DB: get_file_details(file_path) → schedule_id, duration
   - KHÔNG có trong DB → tạo placeholder (schedule_id mặc định) để dev/test
6. submit_task(file_path, schedule_id, duration, config_overrides)
```

> **Điểm hay:** logic **tra cứu metadata trong DB dựa trên input** là bước trung gian bắt buộc.
> Task chỉ mang định danh (file path), còn thông tin nghiệp vụ (schedule_id, duration) được lấy từ nguồn chân lý (DB).

---

## 4. Xử lý song song: `TaskRunner`

### 4.1 Thread pool thay vì process

```python
self.executor = ThreadPoolExecutor(
    max_workers=config.get('max_workers', 2),
    thread_name_prefix="ocr-worker-"
)

def submit_task(self, file_path, schedule_id, duration, config_overrides):
    self.executor.submit(self._process_task, file_path, schedule_id, duration, config_overrides)
```

**Đặc điểm thiết kế:**

- Mỗi task chạy trong **1 thread**, tối đa `max_workers` task song song.
- `submit_task` **trả về ngay**, không block listener → listener tiếp tục nhận message mới.
- Task chạy ngầm; kết quả/lỗi được log + ghi DB, **không quăng lỗi ra ngoài làm crash thread**.

> ⚠️ **Lưu ý khi dùng ThreadPoolExecutor với xử lý video/OCR:**
>
> - Nếu nghiệp vụ **CPU-bound** (như OCR/PaddleOCR), thread pool bị giới hạn bởi CPU → tăng `max_workers`
>   quá mức sẽ không tăng throughput mà còn tốn RAM, gây ảnh hưởng các process khác trên server, chỉ nên để `max_workers` khoảng 4 hoặc 8.
> - Nếu nghiệp vụ **IO-bound** (download/upload file, gọi API), thread pool scale tốt hơn nhiều.
> - Trong module OCR này, các model (PaddleOCR) là **static/shared** (`GoalDetector._shared_ocr_pipeline`)
>   → tránh load model lại cho từng task, chỉ khởi tạo **1 lần cho cả process**.

### 4.2 Cấu trúc một `_process_task` chuẩn

```python
def _process_task(self, video_path, schedule_id, duration, config_overrides):
    try:
        # 1. (tuỳ chọn) resolve output dir theo input
        video_dir = os.path.dirname(video_path)
        output_dir = config_overrides.get('output_dir') or os.path.join(video_dir, 'ocr_output')

        # 2. validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(...)

        # 3. chạy logic nghiệp vụ
        detector = GoalDetector(video_path, output_dir, config_overrides)
        detector.calibrate()
        results = detector.track()

        # 4. format kết quả theo schema output
        db_events = [self._format_event(...) for ...]

        # 5. ghi kết quả vào DB (điểm cuối của pipeline)
        self.db.complete_task(schedule_id, db_events)

    except Exception as err:
        logger.error("Error processing ...", exc_info=True)   # KHÔNG raise
```

**Quy tắc vàng cho worker:**

1. **Bọc toàn bộ thân task trong try/except**, log lỗi `exc_info=True`, **không re-raise**.
   Nếu re-raise, future sẽ lưu lỗi nhưng không ai đọc → log bị nuốt, khó debug.
2. Phân tách rõ 3 bước: **validate input → xử lý → persist output**.
3. **Output của pipeline = bản ghi DB**, không phải file hay print.

---

## 5. Output: Ghi vào MySQL

Tầng DB nằm trong **`src/database/db_access.py`** — đây là **template dùng chung**, copy nguyên sang module mới và chỉ thay schema bảng. Phần này mô tả **hợp đồng** giữa worker và DB (worker cần gì, ghi ra đâu), **không mô tả** cách file đó chạy (mở connection thế nào, cursor ra sao) — chi tiết đó đã nằm sẵn trong template.

### 5.1 Hợp đồng: worker chỉ cần đúng 3 thao tác

Tầng DB chỉ phơi ra **3 phương thức** — đủ cho mọi module, đừng thêm bớt nếu không cần:

| Phương thức                   | Vai trò trong pipeline                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `get_file_details(input_path)` | Tra metadata (id buổi phát, duration) theo input nhận từ Kafka                         |
| `insert_file_placeholder(...)` | Tạo bản ghi giả khi input chưa có trong DB (chỉ để dev/test)                       |
| `complete_task(..., events)`   | Ghi toàn bộ kết quả của task vào bảng output —**điểm cuối của pipeline** |

Luồng dùng chúng (đã nêu ở mục 3.5 và 4.2): *message → tra metadata → xử lý → format theo schema → `complete_task`*.
Worker **không** tự viết SQL rải rác ở logic nghiệp vụ; mọi thao tác DB đều đi qua template này.

### 5.2 Quy ước cần giữ khi copy template

- **Kết quả ghi theo từng task, thành batch, trong 1 transaction** — nếu task lỗi giữa chừng thì không để lại nửa chừng dữ liệu.
- **Idempotent:** dùng `ON DUPLICATE KEY UPDATE` khi cần chạy lại task mà không sinh bản ghi trùng (kết hợp `auto.offset.reset=earliest` + Kafka replay).
- **Ghi status hoàn tất** (trong repo này là `status=2` = MERGED) để hệ thống đọc biết task đã xong, không phụ thuộc vào "có dữ liệu hay không".
- **Schema output là hợp đồng** giữa worker và hệ thống đọc dữ liệu → chỉ đổi tên bảng + cột theo nghiệp vụ module mới, giữ nguyên cấu trúc class.

### 5.3 Schema output `match_events` (hợp đồng của repo này)

```sql
INSERT INTO match_events
  (schedule_id, time, type, start_time, end_time, status, full_path, final_path, comment)
VALUES (%s, %s, %s, %s, %s, 2, %s, %s, %s);
```

| Cột                          | Ý nghĩa                           | Ví dụ                              |
| ----------------------------- | ----------------------------------- | ------------------------------------ |
| `schedule_id`               | Định danh buổi phát/sự kiện   | `12345`                            |
| `time`                      | Thời điểm sự kiện (display)    | `"00:23:45"`                       |
| `type`                      | Loại sự kiện                     | `"GOAL"`                           |
| `start_time` / `end_time` | Khoảng thời gian (giây) cho clip | `1400` / `1430`                  |
| `status`                    | Trạng thái (2 = hoàn tất)       | `2`                                |
| `full_path`                 | File nguồn                         | đường dẫn video                  |
| `final_path`                | File output (clip đã cắt)        | `null` nếu không có             |
| `comment`                   | Mô tả                             | `"Goal 1-0 -> 2-0 (Scorer: home)"` |

### 5.4 Schema thực tế của 2 bảng hệ thống này dùng (tham khảo)

Hệ thống highlight động tới đúng **2 bảng**: một bảng **input** (tra metadata) và một bảng **output** (ghi kết quả).

**a) `full_path_contents` — bảng input (metadata video)**

Bảng chứa các video đã được ghi. Worker tra metadata ở đây khi nhận task từ Kafka.

| Cột                            | Kiểu                      | Mô tả                                                               |
| ------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| `id`                          | `int` PK, AUTO_INCREMENT | ID                                                                    |
| `schedule_id`                 | `int NOT NULL`           | FK →`schedules.id` (lịch ghi trận đấu)                         |
| `start_time` / `end_time`   | `datetime NULL`          | Thời gian bắt đầu / kết thúc ghi                                |
| `final_file`                  | `varchar(255) NULL`      | **Đường dẫn video trên local** — khóa lookup của worker |
| `duration`                    | `int DEFAULT 0`          | Thời lượng video (giây)                                           |
| `last_segment_index`          | `int NULL`               | Segment cuối đã ghi                                                |
| `audio_detected`              | `int DEFAULT 0`          | 0-Waiting, 1-Detecting, 2-Detected, 3-Failed                          |
| `commentary_detected`         | `int DEFAULT 0`          | 0-Waiting, 1-Detecting, 2-Detected, 3-Failed                          |
| `created_at` / `updated_at` | `datetime`               | Timestamp                                                             |

```sql
CREATE TABLE `full_path_contents` (
  `id` int NOT NULL AUTO_INCREMENT,
  `schedule_id` int NOT NULL,
  `start_time` datetime DEFAULT NULL,
  `end_time` datetime DEFAULT NULL,
  `final_file` varchar(255) DEFAULT NULL COMMENT 'Đường dẫn lưu file video trên local',
  `duration` int DEFAULT '0' COMMENT 'Thời lượng của video (giây)',
  `last_segment_index` int DEFAULT NULL,
  `audio_detected` int NOT NULL DEFAULT '0',
  `commentary_detected` int NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `content_schedule_id_fk` (`schedule_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

> ⚠️ **Điểm thiết kế quan trọng:** worker lookup theo **`final_file`** (không phải `id`),
> vì Kafka message chỉ mang đường dẫn video. Message = định danh, metadata = nguồn chân lý trong DB.

**b) `match_events` — bảng output (sự kiện highlight)**

Bảng lưu các sự kiện highlight — điểm cuối của pipeline worker.

| Cột                            | Kiểu                      | Mô tả                                                                                                                                               |
| ------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                          | `int` PK, AUTO_INCREMENT | ID                                                                                                                                                    |
| `schedule_id`                 | `int NOT NULL`           | FK →`schedules.id`                                                                                                                                 |
| `time`                        | `varchar(20) NULL`       | Thời điểm sự kiện (vd:`35`, `60`, `90+3`)                                                                                                  |
| `type`                        | `varchar(100) NULL`      | Loại sự kiện (vd:`GOAL`, `Auto Highlight`)                                                                                                     |
| `start_time` / `end_time`   | `int NULL`               | Khoảng thời gian segment (giây)                                                                                                                    |
| `status`                      | `int DEFAULT 0`          | 0-UNPROCESSED, 1-PROCESSING,**2-MERGED**, 3-NONE/SKIPPED/FAILED, 4-RAW_COMMENTARY, 5-MERGED_INTO_AUDIO, 8-UNMERGED_COMMENTARY, 9-UNMERGED_AUDIO |
| `full_path`                   | `varchar(255) NULL`      | Đường dẫn file local nguồn                                                                                                                       |
| `final_path`                  | `varchar(255) NULL`      | Đường dẫn file output (S3, dạng`2026/04/30/clip_15_001_010.mp4`)                                                                               |
| `comment`                     | `text NULL`              | Mô tả sự kiện                                                                                                                                     |
| `created_at` / `updated_at` | `datetime`               | Timestamp                                                                                                                                             |

```sql
CREATE TABLE `match_events` (
  `id` int NOT NULL AUTO_INCREMENT,
  `schedule_id` int NOT NULL,
  `time` varchar(20) DEFAULT NULL COMMENT 'Thời gian sự kiện xảy ra, ví dụ: 35, 60, 90+3',
  `type` varchar(100) DEFAULT NULL COMMENT 'Loại sự kiện',
  `start_time` int DEFAULT NULL COMMENT 'Thời gian bắt đầu segment tính theo giây',
  `end_time` int DEFAULT NULL COMMENT 'Thời gian kết thúc segment tính theo giây',
  `status` int DEFAULT '0',
  `full_path` varchar(255) DEFAULT NULL,
  `final_path` varchar(255) DEFAULT NULL COMMENT 'Đường dẫn file trên S3',
  `comment` text,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `match_events_schedule_id_fk` (`schedule_id`),
  KEY `match_events_status_IDX` (`status`),
  CONSTRAINT `match_events_schedule_id_fk` FOREIGN KEY (`schedule_id`) REFERENCES `schedules` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**Quan hệ giữa 3 bảng:**

```
schedules (id) 1 ─── * full_path_contents (schedule_id, final_file)
                └─── * match_events (schedule_id)
```

- `full_path_contents` và `match_events` đều có FK → `schedules(id)`, nhưng **không có FK trực tiếp** giữa chúng — chúng nối với nhau qua `schedule_id`.
- Worker **không sửa** bảng `schedules`; nó chỉ đọc `full_path_contents` và ghi `match_events`.
- Module mới nên copy mô hình này: **1 bảng input (chỉ đọc) + 1 bảng output (chỉ ghi)**, nối qua một khóa chung (`schedule_id`), tránh FK chằng chịt giữa các bảng nghiệp vụ.

---

## 6. Các file "template hỗ trợ"

Những file này **không chứa logic nghiệp vụ** và có thể **copy nguyên** sang module mới.

### 6.1 `config.py` — config qua env, có default

Pattern: một dict config với mỗi key đọc từ env, **luôn có default** để chạy được ngay khi thiếu env.

```python
def get_env_int(name, default):
    val = os.environ.get(name)
    if val is None: return default
    try: return int(val)
    except ValueError: return default

CONFIG = {
    'max_workers': get_env_int('MAX_WORKERS', 2),
    'processing_fps': get_env_float('PROCESSING_FPS', 0.5),
    ...
}

# Trong class nghiệp vụ:
self.config = DEFAULT_CONFIG.copy()   # copy để không làm hỏng default toàn cục
if overrides: self.config.update(overrides)   # ghi đè theo từng task
```

**Quy tắc:**

- Mọi tham số có thể "tune" được → đưa vào config, không hard-code.
- Có helper `get_env_bool/int/float/list` xử lý parse + default an toàn.
- Khi nhận config theo task (`config_overrides`), **copy dict trước khi update** để tránh shared mutable state giữa các thread.
- Vì worker chạy **đa luồng**, config của mỗi task phải là **bản sao riêng**.

### 6.2 `log.py` — logging chuẩn

```python
logger = setup_logger("module-name", "logs/worker.log")
```

| Chế độ                    | Hành vi                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| Bình thường (`DEBUG=0`) | Chỉ log ra**console**                                                                         |
| Debug (`DEBUG=1`)          | Log ra console +**file xoay vòng hằng ngày** (`TimedRotatingFileHandler`, backup 7 ngày) |

**Điểm cần copy:**

- Ngăn duplicate handler khi khởi tạo nhiều lần (`if logger.hasHandlers(): return`).
- Format có `[filename:lineno]` để trace lỗi dễ.
- File handler chỉ bật trong debug để không spam disk ở production.
- Tạo thư mục log tự động (`os.makedirs`).

### 6.3 `AES_cipher.py` — mã hóa secret

Dùng để **giấu password trong config** (ví dụ: password Kafka SASL). Nếu giá trị đã là plaintext thì decrypt trả về nguyên giá trị (fail-safe).

```
AES-256-CBC
format: base64( IV[16 bytes] + ciphertext )
```

**Pattern:** secret không bao giờ nằm trần trong file config. Nếu decrypt fail → trả về chuỗi gốc (hỗ trợ dev local dùng plaintext).
