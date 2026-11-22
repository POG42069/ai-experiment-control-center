# Platform Capabilities & Limitations

Tài liệu này mô tả chính xác những gì **AI Experiment Control Center (ECC)** hỗ trợ, hỗ trợ một phần, hoặc không hỗ trợ theo từng nguồn dữ liệu (căn cứ `docs/ARCHITECTURE.md` §1 và mã nguồn thực tế trong `ecc/` & `monitor/`).

> **Nguyên tắc trung thực dữ liệu:** Dashboard không bao giờ bịa hoặc suy đoán dữ liệu mà nguồn không cung cấp. Trường thiếu luôn hiển thị `—` hoặc `UNKNOWN`.

---

## 1. Feature × Source Matrix

| Tính năng | W&B (`WandbSource` + `experiment_monitor`) | Kaggle (`KaggleSource` via `kaggle` CLI 2.2.5) | Google Colab (`ColabSource` stub — MVP) |
|---|---|---|---|
| **Liệt kê runs / notebooks** | **Supported** — `api.runs(entity/project)` (tối đa `max_runs=200` / project, cache TTL 10s). | **Supported** — `kaggle kernels list --mine --csv` cho từng tài khoản (lọc `kaggle_lookback_hours=72`, kiểm tra status tối đa 15 kernel gần nhất / tài khoản, cache TTL 30s). | **Not supported** — không tích hợp native trong MVP; chỉ hiển thị các run Colab đã gắn `experiment_monitor` gửi lên W&B. |
| **Trạng thái chạy (`QUEUED`, `RUNNING`, `FINISHED`, `FAILED`, ...)** | **Supported** — map `run.state` + luật heartbeat (`STALE`/`OFFLINE`). Không thấy trạng thái `QUEUED` của Kaggle trước khi cell `em.init` chạy. | **Supported** — `kaggle kernels status owner/slug` hỗ trợ đầy đủ `QUEUED`, `NEW_SCRIPT`, `RUNNING`, `COMPLETE`, `ERROR`, `CANCEL_*` (thấy được `QUEUED` trước khi W&B run tồn tại). | **Not supported** — Colab không có API chính thức trả về trạng thái thực thi notebook. Theo dõi gián tiếp qua W&B heartbeat. |
| **Structured metrics (`train/loss`, `eval/macro_f1`, `train/lr`, ...)** | **Supported** — đọc `run.summary` và `run.history(samples=2000)` / `scan_history()` (outer-merge theo `_step` khi truy vấn từng key). | **Not supported** — Kaggle CLI không cung cấp structured metrics. Phải gắn `experiment_monitor` → W&B. | **Not supported** — Colab không cung cấp structured metrics. Phải gắn `experiment_monitor` → W&B. |
| **GPU telemetry (`gpu/mem_alloc_gb`, VRAM total, util)** | **Partial (chỉ qua `experiment_monitor`)** — `experiment_monitor` ghi `ecc/gpu_*` và `gpu/*` khi `gpu_telemetry=True` (dùng `torch.cuda` / NVML nếu có). Run W&B thông thường không dùng `experiment_monitor` có thể thiếu các khóa này. | **Not supported** — Kaggle CLI không cung cấp GPU telemetry. | **Not supported** — Colab không cung cấp GPU telemetry qua API; chỉ có khi gắn `experiment_monitor` → W&B. |
| **Live console logs** | **Partial (near-live)** — W&B public API không có first-class console stream đáng tin cậy cho run đang chạy. ECC dùng `experiment_monitor` (`capture_console=True`) để tee `stdout`/`stderr` vào file `ecc_console.log` (ring-capped mặc định 2 MB, đã redact secret) và sync bằng `wandb.save(..., policy="live")`. Độ trễ hiển thị phụ thuộc tần suất sync file của W&B (và TTL cache 10s của `WandbSource`). Fallback sang `output.log` nếu có. | **Partial (persisted one-shot)** — ECC chỉ gọi `kaggle kernels logs owner/slug` (log đã lưu, lấy 1 lần). Kaggle CLI có hỗ trợ live SSE stream qua `kaggle kernels logs -f`, nhưng **dashboard cố ý không dùng** (`KaggleCLI._validate_command` chặn `-f` / `--follow` để tránh giữ subprocess dài hạn trong mô hình read-only polling). | **Not supported** — Colab không có API chính thức cho execution logs. Dùng `experiment_monitor` (`capture_console=True`) → W&B để xem console log. |
| **Checkpoints metadata** | **Supported (chỉ qua `experiment_monitor`)** — ghi `{step, path, size_mb, metric}` vào `run.summary["ecc/checkpoints"]` (không upload trọng số model lên W&B). | **Not supported** — không có trong Kaggle CLI status/list. | **Not supported** — chỉ có qua `experiment_monitor` → W&B. |
| **Notebook URL & liên kết project** | **Supported** — đọc `ecc/notebook_url`, `ecc/kaggle_kernel` từ `run.config`; tự động gộp với `KaggleSource`. | **Supported** — tự sinh URL `https://www.kaggle.com/code/<ref>`; project mặc định `"unassigned"` nếu chưa liên kết qua `ecc/kaggle_kernel` hoặc `Service.link_kernel()`. | **Partial** — không tự phát hiện notebook URL; người dùng truyền tay `notebook_url=` vào `em.init(...)`. |
| **Multi-account isolation** | **Supported (single workspace)** — mọi notebook trên nhiều tài khoản Kaggle/Colab log vào 1 W&B entity/project chung, phân biệt bằng nhãn `ecc/account`. | **Supported** — mỗi tài khoản chạy subprocess với `KAGGLE_API_TOKEN` (qua `env=`, không qua argv) + `KAGGLE_CONFIG_DIR` riêng tại `%LOCALAPPDATA%\ai-ecc\kaggle\<nickname>`, không đụng `~/.kaggle`. | **Partial (nhãn only)** — không có native auth; `colab-1`/`colab-2` chỉ là nhãn `account=` trong `em.init(...)`. |
| **Remote control (start, stop, cancel, execute code)** | **Not supported (by design)** — ECC là hệ thống chỉ-đọc. | **Not supported (by design)** — allow-list chỉ cho phép `kernels list`, `kernels status`, `kernels logs`. | **Not supported (by design)** — không gọi endpoint tạo/xóa runtime hay `execute_code`. |

---

## 2. Stale & Offline Detection Semantics (`ecc/status.py`)

ECC chuẩn hóa trạng thái qua duy nhất một hàm `derive_status(raw_state, last_update, now, stale_after_s=180, offline_after_s=900)`:

1. **Không bao giờ báo `RUNNING` nếu thiếu heartbeat mới:**
   - Nếu `raw_state` là `"running"` nhưng `last_update` là `None` → trả về **`UNKNOWN`**.
2. **Với run W&B (`raw_state == "running"`):**
   - `last_update = max(heartbeatAt, summary["ecc/heartbeat"])` (UTC).
   - `age ≤ 180s` (3 phút, mặc định) → **`RUNNING`**.
   - `180s < age ≤ 900s` (15 phút, mặc định) → **`STALE`** (cảnh báo chậm cập nhật: có thể đang chạy eval nặng, dataloader treo, hoặc W&B tạm chậm sync).
   - `age > 900s` → **`OFFLINE`** (rất có thể runtime Kaggle/Colab đã ngắt đột ngột mà chưa kịp gửi trạng thái kết thúc).
3. **Với Kaggle native kernel (`KaggleSource`):**
   - Endpoint `kaggle kernels status` trả về trạng thái worker trực tiếp (`RUNNING`) nhưng không trả về timestamp heartbeat theo phút.
   - Do đó, khi Kaggle báo `RUNNING`, `KaggleSource` đặt `last_update = now` và ghi rõ `raw_state = "RUNNING (Kaggle API)"` để phân biệt với heartbeat W&B.
4. **Trạng thái kết thúc / hàng đợi:**
   - W&B: `finished` → `FINISHED`, `failed` → `FAILED`, `crashed` → `CRASHED`, `killed`/`preempted` → `KILLED`, `pending` → `QUEUED`.
   - Kaggle: `QUEUED`/`NEW_SCRIPT` → `QUEUED`, `COMPLETE` → `FINISHED`, `ERROR` → `FAILED`, `CANCEL_*` → `KILLED`.
   - Mọi giá trị khác hoặc rỗng → **`UNKNOWN`**.

---

## 3. Operational & Polling Constraints

- **Caching**: `WandbSource` cache trong bộ nhớ 10 giây; `KaggleSource` cache 30 giây. Kết quả thay đổi có thể trễ tối đa 1 chu kỳ cache + 1 chu kỳ refresh của dashboard.
- **Sampling metric**: `WandbSource.get_metrics()` dùng `history(samples=2000)` cho đường cong lớn; `ecc.cli get-metrics` và `compare` downsample thêm theo `--max-points` (mặc định `500`) để giữ JSON gọn cho Claude.
- **Google Colab Phase 3 (chưa bật trong MVP)**: Theo `ecc/adapters/colab.py`, ngay cả khi bật adapter read-only tương lai (`GET /v1beta/runtimes` và `GET /v1beta/subscription` qua Google OAuth riêng theo từng tài khoản), API Colab vẫn **không** cung cấp execution status, console logs hay tiến độ training — các dữ liệu đó vẫn phụ thuộc hoàn toàn vào `experiment_monitor` → W&B.
