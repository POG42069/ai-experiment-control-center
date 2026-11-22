# AI Experiment Control Center (ECC)

Trung tâm điều khiển cục bộ (**read-only**) chạy trên **Windows 11** để theo dõi, so sánh và chẩn đoán các thí nghiệm ML/VLM đang chạy trên nhiều tài khoản **Kaggle** và **Google Colab** (thông qua **Weights & Biases** + **Kaggle CLI**), kèm giao diện JSON CLI để Claude đọc trực tiếp.

> **Nguyên tắc cốt lõi:** Chỉ đọc (read-only), không sửa siêu tham số huấn luyện, không lưu secret trong config/SQLite/git, và chỉ hiển thị những gì nguồn dữ liệu thực sự cung cấp (thiếu dữ liệu hiển thị `—` / `UNKNOWN`).

---

## Architecture

```text
Kaggle Notebooks (kaggle-1..3)     Colab Notebooks (colab-1..2)
  └─ experiment_monitor              └─ experiment_monitor
        │  (ecc/* config, metrics,         │
        │   GPU mem, heartbeat,            │
        │   redacted ecc_console.log)      │
        └──────────────┬───────────────────┘
                       ▼
                 W&B Cloud API           Kaggle CLI (read-only allow-list:
                       │                  kernels list | status | logs)
                       ▼                                │
                 WandbSource                       KaggleSource
                       └──────────────┬─────────────────┘
                                      ▼
                            ecc.service.Service
                        (status.py · store.py · secrets.py)
                        SQLite (%LOCALAPPDATA%\ai-ecc\ecc.db)
                        Windows Credential Manager (keyring: "ai-ecc")
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        Streamlit Dashboard (127.0.0.1)       Read-Only JSON CLI (Claude)
        app/main.py (5 pages)                 python -m ecc.cli <cmd> --json
```

Chi tiết kiến trúc và hợp đồng dữ liệu: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick Start (Windows 11 + PowerShell)

1. **Cài đặt môi trường ảo `.venv`, dependencies, gói `experiment-monitor` (editable) và tạo `ecc_config.toml`:**

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```

   `setup.ps1` có tính **idempotent** (chạy lại an toàn; giữ nguyên `.venv` và `ecc_config.toml` nếu đã tồn tại).

2. **Chạy dashboard ở chế độ Demo (không cần tài khoản W&B/Kaggle):**

   ```powershell
   .\run_dashboard.ps1 -Demo
   ```

   Hoặc khởi chạy bình thường (mặc định tự bật demo nếu chưa cấu hình W&B trong `ecc_config.toml`):

   ```powershell
   .\run_dashboard.ps1
   # Tùy chỉnh cổng:
   .\run_dashboard.ps1 -Port 8502
   ```

   Mở trình duyệt tại `http://127.0.0.1:8501`. Khi dữ liệu tổng hợp đang bật, giao diện luôn hiện banner **DEMO MODE — synthetic data, not real runs**.

   > **Lưu ý về cấu hình Demo Mode:** `ecc/config.py` đọc `demo_mode` (`"auto"` | `"on"` | `"off"`) từ `ecc_config.toml` (hoặc file được trỏ bởi biến môi trường `ECC_CONFIG`), không dùng biến `ECC_DEMO_MODE`. Cờ `-Demo` của `run_dashboard.ps1` tự tạo file TOML tạm có `demo_mode = "on"` và trỏ `ECC_CONFIG` trong phiên chạy đó.

---

## Connect W&B

1. Sửa `ecc_config.toml` (đã được `.gitignore` bỏ qua, **tuyệt đối không ghi API key vào file này**):

   ```toml
   [general]
   refresh_seconds = 15
   stale_after_s = 180
   offline_after_s = 900
   demo_mode = "auto"

   [wandb]
   entity = "your-wandb-entity"
   projects = []   # rỗng = tự phát hiện mọi project của entity; hoặc liệt kê, ví dụ ["shared-task-4"]
   ```

2. Lưu W&B API key (lấy tại `https://wandb.ai/authorize`) vào Windows Credential Manager. Dashboard dùng chính key này để đọc W&B (không cần `wandb login` hay `.netrc`):

   ```powershell
   .\.venv\Scripts\python.exe scripts\add_account.py --platform wandb --nickname wandb-main --from-clipboard
   ```

3. (Tùy chọn) Nếu không lưu key bằng bước 2, `WandbSource` sẽ dùng cơ chế mặc định của wandb: biến môi trường `WANDB_API_KEY` hoặc `.\.venv\Scripts\wandb.exe login`.

---

## Add Accounts (Multi-Account Kaggle & Colab)

Quản lý tài khoản an toàn qua terminal. Secret không bao giờ nằm trong argv hay file: hoặc dùng `--from-clipboard` (script chờ bạn copy token rồi nhấn Enter, đọc clipboard và xóa clipboard ngay), hoặc bỏ cờ này để dán vào ô nhập ẩn (`getpass`, hiện `*`). `--username` chỉ là nhãn hiển thị, có thể bỏ.

```powershell
# Thêm từng tài khoản Kaggle (mỗi tài khoản 1 token riêng, cô lập hoàn toàn)
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-1 --from-clipboard
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-2 --from-clipboard
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-3 --from-clipboard

# Kiểm tra mỗi nickname trỏ tới một tài khoản Kaggle khác nhau (chỉ đọc)
.\.venv\Scripts\python.exe scripts\verify_accounts.py

# Xem danh sách tài khoản & kết quả kiểm tra (không bao giờ in secret)
.\.venv\Scripts\python.exe scripts\add_account.py --list

# Xóa tài khoản và xóa secret khỏi Windows Credential Manager
.\.venv\Scripts\python.exe scripts\add_account.py --remove kaggle-3
```

Hướng dẫn chi tiết cho W&B, 3 tài khoản Kaggle, 2 tài khoản Colab và cách liên kết Kaggle kernel với project: **[docs/SETUP_ACCOUNTS.md](docs/SETUP_ACCOUNTS.md)**.

---

## Instrument a Notebook

Chỉ cần thêm vài cell quan sát vào notebook training hiện có (**không thay đổi siêu tham số**):

- Copy-paste nhanh: **[notebooks/snippets.md](notebooks/snippets.md)**
- Notebook mẫu Kaggle: **[notebooks/kaggle_example.ipynb](notebooks/kaggle_example.ipynb)**
- Notebook mẫu Google Colab: **[notebooks/colab_example.ipynb](notebooks/colab_example.ipynb)**
- Tài liệu gói `experiment-monitor`: **[monitor/README.md](monitor/README.md)**

---

## Dashboard Pages Tour

1. **📊 Overview** (`app/pages/overview.py`):
   - 5 thẻ KPI (`Running`, `Stale / Offline`, `Queued`, `Finished (24h)`, `Failed (24h)`).
   - Thẻ **Active Runs** hiển thị tiến độ step/epoch, 5 metric chính, duration, ETA, nút mở nhanh Details / Notebook / W&B.
   - Bảng **All Runs** có lọc theo status, tìm kiếm theo tên/ID/model/tag, chọn 2–10 run để so sánh hoặc 1 run để mở chi tiết.
   - Sidebar chung: chọn Project, bật/tắt auto-refresh (10–60s), xem **Source health**.
2. **🔍 Run Detail** (`app/pages/run_detail.py`):
   - Bảng metadata đầy đủ (GPU, model, git branch/SHA, dataset, seed, eval signature), bảng checkpoints, ghi chú/tag cục bộ + pin run (lưu SQLite).
   - 4 biểu đồ Plotly theo `step` hoặc `epoch`: Loss (`train/loss`, `eval/loss`), Eval metrics (`eval/macro_f1`, `eval/accuracy`), Learning Rate, GPU Memory (kèm đường tổng VRAM).
   - 3 tab log: **Structured metrics (latest)**, **Console (stdout/stderr)** có tìm kiếm/lọc level/stream và tự động highlight dòng OOM/Exception, và **Kaggle native log** (khi có Kaggle kernel liên kết).
3. **📈 Compare** (`app/pages/compare.py`):
   - So sánh 2–10 runs theo `step` hoặc `epoch`, làm mượt hiển thị bằng EMA (không đổi dữ liệu gốc).
   - Tự động cảnh báo tính tương thích (`ecc/eval_signature`, `ecc/dataset_version`, `ecc/dataset_split`, eval cadence, trộn demo/real, thiếu metric).
   - Bảng diff siêu tham số (tô sáng cột khác biệt + nút chỉ hiện cột khác biệt), bảng best epoch/metric và nút tải CSV.
4. **📁 Projects** (`app/pages/projects.py`):
   - Tổng hợp số lượng run theo trạng thái, tạo/sửa metadata project, chọn **Baseline run** và tính **Delta** giữa Best vs Baseline, lịch sử run hoàn thành/lỗi.
5. **🔑 Connected Accounts** (`app/pages/accounts.py`):
   - Xem trạng thái các tài khoản đã kết nối, chạy kiểm tra sức khỏe (`Check` / `Check all`), thêm/cập nhật hoặc xóa tài khoản (secret lưu thẳng vào `keyring`).

---

## Claude Integration (Read-Only CLI)

Claude đọc trực tiếp trạng thái, metric, log và so sánh thí nghiệm qua CLI xuất JSON UTF-8:

```powershell
.\.venv\Scripts\python.exe -m ecc.cli health --json
.\.venv\Scripts\python.exe -m ecc.cli list-active-runs --project shared-task-4 --json
.\.venv\Scripts\python.exe -m ecc.cli compare --run <run_id_1> <run_id_2> --metric eval/macro_f1 --axis step --json
.\.venv\Scripts\python.exe -m ecc.cli best --project shared-task-4 --metric eval/macro_f1 --mode max --json
```

Xem danh sách đầy đủ 12 lệnh, JSON mẫu thực tế và quy trình hỏi–đáp mẫu tại **[docs/CLAUDE_INTEGRATION.md](docs/CLAUDE_INTEGRATION.md)**.

---

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Chạy thêm unit test riêng của gói `monitor`:

```powershell
.\.venv\Scripts\python.exe -m pytest monitor/tests -q
```

---

## Security Model

- **Lưu trữ secret**: Chỉ lưu trong **Windows Credential Manager** qua `keyring` (service `ai-ecc`, username `<platform>:<nickname>`) hoặc biến môi trường (`ECC_SECRET_<PLATFORM>_<NICKNAME>`).
- **Chặn rò rỉ trong config**: `ecc/config.py` từ chối mọi key trong `ecc_config.toml` chứa chuỗi `key`, `token`, `password`, `secret`.
- **Không bao giờ lưu vào**: SQLite (`ecc.db` chỉ chứa `secret_ref`), `ecc_config.toml`, file log, stdout của CLI, giao diện Streamlit, hay git (`.gitignore` chặn `.venv/`, `*.db`, `.env`, `ecc_config.toml`, `wandb/`, `**/kaggle.json`, `**/access_token*`).
- **Làm sạch log & lỗi**: `ecc/redact.py` và `experiment_monitor/redact.py` tự động thay thế token (`KGAT_...`, `hf_...`, `sk-...`, `ghp_...`, `wandb_v1_...`, chuỗi 40 hex, `Bearer ...`, `key=...`) thành `[REDACTED]`.
- **Chỉ đọc (Read-only)**: Kaggle adapter chỉ cho phép 3 lệnh con trong allow-list (`kernels list`, `kernels status`, `kernels logs` không có `-f`); truyền `KAGGLE_API_TOKEN` qua `env=` (không bao giờ qua argv) và cô lập `KAGGLE_CONFIG_DIR` theo từng nickname. Dashboard chỉ lắng nghe trên `127.0.0.1`.

---

## Limitations Summary

| Tính năng | W&B (`experiment_monitor`) | Kaggle (CLI native) | Google Colab (native) |
|---|---|---|---|
| Trạng thái chạy (`QUEUED`/`RUNNING`/...) | Hỗ trợ (qua heartbeat) | Hỗ trợ (cả `QUEUED` trước W&B) | Không hỗ trợ native (dùng W&B) |
| Metric & đường cong loss/F1 | Hỗ trợ đầy đủ | Không hỗ trợ | Không hỗ trợ native (dùng W&B) |
| GPU telemetry (VRAM, util) | Hỗ trợ (chỉ qua `experiment_monitor`) | Không hỗ trợ | Không hỗ trợ native (dùng W&B) |
| Console logs | Một phần (gần real-time qua `ecc_console.log`) | Một phần (log đã lưu, 1 lần; không dùng `-f`) | Không hỗ trợ native (dùng W&B) |
| Điều khiển từ xa (start/stop/execute) | Không hỗ trợ (thiết kế read-only) | Không hỗ trợ (thiết kế read-only) | Không hỗ trợ (thiết kế read-only) |

Chi tiết đầy đủ và lý do kỹ thuật: **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)**.

---

## Troubleshooting

### 1. Ý nghĩa `STALE` vs `OFFLINE` và ngưỡng thời gian
`ecc/status.py` (`derive_status`) **không bao giờ** báo `RUNNING` nếu thiếu heartbeat mới. Với run W&B có `raw_state == "running"`, tuổi heartbeat tính từ `max(heartbeatAt, summary["ecc/heartbeat"])`:
- `age ≤ stale_after_s` (mặc định **180 giây / 3 phút**) → **`RUNNING`**
- `180s < age ≤ offline_after_s` (mặc định **900 giây / 15 phút**) → **`STALE`** (có thể đang bận eval dài, nghẽn I/O, hoặc mất mạng tạm thời)
- `age > 900s` → **`OFFLINE`** (khả năng cao session Kaggle/Colab đã chết đột ngột hoặc ngắt kết nối)
- Không có `last_update` → **`UNKNOWN`**

Bạn có thể chỉnh `stale_after_s` và `offline_after_s` trong `[general]` của `ecc_config.toml` nếu bước evaluation mất hơn 3 phút. (Riêng trạng thái `RUNNING` từ Kaggle CLI được đánh dấu `RUNNING (Kaggle API)` vì Kaggle báo trực tiếp trạng thái worker.)

### 2. W&B Rate Limits (HTTP 429 / timeout)
- `WandbSource` đã có bộ nhớ đệm TTL **10 giây** trong tiến trình; dashboard mặc định tự làm mới mỗi **15 giây** (`refresh_seconds` trong config cho phép `10..300`, thanh trượt UI `10..60`).
- Nếu theo dõi nhiều project cùng lúc và gặp lỗi rate limit từ W&B API: tăng `refresh_seconds` trong `ecc_config.toml` (ví dụ `30` hoặc `60`), tắt **Enable auto-refresh** trên sidebar khi chỉ đang phân tích log/biểu đồ, hoặc lọc bớt số project trong `[wandb] projects`.

### 3. Lỗi Kaggle Token / Xác thực
- Chạy `.\.venv\Scripts\python.exe scripts\add_account.py --list` hoặc bấm **Check** tại trang **Connected Accounts** để xem thông báo lỗi (đã redact secret).
- Nếu báo `No Kaggle API token available` hoặc HTTP `401 Unauthorized`: tạo token mới tại `https://www.kaggle.com/settings/api` rồi chạy lại `scripts\add_account.py --platform kaggle --nickname <nick>` để cập nhật vào Windows Credential Manager.
- ECC không đọc hay sửa `~/.kaggle/kaggle.json`; mỗi tài khoản chạy với `KAGGLE_API_TOKEN` riêng và thư mục `KAGGLE_CONFIG_DIR` cô lập tại `%LOCALAPPDATA%\ai-ecc\kaggle\<nickname>`.
