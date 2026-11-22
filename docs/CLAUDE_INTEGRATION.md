# Claude Integration Guide (`ecc.cli`)

Claude (Research Lead / Orchestrator) đọc dữ liệu thí nghiệm trực tiếp từ backend dùng chung `ecc.service` thông qua giao diện CLI chỉ-đọc:

```powershell
.\.venv\Scripts\python.exe -m ecc.cli <command> [options] --json
```

### Quy ước đầu ra & mã thoát (`ecc/cli.py`)
- Luôn xuất **JSON UTF-8** ra `stdout` (cờ `--json` được chấp nhận để tương thích và mặc định luôn bật).
- Mọi timestamp ở định dạng **ISO-8601 UTC**; giá trị `NaN`/`Inf` được chuyển thành `null`.
- Không bao giờ in secrets ra stdout/stderr.
- Mã thoát (exit code):
  - `0`: thành công.
  - `1`: lỗi tham số hoặc cấu hình (`{"error": "invalid_argument", ...}`).
  - `2`: không tìm thấy `run_id` (`{"error": "run_not_found", ...}`).

> Tất cả các ví dụ JSON dưới đây là **output thực tế** khi chạy `.\.venv\Scripts\python.exe -m ecc.cli ...` ở chế độ Demo (`demo_mode = "auto"` khi chưa cấu hình W&B), đã cắt gọn các phần lặp dài để dễ đọc.

---

## 1. Command Reference

| Command | Mục đích | Cờ chính |
|---|---|---|
| `health` | Trạng thái tổng thể, `demo_mode`, `db_path`, sức khỏe từng source | `--config`, `--json` |
| `list-projects` | Liệt kê tất cả project (từ SQLite + các source) | `--config`, `--json` |
| `list-runs` | Liệt kê run, sắp xếp active trước rồi mới nhất trước | `--project`, `--status` |
| `list-active-runs` | Liệt kê các run đang `QUEUED`, `RUNNING`, hoặc `STALE` | `--project` |
| `get-run` | Chi tiết đầy đủ 1 run (kèm `kaggle_status` nếu có liên kết) | `--run RUN_ID` |
| `get-metrics` | Chuỗi thời gian metric dạng list các bản ghi JSON | `--run RUN_ID`, `--metric k1,k2` (hoặc lặp `--metric`), `--max-points` (mặc định 500) |
| `get-logs` | Lấy đuôi console log (`ecc_console.log` hoặc native Kaggle log) | `--run RUN_ID`, `--tail` (500), `--query`, `--level INFO\|WARNING\|ERROR`, `--source kaggle` |
| `compare` | So sánh 2–10 run: trả về `runs`, `table`, `curves`, `warnings` | `--run ID1 ID2 ...`, `--metric` (mặc định `eval/macro_f1`), `--axis step\|epoch`, `--max-points` |
| `failed` | Các run `FAILED` / `CRASHED` / `KILLED` trong cửa sổ thời gian | `--project`, `--since-hours` (mặc định 72) |
| `best` | Run tốt nhất theo metric + so sánh `delta` với baseline của project | `--project` (bắt buộc), `--metric` (`eval/macro_f1`), `--mode max\|min` |
| `accounts` | Danh sách tài khoản đã cấu hình, trạng thái secret (bool), capabilities/limitations | `--config`, `--json` |
| `kaggle-kernels` | Danh sách Kaggle kernels từ các tài khoản Kaggle đã bật | `--account NICKNAME` |

---

## 2. Example Outputs (Demo Mode)

### `health`
```powershell
.\.venv\Scripts\python.exe -m ecc.cli health --json
```
```json
{
  "ok": true,
  "demo_mode": "auto",
  "db_path": "C:\\Users\\<USER>\\AppData\\Local\\Temp\\ai-ecc\\ecc.db",
  "sources": {
    "demo": {
      "ok": true,
      "message": "Demo source active (7 synthetic runs across 2 projects)"
    }
  }
}
```

### `list-projects`
```powershell
.\.venv\Scripts\python.exe -m ecc.cli list-projects --json
```
```json
[
  {
    "slug": "demo-cv",
    "name": "demo-cv",
    "description": "",
    "tags": [],
    "notes": "",
    "baseline_run_id": null
  },
  {
    "slug": "demo-shared-task",
    "name": "demo-shared-task",
    "description": "",
    "tags": [],
    "notes": "",
    "baseline_run_id": null
  }
]
```

### `list-active-runs` (rút gọn 1 phần tử mẫu)
```powershell
.\.venv\Scripts\python.exe -m ecc.cli list-active-runs --project demo-shared-task --json
```
```json
[
  {
    "run_id": "demo:demo/demo-shared-task/run-a",
    "source": "demo",
    "project": "demo-shared-task",
    "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32",
    "status": "RUNNING",
    "raw_state": "running",
    "platform": "kaggle",
    "account": "kaggle-1",
    "gpu": "NVIDIA Tesla T4",
    "model_name": "Qwen/Qwen3-VL-4B-Instruct",
    "step": 620,
    "total_steps": 1000,
    "epoch": 1.86,
    "total_epochs": 3.0,
    "progress": 0.62,
    "latest": {
      "train/loss": 0.6856,
      "eval/loss": 0.70413,
      "eval/accuracy": 0.78971,
      "eval/macro_f1": 0.77093,
      "train/lr": 7.58078e-05,
      "gpu/mem_alloc_gb": 11.381,
      "gpu/max_mem_alloc_gb": 11.644
    },
    "best_metric_name": "eval/macro_f1",
    "best_metric": 0.77093,
    "best_step": 600,
    "best_epoch": 1.8,
    "duration_s": 1984.0,
    "eta_s": 1216.0,
    "is_demo": true
  }
]
```
*(Kết quả đầy đủ ở demo mode gồm 3 active runs trong `demo-shared-task`: `run-a` `RUNNING`, `run-b` `RUNNING`, và `run-c` `STALE`.)*

### `get-metrics`
```powershell
.\.venv\Scripts\python.exe -m ecc.cli get-metrics --run demo:demo/demo-shared-task/run-a --metric train/loss,eval/loss,eval/macro_f1 --max-points 5 --json
```
```json
[
  {
    "step": 10,
    "epoch": 0.03,
    "train/loss": 2.07037,
    "eval/loss": null,
    "eval/macro_f1": null
  },
  {
    "step": 160,
    "epoch": 0.48,
    "train/loss": 1.47199,
    "eval/loss": null,
    "eval/macro_f1": null
  },
  {
    "step": 310,
    "epoch": 0.93,
    "train/loss": 1.0255,
    "eval/loss": null,
    "eval/macro_f1": null
  },
  {
    "step": 460,
    "epoch": 1.38,
    "train/loss": 0.81296,
    "eval/loss": null,
    "eval/macro_f1": null
  },
  {
    "step": 620,
    "epoch": 1.86,
    "train/loss": 0.6856,
    "eval/loss": null,
    "eval/macro_f1": null
  }
]
```

### `get-logs` (ví dụ run bị OOM)
```powershell
.\.venv\Scripts\python.exe -m ecc.cli get-logs --run demo:demo/demo-shared-task/run-e --tail 5 --json
```
```json
[
  {
    "ts": "2026-11-22T12:37:13.642658+00:00",
    "stream": "stderr",
    "text": "WARN: DataLoader worker heartbeat delayed; waiting on I/O.",
    "level": "WARNING"
  },
  {
    "ts": "2026-11-22T12:37:12.642658+00:00",
    "stream": "stderr",
    "text": "Traceback (most recent call last):",
    "level": "ERROR"
  },
  {
    "ts": "2026-11-22T12:37:13.642658+00:00",
    "stream": "stderr",
    "text": "  File \"/kaggle/working/train.py\", line 184, in train_step",
    "level": "INFO"
  },
  {
    "ts": "2026-11-22T12:37:13.642658+00:00",
    "stream": "stderr",
    "text": "    loss.backward()",
    "level": "INFO"
  },
  {
    "ts": "2026-11-22T12:37:14.642658+00:00",
    "stream": "stderr",
    "text": "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 14.75 GiB total capacity; 13.84 GiB already allocated; 412.00 MiB free).",
    "level": "ERROR"
  }
]
```

### `compare` (rút gọn `runs` và chỉ giữ `table`, mẫu `curves`, `warnings`)
```powershell
.\.venv\Scripts\python.exe -m ecc.cli compare --run demo:demo/demo-shared-task/run-a demo:demo/demo-shared-task/run-b demo:demo/demo-shared-task/run-c --metric eval/macro_f1 --axis step --max-points 3 --json
```
```json
{
  "runs": ["...3 RunSummary objects..."],
  "table": [
    {
      "run_id": "demo:demo/demo-shared-task/run-a",
      "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32",
      "platform": "kaggle",
      "account": "kaggle-1",
      "gpu": "NVIDIA Tesla T4",
      "model": "Qwen/Qwen3-VL-4B-Instruct",
      "lora_r": 16,
      "lora_alpha": 32,
      "learning_rate": 0.0002,
      "prompt_strategy": "vi-default",
      "seed": 42,
      "dataset_version": "v2",
      "best_metric": 0.77093,
      "best_step": 600,
      "best_epoch": 1.8,
      "duration_s": 1984.0,
      "final train/loss": 0.6856,
      "final eval/loss": 0.70413,
      "eval/accuracy": 0.78971,
      "eval/macro_f1": 0.77093,
      "max gpu/mem_alloc_gb": 11.644
    },
    {
      "run_id": "demo:demo/demo-shared-task/run-b",
      "name": "[DEMO] Qwen3-VL-4B LoRA r32/a64",
      "platform": "kaggle",
      "account": "kaggle-2",
      "gpu": "NVIDIA Tesla P100",
      "model": "Qwen/Qwen3-VL-4B-Instruct",
      "lora_r": 32,
      "lora_alpha": 64,
      "learning_rate": 0.00015,
      "prompt_strategy": "vi-default",
      "seed": 42,
      "dataset_version": "v2",
      "best_metric": 0.72769,
      "best_step": 400,
      "best_epoch": 1.2,
      "duration_s": 1392.0,
      "final train/loss": 0.76722,
      "final eval/loss": 0.99261,
      "eval/accuracy": 0.74728,
      "eval/macro_f1": 0.72769,
      "max gpu/mem_alloc_gb": 13.417
    },
    {
      "run_id": "demo:demo/demo-shared-task/run-c",
      "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32 + EN prompt",
      "platform": "colab",
      "account": "colab-1",
      "gpu": "NVIDIA L4",
      "model": "Qwen/Qwen3-VL-4B-Instruct",
      "lora_r": 16,
      "lora_alpha": 32,
      "learning_rate": 0.0002,
      "prompt_strategy": "en-prompt",
      "seed": 42,
      "dataset_version": "v2",
      "best_metric": 0.73268,
      "best_step": 500,
      "best_epoch": 1.5,
      "duration_s": 1350.0,
      "final train/loss": 0.71171,
      "final eval/loss": 0.87881,
      "eval/accuracy": 0.75659,
      "eval/macro_f1": 0.73268,
      "max gpu/mem_alloc_gb": 12.322
    }
  ],
  "curves": {
    "eval/macro_f1": [
      { "run_id": "demo:demo/demo-shared-task/run-a", "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32", "x": 100, "value": 0.52966 },
      { "run_id": "demo:demo/demo-shared-task/run-a", "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32", "x": 300, "value": 0.66151 },
      { "run_id": "demo:demo/demo-shared-task/run-a", "name": "[DEMO] Qwen3-VL-4B LoRA r16/a32", "x": 600, "value": 0.77093 }
    ]
  },
  "warnings": [
    "Runs have differing ecc/eval_signature values (['v2-en-prompt-90-10-seed42', 'v2-vi-90-10-seed42']); validation comparability is not guaranteed."
  ]
}
```

### `failed` & `best`
```powershell
.\.venv\Scripts\python.exe -m ecc.cli failed --project demo-shared-task --since-hours 72 --json
.\.venv\Scripts\python.exe -m ecc.cli best --project demo-shared-task --metric eval/macro_f1 --mode max --json
```
Kết quả `best` (rút gọn phần `config` trong `best`):
```json
{
  "project": "demo-shared-task",
  "metric": "eval/macro_f1",
  "mode": "max",
  "best": {
    "run_id": "demo:demo/demo-shared-task/run-d",
    "name": "[DEMO] Qwen3-VL-4B LoRA r8/a16 baseline",
    "status": "FINISHED",
    "best_metric_name": "eval/macro_f1",
    "best_metric": 0.77432,
    "best_step": 900,
    "best_epoch": 2.7
  },
  "best_value": 0.77432,
  "baseline": null,
  "baseline_value": null,
  "delta": null
}
```

### `accounts` & `kaggle-kernels`
Khi chưa cấu hình tài khoản nào trong SQLite cục bộ:
```powershell
.\.venv\Scripts\python.exe -m ecc.cli accounts --json
# []
.\.venv\Scripts\python.exe -m ecc.cli kaggle-kernels --json
# []
```

---

## 3. Example Question → Command Sequence

**Câu hỏi của người dùng:**
> *"Ba runs Qwen3-VL mới nhất tiến triển thế nào? Run nào macro F1 tốt nhất? Có overfitting không?"*

**Chuỗi lệnh Claude chạy:**

1. **Liệt kê các runs trong project** để lấy 3 `run_id` Qwen3-VL mới nhất, trạng thái (`RUNNING`/`STALE`/`FINISHED`/`FAILED`), tiến độ và ETA:
   ```powershell
   .\.venv\Scripts\python.exe -m ecc.cli list-runs --project demo-shared-task --json
   ```
2. **So sánh trực tiếp 3 runs** (bảng siêu tham số, đường cong `eval/macro_f1` & `train/loss`/`eval/loss`, và cảnh báo comparability):
   ```powershell
   .\.venv\Scripts\python.exe -m ecc.cli compare --run demo:demo/demo-shared-task/run-a demo:demo/demo-shared-task/run-b demo:demo/demo-shared-task/run-c --metric eval/macro_f1 --axis step --json
   ```
3. **Kiểm tra run có macro F1 cao nhất toàn project** (bao gồm cả các run đã `FINISHED` như baseline):
   ```powershell
   .\.venv\Scripts\python.exe -m ecc.cli best --project demo-shared-task --metric eval/macro_f1 --mode max --json
   ```
4. **Lấy chi tiết chuỗi `train/loss`, `eval/loss`, `eval/macro_f1`** của run nghi ngờ overfitting (không giảm `--max-points` quá thấp để không bỏ qua các bước eval):
   ```powershell
   .\.venv\Scripts\python.exe -m ecc.cli get-metrics --run demo:demo/demo-shared-task/run-b --metric train/loss,eval/loss,eval/macro_f1 --max-points 500 --json
   ```

**Cách Claude tổng hợp từ output demo ở trên:**
- **Tiến độ 3 runs mới nhất**: `run-a` (`LoRA r16/a32`, `kaggle-1`, T4) đang `RUNNING` ở step `620/1000` (62%, ETA ~20m); `run-b` (`LoRA r32/a64`, `kaggle-2`, P100) đang `RUNNING` ở step `480/1000` (48%, ETA ~25m); `run-c` (`EN prompt`, `colab-1`, L4) ở step `540/1000` nhưng đã **`STALE`** (quá 3 phút chưa có heartbeat).
- **Macro F1 tốt nhất**: Toàn project là `run-d` (`FINISHED`, `0.77432` tại step 900); trong 3 runs đang chạy là `run-a` (`0.77093` tại step 600). Cần lưu ý cảnh báo `ecc/eval_signature` của `run-c` khác `run-a`/`run-b` nên không so sánh trực tiếp ngang hàng.
- **Dấu hiệu overfitting**: Kiểm tra khoảng cách `eval/loss - train/loss` và xu hướng `eval/loss` qua các bước eval (`run-a`: `train/loss=0.6856` vs `eval/loss=0.7041` khá sát; `run-b`: `train/loss=0.7672` vs `eval/loss=0.9926` chênh lệch lớn hơn — cần theo dõi các mốc eval tiếp theo).

---

## 4. Optional: Official W&B MCP Server

`python -m ecc.cli` đã đủ cho toàn bộ workflow theo dõi, so sánh, đọc log và kiểm tra sức khỏe trên cả W&B + Kaggle + SQLite cục bộ.

Nếu cần chạy các truy vấn **GraphQL tùy biến (ad-hoc)** trực tiếp vào W&B (ví dụ gom nhóm theo nhiều project ngoài `ecc_config.toml` hoặc truy vấn bảng artifacts), bạn có thể tùy chọn cài thêm **official W&B MCP server**, nhưng **không bắt buộc** cho ECC.
