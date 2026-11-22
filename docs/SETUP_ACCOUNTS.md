# Account & Platform Setup Guide

Hướng dẫn từng bước cấu hình **Weights & Biases (W&B)**, **3 tài khoản Kaggle**, **2 tài khoản Google Colab**, và cách liên kết Kaggle kernel vào project trên ECC.

> **Quy tắc bảo mật:** Không bao giờ dán API key/token vào `ecc_config.toml`, notebook source, SQLite, hay git. Không bao giờ dùng địa chỉ e-mail làm `nickname`.

---

## 1. Weights & Biases (Single Owner Workspace)

### Mô hình lưu trữ key (quan trọng)
Bạn chỉ cần **1 tài khoản/workspace W&B duy nhất** để gom toàn bộ thí nghiệm từ mọi notebook Kaggle và Colab về cùng một nơi (khuyến nghị tạo một **W&B entity / project riêng** cho từng bài toán, ví dụ `shared-task-4`).

W&B API key (lấy tại `https://wandb.ai/authorize`) được lưu ở hai nơi:
1. **Một lần trên máy tính Windows (PC chạy ECC):**
   - Lưu vào Windows Credential Manager qua `scripts\add_account.py` để trang **Connected Accounts** kiểm tra trạng thái xác thực.
   - Đồng thời đăng nhập W&B cục bộ (`wandb login` hoặc đặt biến môi trường `WANDB_API_KEY`) để `WandbSource` (gọi `wandb.Api()`) có quyền đọc danh sách run, metric và file log.
2. **Một bản sao trong trình quản lý Secret của từng tài khoản notebook (Kaggle / Colab):**
   - Mỗi tài khoản Kaggle (`kaggle-1`, `kaggle-2`, `kaggle-3`) và mỗi tài khoản Google Colab (`colab-1`, `colab-2`) lưu riêng một bản sao `WANDB_API_KEY` của bạn trong trình quản lý Secret gốc của nền tảng đó.
   - Nhờ vậy, tất cả các notebook trên 5 tài khoản khác nhau đều ghi dữ liệu thẳng vào **cùng 1 W&B entity/project** mà không để lộ key trong mã nguồn notebook.

### Bước thực hiện trên PC Windows

1. Chỉnh `ecc_config.toml` (tạo từ `ecc_config.example.toml` khi chạy `.\setup.ps1`):

   ```toml
   [general]
   refresh_seconds = 15
   stale_after_s = 180
   offline_after_s = 900
   demo_mode = "auto"

   [wandb]
   entity = "<your-wandb-entity>"
   projects = []   # rỗng = tự phát hiện mọi project của entity
   ```

2. Đăng ký tài khoản W&B vào Windows Credential Manager (service `ai-ecc`, username `wandb:wandb-main`):

   ```powershell
   .\.venv\Scripts\python.exe scripts\add_account.py --platform wandb --nickname wandb-main --from-clipboard
   ```

   Script chờ bạn copy API key rồi nhấn Enter, đọc clipboard, xóa clipboard và kiểm tra xác thực W&B. Dashboard (`WandbSource`) dùng chính key này để đọc run — không cần `wandb login`.

3. (Tùy chọn) Không lưu key ở bước 2 thì `WandbSource` dùng cơ chế mặc định của wandb: `$env:WANDB_API_KEY` hoặc `.\.venv\Scripts\wandb.exe login`.

---

## 2. Three Kaggle Accounts (`kaggle-1`, `kaggle-2`, `kaggle-3`)

### Bước 1: Tạo API Token cho từng tài khoản
1. Đăng nhập tài khoản Kaggle thứ nhất trên trình duyệt (hoặc profile/incognito riêng).
2. Mở **`https://www.kaggle.com/settings/api`** và tạo API token.
3. Lặp lại cho tài khoản Kaggle thứ hai và thứ ba.

### Bước 2: Thêm lần lượt 3 tài khoản vào ECC
Chạy 3 lệnh sau trong một cửa sổ PowerShell thường (khi script nhắc, copy token của đúng tài khoản rồi nhấn Enter; `--username` là nhãn tùy chọn, không dùng e-mail):

```powershell
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-1 --from-clipboard
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-2 --from-clipboard
.\.venv\Scripts\python.exe scripts\add_account.py --platform kaggle --nickname kaggle-3 --from-clipboard
.\.venv\Scripts\python.exe scripts\verify_accounts.py   # xác nhận 3 nickname = 3 tài khoản khác nhau
```

> Terminal tích hợp trong một số app không dán được vào ô nhập ẩn — khi đó dùng `--from-clipboard` như trên.

Mỗi lệnh sẽ:
- Kiểm tra nickname hợp lệ (`^[a-z0-9][a-z0-9_-]{1,31}$`).
- Lưu token vào **Windows Credential Manager** dưới service `ai-ecc`, username `kaggle:kaggle-1` (không lưu token vào SQLite).
- Chạy lệnh kiểm tra chỉ-đọc `kaggle kernels list --mine --page-size 1` để xác minh token hợp lệ.

Kiểm tra lại toàn bộ tài khoản đã thêm:

```powershell
.\.venv\Scripts\python.exe scripts\add_account.py --list
```

### Cách cơ chế cô lập nhiều tài khoản Kaggle hoạt động (`ecc/adapters/kaggle.py`)
- Mỗi khi ECC gọi `kaggle` CLI cho một tài khoản, adapter **xóa toàn bộ biến môi trường `KAGGLE_*` kế thừa** từ shell hiện tại.
- Adapter tiêm trực tiếp:
  - `KAGGLE_API_TOKEN` qua `subprocess.run(..., env=env)` (**không bao giờ** truyền qua dòng lệnh `argv`).
  - `KAGGLE_CONFIG_DIR` riêng biệt cho từng tài khoản tại `%LOCALAPPDATA%\ai-ecc\kaggle\<nickname>` (ví dụ `...\ai-ecc\kaggle\kaggle-1`).
- Thư mục mặc định `~/.kaggle` của máy bạn **không bao giờ bị đọc hay ghi đè**.
- Bạn **không cần logout/login** giữa 3 tài khoản Kaggle; `KaggleSource` truy vấn đồng thời tối đa 3 tài khoản song song (`max_workers=3`).
- Chỉ cho phép 3 subcommand read-only trong allow-list: `kernels list`, `kernels status`, và `kernels logs` (chặn `-f` / `--follow`).

### Bước 3: Thêm `WANDB_API_KEY` vào Kaggle Secrets (cho từng tài khoản)
Trên **mỗi tài khoản Kaggle** (`kaggle-1`, `kaggle-2`, `kaggle-3`):
1. Mở notebook training → chọn menu **Add-ons → Secrets**.
2. Thêm secret có Label = `WANDB_API_KEY`, Value = W&B API key của bạn.
3. Bật công tắc **Attach to notebook** cho `WANDB_API_KEY`.
4. Trong cell `em.init(...)`, truyền đúng `account="kaggle-1"` (hoặc `"kaggle-2"`, `"kaggle-3"`) và `kaggle_kernel="<owner>/<slug>"`.

---

## 3. Two Google Colab Accounts (`colab-1`, `colab-2`)

### Không cần credential Colab native trong MVP
ECC MVP **không tích hợp API Colab native** (vì Colab không có API chính thức cho trạng thái thực thi hay log notebook). Do đó bạn **không cần** cấp OAuth Google hay lưu token Colab nào trên máy PC.

### Các bước thiết lập
1. Trên **mỗi tài khoản Google Colab** (`colab-1`, `colab-2`):
   - Mở biểu tượng **🔑 Secrets** ở thanh công cụ bên trái của Colab.
   - Bấm **Add new secret**: **Name** = `WANDB_API_KEY`, **Value** = W&B API key của bạn.
   - Bật công tắc **Notebook access** cho `WANDB_API_KEY`.
2. Trong notebook Colab, truyền `account="colab-1"` (hoặc `"colab-2"`) và `notebook_url="https://colab.research.google.com/drive/<file_id>"` vào `em.init(...)`. Nickname `colab-1`/`colab-2` chỉ là nhãn nhận diện không nhạy cảm hiển thị trên dashboard và CLI.
3. *(Tùy chọn)* Đăng ký nhãn tài khoản Colab vào bảng Connected Accounts để tiện theo dõi (dùng `--no-check` vì Colab không có native health check):

   ```powershell
   .\.venv\Scripts\python.exe scripts\add_account.py --platform colab --nickname colab-1 --username <colab-alias-1> --no-check
   .\.venv\Scripts\python.exe scripts\add_account.py --platform colab --nickname colab-2 --username <colab-alias-2> --no-check
   ```

---

## 4. Linking a Kaggle Kernel to an ECC Project

Mặc định, một Kaggle kernel chưa gắn với project nào (ví dụ kernel đang `QUEUED` trước khi chạy tới `em.init`, hoặc kernel chưa gắn `experiment_monitor`) sẽ xuất hiện dưới project `"unassigned"`. Có 2 cách để liên kết kernel với project:

### Cách 1 (Khuyến nghị — tự động qua `ecc/kaggle_kernel`)
Trong notebook Kaggle, truyền `kaggle_kernel="<owner>/<slug>"` vào `em.init(...)`:

```python
mon = em.init(
    project="shared-task-4",
    name="qwen3-vl-lora-r16",
    account="kaggle-1",
    notebook_url="https://www.kaggle.com/code/<owner>/<slug>",
    kaggle_kernel="<owner>/<slug>",
    ...
)
```

`experiment_monitor` ghi khóa `ecc/kaggle_kernel` vào W&B run config. Khi `Service.list_runs()` chạy:
- ECC tự ghép trạng thái Kaggle (`kaggle_status`) và tab **Kaggle native log** vào đúng W&B run đó.
- Dòng Kaggle độc lập trùng lặp tự động được ẩn đi.

### Cách 2 (Thủ công qua `Service.link_kernel` cho kernel chưa gắn W&B)
`python -m ecc.cli` được thiết kế **chỉ đọc (read-only)** nên không có lệnh ghi `link-kernel`. Để gắn thủ công một Kaggle kernel (`<owner>/<slug>`) vào project (và tùy chọn vào một `run_id` W&B cụ thể) trong SQLite cục bộ (`kernel_links`), chạy một lệnh Python ngắn từ thư mục repo:

```powershell
.\.venv\Scripts\python.exe -c "from ecc.service import build_service; svc = build_service(); print(svc.link_kernel(kernel_ref='<owner>/<slug>', account='kaggle-1', project='shared-task-4', run_id=None))"
```

Nếu muốn gắn với một W&B run đã tồn tại, thay `run_id=None` bằng `run_id='wandb:<entity>/<project>/<id>'`.
