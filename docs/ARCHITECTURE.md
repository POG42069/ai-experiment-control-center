# AI Experiment Control Center (ECC) — Architecture & Data Contract

Status: v0.1 MVP design · Owner: Claude (Research Lead) · Implementation: Gemini 4 Neon

## 1. Feasibility summary (verified 2026-11-22)

| Source | What is officially possible | What is NOT possible | Used for |
|---|---|---|---|
| **W&B** (`wandb` 0.31, public API `wandb.Api`) | List runs per project, `state`, `config`, `summary`, `history`/`scan_history`, `heartbeatAt`, `createdAt`, run files (`run.files()`, `run.file(name)`), tags, notes, URL. Any number of machines can log into the same entity/project with the owner's API key. `wandb.save(path, policy="live")` re-uploads a file whenever it changes. | No first-class "console log" API for live runs (the UI's console stream is not exposed reliably by the public API). | **Primary source of truth** for metrics, metadata, status, live console tail (via our live-synced `ecc_console.log`). |
| **Kaggle** (`kaggle` CLI 2.2.5) | `kernels list --mine --csv`, `kernels status owner/slug` (`QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT`), `kernels logs owner/slug` (persisted log) and `kernels logs -f` (live SSE stream of a running session). Auth via `KAGGLE_API_TOKEN` env var (wins over `~/.kaggle/access_token`) — so each account = one token injected into a subprocess env + isolated `KAGGLE_CONFIG_DIR`. | GPU telemetry, structured metrics. | Native status (incl. QUEUED before W&B run exists), native console logs, notebook listing/linking. |
| **Colab** (`colaboratory.googleapis.com` v1beta) | `GET /v1beta/runtimes` (runtime name + spec: variant, accelerator e.g. T4/L4/A100, shape), `GET /v1beta/subscription`, MCP `list_sessions` (session names only). | No execution status, no logs, no progress, no notebook↔runtime mapping. Auth = Google OAuth per account (Cloud MCP/API enablement required; consumer-account support not documented). Many endpoints are **destructive** (create/delete runtime, execute_code). | MVP: **not integrated natively** (read-only adapter is a documented Phase-3 option). Colab runs are monitored via W&B instrumentation + stored notebook URL. |

Rule: the dashboard never claims a capability the source does not provide. Missing data is shown as `—` / `UNKNOWN`.

## 2. Components

```
ai-experiment-control-center/
├─ ecc/                     # backend library (dashboard + Claude CLI share it)
│  ├─ models.py             # dataclasses (section 4)
│  ├─ status.py             # derive_status()  (section 5)
│  ├─ config.py             # load ecc_config.toml (no secrets)
│  ├─ secrets.py            # keyring wrapper, service "ai-ecc"
│  ├─ store.py              # SQLite local metadata (section 6)
│  ├─ sources/base.py       # RunSource protocol
│  ├─ sources/demo.py       # synthetic data, every run flagged is_demo=True
│  ├─ sources/wandb_source.py
│  ├─ adapters/kaggle.py    # read-only Kaggle CLI subprocess adapter, multi-account
│  ├─ adapters/colab.py     # capability stub: reports "not supported natively"
│  ├─ service.py            # high-level read-only API (section 7)
│  └─ cli.py                # `python -m ecc.cli <cmd> --json` for Claude
├─ app/                     # Streamlit UI (multipage)
├─ monitor/                 # pip-installable notebook package `experiment_monitor`
├─ notebooks/               # Kaggle & Colab example notebooks
├─ tests/                   # backend tests;  monitor/tests/ for the package
└─ ecc_config.example.toml
```

Data flow: notebook → `experiment_monitor` → W&B cloud ← `WandbSource` ← `service` ← (Streamlit UI | `ecc.cli` for Claude). Kaggle adapter enriches runs whose `ecc/kaggle_kernel` config is set, and lists Kaggle kernels that have no W&B run yet.

## 3. W&B logging contract (written by `experiment_monitor`, read by `WandbSource`)

W&B project = one ECC project (e.g. `shared-task-4`). Entity = the user's W&B entity.

**Config keys** (`run.config`, flat, prefix `ecc/` for monitor-owned fields):

| key | type | example |
|---|---|---|
| `ecc/schema_version` | int | `1` |
| `ecc/project` | str | `shared-task-4` |
| `ecc/platform` | `kaggle` \| `colab` \| `local` \| `other` | auto-detected |
| `ecc/account` | str nickname (never e-mail) | `kaggle-1` |
| `ecc/notebook_url` | str | `https://www.kaggle.com/code/u/slug` |
| `ecc/kaggle_kernel` | str `owner/slug` or empty | |
| `ecc/git_branch`, `ecc/git_sha` | str | |
| `ecc/dataset_version`, `ecc/dataset_split` | str | `v2`, `train/val=90/10 seed42` |
| `ecc/seed` | int | |
| `ecc/model_name` | str | `Qwen/Qwen3-VL-4B-Instruct` |
| `ecc/prompt_strategy` | str | `vi-default` / `en-prompt` |
| `ecc/total_steps`, `ecc/total_epochs` | int / float | |
| `ecc/gpu_name`, `ecc/gpu_count`, `ecc/gpu_mem_total_gb` | str/int/float | |
| `ecc/eval_signature` | str | hash/string of validation set + eval settings, used for comparability |
| any user hyperparameters | as given | `learning_rate`, `lora_r`, `lora_alpha`, … |

**Metric keys** (`wandb.log`):
`train/loss`, `train/lr`, `train/grad_norm`, `train/epoch`, `train/global_step`, `eval/loss`, `eval/accuracy`, `eval/macro_f1`, `eval/<any>`, `gpu/mem_alloc_gb`, `gpu/mem_reserved_gb`, `gpu/max_mem_alloc_gb`, `gpu/util_pct` (optional), `ecc/heartbeat` (unix ts, logged ≤ every 30 s by a background thread), `ecc/progress` (0–1).
Step axis: `train/global_step` (also used as W&B `step`).

**Summary keys**: `ecc/best_metric_name`, `ecc/best_metric`, `ecc/best_step`, `ecc/best_epoch`, `ecc/end_reason` (`finished` \| `exception` \| `interrupted`), `ecc/exception_type`, `ecc/exception_message` (first 500 chars, secrets redacted), `ecc/checkpoints` (list of `{step, path, size_mb, metric}`).

**Files**: `ecc_console.log` — tee of stdout/stderr, ring-capped (default 2 MB, keeps tail), lines prefixed `ISO8601 [stdout|stderr] `, synced with `wandb.save(..., policy="live")`. Secret-looking strings redacted before writing.

## 4. Backend models (`ecc/models.py`)

```python
class RunStatus(str, Enum): QUEUED, RUNNING, STALE, OFFLINE, FINISHED, FAILED, CRASHED, KILLED, UNKNOWN

@dataclass
class RunSummary:
    run_id: str            # globally unique: f"{source}:{entity}/{project}/{id}" e.g. "wandb:me/shared-task-4/abc123", "demo:demo/qwen/run-a"
    source: str            # "wandb" | "demo" | "kaggle"
    project: str
    name: str
    status: RunStatus
    raw_state: str | None  # source's own state string
    platform: str | None; account: str | None; gpu: str | None
    model_name: str | None
    config: dict           # full config
    tags: list[str]
    step: int | None; total_steps: int | None
    epoch: float | None; total_epochs: float | None
    progress: float | None # 0..1, from ecc/progress or step/total_steps
    latest: dict[str, float]   # latest train/loss, eval/loss, eval/accuracy, eval/macro_f1, train/lr ...
    best_metric_name: str | None; best_metric: float | None; best_step: int | None; best_epoch: float | None
    started_at: datetime | None; ended_at: datetime | None; last_update: datetime | None  # tz-aware UTC
    duration_s: float | None; eta_s: float | None
    notebook_url: str | None; source_url: str | None
    git_branch: str | None; git_sha: str | None
    dataset_version: str | None; seed: int | None
    is_demo: bool

@dataclass
class LogLine:  ts: datetime | None; stream: str  # "stdout"|"stderr"|"kaggle"|"system"
               text: str; level: str             # "INFO"|"WARNING"|"ERROR"  (derived)
@dataclass
class Project: slug: str; name: str; description: str; tags: list[str]; notes: str; baseline_run_id: str | None
```
`metrics` are returned as a `pandas.DataFrame` with columns `step`, `epoch`, `<metric keys>` (NaN when absent).

## 5. Status derivation (`ecc/status.py`) — single function, fully unit tested

```python
derive_status(raw_state: str | None, last_update: datetime | None, now: datetime,
              stale_after_s=180, offline_after_s=900) -> RunStatus
```
- W&B `finished`→FINISHED, `failed`→FAILED, `crashed`→CRASHED, `killed`/`preempted`→KILLED, `pending`→QUEUED.
- Kaggle `QUEUED`/`NEW_SCRIPT`→QUEUED, `COMPLETE`→FINISHED, `ERROR`→FAILED, `CANCEL_*`→KILLED, `RUNNING`→(heartbeat rule).
- `running`: no `last_update`→UNKNOWN; age ≤ stale_after→RUNNING; ≤ offline_after→STALE; else OFFLINE.
- anything else / None → UNKNOWN. Never returns RUNNING without a fresh heartbeat.
`last_update` for W&B = max(`heartbeatAt`, summary `ecc/heartbeat`).

## 6. Local store (`ecc/store.py`, SQLite at `%LOCALAPPDATA%\ai-ecc\ecc.db`, overridable)

Only user-authored metadata; never metrics or secrets.
```sql
projects(slug TEXT PK, name TEXT, description TEXT, wandb_entity TEXT, wandb_project TEXT,
         tags TEXT /*json*/, notes TEXT, baseline_run_id TEXT, created_at TEXT, updated_at TEXT)
run_notes(run_id TEXT PK, notes TEXT, tags TEXT /*json*/, pinned INTEGER, updated_at TEXT)
accounts(nickname TEXT PK, platform TEXT CHECK(platform IN ('kaggle','colab','wandb')),
         username TEXT, secret_ref TEXT /*keyring key name, never the secret*/, enabled INTEGER,
         last_check_at TEXT, last_check_ok INTEGER, last_check_msg TEXT)
kernel_links(kernel_ref TEXT PK /*owner/slug*/, account TEXT, project TEXT, run_id TEXT)
```

## 7. Read-only service API (`ecc/service.py`) — mirrored 1:1 by `python -m ecc.cli`

`list_projects()`, `list_runs(project=None, status=None)`, `list_active_runs(project=None)` (QUEUED/RUNNING/STALE),
`get_run_details(run_id)`, `get_run_metrics(run_id, keys=None)`, `get_run_logs(run_id, tail=500, query=None, level=None)`,
`compare_runs(run_ids, metric="eval/macro_f1", axis="step"|"epoch")` → `{runs, table, curves, warnings}`,
`get_failed_runs(project=None, since_hours=72)`, `get_best_experiment(project, metric="eval/macro_f1", mode="max")`.

Comparability warnings (`compare_runs`): differing `ecc/eval_signature`, `ecc/dataset_version`, `ecc/dataset_split`, or eval cadence; any run is demo while others are real; metric missing in a run.

CLI output: JSON to stdout, UTF-8, datetimes ISO-8601, NaN→null. Never prints secrets. Exit code 2 on unknown run.

## 8. Security rules (enforced in code review)

- Secrets only in Windows Credential Manager via `keyring` (service `ai-ecc`, username = `<platform>:<nickname>`), or env vars. Never in SQLite, config, logs, Streamlit session state shown to the page, or git.
- Kaggle subprocesses get `KAGGLE_API_TOKEN` + per-account `KAGGLE_CONFIG_DIR` via `env=`; never in argv. Only `kernels list|status|logs` subcommands are allowed (allow-list in code).
- No scraping, no cookies, no write/execute calls to Kaggle/Colab/W&B. Default read-only.
- `.gitignore`: `.venv/`, `*.db`, `.env`, `ecc_config.toml`, `wandb/`, `**/kaggle.json`, `**/access_token*`.
