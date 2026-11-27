# ECC tools — shared helpers for every project

Proven helpers that used to live inside one project's `.git/` (MMCultureQA, 2026-11-23 → 27). Use these instead of
writing new ones per project; extend them here if something is missing. All need the ECC account DB / credential
store, so run them **in a real terminal tab** (not Claude's sandbox), from any directory.

| Tool | What | Notes |
|---|---|---|
| `launch.ps1` | `launch plan` → print summary → `launch execute` for one run | Always `--commit`, `--requirements requirements.txt` (unless `-NoRequirements`), `--monitor-mode script`. `-PlanOnly` makes a plan without using quota. Refuses commands with `;` `|` `$` (make the plan by hand then, see the script help). |
| `watch.ps1` | Poll Kaggle kernel status + W&B run state every N s, append lines to a log | Read the log with a snapshot Monitor (never `tail -f`). `-Once` for a single check. |
| `fetch_artifacts.py` | Download W&B artifacts with the ECC W&B key | Run with `.venv\Scripts\python.exe` of ECC (needs `keyring`). Cache goes inside the output's parent dir, `skip_cache=True`. |
| `schedule_task.ps1` | Register a one-shot Windows Task Scheduler task (e.g. launch after the Saturday 07:00 quota reset) | Wakes the PC, runs late if missed. Only on explicit user request. Check the result afterwards. |

Before every launch: copy the shape of the last successful launch (`python -m ecc.cli launch history`), then after the
push check QUEUED at 30–60 s and that the log got past model loading.
