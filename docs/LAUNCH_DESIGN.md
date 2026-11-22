# Launch Control (Kaggle) — design v1

Goal: from the dashboard, the user picks **Kaggle account → GitHub repo → branch → commit → command → GPU**, confirms, and a training run starts on that account. Claude never launches anything; only a human click in the UI does.

## Mechanism (official APIs only)

```
Dashboard (Launch page) ──confirm──▶ ecc.launch.LaunchService
   │  gh CLI (local, read-only): repos / branches / commits
   │  build runner.py  (template + embedded experiment_monitor wheel, base64)
   │  write kernel-metadata.json  { id: "<owner>/ecc-runner", kernel_type: script, enable_gpu, enable_internet: true, is_private: true }
   └─ kaggle kernels push -p <tmpdir> [--accelerator NvidiaTeslaT4] [-t <seconds>]   (token via env, per account)
Kaggle session runs runner.py:
   1. read secrets WANDB_API_KEY (+ optional GITHUB_TOKEN for private repos) via kaggle_secrets
   2. git fetch the exact commit SHA of owner/repo → /kaggle/working/src
   3. pip install embedded wheel (+ optional requirements.txt)
   4. export ECC_* env (project, account, kernel ref, notebook url, git branch/sha, launch id)
   5. monitor_mode = "runner": runner opens the W&B run, streams child stdout/stderr into it, heartbeats, records exit code
      monitor_mode = "script": runner only sets env; the training script calls em.init() itself (env defaults)
   6. run the user command in the repo dir, exit with its exit code
Dashboard sees it through the existing read paths: Kaggle status/logs (kernel ref) + W&B run (ecc/kaggle_kernel = ref).
```

## Decisions
- **Runner slug template** (`[launch].runner_slug`, placeholders `{owner}`/`{account}`). Current config:
  `mmculture-qa-new-{owner}` → `alice/mmculture-qa-new-alice`, `bob/mmculture-qa-new-bob`,
  `carol/mmculture-qa-new-carol`. Each launch **overwrites** that notebook's code with the runner.
- **Runner is pushed as a one-cell notebook** (`kernel_type: notebook`) so it can reuse notebooks created
  in the Kaggle UI; a non-zero exit raises `RuntimeError` so Kaggle reports ERROR (not COMPLETE).
- **One runner kernel per account** (originally `<owner>/ecc-runner`, now the template above). Each launch = a new version of it. Reason: Kaggle secrets are attached per notebook in the UI; with a fixed kernel the user attaches `WANDB_API_KEY` (and `GITHUB_TOKEN`) **once** per account. Consequence: one launch at a time per account (the UI refuses when the runner kernel is QUEUED/RUNNING) — matches "3 accounts = 3 parallel runs".
- **Preflight launch** (CPU, no GPU, `--accelerator` omitted, `enable_gpu=false`): checks secrets, clone, wheel install, `import torch` and prints versions, no training. Used first on each account and after attaching secrets — costs no GPU quota.
- **No secrets in pushed code.** The runner reads them from Kaggle Secrets at run time; clone URL with token is never printed (redacted).
- **Safety**: `[launch] enabled = false` by default; UI needs an explicit confirmation checkbox naming account + GPU + commit; every push is written to an audit table `launches`; `ecc.cli` exposes NO launch command (Claude cannot launch); the write allow-list lives in a separate class (`KaggleLauncher`) and allows only `kernels push`; read adapter unchanged.
- **Not supported**: stopping a run (no official cancel API → UI links to the Kaggle page), GPU quota numbers (no API), Colab launch.
