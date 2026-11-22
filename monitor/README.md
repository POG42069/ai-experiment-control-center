# `experiment-monitor`

Lightweight observation-only W&B logging wrapper for **AI Experiment Control Center (ECC)**.
Designed for Kaggle and Google Colab notebooks, it writes standardized `ecc/*` config, metric, summary, GPU telemetry, heartbeat, and redacted live console logs (`ecc_console.log`) without altering any training hyperparameters or behavior.

## Installation

Inside a Kaggle or Colab notebook (or locally):

```bash
pip install -e ./monitor
# Optional: with HuggingFace Transformers support
pip install -e "./monitor[hf]"
```

## Authentication in Kaggle / Colab

Store your W&B API key in Kaggle Secrets or Colab Secrets under `WANDB_API_KEY`, then call:

```python
import experiment_monitor as em

ok = em.login_from_notebook_secrets("WANDB_API_KEY")
```

Resolution order: Kaggle `UserSecretsClient` → Colab `google.colab.userdata` → `WANDB_API_KEY` environment variable. The key is never printed or returned.

## Plain PyTorch Example

```python
import experiment_monitor as em

cfg = {"lr": 3e-4, "batch_size": 32, "epochs": 3}
with em.init("shared-task-4", name="baseline-pt", config=cfg, account="kaggle-1",
             total_steps=300, total_epochs=3, seed=42, model_name="resnet18") as mon:
    mon.track_best("eval/macro_f1", mode="max")
    for step in range(1, 301):
        loss = train_step(step)
        mon.log({"train/loss": loss, "train/lr": cfg["lr"]}, step=step)
        mon.set_progress(step=step, epoch=step / 100)
        if step % 100 == 0:
            val_f1 = evaluate()
            mon.log({"eval/macro_f1": val_f1}, step=step)
            ckpt = save_checkpoint(step)
            mon.log_checkpoint(ckpt, step=step, metric=val_f1)
```

If an uncaught exception occurs inside the `with` block, `Monitor` records `ecc/end_reason="exception"`, `ecc/exception_type`, redacted `ecc/exception_message` (≤500 chars), writes the full traceback to `ecc_console.log`, restores `sys.stdout`/`sys.stderr`, calls `finish(exit_code=1)`, and re-raises. On `KeyboardInterrupt`, `ecc/end_reason="interrupted"`.

## HuggingFace `Trainer` & TRL `SFTTrainer`

`make_hf_callback(mon)` builds a `transformers.TrainerCallback` lazily (so `experiment_monitor` works without `transformers` installed). Because TRL's `SFTTrainer` subclasses `transformers.Trainer`, the exact same callback works for `SFTTrainer`.

```python
import experiment_monitor as em
from transformers import Trainer, TrainingArguments

with em.init(
    "shared-task-4",
    name="qwen3-lora-v1",
    config={"lora_r": 16, "lr": 2e-4},
    account="colab-1",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    dataset_version="v2",
) as mon:
    mon.track_best("eval/macro_f1", mode="max")
    args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        bf16=True,
        report_to="none",  # Recommended: let ECC own metric logging
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[em.make_hf_callback(mon)],
    )
    trainer.train()
```

### Important Notes for HF `Trainer` / `SFTTrainer`

- **`report_to="none"` vs `report_to="wandb"`**: Setting `report_to="none"` is recommended so metrics are logged once under the standardized ECC keys (`train/loss`, `train/lr`, `eval/macro_f1`, etc.). If you already have `report_to="wandb"` or an active `wandb.init()` session, `em.init(...)` automatically reuses the active `wandb.run` instead of opening a second run.
- **Gradient accumulation & mixed precision (`fp16` / `bf16`)**: No special monitor configuration is required. HuggingFace `state.global_step` already counts optimizer update steps (not micro-batches), so `train/global_step`, `ecc/progress`, and checkpoint steps align cleanly with optimizer steps.

## Contract Summary (ECC §3)

- **Config (`ecc/*`)**: `ecc/schema_version` (`1`), `ecc/project`, `ecc/platform` (`kaggle|colab|local`), `ecc/account`, `ecc/notebook_url`, `ecc/kaggle_kernel`, `ecc/git_branch`, `ecc/git_sha`, `ecc/git_repo`, `ecc/launch_id`, `ecc/dataset_version`, `ecc/dataset_split`, `ecc/seed`, `ecc/model_name`, `ecc/prompt_strategy`, `ecc/total_steps`, `ecc/total_epochs`, `ecc/gpu_name`, `ecc/gpu_count`, `ecc/gpu_mem_total_gb`, `ecc/eval_signature`, plus all user config keys untouched.
- **Metrics**: `train/loss`, `train/lr`, `train/grad_norm`, `train/epoch`, `train/global_step`, `eval/loss`, `eval/accuracy`, `eval/macro_f1`, `eval/<any>`, `gpu/mem_alloc_gb`, `gpu/mem_reserved_gb`, `gpu/max_mem_alloc_gb`, `gpu/util_pct`, `ecc/progress`.
- **Summary**: `ecc/heartbeat`, `ecc/best_metric_name`, `ecc/best_metric`, `ecc/best_step`, `ecc/best_epoch`, `ecc/end_reason`, `ecc/exception_type`, `ecc/exception_message`, `ecc/checkpoints` (metadata only; checkpoints are never uploaded).
- **Live console log**: `<run.dir>/ecc_console.log` (timestamped `[stdout]`/`[stderr]`, secrets redacted, ring-capped at `console_max_bytes`, synced via `wandb.save(..., policy="live")`).

## Environment Variable Defaults

When training scripts are launched via ECC's Kaggle runner (`monitor_mode="script"`) or invoked with `ECC_*` environment variables, you can call `em.init()` with no arguments and let `experiment_monitor` fill missing fields from the environment. **Explicit arguments always take precedence over environment variables.**

| Argument / Config key | Environment fallback | Notes |
|---|---|---|
| `project` | `ECC_PROJECT`, then `WANDB_PROJECT` | Raises `ValueError` if neither argument nor env is set |
| `name` | `ECC_RUN_NAME` | W&B run display name |
| `account` | `ECC_ACCOUNT` | Stored as `ecc/account` |
| `kaggle_kernel` | `ECC_KAGGLE_KERNEL` | Stored as `ecc/kaggle_kernel` (`owner/slug`) |
| `notebook_url` | `ECC_NOTEBOOK_URL` | Stored as `ecc/notebook_url` |
| `ecc/git_branch` | `ECC_GIT_BRANCH` | Overrides `detect_git_info()` when set |
| `ecc/git_sha` | `ECC_GIT_SHA` | Overrides `detect_git_info()` when set |
| `ecc/launch_id` | `ECC_LAUNCH_ID` | Added to config when present (unless provided in `config`) |
| `ecc/git_repo` | `ECC_GIT_REPO` | Added to config when present (unless provided in `config`) |
