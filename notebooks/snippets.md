# Copy-Paste Notebook Snippets (`experiment_monitor`)

> **Rule:** Do not change your hyperparameters; these cells only observe.

Full example notebooks:
- Kaggle: [kaggle_example.ipynb](kaggle_example.ipynb)
- Google Colab: [colab_example.ipynb](colab_example.ipynb)

---

## 1. Install `experiment-monitor`

### From GitHub (Kaggle & Colab)
```python
!pip install -q "git+https://github.com/<YOUR_GH_USER>/ai-experiment-control-center.git#subdirectory=monitor"
```

### Alternative: Kaggle Dataset (offline / private)
Upload the repo's `monitor/` folder as a Kaggle Dataset (e.g. `experiment-monitor`) and attach it to your notebook:
```python
!cp -r /kaggle/input/experiment-monitor/monitor /kaggle/working/monitor
!pip install -q -e /kaggle/working/monitor
```

### Alternative: Google Drive (Colab)
Upload `monitor/` to `MyDrive/ecc/monitor`:
```python
from google.colab import drive
drive.mount("/content/drive")
!pip install -q -e "/content/drive/MyDrive/ecc/monitor"
```

---

## 2. Login from Notebook Secrets

- **Kaggle:** **Add-ons → Secrets** → add `WANDB_API_KEY` → toggle **Attach to notebook**.
- **Colab:** **🔑 Secrets** panel (left sidebar) → add `WANDB_API_KEY` → turn ON **Notebook access**.

```python
import experiment_monitor as em

assert em.login_from_notebook_secrets("WANDB_API_KEY"), "Missing WANDB_API_KEY in Secrets"
```

---

## 3. Initialize Monitor (`em.init`)

### Kaggle (`kaggle-1`, `kaggle-2`, `kaggle-3`)
```python
mon = em.init(
    project="shared-task-4",
    name="qwen3-vl-lora-r16",
    account="kaggle-1",
    notebook_url="https://www.kaggle.com/code/<owner>/<slug>",
    kaggle_kernel="<owner>/<slug>",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    seed=42,
    dataset_version="v2",
    dataset_split="train/val=90/10 seed42",
    prompt_strategy="vi-default",
    eval_signature="v2-vi-90-10-seed42",
    config=existing_config_dict,
)
```

### Google Colab (`colab-1`, `colab-2`)
```python
mon = em.init(
    project="shared-task-4",
    name="qwen3-vl-lora-r16-colab",
    account="colab-1",
    notebook_url="https://colab.research.google.com/drive/<YOUR_COLAB_FILE_ID>",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    seed=42,
    dataset_version="v2",
    dataset_split="train/val=90/10 seed42",
    prompt_strategy="en-prompt",
    eval_signature="v2-en-prompt-90-10-seed42",
    config=existing_config_dict,
)
```

---

## 4A. HuggingFace `Trainer` / TRL `SFTTrainer`

- Set `report_to="none"` in `TrainingArguments` / `SFTConfig` (recommended) so ECC logs once under standardized `train/*` and `eval/*` keys.
- If you already use `report_to="wandb"`, `em.init(...)` reuses the active `wandb.run`.
- TRL `SFTTrainer` subclasses `transformers.Trainer`, so the same callback works unchanged.

```python
from experiment_monitor import make_hf_callback

mon.track_best("eval/macro_f1", mode="max")
trainer.add_callback(make_hf_callback(mon))

with mon:
    trainer.train()
```

---

## 4B. Plain PyTorch Training Loop

Wrap the loop in `with mon:` so exceptions and `KeyboardInterrupt` automatically log `ecc/end_reason`, `ecc/exception_type`, redacted `ecc/exception_message`, and the traceback in `ecc_console.log`.

```python
with mon:
    mon.track_best("eval/macro_f1", mode="max")
    for step in range(1, total_steps + 1):
        loss = train_step(...)
        mon.log({"train/loss": float(loss), "train/lr": current_lr, "train/epoch": epoch}, step=step)
        mon.set_progress(step=step, total_steps=total_steps, epoch=epoch)

        if step % eval_every == 0:
            eval_loss, eval_f1 = evaluate(...)
            mon.log({"eval/loss": eval_loss, "eval/macro_f1": eval_f1}, step=step)
            ckpt_path = save_checkpoint(step)
            mon.log_checkpoint(ckpt_path, step=step, metric=eval_f1, metric_name="eval/macro_f1")
```

---

## 5. Resume an Interrupted Run

```python
with em.init(
    project="shared-task-4",
    name="qwen3-vl-lora-r16",
    account="kaggle-1",
    notebook_url="https://www.kaggle.com/code/<owner>/<slug>",
    kaggle_kernel="<owner>/<slug>",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    seed=42,
    dataset_version="v2",
    resume_id="<YOUR_WANDB_RUN_ID>",
    config=existing_config_dict,
) as mon:
    trainer.add_callback(make_hf_callback(mon))
    trainer.train(resume_from_checkpoint=True)
```
