"""HuggingFace Transformers / TRL Trainer callback (lazy import of transformers)."""

from __future__ import annotations

import os
from typing import Any

_DIRECT_KEY_MAP: dict[str, str] = {
    "loss": "train/loss",
    "learning_rate": "train/lr",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
    "eval_loss": "eval/loss",
    "eval_accuracy": "eval/accuracy",
    "eval_macro_f1": "eval/macro_f1",
    "eval_f1_macro": "eval/macro_f1",
}


def map_hf_logs(logs: dict[str, Any]) -> dict[str, Any]:
    """Pure mapping from HuggingFace Trainer log keys to ECC §3 contract metric keys."""
    mapped: dict[str, Any] = {}
    for key, value in logs.items():
        if key in _DIRECT_KEY_MAP:
            mapped[_DIRECT_KEY_MAP[key]] = value
        elif key.startswith("eval_"):
            suffix = key[len("eval_") :]
            if suffix:
                mapped[f"eval/{suffix}"] = value
        else:
            mapped[key] = value
    return mapped


_ECC_CALLBACK_CLASS: type[Any] | None = None


def _build_callback_class() -> type[Any]:
    global _ECC_CALLBACK_CLASS
    if _ECC_CALLBACK_CLASS is not None:
        return _ECC_CALLBACK_CLASS

    try:
        from transformers import TrainerCallback  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ECCCallback. "
            "Install with: pip install 'experiment-monitor[hf]'"
        ) from exc

    class ECCCallback(TrainerCallback):
        """HuggingFace ``TrainerCallback`` (also works with TRL ``SFTTrainer``)."""

        def __init__(self, monitor: Any) -> None:
            super().__init__()
            self.monitor = monitor

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            max_steps = getattr(state, "max_steps", None)
            total_steps = int(max_steps) if max_steps and max_steps > 0 else None
            num_epochs = getattr(args, "num_train_epochs", None)
            total_epochs = float(num_epochs) if num_epochs is not None else None
            self.monitor.set_totals(total_steps=total_steps, total_epochs=total_epochs)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs:
                return
            mapped = map_hf_logs(logs)
            step = int(getattr(state, "global_step", 0))
            epoch = mapped.get("train/epoch", getattr(state, "epoch", None))
            max_steps = getattr(state, "max_steps", None)
            total_steps = int(max_steps) if max_steps and max_steps > 0 else None

            self.monitor.log(mapped, step=step, commit=False)
            self.monitor.set_progress(step=step, total_steps=total_steps, epoch=epoch)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(getattr(state, "global_step", 0))
            output_dir = getattr(args, "output_dir", "") or ""
            ckpt_path = os.path.join(output_dir, f"checkpoint-{step}")
            self.monitor.log_checkpoint(ckpt_path, step=step)

    _ECC_CALLBACK_CLASS = ECCCallback
    return _ECC_CALLBACK_CLASS


def make_hf_callback(monitor: Any) -> Any:
    """Create an ``ECCCallback(monitor)`` instance with lazy ``transformers`` import."""
    cls = _build_callback_class()
    return cls(monitor)


def __getattr__(name: str) -> Any:
    if name == "ECCCallback":
        return _build_callback_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
