"""experiment_monitor: lightweight W&B logging wrapper for AI Experiment Control Center."""

from .core import Monitor, init, login_from_notebook_secrets
from .gpu import gpu_snapshot, gpu_static_info
from .hf import make_hf_callback, map_hf_logs
from .platform import detect_git_info, detect_platform
from .redact import redact

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Monitor",
    "init",
    "login_from_notebook_secrets",
    "detect_platform",
    "detect_git_info",
    "gpu_snapshot",
    "gpu_static_info",
    "make_hf_callback",
    "map_hf_logs",
    "redact",
]
