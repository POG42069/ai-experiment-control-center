"""Platform adapters for Kaggle (read-only CLI) and Colab (capability stub)."""

from ecc.adapters.colab import COLAB_CAPABILITY, ColabCapability, ColabSource
from ecc.adapters.kaggle import (
    KaggleCLI,
    KaggleCLIError,
    KaggleKernel,
    KaggleSource,
    parse_kaggle_logs,
    parse_status_output,
)

__all__ = [
    "COLAB_CAPABILITY",
    "ColabCapability",
    "ColabSource",
    "KaggleCLI",
    "KaggleCLIError",
    "KaggleKernel",
    "KaggleSource",
    "parse_kaggle_logs",
    "parse_status_output",
]
