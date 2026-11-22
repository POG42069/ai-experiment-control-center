"""Unit tests for experiment_monitor.redact."""

from __future__ import annotations

from experiment_monitor.redact import REDACTED, redact


def test_redact_wandb_40_hex() -> None:
    fake_wandb = "a1b2c3d4e5" * 4  # 40 hex chars
    out = redact(f"using key {fake_wandb} now")
    assert fake_wandb not in out
    assert REDACTED in out


def test_redact_kaggle_kgat_token() -> None:
    fake_kgat = "KGAT_" + "ab12cd34ef56gh78"
    out = redact(f"auth={fake_kgat}")
    assert fake_kgat not in out
    assert REDACTED in out


def test_redact_hf_token() -> None:
    fake_hf = "hf_" + "A" * 34
    out = redact(f"downloading with {fake_hf}")
    assert fake_hf not in out
    assert REDACTED in out


def test_redact_sk_token() -> None:
    fake_sk = "sk-" + "proj_" + "xY9z" * 6
    out = redact(f"openai key: {fake_sk}")
    assert fake_sk not in out
    assert REDACTED in out


def test_redact_github_tokens() -> None:
    fake_ghp = "ghp_" + "9AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    fake_gho = "gho_" + "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    fake_pat = "github_pat_" + "11ABCDEFG_" + "0" * 24
    for tok in (fake_ghp, fake_gho, fake_pat):
        out = redact(f"git push {tok}")
        assert tok not in out
        assert REDACTED in out


def test_redact_bearer_header() -> None:
    fake_bearer = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
    out = redact(f"Authorization: Bearer {fake_bearer}")
    assert fake_bearer not in out
    assert f"Bearer {REDACTED}" in out


def test_redact_assignments() -> None:
    cases = [
        ("password=mySuperSecret123!", "mySuperSecret123!"),
        ('token="tok_val_98765"', "tok_val_98765"),
        ("api_key=abc123xyz", "abc123xyz"),
        ('{"api_key": "jsonSecretVal"}', "jsonSecretVal"),
        ("WANDB_API_KEY=custom_secret_value", "custom_secret_value"),
    ]
    for text, secret in cases:
        out = redact(text)
        assert secret not in out
        assert REDACTED in out


def test_redact_preserves_normal_training_logs() -> None:
    line = "step=100 loss=0.4321 lr=3e-04 max_tokens=512 epoch=1.50"
    assert redact(line) == line
    assert redact("") == ""
    assert redact(None) == ""
