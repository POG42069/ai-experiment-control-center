"""Shared fixtures for ECC backend unit and CLI tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ecc.config import GeneralSettings, Settings, WandbSettings
from ecc.service import Service
from ecc.sources.demo import DemoSource
from ecc.store import Store


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def temp_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "test_ecc.db")
    yield store
    store.close()


@pytest.fixture
def demo_source(fixed_now: datetime) -> DemoSource:
    return DemoSource(now_fn=lambda: fixed_now, seed=0)


@pytest.fixture
def demo_service(
    demo_source: DemoSource,
    temp_store: Store,
    tmp_path: Path,
    fixed_now: datetime,
) -> Service:
    settings = Settings(
        general=GeneralSettings(demo_mode="on", db_path=tmp_path / "test_ecc.db"),
        wandb=WandbSettings(),
    )
    return Service(
        sources=[demo_source],
        store=temp_store,
        settings=settings,
        now_fn=lambda: fixed_now,
    )
