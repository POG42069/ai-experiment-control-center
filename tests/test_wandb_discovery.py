"""WandbSource project auto-discovery when no projects are configured."""

from types import SimpleNamespace

from ecc.sources.wandb_source import WandbSource


class _FakeApi:
    def __init__(self, names, fail=False):
        self._names = names
        self._fail = fail
        self.project_calls = 0
        self.runs_paths = []

    def projects(self, entity):
        self.project_calls += 1
        if self._fail:
            raise RuntimeError("network down token=abc123secretvalue")
        return [SimpleNamespace(name=n) for n in self._names]

    def runs(self, path, **kwargs):
        self.runs_paths.append(path)
        return []


def test_configured_projects_take_precedence():
    api = _FakeApi(["a", "b"])
    src = WandbSource("me", ["only-this"], api_factory=lambda: api)
    assert src.list_projects() == ["only-this"]
    assert api.project_calls == 0


def test_empty_projects_auto_discovers_and_caches():
    api = _FakeApi(["zeta", "alpha"])
    src = WandbSource("me", [], api_factory=lambda: api)
    assert src.list_projects() == ["alpha", "zeta"]
    assert src.list_projects() == ["alpha", "zeta"]
    assert api.project_calls == 1
    src.list_runs()
    assert sorted(api.runs_paths) == ["me/alpha", "me/zeta"]


def test_discovery_failure_is_contained_and_redacted():
    api = _FakeApi([], fail=True)
    src = WandbSource("me", [], api_factory=lambda: api)
    assert src.list_projects() == []
    assert src.list_runs() == []
    ok, msg = src.health()
    assert "abc123secretvalue" not in (msg or "")


def test_no_entity_means_no_discovery():
    api = _FakeApi(["x"])
    src = WandbSource("", [], api_factory=lambda: api)
    assert src.list_projects() == []
    assert api.project_calls == 0
