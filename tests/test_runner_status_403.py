"""Kaggle answers 403 'kernels.get was denied' for a runner kernel that does not exist yet."""

from types import SimpleNamespace

import pytest

from ecc.adapters.kaggle import KaggleCLIError
from ecc.launch.service import LaunchRefused, LaunchService

DENIED = (
    "Kaggle CLI failed (exit 1) for account 'kaggle-1': Cannot access kernel "
    "'alice/ecc-runner' (Permission 'kernels.get' was denied)."
)


class _FakeCli:
    def __init__(self, own_refs, error=DENIED):
        self._own = own_refs
        self._error = error

    def status(self, ref):
        raise KaggleCLIError(self._error)

    def list_kernels(self, page_size=100):
        return [SimpleNamespace(ref=r) for r in self._own]


def _svc(cli):
    ksrc = SimpleNamespace(_clients={"kaggle-1": cli})
    read = SimpleNamespace(_get_kaggle_source=lambda: ksrc)
    svc = LaunchService.__new__(LaunchService)
    svc.read_service = read
    svc.runner_ref = lambda nick: "alice/ecc-runner"
    return svc


def test_denied_and_absent_from_own_list_means_not_created():
    assert _svc(_FakeCli(["alice/nb1"])).runner_status("kaggle-1") == "NOT_CREATED"


def test_denied_but_present_in_own_list_is_a_real_error():
    with pytest.raises(LaunchRefused):
        _svc(_FakeCli(["alice/ecc-runner"])).runner_status("kaggle-1")


def test_other_errors_still_refuse():
    with pytest.raises(LaunchRefused):
        _svc(_FakeCli([], error="network timeout")).runner_status("kaggle-1")
