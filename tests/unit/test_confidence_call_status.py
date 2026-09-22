"""Keep a failed trainer from stopping publication for the other model."""

import modal
import pytest

from tools.confidence.call_status import training_call_status


@pytest.mark.parametrize("error,expected", [(None, "complete"), (TimeoutError(), None), (RuntimeError("incomplete updates"), "failed"), (ValueError("invalid loss"), "failed"), (modal.exception.RemoteError("cancelled"), "failed"), (modal.exception.FunctionTimeoutError("expired"), "failed")])
def test_training_result_status(monkeypatch: pytest.MonkeyPatch, error: Exception | None, expected: str | None) -> None:
    class Call:
        def get(self, *, timeout: int) -> None:
            assert timeout == 0
            if error is not None:
                raise error

    monkeypatch.setattr(modal.FunctionCall, "from_id", lambda call_id: Call())
    assert training_call_status("fc-test") == expected


@pytest.mark.parametrize("error", [modal.exception.ServiceError("unavailable"), ConnectionError("disconnected")])
def test_service_error_is_not_a_terminal_training_failure(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    class Call:
        def get(self, *, timeout: int) -> None:
            raise error

    monkeypatch.setattr(modal.FunctionCall, "from_id", lambda call_id: Call())
    with pytest.raises(type(error)):
        training_call_status("fc-test")
