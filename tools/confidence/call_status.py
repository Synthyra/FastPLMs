"""Distinguish terminal training failures from pending calls and service failures."""

from __future__ import annotations

import modal


def training_call_status(call_id: str) -> str | None:
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return None
    except (modal.exception.RemoteError, modal.exception.FunctionTimeoutError):
        return "failed"
    except (modal.exception.Error, ConnectionError):
        raise  # A service failure does not establish whether the trainer has stopped.
    except Exception:
        # Modal re-raises deserialized user exceptions such as RuntimeError directly.
        return "failed"
    return "complete"
