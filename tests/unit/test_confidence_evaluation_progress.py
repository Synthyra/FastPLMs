"""Evaluation keeps sample ordering and preserves completed targets during failures."""

import json
import weakref
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.confidence import test_evaluation as evaluation


class _Executor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, argument):
        future = Future()
        try:
            future.set_result(function(argument))
        except (RuntimeError, ValueError) as error:
            future.set_exception(error)
        return future


class _DeferredFuture(Future):
    def __init__(self, function, argument):
        super().__init__()
        self.function = function
        self.argument = argument

    def result(self, timeout=None):
        if not self.done():
            self.set_result(self.function(self.argument))
        return super().result(timeout)


class _DeferredExecutor(_Executor):
    def __init__(self, *args, **kwargs):
        self.pending = []

    def __exit__(self, *args):
        # ProcessPoolExecutor also waits for already submitted work on an exceptional exit.
        for future in reversed(self.pending):
            future.result()

    def submit(self, function, argument):
        future = _DeferredFuture(function, argument)
        self.pending.append(future)
        return future


class _Rollout:
    def __init__(self, target_id):
        self.target_id = target_id
        self.quality = [{"lddt": 0.1 * sample} for sample in range(evaluation.EVALUATION_SAMPLES)]


def _targets():
    return [
        {
            "target_id": name,
            "sequences": ["AC"],
            "stratum": "monomer_short",
            "num_tokens": 2,
        }
        for name in ("first", "second", "third")
    ]


def _stub_evaluation(monkeypatch, fold, executor=_Executor):
    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(
        evaluation,
        "structure",
        lambda pool, target: SimpleNamespace(
            target_id=target["target_id"],
            positions=np.zeros((2, 14, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(evaluation, "fold", fold)
    monkeypatch.setattr(evaluation, "sample_structures", lambda rollout, sequences, sample: sample)
    monkeypatch.setattr(
        evaluation,
        "structure_scores",
        lambda sample: {"tm_score": sample / 10, "dockq": None},
    )
    monkeypatch.setattr(
        evaluation,
        "head_sample_predictions",
        lambda head, rollout, sample: {"mean_plddt": sample / 10},
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.mark.parametrize("executor", [_Executor, _DeferredExecutor])
def test_evaluation_preserves_target_order_sample_order_and_seeds(monkeypatch, tmp_path, executor):
    calls = []

    def fold(model, target, samples, seed, loops, steps, **kwargs):
        calls.append((target.target_id, samples, seed, loops, steps))
        return _Rollout(target.target_id)

    _stub_evaluation(monkeypatch, fold, executor)
    output = tmp_path / "records.json"
    records = evaluation.fold_and_score(
        object(), tmp_path, _targets(), {"head": object()}, output, lambda message: None
    )
    assert calls == [
        (target["target_id"], 5, 1000 + index, 3, 50) for index, target in enumerate(_targets())
    ]
    assert [(record["target_id"], record["sample"]) for record in records] == [
        (target["target_id"], sample) for target in _targets() for sample in range(5)
    ]
    assert json.loads(output.read_text()) == records
    assert all(record["tm_score"] == record["sample"] / 10 for record in records)


def test_completed_rollout_is_released_before_next_fold_including_oom(monkeypatch, tmp_path):
    previous = None

    def fold(model, target, *args, **kwargs):
        nonlocal previous
        assert previous is None or previous() is None, "Previous rollout is still resident"
        if target.target_id == "second":
            raise torch.OutOfMemoryError("synthetic OOM")
        rollout = _Rollout(target.target_id)
        previous = weakref.ref(rollout)
        return rollout

    _stub_evaluation(monkeypatch, fold)
    records = evaluation.fold_and_score(
        object(),
        tmp_path,
        _targets(),
        {"head": object()},
        tmp_path / "records.json",
        lambda message: None,
    )
    assert {record["target_id"] for record in records} == {"first", "third"}
    assert json.loads((tmp_path / "skipped.json").read_text()) == ["second"]
    skipped = json.loads((tmp_path / "partial-records" / "000001.json").read_text())
    assert skipped["skipped_out_of_memory"] is True
    assert skipped["records"] == []


@pytest.mark.parametrize("executor", [_Executor, _DeferredExecutor])
@pytest.mark.parametrize("failure_stage", ["fold", "head"])
def test_completed_targets_survive_later_failure_without_final_records(
    monkeypatch, tmp_path, executor, failure_stage
):
    def fold(model, target, *args, **kwargs):
        if target.target_id == "second" and failure_stage == "fold":
            raise RuntimeError("fold failed")
        return _Rollout(target.target_id)

    _stub_evaluation(monkeypatch, fold, executor)
    if failure_stage == "head":

        def predict(head, rollout, sample):
            if rollout.target_id == "second":
                raise RuntimeError("head failed")
            return {"mean_plddt": sample / 10}

        monkeypatch.setattr(evaluation, "head_sample_predictions", predict)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        evaluation.fold_and_score(
            object(),
            tmp_path,
            _targets(),
            {"head": object()},
            tmp_path / "records.json",
            lambda message: None,
        )
    assert not (tmp_path / "records.json").exists()
    assert not (tmp_path / "completion.json").exists()
    partials = sorted((tmp_path / "partial-records").glob("*.json"))
    assert len(partials) == 1
    saved = json.loads(partials[0].read_text())
    assert saved["status"] == "partial_evaluation"
    assert saved["target_index"] == 0
    assert saved["target_id"] == "first"
    assert [record["sample"] for record in saved["records"]] == list(range(5))


def test_existing_partial_evaluation_is_refused_before_folding(monkeypatch, tmp_path):
    progress = tmp_path / "partial-records"
    progress.mkdir()
    (progress / "000000.json").write_text("retained", encoding="utf-8")

    def unexpected_fold(*args, **kwargs):
        pytest.fail("must refuse existing partial evaluation before folding")

    _stub_evaluation(monkeypatch, unexpected_fold)
    with pytest.raises(FileExistsError):
        evaluation.fold_and_score(
            object(),
            tmp_path,
            _targets(),
            {"head": object()},
            tmp_path / "records.json",
            lambda message: None,
        )
    assert (progress / "000000.json").read_text() == "retained"


def test_partial_target_is_atomically_published_only_after_complete_write(monkeypatch, tmp_path):
    original_write = evaluation.write_new_json
    destination = tmp_path / "000000.json"

    def check_temporary(path, payload):
        assert not destination.exists()
        assert path.name == ".000000.json.tmp"
        original_write(path, payload)
        assert not destination.exists()

    monkeypatch.setattr(evaluation, "write_new_json", check_temporary)
    evaluation._write_partial_target(tmp_path, 0, "first", [{"sample": 0}])
    assert destination.is_file()
    assert not (tmp_path / ".000000.json.tmp").exists()
