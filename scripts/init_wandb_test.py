"""Tests for `init_wandb`'s run-resumption, which is what keeps a preempted job on one W&B chart.

The run id is written to `wandb_id.txt` inside the checkpoint directory, so a job that resumes from
its checkpoints also rejoins its own W&B run. On a preemptible TPU the checkpoint dir is restored
from durable storage before the retry starts (see tpu/server/lib/openpi.js), which is what carries
the id across attempts.
"""

import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


class _FakeRun:
    def __init__(self, run_id: str):
        self.id = run_id


class _FakeWandb:
    """Records what wandb.init was called with, and hands back a run id like the real thing."""

    def __init__(self, new_id: str = "fresh-run-id"):
        self.calls: list[dict] = []
        self.run: _FakeRun | None = None
        self._new_id = new_id

    def init(self, **kwargs):
        self.calls.append(kwargs)
        # A resumed run keeps the id it was given; a new one is assigned a fresh id by W&B.
        self.run = _FakeRun(kwargs.get("id") or self._new_id)
        return self.run


@pytest.fixture
def config(tmp_path: pathlib.Path):
    cfg = dataclasses.replace(
        _config.get_config("debug"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        exp_name="myrun",
        project_name="openpi",
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setattr(train, "wandb", fake)
    return fake


def test_a_fresh_run_records_its_id_next_to_the_checkpoints(config, fake_wandb):
    train.init_wandb(config, resuming=False)

    (call,) = fake_wandb.calls
    assert call["name"] == "myrun"
    assert call["project"] == "openpi"
    assert "id" not in call, "a fresh run must not claim an id"
    # The id has to be persisted, or a later attempt has nothing to rejoin.
    assert (config.checkpoint_dir / "wandb_id.txt").read_text() == "fresh-run-id"


def test_resuming_rejoins_the_same_run(config, fake_wandb):
    (config.checkpoint_dir / "wandb_id.txt").write_text("run-abc123\n")

    train.init_wandb(config, resuming=True)

    (call,) = fake_wandb.calls
    assert call["id"] == "run-abc123", "the preempted job must land back on its own W&B run"
    assert call["project"] == "openpi"
    assert call["name"] == "myrun"


def test_resuming_allows_an_id_wandb_has_never_seen(config, fake_wandb):
    """`resume="must"` would hard-fail here and kill a retry that is otherwise fine.

    The id outlives the run whenever a first attempt is preempted before its first log, or ran with
    WANDB_MODE=offline: the file is on disk but W&B has no such run. `allow` (re)creates it under
    the same id instead of raising.
    """
    (config.checkpoint_dir / "wandb_id.txt").write_text("run-abc123")

    train.init_wandb(config, resuming=True)

    (call,) = fake_wandb.calls
    assert call["resume"] == "allow"


def test_resuming_twice_keeps_the_same_id(config, fake_wandb):
    """Preemption can happen more than once; every attempt must land on the same run."""
    train.init_wandb(config, resuming=False)
    first_id = (config.checkpoint_dir / "wandb_id.txt").read_text().strip()

    for _ in range(3):
        train.init_wandb(config, resuming=True)

    assert [c.get("id") for c in fake_wandb.calls[1:]] == [first_id] * 3
    assert (config.checkpoint_dir / "wandb_id.txt").read_text().strip() == first_id, (
        "resuming must not rewrite the id file"
    )


def test_disabled_wandb_does_not_touch_the_id_file(config, fake_wandb):
    train.init_wandb(config, resuming=False, enabled=False)

    assert fake_wandb.calls == [{"mode": "disabled"}]
    assert not (config.checkpoint_dir / "wandb_id.txt").exists()
