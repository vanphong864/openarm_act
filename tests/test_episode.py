import numpy as np
import pytest

from openarm_data import constants as C
from openarm_data.episode import EpisodeReader
from openarm_data.episode import command_lag
from openarm_data.episode import episode_id
from openarm_data.episode import find_episodes
from openarm_data.manifest import Manifest


def test_find_skips_partial(episode_factory, tmp_path):
    episode_factory("2026-10-08/a")
    episode_factory("2026-10-08/b")
    (tmp_path / "raw/2026-10-08/c.partial.hdf5").write_bytes(b"")
    found = find_episodes(tmp_path / "raw")
    assert [episode_id(p, tmp_path / "raw") for p in found] == ["2026-10-08/a", "2026-10-08/b"]


def test_commands_are_the_labels(episode_factory):
    with EpisodeReader(episode_factory()) as ep:
        assert ep.problems() == []
        labels = ep.labels("commands")
        commands = ep.commands()
    assert labels.source == "commands"
    np.testing.assert_allclose(labels.actions, commands[labels.indices])
    # The command leads the arm, so it is not the state.
    assert not np.allclose(labels.actions, labels.state)


def test_next_qpos_and_qpos(episode_factory):
    with EpisodeReader(episode_factory(n=100)) as ep:
        qpos = ep.qpos()
        nxt = ep.labels("next_qpos")
        same = ep.labels("qpos")
    assert len(nxt.indices) == 99
    np.testing.assert_allclose(nxt.actions, qpos[1:])
    np.testing.assert_allclose(same.actions, same.state)


def test_fps_25_takes_every_other_frame(episode_factory):
    with EpisodeReader(episode_factory(n=100)) as ep:
        qpos = ep.qpos()
        labels = ep.labels("next_qpos", fps=25)
    np.testing.assert_array_equal(labels.indices, np.arange(0, 98, 2))
    np.testing.assert_allclose(labels.actions, qpos[labels.indices + 2])
    with EpisodeReader(episode_factory("x/y")) as ep, pytest.raises(ValueError):
        ep.labels(fps=30)  # does not divide 50


def test_grippers_never_commanded_fall_back_per_column(episode_factory):
    with EpisodeReader(episode_factory(commands="no_grippers", n=100)) as ep:
        qpos, commands = ep.qpos(), ep.commands()
        labels = ep.labels("commands")
    assert labels.source == "commands"
    assert any("left_gripper" in note for note in labels.notes)
    grippers, arm = list(C.GRIPPERS), list(C.ARM_JOINTS)
    np.testing.assert_allclose(labels.actions[:, grippers], qpos[labels.indices + 1][:, grippers])
    np.testing.assert_allclose(labels.actions[:, arm], commands[labels.indices][:, arm])


def test_sparse_nan_trimmed_and_interpolated(episode_factory):
    with EpisodeReader(episode_factory(commands="sparse_nan", n=120)) as ep:
        commands = ep.commands()
        labels = ep.labels("commands", max_nan_fraction=0.05)
    assert labels.source == "commands"
    assert labels.indices[0] == 5
    assert np.all(np.isfinite(labels.actions))
    row = int(np.flatnonzero(labels.indices == 40)[0])
    expected = commands[39] + (commands[42] - commands[39]) / 3
    np.testing.assert_allclose(labels.actions[row], expected, rtol=1e-5, atol=1e-6)


def test_mostly_nan_falls_back_whole(episode_factory):
    with EpisodeReader(episode_factory(commands="mostly_nan")) as ep:
        labels = ep.labels("commands")
    assert labels.source == "next_qpos"
    with EpisodeReader(episode_factory("x/none", commands=None)) as ep:
        assert ep.labels("commands").source == "next_qpos"


def test_problems(episode_factory):
    with EpisodeReader(episode_factory(complete=False)) as ep:
        assert any("not complete" in p for p in ep.problems())
    with EpisodeReader(episode_factory("x/short", n=10)) as ep:
        assert any("fewer than" in p for p in ep.problems(min_frames=50))


def test_command_lag_finds_the_lead(episode_factory):
    with EpisodeReader(episode_factory(n=400)) as ep:
        lags = command_lag(ep.qpos(), ep.commands())
    assert all(lag == 3 for lag in lags.values())


def test_manifest_priority():
    manifest = Manifest(
        {
            "defaults": {"task": "default task"},
            "rules": [
                {"match": "2026-10-08/*", "task": "rule task"},
                {"match": "2026-10-08/bad*", "exclude": True, "note": "stalled"},
                {"match": "2026-10-09/*", "split": "val"},
            ],
            "episodes": {"2026-10-08/fixed": {"task": "manifest wins", "success": False}},
        }
    )
    assert manifest.resolve("2026-10-07/x").task == "default task"
    assert manifest.resolve("2026-10-08/x").task == "rule task"
    assert manifest.resolve("2026-10-08/x", file_task="from file").task == "from file"
    fixed = manifest.resolve("2026-10-08/fixed", file_task="from file")
    assert fixed.task == "manifest wins"
    assert fixed.extra == {"success": False}
    bad = manifest.resolve("2026-10-08/bad1")
    assert bad.exclude
    assert bad.extra["note"] == "stalled"
    assert manifest.resolve("2026-10-09/x").split == "val"
    with pytest.raises(ValueError):
        Manifest({"rules": [{"task": "no match"}]})
