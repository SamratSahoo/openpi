"""Unit tests for the pure sharding/mixing logic in ``streaming_dataset``.

These deliberately avoid the network: they exercise ``_v3_shard_plan`` (which files/episode-group a shard
reads), ``_episode_owner`` (how a shard's group maps onto a file's episodes -- the guard against a shard
owning zero episodes and hanging), ``_mixing_weight`` (explicit vs size-proportional weights), and the
``sampling_weights`` construction-time validation, which all run without contacting the Hub.
"""

import types

import pytest

from openpi.training.streaming_dataset import StreamingLeRobotDataset
from openpi.training.streaming_dataset import _episode_owner
from openpi.training.streaming_dataset import _mixing_weight
from openpi.training.streaming_dataset import _v3_shard_plan


def _owned_episodes(num_files: int, shard_id: int, num_shards: int, episodes_per_file):
    """Simulate which (file, episode) pairs one shard emits, mirroring _stream_frames_v3 exactly.

    ``episodes_per_file`` maps file_idx -> episode count. Returns the set of (file_idx, episode_idx) this
    shard owns, using _v3_shard_plan for files/group and _episode_owner for the within-file episode split.
    """
    file_idxs, group, num_groups = _v3_shard_plan(num_files, shard_id, num_shards)
    owned = set()
    for file_idx in file_idxs:
        n_ep = episodes_per_file[file_idx]
        divisor, remainder = _episode_owner(n_ep, group, num_groups)
        for episode_idx in range(n_ep):
            if episode_idx % divisor == remainder:
                owned.add((file_idx, episode_idx))
    return owned


@pytest.mark.parametrize("num_files", [1, 2, 3, 7, 49])
@pytest.mark.parametrize("num_shards", [1, 2, 4, 8, 16, 64])
def test_every_episode_is_covered_and_no_shard_is_idle(num_files, num_shards):
    """The two invariants: every (file, episode) is owned by >=1 shard, and NO shard owns zero episodes.

    A shard owning zero episodes is the critical bug: its generator would loop forever without yielding and
    hang the dataloader worker. episodes_per_file=5 makes num_shards=64 exceed a single file's episodes,
    exercising the overflow/duplication path.
    """
    episodes_per_file = dict.fromkeys(range(num_files), 5)
    per_shard = [_owned_episodes(num_files, s, num_shards, episodes_per_file) for s in range(num_shards)]

    # No idle shard: every shard that is assigned a file owns at least one episode.
    for shard_id, owned in enumerate(per_shard):
        file_idxs, _group, _num_groups = _v3_shard_plan(num_files, shard_id, num_shards)
        if file_idxs:  # a shard with no files legitimately contributes nothing (handled by `continue`)
            assert owned, f"shard {shard_id} owns a file but zero episodes -> would hang"

    # Completeness: the union across shards covers every (file, episode) pair.
    covered = set().union(*per_shard) if per_shard else set()
    expected = {(f, e) for f in range(num_files) for e in range(episodes_per_file[f])}
    assert covered == expected


def test_no_overflow_case_is_a_disjoint_partition():
    """When each file has >= as many episodes as sharing shards, ownership is disjoint (no duplication)."""
    num_files, num_shards = 1, 8
    episodes_per_file = {0: 40}  # 40 episodes, 8 sharing shards -> 5 each, disjoint
    per_shard = [_owned_episodes(num_files, s, num_shards, episodes_per_file) for s in range(num_shards)]
    counts = [len(o) for o in per_shard]
    assert counts == [5] * 8  # even, non-empty split
    all_pairs = [p for o in per_shard for p in o]
    assert len(all_pairs) == len(set(all_pairs)) == 40  # disjoint and complete


def test_overflow_duplicates_instead_of_leaving_shards_idle():
    """Single file, more shards than episodes: every shard still owns exactly one episode (via duplication)."""
    num_files, num_shards = 1, 32
    episodes_per_file = {0: 20}  # 20 episodes, 32 shards -> shards 20..31 duplicate episodes 0..11
    per_shard = [_owned_episodes(num_files, s, num_shards, episodes_per_file) for s in range(num_shards)]
    assert all(len(o) == 1 for o in per_shard), "some shard owns zero episodes -> would hang"
    covered = set().union(*per_shard)
    assert covered == {(0, e) for e in range(20)}  # all 20 episodes still covered
    assert per_shard[24] == {(0, 4)}  # shard 24 duplicates episode 4 (24 % 20)


def test_v3_shard_plan_common_case_disjoint_whole_files():
    """When shards <= files each shard owns disjoint whole files and emits all episodes (group 0 of 1)."""
    num_files, num_shards = 49, 8
    all_files = []
    for shard_id in range(num_shards):
        file_idxs, group, num_groups = _v3_shard_plan(num_files, shard_id, num_shards)
        assert (group, num_groups) == (0, 1)  # no episode splitting
        assert file_idxs == list(range(shard_id, num_files, num_shards))
        all_files.extend(file_idxs)
    assert sorted(all_files) == list(range(num_files))  # every file read exactly once


def test_v3_shard_plan_single_file_spread_across_all_shards():
    """A 1-file repo over 8 shards: all shards read file 0 and take distinct episode groups 0..7."""
    num_files, num_shards = 1, 8
    groups = []
    for shard_id in range(num_shards):
        file_idxs, group, num_groups = _v3_shard_plan(num_files, shard_id, num_shards)
        assert file_idxs == [0]
        assert num_groups == num_shards
        groups.append(group)
    assert sorted(groups) == list(range(num_shards))  # every group represented once


def test_v3_shard_plan_no_files():
    assert _v3_shard_plan(0, 0, 4) == ([], 0, 1)


@pytest.mark.parametrize("num_episode_groups", [1, 2, 5, 8, 32])
@pytest.mark.parametrize("num_episodes", [1, 3, 8, 20])
def test_episode_owner_never_leaves_a_group_empty(num_episodes, num_episode_groups):
    """Every group in [0, num_episode_groups) must own >= 1 of the file's episodes (the anti-hang guarantee)."""
    for group in range(num_episode_groups):
        divisor, remainder = _episode_owner(num_episodes, group, num_episode_groups)
        owned = [e for e in range(num_episodes) if e % divisor == remainder]
        assert owned, f"group {group}/{num_episode_groups} owns no episode of a {num_episodes}-episode file"


def test_episode_owner_whole_file_and_empty_file():
    assert _episode_owner(10, 0, 1) == (1, 0)  # whole-file shard owns everything
    assert _episode_owner(0, 3, 8) == (1, 0)  # empty file -> harmless no-op (nothing to iterate)


def _fake_source(repo_id, sampleable_frames):
    return types.SimpleNamespace(repo_id=repo_id, sampleable_frames=sampleable_frames)


def test_mixing_weight_defaults_to_frame_count():
    src = _fake_source("a", 12345)
    assert _mixing_weight(src, None) == 12345.0
    assert _mixing_weight(src, {}) == 12345.0  # empty dict == size-proportional


def test_mixing_weight_zero_frames_floored_to_one():
    # A source that reports 0 sampleable frames must not get weight 0 (which rng.choices rejects).
    assert _mixing_weight(_fake_source("a", 0), None) == 1.0


def test_mixing_weight_explicit_overrides_size():
    droid = _fake_source("lerobot/droid_1.0.1", 27_630_375)
    toys = _fake_source("SamratSahoo/toys300_sim", 411_593)
    weights = {"lerobot/droid_1.0.1": 0.5, "SamratSahoo/toys300_sim": 0.5}
    assert _mixing_weight(droid, weights) == 0.5
    assert _mixing_weight(toys, weights) == 0.5  # equal despite 67x the frames


def test_sampling_weights_missing_repo_raises():
    with pytest.raises(ValueError, match="missing entries"):
        StreamingLeRobotDataset(["a", "b"], action_horizon=10, sampling_weights={"a": 0.5})


def test_sampling_weights_extra_repo_raises():
    with pytest.raises(ValueError, match="not in the mixture"):
        StreamingLeRobotDataset(["a", "b"], action_horizon=10, sampling_weights={"a": 1, "b": 1, "c": 1})


def test_sampling_weights_non_positive_raises():
    with pytest.raises(ValueError, match="must be positive"):
        StreamingLeRobotDataset(["a", "b"], action_horizon=10, sampling_weights={"a": 1, "b": 0})
