"""Unit tests for the pure sharding/mixing logic in ``streaming_dataset``.

These deliberately avoid the network: they exercise ``_v3_shard_plan`` (which files/episode-group a shard
reads), ``_episode_owner`` (how a shard's group maps onto a file's episodes -- the guard against a shard
owning zero episodes and hanging), ``_mixing_weight`` (explicit vs size-proportional weights), and the
``sampling_weights`` construction-time validation, which all run without contacting the Hub.
"""

import types

import numpy as np
import pytest

from openpi.training.streaming_dataset import StreamingLeRobotDataset
from openpi.training.streaming_dataset import _episode_owner
from openpi.training.streaming_dataset import _HubSource
from openpi.training.streaming_dataset import _mixing_weight
from openpi.training.streaming_dataset import _v3_shard_plan
from openpi.training.streaming_dataset import _v21_only_starves_ranks


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


def _owned_kept_episodes(num_files, shard_id, num_shards, kept_per_file):
    """Simulate which (file, episode) pairs a shard emits UNDER A CAP, mirroring _stream_frames_v3.

    ``kept_per_file`` maps file_idx -> list of kept episode indices (the ``allowed`` list the generator
    builds via source.kept_episodes membership). The split is _episode_owner over the KEPT episodes, so a
    dropped episode is never emitted and no group is left empty when a file has >=1 kept episode.
    """
    file_idxs, group, num_groups = _v3_shard_plan(num_files, shard_id, num_shards)
    owned = set()
    for file_idx in file_idxs:
        kept = kept_per_file[file_idx]
        if not kept:  # _apply_episode_cap prunes such files, and the generator has `if not allowed: continue`
            continue
        divisor, remainder = _episode_owner(len(kept), group, num_groups)
        for local_ordinal, episode in enumerate(kept):
            if local_ordinal % divisor == remainder:
                owned.add((file_idx, episode))
    return owned


@pytest.mark.parametrize("num_shards", [1, 2, 3, 8, 16])
def test_capped_v3_covers_kept_episodes_drops_the_rest_and_no_shard_idle(num_shards):
    """Under a cap: a boundary file emits only its kept episodes, every kept episode is covered exactly by
    the sharding, no dropped episode leaks, and no shard owning a kept file is idle (would hang)."""
    # file0: episodes {0,1,2} all kept; file1: episodes {3,4} but only {3} kept (4 dropped by the cap).
    num_files = 2
    kept_per_file = {0: [0, 1, 2], 1: [3]}
    per_shard = [_owned_kept_episodes(num_files, s, num_shards, kept_per_file) for s in range(num_shards)]

    for shard_id, owned in enumerate(per_shard):
        file_idxs, _g, _ng = _v3_shard_plan(num_files, shard_id, num_shards)
        if any(kept_per_file[f] for f in file_idxs):
            assert owned, f"shard {shard_id} owns a kept file but zero kept episodes -> would hang"

    covered = set().union(*per_shard) if per_shard else set()
    assert covered == {(0, 0), (0, 1), (0, 2), (1, 3)}  # every kept episode, and dropped episode 4 absent


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


# --- max_episodes (first-N-episodes cap) ----------------------------------------------------------
# Construction-time validation raises before any Hub access (like sampling_weights), and
# _HubSource._apply_episode_cap prunes files / rescales frame counts without the network.


def test_max_episodes_extra_repo_raises():
    with pytest.raises(ValueError, match="not in the mixture"):
        StreamingLeRobotDataset(["a", "b"], action_horizon=10, max_episodes={"a": 5, "c": 5})


def test_max_episodes_non_positive_raises():
    with pytest.raises(ValueError, match="must be positive"):
        StreamingLeRobotDataset(["a", "b"], action_horizon=10, max_episodes={"a": 0})


def _bare_source(**attrs):
    """A _HubSource instance without running __init__ (no Hub), pre-seeded for _apply_episode_cap."""
    src = object.__new__(_HubSource)
    src.repo_id = "repo"
    src.max_episodes = None
    src.kept_episodes = None
    for k, v in attrs.items():
        setattr(src, k, v)
    return src


def test_apply_episode_cap_v21_prunes_files_and_scales_frames():
    # v2.1: one episode per parquet file -> first-N cap keeps the first N files and scales total_frames.
    src = _bare_source(
        is_v30=False,
        files=[f"data/chunk-000/file-{i:03d}.parquet" for i in range(10)],
        total_frames=1000,
    )
    src._apply_episode_cap(3)  # noqa: SLF001 - unit-testing the internal cap helper directly
    assert src.max_episodes == 3
    assert src.kept_episodes == {0, 1, 2}
    assert src.files == [f"data/chunk-000/file-{i:03d}.parquet" for i in range(3)]
    assert src.total_frames == 300  # round(1000 * 3 / 10)


def test_apply_episode_cap_v21_over_count_is_noop():
    src = _bare_source(is_v30=False, files=["data/f0.parquet", "data/f1.parquet"], total_frames=50)
    src._apply_episode_cap(9)  # noqa: SLF001 - cap >= episode count -> "use all"
    assert src.max_episodes is None
    assert src.files == ["data/f0.parquet", "data/f1.parquet"]
    assert src.total_frames == 50


def test_apply_episode_cap_v30_keeps_boundary_file_and_exact_frames():
    # 5 episodes in 3 data files: eps {0,1}->(0,0), {2,3}->(0,1), {4}->(0,2). Cap 3 keeps eps 0,1,2, so
    # files (0,0) and the boundary file (0,1) [holds kept ep 2 and dropped ep 3] survive; (0,2) is pruned.
    src = _bare_source(
        is_v30=True,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        episode_length=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        _episode_index_sorted=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        _ep_data_chunk=np.array([0, 0, 0, 0, 0], dtype=np.int64),
        _ep_data_file=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        total_frames=150,
    )
    src._apply_episode_cap(3)  # noqa: SLF001 - unit-testing the internal cap helper directly
    assert src.max_episodes == 3
    assert src.kept_episodes == {0, 1, 2}
    assert src.files == ["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]
    assert src.total_frames == 60  # 10 + 20 + 30, exact from per-episode lengths


def test_apply_episode_cap_v30_noncontiguous_episode_indices():
    # Episode indices are NOT 0-based/contiguous (e.g. a renumbered/filtered dataset): [0, 5, 10, 15, 20].
    # The cap must keep the first N BY ASCENDING INDEX (values {0, 5, 10} for N=3) -- a count-as-threshold
    # filter ("index < 3") would wrongly keep only episode 0. Files/frames/kept_episodes must all agree.
    src = _bare_source(
        is_v30=True,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        episode_length=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        _episode_index_sorted=np.array([0, 5, 10, 15, 20], dtype=np.int64),
        _ep_data_chunk=np.array([0, 0, 0, 0, 0], dtype=np.int64),
        _ep_data_file=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        total_frames=150,
    )
    src._apply_episode_cap(3)  # noqa: SLF001 - unit-testing the internal cap helper directly
    assert src.max_episodes == 3
    assert src.kept_episodes == {0, 5, 10}  # by index order, NOT {0, 1, 2}
    assert src.files == ["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]
    assert src.total_frames == 60


def test_apply_episode_cap_v30_over_count_is_noop():
    src = _bare_source(
        is_v30=True,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        episode_length=np.array([10, 20], dtype=np.int64),
        _episode_index_sorted=np.array([0, 1], dtype=np.int64),
        _ep_data_chunk=np.array([0, 0], dtype=np.int64),
        _ep_data_file=np.array([0, 0], dtype=np.int64),
        files=["data/chunk-000/file-000.parquet"],
        total_frames=30,
    )
    src._apply_episode_cap(2)  # noqa: SLF001 - cap == episode count -> "use all"
    assert src.max_episodes is None
    assert src.kept_episodes is None
    assert src.total_frames == 30


def test_apply_episode_cap_rejects_non_positive():
    src = _bare_source(is_v30=False, files=["data/f0.parquet"], total_frames=10)
    with pytest.raises(ValueError, match="must be positive"):
        src._apply_episode_cap(0)  # noqa: SLF001 - unit-testing the internal cap helper directly


# --- DDP rank-starvation guard --------------------------------------------------------------------
# Under DDP a v2.1 source only covers ranks < len(files); a v3.0 source covers every rank. A rank that
# no source covers gets an empty generator list and stalls the collective, so we fail fast for it.


def _v30_source(n_files):
    return types.SimpleNamespace(is_v30=True, files=list(range(n_files)))


def _v21_source(n_files):
    return types.SimpleNamespace(is_v30=False, files=list(range(n_files)))


def test_v21_only_starves_ranks_single_process_never_starves():
    # world_size <= 1 (the JAX single-process path) can never starve, regardless of file counts.
    assert _v21_only_starves_ranks([_v21_source(1)], world_size=1) is False


def test_v21_only_starves_ranks_v21_only_capped_below_world_size():
    # v2.1-only mixture with fewer files than ranks -> top ranks own zero files -> would hang.
    assert _v21_only_starves_ranks([_v21_source(3)], world_size=8) is True


def test_v21_only_starves_ranks_v30_source_covers_every_rank():
    # A v3.0 source (e.g. DROID) covers every rank, so a tiny v2.1 repo alongside it is safe.
    assert _v21_only_starves_ranks([_v30_source(2), _v21_source(3)], world_size=8) is False


def test_v21_only_starves_ranks_v21_with_enough_files_is_safe():
    # A v2.1 source with >= world_size files covers every rank.
    assert _v21_only_starves_ranks([_v21_source(8)], world_size=8) is False
    assert _v21_only_starves_ranks([_v21_source(20)], world_size=8) is False
