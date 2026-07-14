"""Unit tests for the pure sharding/mixing logic in ``streaming_dataset``.

These deliberately avoid the network: they exercise ``_v3_shard_plan`` (which files/episode-group a shard
reads), ``_episode_owner`` (how a shard's group maps onto a file's episodes -- the guard against a shard
owning zero episodes and hanging), ``_mixing_weight`` (explicit vs size-proportional weights), and the
``sampling_weights`` construction-time validation, which all run without contacting the Hub.
"""

import json
import types

import numpy as np
import pytest

from openpi.training import streaming_dataset
from openpi.training.streaming_dataset import StreamingLeRobotDataset
from openpi.training.streaming_dataset import _episode_owner
from openpi.training.streaming_dataset import _HubSource
from openpi.training.streaming_dataset import _mixing_weight
from openpi.training.streaming_dataset import _retry_kind
from openpi.training.streaming_dataset import _v3_shard_plan
from openpi.training.streaming_dataset import _v21_only_starves_ranks
from openpi.training.streaming_dataset import retry_call


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
    src.mirror_root = None
    for k, v in attrs.items():
        setattr(src, k, v)
    return src


# ---------------------------------------------------------------------------------------------
# Mirror backend: the same v3.0 reader, pointed at a bucket/directory instead of the Hub. Exists
# because the Hub's CDN is unusable from GCP (it routes GCP clients to an edge that rejects its own
# signed URLs), and because the map-style "just download it" path CANNOT read these v3.0 datasets
# under the pinned lerobot (CODEBASE_VERSION "v2.1").
# ---------------------------------------------------------------------------------------------


def test_without_a_mirror_the_source_still_reads_from_the_hub():
    src = _bare_source(repo_id="lerobot/droid_1.0.1")
    assert src.fs_path("meta/info.json") == "datasets/lerobot/droid_1.0.1/meta/info.json"
    assert src.urls(["data/f.parquet"]) == ["hf://datasets/lerobot/droid_1.0.1/data/f.parquet"]
    assert src.origin == "lerobot/droid_1.0.1"


def test_a_gcs_mirror_rewrites_every_path_into_the_bucket():
    """Paths must land at <mirror_root>/<repo_id>/<rel> -- the layout the mirror job writes."""
    src = _bare_source(repo_id="lerobot/droid_1.0.1", mirror_root="gs://prpl-data-us-central2/ss1824/datasets")
    # gcsfs wants bucket-relative paths (no scheme); the parquet builder wants the full gs:// URL.
    assert src.fs_path("meta/info.json") == "prpl-data-us-central2/ss1824/datasets/lerobot/droid_1.0.1/meta/info.json"
    assert src.urls(["data/f.parquet"]) == [
        "gs://prpl-data-us-central2/ss1824/datasets/lerobot/droid_1.0.1/data/f.parquet"
    ]
    assert "gs://" in src.origin


def test_a_trailing_slash_on_the_mirror_root_does_not_double_up(tmp_path):
    src = _bare_source(repo_id="a/b", mirror_root=f"{tmp_path}/")
    assert "//" not in src.fs_path("meta/info.json").removeprefix("/")


def test_a_directory_mirror_lists_files_relative_to_the_repo_root(tmp_path):
    """_list_files must return repo-relative paths ("meta/info.json"), not absolute ones.

    Everything downstream (the v3.0 episode-meta glob, the canonical-file selection) matches on
    repo-relative prefixes, so an absolute path here would silently find zero data files.
    """
    repo = tmp_path / "lerobot" / "droid_1.0.1"
    (repo / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (repo / "data" / "chunk-000").mkdir(parents=True)
    (repo / "meta" / "info.json").write_text("{}")
    (repo / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_text("x")
    (repo / "data" / "chunk-000" / "file-000.parquet").write_text("x")

    src = _bare_source(repo_id="lerobot/droid_1.0.1", mirror_root=str(tmp_path))
    assert src._list_files() == [  # noqa: SLF001
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/info.json",
    ]


def test_an_empty_mirror_fails_loudly_instead_of_training_on_nothing(tmp_path):
    """A missing/half-built mirror must be a clean error, not an empty file list + a silent no-op."""
    (tmp_path / "a" / "b").mkdir(parents=True)
    src = _bare_source(repo_id="a/b", mirror_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mirror is missing or incomplete"):
        src._list_files()  # noqa: SLF001


def test_the_mirrored_source_reads_a_real_file_off_the_filesystem(tmp_path):
    """End-to-end through fsspec: _download must return a readable local copy of a mirrored file."""
    repo = tmp_path / "a" / "b" / "meta"
    repo.mkdir(parents=True)
    (repo / "info.json").write_text('{"codebase_version": "v3.0", "total_frames": 7}')

    src = _bare_source(repo_id="a/b", mirror_root=str(tmp_path))
    with open(src._download("meta/info.json")) as f:  # noqa: SLF001
        assert json.load(f)["total_frames"] == 7


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


# ---------------------------------------------------------------------------------------------
# Retry classification.
#
# A real run streamed lerobot/droid_1.0.1 for 3h09m (3.9k steps), absorbing hundreds of transient
# Hub errors, and was then killed at step 3900 by a single 403 from the CAS/CDN bridge:
#
#   huggingface_hub.errors.HfHubHTTPError: 403 Forbidden: None.
#   Cannot access content at: https://cas-bridge.xethub.hf.co/xet-bridge-us/...file-064.mp4
#
# The signed URL was 16 seconds old and the dataset is public, so that 403 was not a permissions
# verdict -- it was the CDN. It reached `retry_call` (the same one already retrying the decode of
# that very file) and was re-raised, because 403 was classified as terminal.
# ---------------------------------------------------------------------------------------------

CAS_URL = (
    "https://cas-bridge.xethub.hf.co/xet-bridge-us/6877d314/58540e7d?Expires=1783982474"
    "&X-Amz-Signature=e181ba77&X-Amz-Expires=3600&X-Xet-Cas-Uid=public"
)

# The edge GCP-hosted clients (our TPU workers) actually get routed to. Google Cloud CDN dialect:
# bare `Signature`, ULID `Key-Pair-Id`, no `X-Amz-*` anywhere -- so a CloudFront-shaped marker list
# does not recognize it. It served a 403 "SignatureError: invalid key pair id" on a URL it had just
# minted, and because it went unrecognized the run died on attempt 1 instead of retrying.
GCP_CDN_URL = (
    "https://us.gcp.cdn.hf.co/xet-bridge-us/6877d314/f0583c8d?X-Xet-Cas-Uid=6682d201"
    "&user_id=6682d201&Expires=1784013127&Policy=eyJTdGF0ZW1lbnQ&Signature=MEUCIFS0"
    "&Key-Pair-Id=01KXEF4KZ1B6FV465MAWR4M21F"
)


def _http_error(status: int, url: str, exc_type=None):
    """An exception shaped like the ones huggingface_hub raises (it carries `.response`)."""
    response = types.SimpleNamespace(status_code=status, url=url, headers={}, request=None)
    exc = (exc_type or RuntimeError)(f"{status} Forbidden: None.\nCannot access content at: {url}")
    exc.response = response
    return exc


@pytest.mark.parametrize("status", [401, 403])
def test_an_auth_rejection_from_the_content_cdn_is_retried(status):
    """These are the exact errors that killed two multi-hour TPU runs.

    The bridge rejects a stale signed URL with either code; both mean "mint a fresh one", not
    "your credentials are bad" -- authorization to the bridge is in the URL signature, not the token.
    """
    assert _retry_kind(_http_error(status, CAS_URL)) == "presigned-auth"


@pytest.mark.parametrize("status", [401, 403])
def test_an_auth_rejection_from_the_gcp_cdn_edge_is_retried(status):
    """GCP-hosted workers get a different CDN edge; its signed URLs must be recognized too.

    This is the "SignatureError: invalid key pair id" 403 that killed a run on attempt 1: the URL is
    just as re-mintable as a CloudFront one, but the vendor-specific marker list did not match it.
    """
    assert _retry_kind(_http_error(status, GCP_CDN_URL)) == "presigned-auth"


@pytest.mark.parametrize("status", [401, 403])
def test_an_auth_rejection_from_the_hub_api_still_fails_fast(status):
    """A real permissions/token error must NOT be retried -- that would hang the job forever."""
    exc = _http_error(status, "https://huggingface.co/api/datasets/someone/private")
    assert _retry_kind(exc) is None


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_rate_limits_and_5xx_are_retried_forever(status):
    assert _retry_kind(_http_error(status, "https://huggingface.co/api/x")) == "transient"


@pytest.mark.parametrize("status", [404, 410])
def test_genuinely_terminal_statuses_still_fail_fast(status):
    assert _retry_kind(_http_error(status, CAS_URL)) is None


@pytest.mark.parametrize("status", [401, 403])
def test_a_cdn_auth_rejection_recovers_when_the_next_open_mints_a_fresh_url(monkeypatch, status):
    """`_decode_video_frames` re-opens the file on every call, so a retry gets a new signed URL.

    The 401 case is the step-3200 crash: a DataLoader worker decoding file-067.mp4 took a 401 from
    the bridge and, because 401 was classified terminal, threw away 2.5 hours of training.
    """
    monkeypatch.setattr(streaming_dataset.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:  # the CDN rejects twice, then serves it
            raise _http_error(status, CAS_URL)
        return "frames"

    assert retry_call(flaky, what="decoding file-067.mp4") == "frames"
    assert calls["n"] == 3


@pytest.mark.parametrize("status", [401, 403])
def test_an_auth_rejection_that_never_clears_gives_up_instead_of_hanging(monkeypatch, status):
    """A dataset flipped to private would reject forever; retrying it forever would stall the run."""
    monkeypatch.setattr(streaming_dataset.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def always_rejected():
        calls["n"] += 1
        raise _http_error(status, CAS_URL)

    with pytest.raises(RuntimeError, match=str(status)):
        retry_call(always_rejected, what="decoding file-064.mp4")
    assert calls["n"] == streaming_dataset._PRESIGNED_AUTH_MAX_ATTEMPTS + 1  # noqa: SLF001


def test_a_rate_limit_is_not_capped_by_the_presigned_budget(monkeypatch):
    """The bounded budget applies to signed-URL rejections only; 429s keep retrying forever."""
    monkeypatch.setattr(streaming_dataset.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def rate_limited():
        calls["n"] += 1
        if calls["n"] <= streaming_dataset._PRESIGNED_AUTH_MAX_ATTEMPTS + 5:  # noqa: SLF001
            raise _http_error(429, "https://huggingface.co/api/x")
        return "ok"

    assert retry_call(rate_limited, what="reading meta") == "ok"
