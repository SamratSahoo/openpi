"""Stream LeRobot datasets directly from the HuggingFace Hub.

This module implements :class:`StreamingLeRobotDataset`, a ``torch`` iterable dataset that
streams one or more LeRobot (v2.x) datasets straight from the Hub *without* persisting them
to local disk. It is designed as a drop-in replacement for the map-style
``LeRobotDataset`` path in ``data_loader.py`` for the (common) case where the dataset stores
camera frames inline in the parquet files as HuggingFace ``Image`` features (rather than as
separate MP4 videos).

Key properties:

* **No local storage.** Parquet row groups are streamed over HTTP via the generic
  ``datasets`` parquet builder pointed at ``hf://`` paths. Nothing is written to the
  ``datasets``/LeRobot cache.
* **Dataset mixtures.** Multiple repos can be mixed. Samples are drawn *as if the repos were
  a single concatenated dataset*: each source is picked with probability proportional to its
  number of frames (proportional / "sample as one dataset" mixing), which matches uniform
  sampling over a concatenation of the datasets in expectation.
* **Action-horizon chunking.** For each anchor frame we build the ``action_horizon`` window of
  future actions (clamped/padded at episode boundaries, exactly like LeRobot's
  ``delta_timestamps`` logic), producing an ``actions`` array of shape ``[action_horizon, action_dim]``
  plus an ``actions_is_pad`` mask. This is done with an episode-contiguous sliding window, so it
  is correct even though the stream is iterable.
* **Rate-limit robustness.** Hub 429 / 5xx / transient network errors are caught, ``Retry-After``
  (or a capped exponential backoff) is honored, and the source stream is transparently restarted.
  Training never crashes on a rate limit -- it waits.
* **Shuffling.** A reservoir-style shuffle buffer approximates a global shuffle over the
  otherwise-ordered stream.
* **Sharding.** DDP ranks are sharded here (``datasets`` is not told the rank in this setup);
  sharding across dataloader workers is delegated to ``datasets``, which splits the file list across
  workers as ``files[worker_id::num_workers]``. Because ``datasets`` does this itself, we must NOT
  also slice files per worker (that would double-shard and silently drop ~1-1/num_workers of the
  data); instead we pass every worker of a rank the same file order so the split is disjoint and
  complete.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Iterator, Mapping, Sequence
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback

import datasets
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:  # aiohttp is pulled in transitively by datasets/fsspec; guard just in case.
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# Set (via a mutable holder, to avoid a module-level ``global`` statement) when any
# StreamingLeRobotDataset is constructed. Used by ``run_main`` to decide whether the special
# hard-exit shutdown is needed. Non-streaming runs never set this and are therefore unaffected.
_streaming_used = {"value": False}


def streaming_was_used() -> bool:
    return _streaming_used["value"]


def run_main(fn: Callable[[], object]) -> None:
    """Run a script entrypoint, then exit with the correct code.

    When Hub streaming is used, the background fsspec/aiohttp event-loop threads spun up by
    ``datasets`` streaming can crash during interpreter finalization (a benign but well-known
    fsspec/datasets issue that shows up as a SIGABRT/segfault at process exit). The data is already
    fully produced by then, but the non-zero exit code breaks job schedulers and ``&&`` chains, and
    the buggy teardown can also hang torch's persistent dataloader workers.

    To avoid this we run the registered ``atexit`` handlers ourselves (so e.g. wandb/jax cleanup
    still happens) and then hard-exit via ``os._exit``, bypassing the crashy C-level finalizers,
    while preserving the intended exit code. This only kicks in when streaming was actually used, so
    ordinary (non-streaming) runs behave exactly as before.
    """
    code: int = 0
    try:
        fn()
    except KeyboardInterrupt:
        code = 130  # Conventional SIGINT exit code, so schedulers can tell user-interrupt from failure.
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1

    if not streaming_was_used():
        sys.exit(code)

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        atexit._run_exitfuncs()  # noqa: SLF001 - intentionally run atexit handlers before hard-exit.
    except Exception:
        logger.warning("atexit cleanup raised during streaming hard-exit", exc_info=True)
    os._exit(code)


# Retry tuning. The user preference is "waiting is fine, crashing is not", so we retry
# effectively forever with a capped backoff.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_BASE_WAIT_S = 5.0
_MAX_WAIT_S = 300.0
# Largest header-provided wait we trust as "seconds until reset"; larger values are treated as an
# (unusable) absolute epoch timestamp and ignored in favor of backoff.
_MAX_TRUSTED_HEADER_WAIT_S = 3600.0
# Backstop sleep if a worker is assigned no data by datasets (avoids a busy loop).
_EMPTY_PASS_WAIT_S = 5.0


def _http_status(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            return int(status)
    # aiohttp / fsspec sometimes surface the status directly.
    for attr in ("status", "code"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            return status
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Return True for Hub rate-limit / transient network errors that we should wait out.

    A definite HTTP status is authoritative: only transient statuses (429/5xx/408/425) are retried,
    so terminal errors like 401/403/404 fail fast instead of spinning forever (this also correctly
    handles aiohttp.ClientResponseError, whose status we extract). Exceptions with no status are
    retried only if they are recognized transport/connection errors, plus a narrow phrase-based
    fallback (deliberately not matching bare numbers like "503", which would false-match "50302").
    """
    status = _http_status(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    if requests is not None and isinstance(
        exc,
        requests.exceptions.ConnectionError | requests.exceptions.Timeout | requests.exceptions.ChunkedEncodingError,
    ):
        return True
    if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
        return True
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    message = str(exc).lower()
    return any(needle in message for needle in ("too many requests", "rate limit", "ratelimit", "timed out"))


def _retry_wait_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before retrying, honoring rate-limit headers when present.

    Headers are interpreted as "seconds until reset". A value that is negative (clock skew) or
    implausibly large (an absolute epoch timestamp rather than a delta, e.g. ``X-RateLimit-Reset``
    on some backends) is rejected in favor of exponential backoff, so we never sleep a negative
    duration (which ``time.sleep`` rejects) or stall the full 5-minute cap on a misparsed epoch.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("Retry-After", "retry-after", "X-RateLimit-Reset", "RateLimit-Reset", "ratelimit-reset"):
        value = headers.get(key)
        if value:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= seconds <= _MAX_TRUSTED_HEADER_WAIT_S:
                return min(seconds, _MAX_WAIT_S)
    return min(_BASE_WAIT_S * (2 ** min(attempt, 6)), _MAX_WAIT_S)


def retry_call(fn, *, what: str):
    """Call ``fn`` with unbounded retry-on-transient-error and capped backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            attempt += 1
            wait = _retry_wait_seconds(exc, attempt)
            logger.warning(
                "Transient Hub error while %s (attempt %d); sleeping %.1fs then retrying: %r",
                what,
                attempt,
                wait,
                exc,
            )
            time.sleep(wait)


def _load_keep_ranges(path: str) -> dict[int, list[tuple[int, int]]]:
    """Load a non-idle keep-ranges JSON (``str(episode_index) -> [[start, end], ...]``).

    Produced by ``examples/droid/compute_droid_nonidle_ranges_streaming.py``. Only frames inside a
    kept range are sampled during training; empty range lists filter an episode out entirely.
    """
    with Path(path).open() as f:
        raw = json.load(f)
    return {int(ep): [(int(start), int(end)) for start, end in ranges] for ep, ranges in raw.items()}


# --- LeRobot v3.0 (video) support ------------------------------------------------------------------
#
# v3.0 DROID (``lerobot/droid_1.0.1``) stores camera frames as separate MP4 videos and uses a different
# column schema than the v2.1 inline-image converter datasets. The v3.0 streaming path below normalizes
# both a v3.0 dataset's columns AND its decoded video frames into the SAME raw frame-dict keys the v2.1
# path produces (``exterior_image_1_left``, ``joint_position``, ``gripper_position``, ``actions``, ...),
# so the exact same downstream transforms (e.g. LeRobotDROIDDataConfig) work for both. The training
# ``actions`` are built here as ``concat(action.joint_velocity[7], action.gripper_position[1])`` -- i.e.
# joint-velocity actions -- which is what we train on (droid's own ``action`` is joint-position targets).

# v3.0 DROID column name -> the common raw frame-dict image key produced for downstream transforms.
_V3_VIDEO_KEY_MAP = {
    "observation.images.exterior_1_left": "exterior_image_1_left",
    "observation.images.exterior_2_left": "exterior_image_2_left",
    "observation.images.wrist_left": "wrist_image_left",
}
_V3_JOINT_POSITION_KEY = "observation.state.joint_position"
_V3_GRIPPER_POSITION_KEY = "observation.state.gripper_position"
_V3_ACTION_JOINT_VELOCITY_KEY = "action.joint_velocity"
_V3_ACTION_GRIPPER_KEY = "action.gripper_position"
# Low-dim columns read from the v3.0 data parquet (images come from the videos, decoded separately).
_V3_LOWDIM_COLUMNS = (
    "episode_index",
    "frame_index",
    _V3_JOINT_POSITION_KEY,
    _V3_GRIPPER_POSITION_KEY,
    _V3_ACTION_JOINT_VELOCITY_KEY,
    _V3_ACTION_GRIPPER_KEY,
    "task_index",
)


def _as_1d_float(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=np.float32))


def _decode_video_frames(fs, video_path: str, from_ts: float, to_ts: float, num_frames: int) -> np.ndarray:
    """Decode ``num_frames`` RGB frames starting at ``from_ts`` from a streamed MP4 (returns [N,H,W,3] uint8).

    Streams the MP4 over a seekable ``hf://`` handle: PyAV seeks to ``from_ts`` and reads only the needed
    byte ranges, so the (large) video file is never downloaded in full. Defensively pads/truncates to
    exactly ``num_frames`` so images always align 1:1 with the episode's low-dim rows.
    """
    import av  # Imported lazily so v2.1-only usage does not require PyAV.

    handle = fs.open(video_path, "rb")
    frames: list[np.ndarray] = []
    try:
        container = av.open(handle)
        stream = container.streams.video[0]
        container.seek(int(from_ts / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < from_ts - 1e-3:
                continue
            if len(frames) >= num_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    finally:
        handle.close()
    if not frames:
        raise ValueError(f"Decoded 0 frames from {video_path} in [{from_ts:.3f}, {to_ts:.3f}].")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])


class _HubSource:
    """Immutable, picklable metadata for one HF LeRobot dataset repo.

    All Hub access happens here (in the main process, before workers are forked/spawned); the
    resulting object holds only small, picklable data (file list, tasks, counts) so it can be
    shipped to worker processes cheaply.
    """

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi
        from huggingface_hub import hf_hub_download

        self.repo_id = repo_id
        api = HfApi()
        self._all_files = retry_call(
            lambda: api.list_repo_files(repo_id, repo_type="dataset"), what=f"listing files for {repo_id}"
        )

        def _download(rel: str) -> str:
            return hf_hub_download(repo_id, rel, repo_type="dataset")

        info_path = retry_call(lambda: _download("meta/info.json"), what=f"downloading info.json for {repo_id}")
        with open(info_path) as f:
            info = json.load(f)
        self.codebase_version: str = str(info.get("codebase_version", ""))
        self.is_v30: bool = self.codebase_version.startswith("v3")
        self.total_frames: int = int(info.get("total_frames", 0))
        self.fps: int = int(info.get("fps", 0)) or 1

        if self.is_v30:
            self._init_v30(info)
        else:
            self._init_v21(info, _download)

        # Optional non-idle keep-ranges (episode_index -> list of [start, end) frame ranges). When None,
        # every frame is sampled. Populated by StreamingLeRobotDataset only for repos with a configured
        # filter path. ``sampleable_frames`` is the mixing weight (kept frames if filtered, else total).
        self.keep_ranges: dict[int, list[tuple[int, int]]] | None = None
        self.sampleable_frames: int = self.total_frames

    def _init_v21(self, info: dict, download) -> None:
        """v2.1 LeRobot: inline-image parquet, one episode per file, tasks.jsonl."""
        self.files: list[str] = sorted(f for f in self._all_files if f.startswith("data/") and f.endswith(".parquet"))
        if not self.files:
            raise ValueError(f"No data/*.parquet files found for dataset {self.repo_id!r}.")
        self.image_keys: tuple[str, ...] = tuple(
            key for key, feat in info.get("features", {}).items() if str(feat.get("dtype")) in ("image", "video")
        )
        tasks: dict[int, str] = {}
        try:
            tasks_path = retry_call(
                lambda: download("meta/tasks.jsonl"), what=f"downloading tasks.jsonl for {self.repo_id}"
            )
            with open(tasks_path) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tasks[int(row["task_index"])] = row["task"]
        except Exception:
            logger.warning(
                "Could not load meta/tasks.jsonl for %s; prompts from task will be unavailable.", self.repo_id
            )
        self.tasks: dict[int, str] = tasks

    def _init_v30(self, info: dict) -> None:
        """v3.0 LeRobot (e.g. lerobot/droid_1.0.1): multi-episode parquet + separate MP4 videos.

        Loads authoritative per-episode metadata from ``meta/episodes`` (which canonical data file holds
        each episode, and which video file + timestamp range holds each camera's frames) plus
        ``meta/tasks.parquet``. Only canonical data files (referenced by an episode) are streamed.
        """
        from huggingface_hub import HfFileSystem
        import pyarrow as pa
        import pyarrow.parquet as pq

        fs = HfFileSystem()
        features = info.get("features", {})
        self.data_path: str = info["data_path"]
        self.video_path: str = info["video_path"]
        # Raw v3.0 video column -> common frame-dict image key (only those present in this dataset).
        self.video_key_map: dict[str, str] = {k: v for k, v in _V3_VIDEO_KEY_MAP.items() if k in features}
        if not self.video_key_map:
            raise ValueError(f"v3.0 dataset {self.repo_id!r} has none of the expected DROID video keys.")
        self.image_keys: tuple[str, ...] = tuple(self.video_key_map.values())

        def _read_meta(rel_paths, columns):
            tables = []
            for rel in rel_paths:

                def _read(rel=rel):
                    with fs.open(f"datasets/{self.repo_id}/{rel}", "rb") as fh:
                        return pq.ParquetFile(fh).read(columns=columns)

                tables.append(retry_call(_read, what=f"reading {rel}"))
            return pa.concat_tables(tables).to_pydict()

        episode_meta_files = sorted(
            f for f in self._all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")
        )
        if not episode_meta_files:
            raise ValueError(f"v3.0 dataset {self.repo_id!r} has no meta/episodes parquet files.")
        ep_columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        for raw_key in self.video_key_map:
            ep_columns += [
                f"videos/{raw_key}/chunk_index",
                f"videos/{raw_key}/file_index",
                f"videos/{raw_key}/from_timestamp",
                f"videos/{raw_key}/to_timestamp",
            ]
        meta = _read_meta(episode_meta_files, ep_columns)
        order = np.argsort(np.asarray(meta["episode_index"]))  # index everything by episode_index
        self.episode_length: np.ndarray = np.asarray(meta["length"])[order].astype(np.int64)
        ep_data_chunk = np.asarray(meta["data/chunk_index"])[order].astype(np.int64)
        ep_data_file = np.asarray(meta["data/file_index"])[order].astype(np.int64)
        self.episode_video: dict[str, dict[str, np.ndarray]] = {}
        for raw_key in self.video_key_map:
            self.episode_video[raw_key] = {
                "chunk": np.asarray(meta[f"videos/{raw_key}/chunk_index"])[order].astype(np.int64),
                "file": np.asarray(meta[f"videos/{raw_key}/file_index"])[order].astype(np.int64),
                "from_ts": np.asarray(meta[f"videos/{raw_key}/from_timestamp"])[order].astype(np.float64),
                "to_ts": np.asarray(meta[f"videos/{raw_key}/to_timestamp"])[order].astype(np.float64),
            }

        # Canonical data files (referenced by episodes), in order. These are what we shard/stream.
        canonical = sorted({(int(c), int(f)) for c, f in zip(ep_data_chunk, ep_data_file, strict=True)})
        self.files: list[str] = [self.data_path.format(chunk_index=chunk, file_index=file) for chunk, file in canonical]

        tasks: dict[int, str] = {}
        try:
            tasks_meta = _read_meta(["meta/tasks.parquet"], None)
            # The task text lives in "task" (some versions) or "__index_level_0__" (pandas index column,
            # as in lerobot/droid_1.0.1); "task_index" is the numeric id.
            text_key = next((k for k in ("task", "__index_level_0__") if k in tasks_meta), None)
            if text_key is None:
                text_key = next(k for k in tasks_meta if k != "task_index")
            task_strings = tasks_meta[text_key]
            task_indices = tasks_meta.get("task_index", list(range(len(task_strings))))
            for idx, task in zip(task_indices, task_strings, strict=True):
                tasks[int(idx)] = str(task)
        except Exception:
            logger.warning("Could not load meta/tasks.parquet for %s; prompts from task unavailable.", self.repo_id)
        self.tasks: dict[int, str] = tasks

    def video_url(self, raw_key: str, episode: int) -> tuple[str, float, float, int]:
        """Return (fsspec video path, from_ts, to_ts, num_frames) for a camera of one episode (v3.0)."""
        vid = self.episode_video[raw_key]
        path = self.video_path.format(
            video_key=raw_key, chunk_index=int(vid["chunk"][episode]), file_index=int(vid["file"][episode])
        )
        return (
            f"datasets/{self.repo_id}/{path}",
            float(vid["from_ts"][episode]),
            float(vid["to_ts"][episode]),
            int(self.episode_length[episode]),
        )

    def hf_paths(self, files: Sequence[str]) -> list[str]:
        return [f"hf://datasets/{self.repo_id}/{rel}" for rel in files]


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that streams (and mixes) LeRobot datasets from the Hub.

    Each yielded sample is a raw ``dict`` whose keys mirror what the map-style LeRobot path
    produces *before* the repack/data transforms run (image feature keys as HWC uint8 arrays,
    low-dim feature keys, ``actions`` chunked to ``[action_horizon, action_dim]``, ``task_index``,
    and -- when ``prompt_from_task`` is set -- ``prompt``). This lets it reuse the exact same
    downstream transform stack as the non-streaming path.
    """

    def __init__(
        self,
        repo_ids: Sequence[str],
        action_horizon: int,
        *,
        action_sequence_keys: Sequence[str] = ("actions",),
        prompt_from_task: bool = False,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 0,
        filter_paths: Mapping[str, str] | None = None,
    ):
        super().__init__()
        if isinstance(repo_ids, str):
            repo_ids = [repo_ids]
        if not repo_ids:
            raise ValueError("StreamingLeRobotDataset requires at least one repo id.")
        self.repo_ids = list(repo_ids)
        self.action_horizon = int(action_horizon)
        self.action_sequence_keys = tuple(action_sequence_keys)
        self.prompt_from_task = bool(prompt_from_task)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)

        # Capture the DDP rank/world size HERE, in the main process. __iter__ runs inside spawned
        # dataloader workers where the process group is NOT initialized, so querying torch.distributed
        # there would wrongly report rank 0 / world size 1 and make every rank stream identical data.
        self._ddp_rank, self._ddp_world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._ddp_rank = torch.distributed.get_rank()
            self._ddp_world_size = torch.distributed.get_world_size()

        _streaming_used["value"] = True

        # Fetch all Hub metadata up front (main process, before worker fork/spawn).
        self.sources: list[_HubSource] = [_HubSource(repo_id) for repo_id in self.repo_ids]

        # Fail fast at construction (not at the first training batch) if prompts are required but a
        # repo's task metadata could not be loaded.
        if self.prompt_from_task:
            missing = [source.repo_id for source in self.sources if not source.tasks]
            if missing:
                raise ValueError(
                    f"prompt_from_task=True but no task metadata (meta/tasks.jsonl) could be loaded for "
                    f"{missing}. Fix the dataset's tasks metadata or set prompt_from_task=False."
                )

        # Load non-idle keep-ranges ONLY for the repos explicitly configured in ``filter_paths``.
        # Every other repo is sampled in full (no filtering). This is how a DROID repo gets its idle
        # frames filtered out while the rest of the mixture is untouched.
        for source in self.sources:
            path = (filter_paths or {}).get(source.repo_id)
            if path is None:
                continue
            source.keep_ranges = _load_keep_ranges(path)
            source.sampleable_frames = sum(
                end - start for ranges in source.keep_ranges.values() for start, end in ranges
            )
            logger.info(
                "Loaded non-idle filter for %s: %d episodes, %d/%d frames kept (%.1f%%).",
                source.repo_id,
                len(source.keep_ranges),
                source.sampleable_frames,
                source.total_frames,
                100 * source.sampleable_frames / max(1, source.total_frames),
            )

    @property
    def total_frames(self) -> int:
        return sum(source.total_frames for source in self.sources)

    def __len__(self) -> int:
        # Approximate length (one pass over all source frames). The stream itself is infinite;
        # this is only used for logging / bounding norm-stat computation.
        return self.total_frames

    # -- streaming ----------------------------------------------------------------------------

    def _stream_frames(self, source: _HubSource, files: Sequence[str], base_seed: int) -> Iterator[dict]:
        """Yield raw frames for ``source`` in episode-contiguous order, forever.

        Worker-level sharding of ``files`` is performed internally by ``datasets`` (it splits the file
        list across dataloader workers as ``files[worker_id::num_workers]``). We shuffle the file order
        ONCE with a worker-independent ``base_seed`` so every worker of a rank sees the same order and
        ``datasets`` splits it into disjoint, complete per-worker subsets. (A per-pass reshuffle would
        desync workers that hold different numbers of files, breaking that split.) Transient Hub errors
        are waited out and the stream restarted from the start of this worker's shard rather than
        propagated, so training never crashes on a rate limit.
        """
        files = list(files)
        if self.shuffle:
            random.Random(base_seed).shuffle(files)
        data_files = {"train": source.hf_paths(files)}
        attempt = 0
        while True:
            try:
                stream = datasets.load_dataset("parquet", data_files=data_files, split="train", streaming=True)
                produced = False
                for row in stream:
                    attempt = 0  # any successful read resets the backoff.
                    produced = True
                    yield row
                # Completed a full pass; loop to continue (infinite stream).
                if not produced:
                    # datasets assigned this worker no data (more workers than file shards). __iter__
                    # normally skips such workers, so this is only a backstop against a busy loop.
                    time.sleep(_EMPTY_PASS_WAIT_S)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                attempt += 1
                wait = _retry_wait_seconds(exc, attempt)
                logger.warning(
                    "Transient Hub error while streaming %s (attempt %d); sleeping %.1fs then restarting: %r",
                    source.repo_id,
                    attempt,
                    wait,
                    exc,
                )
                time.sleep(wait)

    def _make_v3_frame(self, source: _HubSource, low: dict, row: int, frame_in_ep: int, images: dict) -> dict:
        """Build one raw frame dict from v3.0 low-dim rows + decoded images, using the common keys.

        The training ``actions`` are ``concat(action.joint_velocity[7], action.gripper_position[1])`` --
        joint-velocity actions -- matching the v2.1 datasets so all downstream transforms are shared.
        """
        frame = {
            "episode_index": int(low["episode_index"][row]),
            "frame_index": int(low["frame_index"][row]),
            "joint_position": _as_1d_float(low[_V3_JOINT_POSITION_KEY][row]),
            "gripper_position": _as_1d_float(low[_V3_GRIPPER_POSITION_KEY][row]),
            "actions": np.concatenate(
                [_as_1d_float(low[_V3_ACTION_JOINT_VELOCITY_KEY][row]), _as_1d_float(low[_V3_ACTION_GRIPPER_KEY][row])]
            ),
            "task_index": int(low["task_index"][row]),
        }
        for common_key, decoded in images.items():
            frame[common_key] = decoded[frame_in_ep]
        return frame

    def _stream_frames_v3(self, source: _HubSource, files: Sequence[str], base_seed: int) -> Iterator[dict]:
        """Yield raw frames for a v3.0 (video) source, forever, in episode-contiguous order.

        Reads each canonical data parquet's low-dim columns, splits it into episodes, decodes each
        episode's camera videos from the MP4s (streamed by timestamp), and yields per-frame dicts with
        the common keys. Sharding is done by the caller (disjoint ``files`` per shard). Transient Hub
        errors are waited out and the current file restarted.
        """
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem()
        files = list(files)
        if self.shuffle:
            random.Random(base_seed).shuffle(files)
        attempt = 0
        while True:
            try:
                for rel in files:
                    data_path = f"datasets/{source.repo_id}/{rel}"

                    def _read(data_path=data_path):
                        with fs.open(data_path, "rb") as fh:
                            return pq.ParquetFile(fh).read(columns=list(_V3_LOWDIM_COLUMNS)).to_pydict()

                    low = retry_call(_read, what=f"reading {rel}")
                    attempt = 0
                    episode_col = np.asarray(low["episode_index"])
                    boundaries = np.nonzero(np.diff(episode_col))[0] + 1
                    seg_starts = np.concatenate([[0], boundaries])
                    seg_ends = np.concatenate([boundaries, [len(episode_col)]])
                    for seg_start, seg_end in zip(seg_starts, seg_ends, strict=True):
                        episode = int(episode_col[seg_start])
                        seg_len = int(seg_end - seg_start)
                        images = {}
                        for raw_key, common_key in source.video_key_map.items():
                            video_path, from_ts, to_ts, num_frames = source.video_url(raw_key, episode)

                            def _decode(video_path=video_path, from_ts=from_ts, to_ts=to_ts, num_frames=num_frames):
                                return _decode_video_frames(fs, video_path, from_ts, to_ts, num_frames)

                            images[common_key] = retry_call(_decode, what=f"decoding {video_path}")
                        for i in range(seg_len):
                            yield self._make_v3_frame(source, low, seg_start + i, i, images)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                attempt += 1
                wait = _retry_wait_seconds(exc, attempt)
                logger.warning(
                    "Transient Hub error while streaming (v3) %s (attempt %d); sleeping %.1fs then restarting: %r",
                    source.repo_id,
                    attempt,
                    wait,
                    exc,
                )
                time.sleep(wait)

    # -- chunking -----------------------------------------------------------------------------

    def _make_sample(self, window: Sequence[dict], source: _HubSource) -> dict:
        """Build one training sample anchored at ``window[0]`` with an ``action_horizon`` window."""
        anchor = window[0]
        horizon = self.action_horizon
        sample = dict(anchor)

        # Decode image features to numpy HWC uint8 (matches np.asarray(PIL) used downstream).
        for key in source.image_keys:
            if key in sample:
                sample[key] = np.asarray(sample[key])

        # Chunk each action-sequence key over the (episode-clamped) horizon window.
        for key in self.action_sequence_keys:
            seq = [np.asarray(frame[key], dtype=np.float32) for frame in window[:horizon]]
            is_pad = [False] * len(seq)
            # Pad by repeating the last in-episode frame's value (LeRobot clamps idx to ep_end-1).
            while len(seq) < horizon:
                seq.append(seq[-1])
                is_pad.append(True)
            sample[key] = np.stack(seq, axis=0)
            sample[f"{key}_is_pad"] = np.asarray(is_pad, dtype=bool)

        sample["task_index"] = int(np.asarray(anchor["task_index"]).item())
        if self.prompt_from_task:
            prompt = source.tasks.get(sample["task_index"])
            if prompt is None:
                raise ValueError(f"task_index={sample['task_index']} not found in tasks for {source.repo_id!r}.")
            sample["prompt"] = prompt
        return sample

    def _keep_anchor(self, source: _HubSource, anchor: dict) -> bool:
        """Whether an anchor frame should be sampled, given this source's non-idle keep-ranges.

        Sources without keep-ranges (i.e. every repo not explicitly configured for filtering) keep
        all frames. Filtering only affects the anchor frame; the action-chunk window is still built
        from the surrounding (possibly filtered-out) frames.
        """
        if source.keep_ranges is None:
            return True
        episode = int(np.asarray(anchor["episode_index"]).item())
        ranges = source.keep_ranges.get(episode)
        if ranges is None:
            # Episode absent from the (assumed-complete) filter map: keep it rather than silently
            # dropping data. A complete map has an entry for every episode.
            return True
        frame_index = int(np.asarray(anchor["frame_index"]).item())
        return any(start <= frame_index < end for start, end in ranges)

    def _chunk_frames(self, frames: Iterator[dict], source: _HubSource) -> Iterator[dict]:
        """Turn an ordered frame stream into anchor-chunked samples, respecting episode bounds."""
        horizon = self.action_horizon
        window: list[dict] = []
        current_ep: int | None = None

        def flush_tail() -> Iterator[dict]:
            # Emit remaining anchors (near the episode end) with padding.
            for i in range(len(window)):
                if self._keep_anchor(source, window[i]):
                    yield self._make_sample(window[i:], source)

        for frame in frames:
            episode = int(np.asarray(frame["episode_index"]).item())
            frame_index = int(np.asarray(frame["frame_index"]).item())
            # A new episode begins whenever the per-episode frame index resets to 0 or the episode
            # index changes. Using frame_index == 0 also correctly separates a re-streamed copy of
            # the same episode at a pass seam (where episode_index alone would not change).
            is_boundary = bool(window) and (frame_index == 0 or episode != current_ep)
            if is_boundary:
                yield from flush_tail()
                window = []
            current_ep = episode
            window.append(frame)
            # Emit any anchor that now has a full horizon of in-episode frames available (subject to
            # the non-idle filter for this source).
            while len(window) >= horizon:
                if self._keep_anchor(source, window[0]):
                    yield self._make_sample(window[:horizon], source)
                window.pop(0)
        # Infinite streams never reach here, but be correct if a source ever ends.
        yield from flush_tail()

    # -- mixing + shuffling -------------------------------------------------------------------

    def _shuffle_buffer(self, it: Iterator[dict], rng: random.Random) -> Iterator[dict]:
        size = self.shuffle_buffer_size
        if not self.shuffle or size <= 1:
            yield from it
            return
        buffer: list[dict] = []
        for item in it:
            if len(buffer) < size:
                buffer.append(item)
                continue
            j = rng.randrange(size)
            out, buffer[j] = buffer[j], item
            yield out
        # Drain (only reached for finite streams).
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterator[dict]:
        # DDP rank is captured in the main process at construction; worker id is read here (inside the
        # worker). We shard files by RANK only -- worker sharding is delegated to `datasets` (see the
        # module docstring), which splits each rank's file list across the rank's workers.
        rank, world_size = self._ddp_rank, self._ddp_world_size
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        gens: list[Iterator[dict]] = []
        weights: list[float] = []
        for source_idx, source in enumerate(self.sources):
            if source.is_v30:
                # v3.0 (video): we read parquet + decode video ourselves, so we shard the canonical
                # data files directly across (rank, worker). Files are disjoint per shard, so a
                # per-shard reshuffle seed is fine.
                shard_files = source.files[shard_id::num_shards]
                if not shard_files:
                    continue
                file_seed = self.seed + shard_id * 9973 + source_idx * 104729
                frames = self._stream_frames_v3(source, shard_files, file_seed)
            else:
                # v2.1 (inline image): shard by DDP rank; worker sharding is delegated to datasets, which
                # gives this worker rank_files[worker_id::num_workers] (empty when worker_id >= len).
                rank_files = source.files[rank::world_size]
                if worker_id >= len(rank_files):
                    continue
                # Worker-INDEPENDENT seed so all workers of a rank shuffle identically and datasets'
                # per-worker split stays disjoint and complete.
                file_seed = self.seed + rank * 9973 + source_idx * 104729
                frames = self._stream_frames(source, rank_files, file_seed)
            gens.append(self._chunk_frames(frames, source))
            # Mix proportional to the number of *sampleable* frames: for a filtered source this is its
            # kept-frame count, so the non-idle filter shrinks its share of the mixture accordingly.
            weights.append(float(source.sampleable_frames) or 1.0)

        if not gens:
            # This worker was assigned no files from any source (num_workers exceeds every repo's
            # file/shard count). Return so the DataLoader marks it finished instead of hanging; the
            # remaining workers still cover all of the data.
            return

        rng = random.Random(self.seed + (rank * num_workers + worker_id) * 9973)

        def interleaved() -> Iterator[dict]:
            # Proportional ("sample as one dataset") mixing: pick a source with probability
            # proportional to its size, then draw its next (already-shuffled-in-order) sample.
            if len(gens) == 1:
                yield from gens[0]
                return
            indices = list(range(len(gens)))
            while True:
                (choice,) = rng.choices(indices, weights=weights, k=1)
                yield next(gens[choice])

        yield from self._shuffle_buffer(interleaved(), rng)
