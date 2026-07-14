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
* **Dataset mixtures.** Multiple repos can be mixed. By default samples are drawn *as if the repos
  were a single concatenated dataset*: each source is picked with probability proportional to its
  number of frames (proportional / "sample as one dataset" mixing), which matches uniform sampling
  over a concatenation of the datasets in expectation. Passing ``sampling_weights`` instead fixes
  each repo's per-sample probability to an explicit, size-independent weight -- e.g. equal weights
  make a small target repo contribute the same share as a huge base repo (i.e. oversampling it).
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
  complete. The v3.0 (video) path shards the canonical data files ourselves across (rank, worker);
  when a repo has fewer files than shards (e.g. a small single-file sim repo) those extra shards
  would otherwise stream nothing from it, pinning the whole repo to one worker and making its samples
  arrive in bursts. ``_v3_shard_plan`` avoids this by having the sharing shards split that file's
  *episodes* between them, so every worker holds a proportional slice of every repo.
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
import tempfile
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

# A 401/403 from a *presigned content URL* is not a permissions verdict.
#
# Hub file bytes are served from a CAS/CDN bridge (cas-bridge.xethub.hf.co, cdn-lfs*) behind a
# short-lived signed URL, and that bridge rejects requests transiently -- under load, or on a URL
# that expired mid-read. It is not a statement about your access: a run can stream the same public
# dataset for hours and then take a 403 on one range request. Re-opening the file mints a fresh URL,
# so a retry fixes it, and `_decode_video_frames` re-opens on every call.
#
# The bridge speaks BOTH codes: 403 `AccessDenied` when the signature is rejected, and 401 when it
# considers the request unauthenticated (e.g. an expired signature on a range read). Authorization
# to the bridge lives in the URL signature, not in HF_TOKEN, so a 401 on a signed URL says the same
# thing a 403 does -- "this URL is no longer good" -- and wants the same fix: mint a new one. Only
# the Hub *API* (huggingface.co/api/...) issues a 401 that means "your token is bad"; that one still
# fails fast below, because it is terminal.
#
# But a 401/403 CAN also be real (a dataset flipped to private, a revoked token), and that would
# never clear -- so unlike 429/5xx these are retried a BOUNDED number of times and then fail loudly,
# rather than hanging the job forever behind a 5-minute backoff.
#
# The markers must identify a signed URL by SHAPE, not by CDN vendor. The Hub routes clients to
# different edges -- `cas-bridge.xethub.hf.co` (CloudFront: `X-Amz-Signature`, `Key-Pair-Id=K...`)
# from most networks, but `us.gcp.cdn.hf.co` (Google Cloud CDN: bare `Signature`, ULID `Key-Pair-Id`)
# for GCP-hosted clients like our TPU workers. A vendor-specific list silently misses the edge you
# are actually routed to, classifies its 403 as terminal, and crashes on the first attempt with zero
# retries -- exactly how the GCP edge's "SignatureError: invalid key pair id" killed a run. Matching
# `key-pair-id=` / `signature=` / the `xet-bridge` path covers any edge that speaks this dialect.
# The Hub *API* (huggingface.co/api/...) never carries these, so its 401/403 still fails fast.
_PRESIGNED_AUTH_STATUS = {401, 403}
_PRESIGNED_AUTH_MAX_ATTEMPTS = 8
_PRESIGNED_URL_MARKERS = (
    "xethub.hf.co",
    "xet-bridge",
    "cdn.hf.co",
    "cdn-lfs",
    "key-pair-id=",
    "signature=",  # matches both `Signature=` (GCP) and `X-Amz-Signature=` (CloudFront).
)
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


def _is_presigned_content_url(exc: BaseException) -> bool:
    """Did this error come from a signed Hub content URL (the CAS/CDN bridge), not the Hub API?

    Checked on the response's URL, falling back to the message: huggingface_hub renders the URL into
    HfHubHTTPError's text ("Cannot access content at: https://cas-bridge.xethub.hf.co/…").
    """
    response = getattr(exc, "response", None)
    url = getattr(response, "url", None) or getattr(getattr(response, "request", None), "url", None)
    haystack = f"{url or ''} {exc}".lower()
    return any(marker in haystack for marker in _PRESIGNED_URL_MARKERS)


def _retry_kind(exc: BaseException) -> str | None:
    """How to treat ``exc``: ``None`` (fail), ``"transient"`` (retry forever), or ``"presigned-auth"``.

    A definite HTTP status is authoritative: only transient statuses (429/5xx/408/425) are retried
    forever, so terminal errors like 404 fail fast instead of spinning (this also correctly handles
    aiohttp.ClientResponseError, whose status we extract). The one exception is a 401/403 from a
    signed content URL, which the CAS/CDN bridge hands out transiently -- see
    _PRESIGNED_AUTH_MAX_ATTEMPTS. The same codes from the Hub API are terminal and still fail fast.
    Exceptions with no status are retried only if they are recognized transport/connection errors,
    plus a narrow phrase-based fallback (deliberately not matching bare numbers like "503", which
    would false-match "50302").
    """
    status = _http_status(exc)
    if status is not None:
        if status in _RETRYABLE_STATUS:
            return "transient"
        if status in _PRESIGNED_AUTH_STATUS and _is_presigned_content_url(exc):
            return "presigned-auth"
        return None
    if requests is not None and isinstance(
        exc,
        requests.exceptions.ConnectionError | requests.exceptions.Timeout | requests.exceptions.ChunkedEncodingError,
    ):
        return "transient"
    if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
        return "transient"
    if isinstance(exc, ConnectionError | TimeoutError):
        return "transient"
    message = str(exc).lower()
    if any(needle in message for needle in ("too many requests", "rate limit", "ratelimit", "timed out")):
        return "transient"
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying at all. See ``_retry_kind`` for how long."""
    return _retry_kind(exc) is not None


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
    """Call ``fn``, retrying transient Hub errors with capped backoff.

    Rate limits and 5xx are retried forever ("waiting is fine, crashing is not"). A 401/403 from a
    signed content URL is retried too -- the CDN issues those transiently -- but only
    ``_PRESIGNED_AUTH_MAX_ATTEMPTS`` times, so a rejection that is a genuine permissions change fails
    with a real error instead of stalling the run indefinitely.
    """
    attempt = 0
    presigned_auth_attempts = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            kind = _retry_kind(exc)
            if kind is None:
                raise
            attempt += 1
            if kind == "presigned-auth":
                presigned_auth_attempts += 1
                if presigned_auth_attempts > _PRESIGNED_AUTH_MAX_ATTEMPTS:
                    logger.error(
                        "Hub content URL kept rejecting us (HTTP %s) while %s after %d attempts. A "
                        "signed-URL 401/403 is normally transient, so this looks like a real access "
                        "problem: check the dataset is still public, or that HF_TOKEN is set and "
                        "valid (an unauthenticated stream is throttled far more aggressively).",
                        _http_status(exc),
                        what,
                        presigned_auth_attempts,
                    )
                    raise
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
# Absolute commanded/target joints (DROID 1.0.1 action.joint_position). Only requested + used when
# joint_position_actions=True; datasets lacking this column are unaffected on the default velocity path.
_V3_ACTION_JOINT_POSITION_KEY = "action.joint_position"
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
    """Immutable, picklable metadata for one LeRobot dataset repo.

    All catalog access happens here (in the main process, before workers are forked/spawned); the
    resulting object holds only small, picklable data (file list, tasks, counts) so it can be
    shipped to worker processes cheaply.

    The repo's bytes come from one of two backends, selected by ``mirror_root``:

    * ``None`` (default) -- the HuggingFace Hub (``hf://datasets/<repo_id>/...``).
    * ``gs://bucket/prefix`` -- a *mirror* of the repo at ``<mirror_root>/<repo_id>/...``, read with
      gcsfs. Same files, same layout, same v3.0 reader; only the bytes' origin changes.

    The mirror exists because the Hub's CDN cannot be relied on from GCP: it routes GCP-hosted
    clients to a Google Cloud CDN edge that rejects its own signed URLs ("SignatureError: invalid key
    pair id"), which kills TPU runs outright. Reading from a bucket in the training region also
    removes the cross-cloud hop. Note this is NOT the same as LeRobot's non-streaming path -- that one
    is map-style, needs the whole dataset on local disk, and cannot read v3.0 at all under the pinned
    lerobot (CODEBASE_VERSION "v2.1"). Mirroring keeps this v3.0 reader and just re-points it.

    ``fsspec`` filesystem objects are NOT held on the instance: they are built on demand (see
    ``new_fs``), because this object is pickled out to dataloader worker processes.
    """

    def __init__(self, repo_id: str, *, max_episodes: int | None = None, mirror_root: str | None = None):
        self.repo_id = repo_id
        self.mirror_root: str | None = mirror_root.rstrip("/") if mirror_root else None
        # First-N-episodes cap for this repo (None => use all), set by _apply_episode_cap below.
        # ``max_episodes`` is the clamped COUNT; ``kept_episodes`` is the exact SET of kept episode-index
        # VALUES (the first N by ascending index). _stream_frames_v3 filters by set membership so it keeps
        # exactly the episodes whose files/frames were selected, correct even if episode indices are not
        # 0-based/contiguous. The v2.1 stream needs no in-stream check (one episode per parquet file, so the
        # kept file prefix already contains exactly the kept episodes).
        self.max_episodes: int | None = None
        self.kept_episodes: set[int] | None = None
        self._all_files = retry_call(self._list_files, what=f"listing files for {self.origin}")

        info_path = retry_call(
            lambda: self._download("meta/info.json"), what=f"downloading info.json for {self.origin}"
        )
        with open(info_path) as f:
            info = json.load(f)
        self.codebase_version: str = str(info.get("codebase_version", ""))
        self.is_v30: bool = self.codebase_version.startswith("v3")
        self.total_frames: int = int(info.get("total_frames", 0))
        self.fps: int = int(info.get("fps", 0)) or 1

        if self.is_v30:
            self._init_v30(info)
        else:
            self._init_v21(info, self._download)

        # Restrict to the first N episodes (updating files + total_frames) BEFORE the mixing-weight and
        # keep-range fields are derived from total_frames below, so a capped repo mixes at its subset size.
        if max_episodes is not None:
            self._apply_episode_cap(int(max_episodes))

        # Optional non-idle keep-ranges (episode_index -> list of [start, end) frame ranges). When None,
        # every frame is sampled. Populated by StreamingLeRobotDataset only for repos with a configured
        # filter path. ``sampleable_frames`` is the mixing weight (kept frames if filtered, else total).
        self.keep_ranges: dict[int, list[tuple[int, int]]] | None = None
        self.sampleable_frames: int = self.total_frames

    # -- Backend: the Hub, or a mirror of it. ------------------------------------------------------

    @property
    def origin(self) -> str:
        """Human-readable source of this repo's bytes, for log/error messages."""
        return self.repo_url if self.mirror_root else self.repo_id

    @property
    def repo_url(self) -> str:
        """Root under which this repo's files live, as a full URL (mirror backends only)."""
        return f"{self.mirror_root.rstrip('/')}/{self.repo_id}"

    @property
    def repo_root(self) -> str:
        """Root under which this repo's files live, in fsspec terms (mirror backends only)."""
        # gcsfs takes bucket-relative paths (`find` returns them that way), so strip the scheme here and
        # keep `repo_url` as the only place that speaks a full URL.
        return self.repo_url.removeprefix("gs://")

    def new_fs(self):
        """Build a fresh fsspec filesystem for this repo's backend.

        Called per-process (constructor, and again inside each dataloader worker) rather than cached
        on the instance: fsspec filesystems hold sockets/loops and must not be pickled across the
        worker fork.
        """
        if self.mirror_root is None:
            from huggingface_hub import HfFileSystem

            return HfFileSystem()
        if self.mirror_root.startswith("gs://"):
            import gcsfs

            return gcsfs.GCSFileSystem()
        # Any other mirror root is a plain directory (a local disk, an NFS/Lustre scratch mount).
        import fsspec

        return fsspec.filesystem("file")

    def fs_path(self, rel: str) -> str:
        """fsspec-native path for a repo-relative file (what ``new_fs().open()`` takes)."""
        if self.mirror_root:
            return f"{self.repo_root}/{rel}"
        return f"datasets/{self.repo_id}/{rel}"

    def urls(self, files: Sequence[str]) -> list[str]:
        """Protocol-qualified URLs (what the ``datasets`` parquet builder takes)."""
        if self.mirror_root:
            return [f"{self.repo_url}/{rel}" for rel in files]
        return [f"hf://datasets/{self.repo_id}/{rel}" for rel in files]

    def _list_files(self) -> list[str]:
        """Every repo-relative file path in the repo."""
        if self.mirror_root:
            fs = self.new_fs()
            base = self.repo_root
            found = fs.find(base)
            if not found:
                raise FileNotFoundError(
                    f"No files under mirror {self.repo_url}. The mirror is missing or incomplete -- "
                    f"re-run the dataset mirror before training."
                )
            # gcsfs returns bucket-relative paths ("bucket/prefix/repo/meta/info.json"); strip to "meta/...".
            return sorted(p[len(base) :].lstrip("/") for p in found)
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(self.repo_id, repo_type="dataset")

    def _download(self, rel: str) -> str:
        """Fetch one small metadata file to local disk and return its path."""
        if self.mirror_root:
            fs = self.new_fs()
            local = Path(tempfile.gettempdir()) / "openpi_mirror" / self.repo_id / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            fs.get(self.fs_path(rel), str(local))
            return str(local)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.repo_id, rel, repo_type="dataset")

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
        import pyarrow as pa
        import pyarrow.parquet as pq

        fs = self.new_fs()
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
                    with fs.open(self.fs_path(rel), "rb") as fh:
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
        # All indexed by position 0..E-1 in ascending-episode-index order, so _apply_episode_cap can select
        # the first-N episodes' files, frame counts, and actual index values consistently.
        self._episode_index_sorted: np.ndarray = np.asarray(meta["episode_index"])[order].astype(np.int64)
        self._ep_data_chunk: np.ndarray = ep_data_chunk
        self._ep_data_file: np.ndarray = ep_data_file
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

    def _apply_episode_cap(self, max_episodes: int) -> None:
        """Restrict this source to its first ``max_episodes`` episodes (by ascending episode index).

        Prunes ``self.files`` to only the data files holding a kept episode and shrinks ``self.total_frames``
        to the kept episodes' frame count, so downstream mixing weights reflect the subset. A cap >= the
        repo's episode count is a no-op ("use all"). ``self.kept_episodes`` records the exact index VALUES of
        the kept episodes so ``_stream_frames_v3`` keeps precisely them (a boundary data file may also hold
        dropped episodes) -- by set membership, so it is correct even if episode indices are not
        0-based/contiguous. v2.1 needs no in-stream check (one episode per sorted parquet file, so the kept
        file prefix already contains exactly the kept episodes).
        """
        if max_episodes <= 0:
            raise ValueError(f"max_episodes for {self.repo_id!r} must be positive, got {max_episodes}.")
        if self.is_v30:
            total_eps = int(self.episode_length.shape[0])
            n = min(max_episodes, total_eps)
            if n >= total_eps:
                return
            self.max_episodes = n
            # Position 0..n-1 (ascending index order): their actual index values, files, and frame count.
            # kept_files may include a boundary file that also holds a dropped episode; the stream filter
            # (kept_episodes membership) excludes that episode.
            self.kept_episodes = {int(e) for e in self._episode_index_sorted[:n]}
            kept_files = {
                (int(c), int(f)) for c, f in zip(self._ep_data_chunk[:n], self._ep_data_file[:n], strict=True)
            }
            self.files = [self.data_path.format(chunk_index=c, file_index=f) for c, f in sorted(kept_files)]
            self.total_frames = int(self.episode_length[:n].sum())
        else:
            # v2.1: one episode per parquet file, files sorted by episode index -> first N files == first N
            # episodes (indices 0..n-1). total_frames (from info.json) is over all episodes; scale it by the
            # kept fraction so the size-proportional mixing weight tracks the subset (episode lengths vary,
            # so this is an estimate -- exactly like the whole-repo count it replaces, it only sets a mixing
            # proportion). kept_episodes is recorded for the (rare) filtered-repo weight calc; the v2.1
            # stream itself relies only on the file prefix.
            total_eps = len(self.files)
            n = min(max_episodes, total_eps)
            if n >= total_eps:
                return
            self.max_episodes = n
            self.kept_episodes = set(range(n))
            if self.total_frames and total_eps:
                self.total_frames = round(self.total_frames * n / total_eps)
            self.files = self.files[:n]

    def video_url(self, raw_key: str, episode: int) -> tuple[str, float, float, int]:
        """Return (fsspec video path, from_ts, to_ts, num_frames) for a camera of one episode (v3.0)."""
        vid = self.episode_video[raw_key]
        path = self.video_path.format(
            video_key=raw_key, chunk_index=int(vid["chunk"][episode]), file_index=int(vid["file"][episode])
        )
        return (
            self.fs_path(path),
            float(vid["from_ts"][episode]),
            float(vid["to_ts"][episode]),
            int(self.episode_length[episode]),
        )


def _v3_shard_plan(num_files: int, shard_id: int, num_shards: int) -> tuple[list[int], int, int]:
    """Plan how one v3.0 shard reads a repo's canonical data files: ``(file_indices, group, num_groups)``.

    Two regimes, chosen so every shard gets a proportional slice of the repo (fixing the single-file
    "burst" where a repo with fewer files than shards is pinned to shard 0):

    * ``num_shards <= num_files``: each shard owns a *disjoint set of whole files*
      (``files[shard_id::num_shards]``) and emits all of their episodes (``group 0 of 1``). No file is
      read by more than one shard -- the efficient common case (e.g. 49-file DROID over 8 shards).
    * ``num_shards >  num_files``: files are shared. This shard reads the single file
      ``files[shard_id % num_files]`` and, together with the other shards reading that same file, splits
      its episodes ``num_groups`` ways. So a 1-file repo over 8 shards is spread across all 8 workers
      instead of pinned to one.

    Episode selection within a shared file is applied in ``_stream_frames_v3`` via ``_episode_owner``,
    which maps this shard's ``(group, num_groups)`` onto the file's actual episode count. When a file has
    at least ``num_groups`` episodes this is a disjoint, complete partition; when it has FEWER (more
    shards share it than it has episodes) the excess shards duplicate an episode rather than owning none
    -- so no shard is ever idle (which would hang its dataloader worker) while global coverage is kept.
    """
    if num_files <= 0:
        return [], 0, 1
    if num_shards <= num_files:
        return list(range(shard_id, num_files, num_shards)), 0, 1
    file_idx = shard_id % num_files
    num_groups = len(range(file_idx, num_shards, num_files))  # shards sharing this file
    group = shard_id // num_files  # this shard's index among them, in [0, num_groups)
    return [file_idx], group, num_groups


def _episode_owner(num_episodes: int, episode_group: int, num_episode_groups: int) -> tuple[int, int]:
    """Map a shard's ``(episode_group, num_episode_groups)`` onto a file's episodes: ``(divisor, remainder)``.

    The caller owns local episode ``j`` iff ``j % divisor == remainder``. ``divisor`` is capped at the
    file's episode count so that EVERY group in ``[0, num_episode_groups)`` maps to a real episode:

    * ``num_episodes >= num_episode_groups``: ``divisor = num_episode_groups`` -- a disjoint, complete
      partition; each shard owns ``floor``/``ceil(num_episodes / num_episode_groups)`` episodes (>= 1).
    * ``num_episodes <  num_episode_groups``: ``divisor = num_episodes`` -- there are more sharing shards
      than episodes, so shard ``g`` owns episode ``g % num_episodes`` (exactly one), and shards beyond the
      episode count DUPLICATE earlier episodes instead of owning nothing. A shard owning zero episodes
      would yield forever without producing a sample and hang its dataloader worker; capping the divisor
      is what prevents that. Duplication is harmless here -- these repos are being oversampled anyway.

    ``num_episode_groups == 1`` (the common whole-file shard) always yields ``(1, 0)`` -> owns every
    episode. ``num_episodes == 0`` (an empty file) yields ``(1, 0)`` too, a no-op since there is nothing
    to iterate.
    """
    divisor = min(num_episode_groups, num_episodes) if num_episodes > 0 else 1
    return divisor, episode_group % divisor


def _mixing_weight(source: _HubSource, sampling_weights: Mapping[str, float] | None) -> float:
    """Relative mixing weight for one source: an explicit ``sampling_weights`` entry, else its frame count.

    ``rng.choices`` normalizes these, so explicit weights are size-independent proportions while the
    default (``sampleable_frames``) reproduces size-proportional "sample as one dataset" mixing. The
    same weight is used on every worker; combined with ``_v3_shard_plan`` placing every repo on every
    worker, the per-worker mix equals the intended global mix (no bursts).
    """
    if sampling_weights:
        return float(sampling_weights[source.repo_id])
    return float(source.sampleable_frames) or 1.0


def _v21_only_starves_ranks(sources: Sequence[_HubSource], world_size: int) -> bool:
    """True if some DDP rank would stream nothing (and stall the collective) under this source layout.

    Under DDP each rank streams ``source.files[rank::world_size]`` for a v2.1 source, which is empty for
    ``rank >= len(files)``; a v3.0 source instead covers every rank (``_v3_shard_plan`` gives each shard a
    file). So a rank is starved only when there is NO v3.0 source and every v2.1 source has fewer files than
    ``world_size`` -- then the top ranks own zero files, return an empty generator list, and stall the
    all-reduce. Capping a v2.1 repo small enough can trigger this. ``world_size <= 1`` (the JAX
    single-process path) never starves, so the guard is a no-op there.
    """
    if world_size <= 1:
        return False
    if any(source.is_v30 for source in sources):
        return False
    return max((len(source.files) for source in sources if not source.is_v30), default=0) < world_size


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
        joint_position_actions: bool = False,
        sampling_weights: Mapping[str, float] | None = None,
        max_episodes: Mapping[str, int] | None = None,
        mirror_root: str | None = None,
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
        self.joint_position_actions = bool(joint_position_actions)
        # When set (e.g. "gs://bucket/user/datasets"), read every repo from a mirror under
        # <mirror_root>/<repo_id>/ instead of from the Hub. See _HubSource.
        self.mirror_root: str | None = mirror_root.rstrip("/") if mirror_root else None

        # Explicit per-repo mixing weights (repo_id -> relative weight). Empty/None => size-proportional
        # mixing on frame counts (the default). When provided it must cover EXACTLY the repos in the
        # mixture with positive weights -- mixing an explicit weight with a frame-count fallback (millions
        # vs ~1) would silently wipe out the explicit repo's share, so we fail fast instead of guessing.
        self.sampling_weights = dict(sampling_weights) if sampling_weights else None
        if self.sampling_weights is not None:
            repo_set = set(self.repo_ids)
            missing = [r for r in self.repo_ids if r not in self.sampling_weights]
            if missing:
                raise ValueError(
                    f"sampling_weights was provided but is missing entries for {missing}. Give a weight for "
                    "every repo in the mixture, or leave it empty for size-proportional mixing."
                )
            extra = [k for k in self.sampling_weights if k not in repo_set]
            if extra:
                raise ValueError(f"sampling_weights has entries for repos not in the mixture: {extra}.")
            non_positive = {k: v for k, v in self.sampling_weights.items() if not (float(v) > 0.0)}
            if non_positive:
                raise ValueError(f"sampling_weights values must be positive; got non-positive {non_positive}.")

        # Per-repo first-N-episodes cap (repo_id -> episode count). Empty/None => use every episode of
        # every repo. Entries must reference repos in the mixture and be positive integers; a cap larger
        # than a repo's episode count is clamped to "use all" later (in _HubSource._apply_episode_cap).
        self.max_episodes = {k: int(v) for k, v in max_episodes.items()} if max_episodes else {}
        if self.max_episodes:
            extra = [k for k in self.max_episodes if k not in set(self.repo_ids)]
            if extra:
                raise ValueError(f"max_episodes has entries for repos not in the mixture: {extra}.")
            non_positive = {k: v for k, v in self.max_episodes.items() if v <= 0}
            if non_positive:
                raise ValueError(f"max_episodes values must be positive integers; got {non_positive}.")

        # Joint-position training additionally requests the action.joint_position column; velocity
        # configs keep the original column set so datasets without that column stream unchanged.
        self._v3_lowdim_columns = _V3_LOWDIM_COLUMNS + (
            (_V3_ACTION_JOINT_POSITION_KEY,) if self.joint_position_actions else ()
        )

        # Capture the DDP rank/world size HERE, in the main process. __iter__ runs inside spawned
        # dataloader workers where the process group is NOT initialized, so querying torch.distributed
        # there would wrongly report rank 0 / world size 1 and make every rank stream identical data.
        self._ddp_rank, self._ddp_world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._ddp_rank = torch.distributed.get_rank()
            self._ddp_world_size = torch.distributed.get_world_size()

        _streaming_used["value"] = True

        # Fetch all catalog metadata up front (main process, before worker fork/spawn).
        if self.mirror_root:
            logger.info("Streaming datasets from mirror %s (not the HuggingFace Hub).", self.mirror_root)
        self.sources: list[_HubSource] = [
            _HubSource(repo_id, max_episodes=self.max_episodes.get(repo_id), mirror_root=self.mirror_root)
            for repo_id in self.repo_ids
        ]

        # Fail fast on a DDP layout where some rank would stream nothing and hang the collective -- e.g. a
        # v2.1 repo capped below the GPU count with no v3.0 source to cover the remaining ranks. Only the
        # multi-process (world_size > 1) path can hit this; the JAX single-process flow is unaffected.
        if _v21_only_starves_ranks(self.sources, self._ddp_world_size):
            max_v21 = max((len(s.files) for s in self.sources if not s.is_v30), default=0)
            raise ValueError(
                f"Under DDP with world_size={self._ddp_world_size}, no source covers every rank: there is no "
                f"v3.0 source and the largest v2.1 source has only {max_v21} file(s) (one episode each), so "
                f"ranks beyond that would stream nothing and stall training. Lower world_size, raise the "
                f"max_episodes cap, or include a v3.0 source in the mixture."
            )

        # Fail fast at construction (not at the first training batch) if prompts are required but a
        # repo's task metadata could not be loaded.
        if self.prompt_from_task:
            # Name the file the source actually reads: v3.0 repos carry meta/tasks.parquet and have no
            # tasks.jsonl at all, so a hardcoded "tasks.jsonl" sends you hunting for a file that was
            # never missing -- when the real cause is almost always that the read FAILED (see the
            # "Could not load ..." warning logged just above, e.g. the Hub 401/403-ing an anonymous
            # stream), not that the dataset lacks task metadata.
            missing = [
                f"{source.repo_id} (meta/tasks.{'parquet' if source.is_v30 else 'jsonl'})"
                for source in self.sources
                if not source.tasks
            ]
            if missing:
                raise ValueError(
                    f"prompt_from_task=True but no task metadata could be loaded for {missing}. Check the "
                    f"log above for why the read failed; if the dataset genuinely has no tasks metadata, "
                    f"fix it or set prompt_from_task=False."
                )

        # Joint-position action training is only defined for v3.0 datasets, which expose a dedicated
        # action.joint_position column that _make_v3_frame can select. A v2.1 source has a single
        # (velocity) `actions` column the flag cannot switch, yet the paired DeltaActions transform is
        # applied to the whole mixture -- so a v2.1 repo here would be silently double-transformed. Fail
        # fast instead of corrupting that half of the mixture.
        if self.joint_position_actions:
            non_v3 = [source.repo_id for source in self.sources if not source.is_v30]
            if non_v3:
                raise ValueError(
                    f"joint_position_actions=True requires v3.0 datasets with an "
                    f"{_V3_ACTION_JOINT_POSITION_KEY!r} column, but these repos are v2.1: {non_v3}. "
                    f"Convert them to v3.0 or unset joint_position_actions."
                )

        # Load non-idle keep-ranges ONLY for the repos explicitly configured in ``filter_paths``.
        # Every other repo is sampled in full (no filtering). This is how a DROID repo gets its idle
        # frames filtered out while the rest of the mixture is untouched.
        for source in self.sources:
            path = (filter_paths or {}).get(source.repo_id)
            if path is None:
                continue
            source.keep_ranges = _load_keep_ranges(path)
            # A capped source only samples its kept episodes, so exclude the rest from its kept-frame mixing
            # weight (keep_ranges is keyed by episode index; kept_episodes is the kept index set).
            kept = source.kept_episodes
            source.sampleable_frames = sum(
                end - start
                for episode, ranges in source.keep_ranges.items()
                if kept is None or episode in kept
                for start, end in ranges
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
        data_files = {"train": source.urls(files)}
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
        When ``joint_position_actions`` is set, the joint part comes from ``action.joint_position``
        (absolute commanded targets) instead; the delta transform is added by the data config.
        """
        action_key = _V3_ACTION_JOINT_POSITION_KEY if self.joint_position_actions else _V3_ACTION_JOINT_VELOCITY_KEY
        frame = {
            "episode_index": int(low["episode_index"][row]),
            "frame_index": int(low["frame_index"][row]),
            "joint_position": _as_1d_float(low[_V3_JOINT_POSITION_KEY][row]),
            "gripper_position": _as_1d_float(low[_V3_GRIPPER_POSITION_KEY][row]),
            "actions": np.concatenate(
                [_as_1d_float(low[action_key][row]), _as_1d_float(low[_V3_ACTION_GRIPPER_KEY][row])]
            ),
            "task_index": int(low["task_index"][row]),
        }
        for common_key, decoded in images.items():
            frame[common_key] = decoded[frame_in_ep]
        return frame

    def _stream_frames_v3(
        self,
        source: _HubSource,
        files: Sequence[str],
        base_seed: int,
        *,
        episode_group: int = 0,
        num_episode_groups: int = 1,
    ) -> Iterator[dict]:
        """Yield raw frames for a v3.0 (video) source, forever, in episode-contiguous order.

        Reads each canonical data parquet's low-dim columns, splits it into episodes, decodes each
        episode's camera videos from the MP4s (streamed by timestamp), and yields per-frame dicts with
        the common keys. Sharding is done by the caller: whole disjoint ``files`` per shard in the common
        case, or -- when several shards share one file (a repo with fewer files than shards) -- an episode
        split. Within each file this shard owns the episodes selected by ``_episode_owner`` (which caps the
        split at the file's episode count so this shard is never left owning zero episodes -- that would
        make the generator loop forever without yielding and hang the dataloader worker). Non-owned
        episodes are skipped WITHOUT decoding their videos (the expensive step), so the shared-file case
        stays cheap. The per-file episode ordinal is recomputed each pass, so the split is identical across
        the shards sharing a file and across retries. If a whole pass produces nothing (only possible for a
        degenerate empty file), we sleep before retrying rather than busy-looping on the Hub.
        ``num_episode_groups == 1`` (the default whole-file shard) emits every episode, unchanged.
        """
        import pyarrow.parquet as pq

        fs = source.new_fs()
        files = list(files)
        if self.shuffle:
            random.Random(base_seed).shuffle(files)
        attempt = 0
        while True:
            try:
                produced = False
                for rel in files:
                    data_path = source.fs_path(rel)

                    def _read(data_path=data_path):
                        with fs.open(data_path, "rb") as fh:
                            return pq.ParquetFile(fh).read(columns=list(self._v3_lowdim_columns)).to_pydict()

                    low = retry_call(_read, what=f"reading {rel}")
                    # pyarrow silently drops absent requested columns (no read-time error), so a missing
                    # action.joint_position would otherwise surface as an opaque KeyError in _make_v3_frame.
                    if self.joint_position_actions and _V3_ACTION_JOINT_POSITION_KEY not in low:
                        raise KeyError(
                            f"{source.repo_id!r} lacks column {_V3_ACTION_JOINT_POSITION_KEY!r} required by "
                            "joint_position_actions=True"
                        )
                    attempt = 0
                    episode_col = np.asarray(low["episode_index"])
                    if episode_col.size == 0:
                        # Degenerate empty canonical file: contribute nothing (and, if every file is empty,
                        # leave `produced` False so the pass hits the sleep backstop instead of a busy loop).
                        continue
                    boundaries = np.nonzero(np.diff(episode_col))[0] + 1
                    seg_starts = np.concatenate([[0], boundaries])
                    seg_ends = np.concatenate([boundaries, [len(episode_col)]])
                    # Segments this file contributes: all of them, minus any episode not in the first-N cap
                    # (source.kept_episodes). A capped repo only ever drops segments of a boundary file;
                    # _apply_episode_cap already pruned every file that holds no kept episode, so `allowed`
                    # is non-empty for every file we stream here.
                    allowed = [
                        idx
                        for idx, seg_start in enumerate(seg_starts)
                        if source.kept_episodes is None or int(episode_col[seg_start]) in source.kept_episodes
                    ]
                    if not allowed:
                        continue
                    # Map this shard's (group, num_groups) onto the file's KEPT episode count so it always
                    # owns >= 1 episode (see _episode_owner). Consistent across the shards sharing the file
                    # since they read the same segments in the same order. With no cap `allowed` is every
                    # segment, so this reduces exactly to partitioning the whole file.
                    divisor, remainder = _episode_owner(len(allowed), episode_group, num_episode_groups)
                    for local_ordinal, seg_idx in enumerate(allowed):
                        if local_ordinal % divisor != remainder:
                            continue  # not this shard's episode; skip before any (expensive) video decode.
                        seg_start, seg_end = int(seg_starts[seg_idx]), int(seg_ends[seg_idx])
                        produced = True
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
                if not produced:
                    # Every file was empty (degenerate). Avoid a tight Hub re-read loop; mirrors the v2.1 path.
                    time.sleep(_EMPTY_PASS_WAIT_S)
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
                # data files directly across (rank, worker). When a repo has fewer files than shards,
                # sharing shards split that file's episodes instead of leaving extra shards empty (which
                # would pin a small single-file repo to one worker and make its samples arrive in bursts).
                file_idxs, ep_group, ep_groups = _v3_shard_plan(len(source.files), shard_id, num_shards)
                if not file_idxs:
                    continue
                shard_files = [source.files[i] for i in file_idxs]
                file_seed = self.seed + shard_id * 9973 + source_idx * 104729
                frames = self._stream_frames_v3(
                    source, shard_files, file_seed, episode_group=ep_group, num_episode_groups=ep_groups
                )
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
            # Explicit per-repo weight if configured, else mix proportional to the number of *sampleable*
            # frames (for a filtered source this is its kept-frame count, so the non-idle filter shrinks
            # its share of the mixture accordingly). Every worker uses the same weights and -- thanks to
            # _v3_shard_plan placing every repo on every worker -- yields the same global mix, no bursts.
            weights.append(_mixing_weight(source, self.sampling_weights))

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
