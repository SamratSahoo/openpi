"""Compute DROID non-idle keep-ranges for a LeRobot-format DROID dataset by STREAMING it.

This is a streaming counterpart to ``compute_droid_nonidle_ranges.py`` (which reads a local TFDS/RLDS
copy of DROID). It targets the LeRobot v3.0 dataset ``lerobot/droid_1.0.1`` on the HuggingFace Hub and
streams only the low-dimensional columns it needs (``episode_index``, ``frame_index``,
``action.joint_velocity``) via pyarrow column projection over ``hf://`` -- so the (very large) camera
videos are never downloaded and nothing is written to the dataset cache.

It produces a JSON mapping ``str(episode_index) -> [[start, end], ...]``: for each episode, the ranges of
frame indices that should be sampled during training (all other frames are idle/near-idle and filtered
out). The filtering logic mirrors the RLDS version exactly:

* A frame is "idle" if its joint velocity barely changed from the previous frame
  (``|jv[t] - jv[t-1]| < 1e-3`` on all 7 joints).
* Idle segments of length >= ``min_idle_len`` (7) are filtered out.
* Of what remains, only non-idle segments of length >= ``min_non_idle_len`` (16) are kept.
* The last ``filter_last_n_in_ranges`` (10) frames are trimmed from each kept range (those anchors
  correspond to action chunks that are mostly idle).

The output JSON is consumed at training time by the streaming data loader (see
``openpi.training.streaming_dataset.StreamingLeRobotDataset``) to filter this repo's frames.

Usage:
    uv run python examples/droid/compute_droid_nonidle_ranges_streaming.py \
        --repo-id lerobot/droid_1.0.1 \
        --out filters/lerobot_droid_1.0.1.json

The script is resumable: it periodically writes partial results and, on restart, skips episodes that
are already present in the output file. Transient Hub errors (429/5xx/network) are waited out.
"""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Reuse the streaming module's robust Hub-retry helpers so a 429 mid-compute waits instead of crashing.
from openpi.training import streaming_dataset as _sd

_COLUMNS = ["episode_index", "frame_index", "action.joint_velocity"]


def compute_keep_ranges(
    joint_velocities: np.ndarray,
    *,
    min_idle_len: int,
    min_non_idle_len: int,
    filter_last_n_in_ranges: int,
) -> list[tuple[int, int]]:
    """Non-idle keep-ranges for one episode. Mirrors compute_droid_nonidle_ranges.py exactly."""
    if joint_velocities.ndim != 2 or joint_velocities.shape[0] == 0:
        return []
    num_frames = joint_velocities.shape[0]

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )

    # Start and end with False so idle at the first step counts as a start of motion.
    is_idle_padded = np.concatenate([[False], is_idle_array, [False]])
    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]

    # Keep only idle segments of length at least min_idle_len (those get filtered out below).
    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(num_frames, dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    # Now find contiguous kept (non-idle) segments of length at least min_non_idle_len.
    keep_padded = np.concatenate([[False], keep_mask, [False]])
    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]
    keep_true_ends = np.where(keep_diff == -1)[0]

    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    ranges: list[tuple[int, int]] = []
    for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
        ranges.append((int(start), int(end) - filter_last_n_in_ranges))
    return ranges


def _joint_velocity_numpy(batch, column: str) -> np.ndarray:
    """Extract a list<float> column from a pyarrow RecordBatch as a [N, D] float array.

    Handles both fixed-size and variable-size list types and correctly respects array offsets/nulls
    via ``list_flatten``. Assumes each row has the same width (7 for joint velocity).
    """
    col = batch.column(column)
    num_rows = len(col)
    flat = np.asarray(pc.list_flatten(col).to_numpy(zero_copy_only=False), dtype=np.float32)
    if num_rows == 0:
        return flat.reshape(0, 0)
    if flat.shape[0] % num_rows == 0:
        return flat.reshape(num_rows, flat.shape[0] // num_rows)
    # Ragged / contains nulls (unexpected for joint velocity): fall back to a per-row conversion.
    return np.array(col.to_pylist(), dtype=np.float32)


def _iter_file_batches(fs: HfFileSystem, path: str, batch_size: int):
    """Yield pyarrow RecordBatches (projected to _COLUMNS) for one parquet file, with Hub retry."""

    def _open():
        return pq.ParquetFile(fs.open(path, "rb"))

    parquet_file = _sd.retry_call(_open, what=f"opening {path}")
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=_COLUMNS)


def _canonical_data_files(repo_id: str, fs: HfFileSystem) -> list[str] | None:
    """Return the CANONICAL data parquet paths (in order), or None if there is no episode metadata.

    LeRobot v3.0 DROID ships extra, non-canonical data parquet files (e.g. lerobot/droid_1.0.1 has 156
    data files but only 86 are real training data). Only files referenced by an episode's
    ``data/{chunk,file}_index`` in ``meta/episodes`` are canonical; streaming the rest would duplicate
    frames and corrupt episode grouping. If ``meta/episodes`` is absent (e.g. a v2.1 dataset), returns
    None so the caller falls back to all data files.
    """
    api = HfApi()
    episode_meta_files = sorted(
        f
        for f in _sd.retry_call(lambda: api.list_repo_files(repo_id, repo_type="dataset"), what="listing files")
        if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    if not episode_meta_files:
        return None
    pairs: set[tuple[int, int]] = set()
    for rel in episode_meta_files:

        def _read(rel=rel):
            with fs.open(f"datasets/{repo_id}/{rel}", "rb") as fh:
                return pq.ParquetFile(fh).read(columns=["data/chunk_index", "data/file_index"]).to_pydict()

        table = _sd.retry_call(_read, what=f"reading {rel}")
        pairs.update(zip(table["data/chunk_index"], table["data/file_index"], strict=True))
    return [f"data/chunk-{int(chunk):03d}/file-{int(file):03d}.parquet" for chunk, file in sorted(pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    parser.add_argument("--out", required=True, help="Path to write the keep-ranges JSON.")
    parser.add_argument("--min-idle-len", type=int, default=7)
    parser.add_argument("--min-non-idle-len", type=int, default=16)
    parser.add_argument("--filter-last-n-in-ranges", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint every N episodes.")
    parser.add_argument("--max-files", type=int, default=None, help="For testing: only process this many files.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep_ranges: dict[str, list[tuple[int, int]]] = {}
    if out_path.exists():
        with out_path.open() as f:
            keep_ranges = {k: [tuple(r) for r in v] for k, v in json.load(f).items()}
        print(f"Resuming: {len(keep_ranges)} episodes already computed.")

    def save() -> None:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(keep_ranges, f)
        tmp.replace(out_path)

    api = HfApi()
    fs = HfFileSystem()
    # Prefer the canonical file list from meta/episodes (v3.0); fall back to all data files (v2.1).
    files = _canonical_data_files(args.repo_id, fs)
    if files is None:
        files = sorted(
            f
            for f in _sd.retry_call(
                lambda: api.list_repo_files(args.repo_id, repo_type="dataset"), what="listing files"
            )
            if f.startswith("data/") and f.endswith(".parquet")
        )
        print(f"No meta/episodes found; using all {len(files)} data files.")
    else:
        print(f"Using {len(files)} canonical data files (from meta/episodes).")
    if args.max_files is not None:
        files = files[: args.max_files]
    print(f"Streaming {len(files)} parquet files from {args.repo_id} (low-dim columns only, no video).")
    current_ep: int | None = None
    chunks: list[np.ndarray] = []
    max_ep_seen = -1
    processed = 0
    t0 = time.monotonic()

    def flush() -> None:
        nonlocal chunks
        if current_ep is None:
            return
        key = str(current_ep)
        if key not in keep_ranges:
            jv = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 7), dtype=np.float32)
            keep_ranges[key] = compute_keep_ranges(
                jv,
                min_idle_len=args.min_idle_len,
                min_non_idle_len=args.min_non_idle_len,
                filter_last_n_in_ranges=args.filter_last_n_in_ranges,
            )
        chunks = []

    total_rows = 0
    for file_idx, rel_path in enumerate(files):
        path = f"datasets/{args.repo_id}/{rel_path}"
        for batch in _iter_file_batches(fs, path, args.batch_size):
            episode_index = np.asarray(batch.column("episode_index").to_numpy(zero_copy_only=False))
            joint_velocity = _joint_velocity_numpy(batch, "action.joint_velocity")
            total_rows += len(episode_index)
            if len(episode_index) and int(episode_index[-1]) < max_ep_seen:
                print(f"WARNING: episode_index went backwards in {rel_path}; grouping assumes monotonic order.")
            # Split this batch into contiguous per-episode segments (episodes are stored in order).
            boundaries = np.nonzero(np.diff(episode_index))[0] + 1
            seg_starts = np.concatenate([[0], boundaries])
            seg_ends = np.concatenate([boundaries, [len(episode_index)]])
            for seg_start, seg_end in zip(seg_starts, seg_ends, strict=True):
                ep = int(episode_index[seg_start])
                if ep != current_ep:
                    flush()
                    current_ep = ep
                    processed += 1
                    max_ep_seen = max(max_ep_seen, ep)
                    if processed % args.save_every == 0:
                        save()
                        rate = total_rows / max(1e-6, time.monotonic() - t0)
                        print(
                            f"  episodes={len(keep_ranges)} rows={total_rows:,} "
                            f"({rate:,.0f} rows/s) file {file_idx + 1}/{len(files)}"
                        )
                chunks.append(joint_velocity[seg_start:seg_end])

    flush()
    save()
    total_kept = sum(end - start for ranges in keep_ranges.values() for start, end in ranges)
    print(
        f"Done. {len(keep_ranges)} episodes, {total_rows:,} frames streamed, "
        f"{total_kept:,} kept frames ({100 * total_kept / max(1, total_rows):.1f}%). Wrote {out_path}."
    )


if __name__ == "__main__":
    _sd.run_main(main)
