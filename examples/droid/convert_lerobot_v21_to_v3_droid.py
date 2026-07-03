"""Convert a v2.1 inline-image LeRobot dataset (e.g. SamratSahoo/toys*_sim) to a v3.0 video dataset
that matches ``lerobot/droid_1.0.1``'s schema, by STREAMING the source (no full local download).

Why: the openpi streaming loader (openpi.training.streaming_dataset) can mix v3.0 (video) and v2.1
(inline-image) datasets, but inline-image datasets are huge (toys300_sim is ~98 GB of PNG); re-encoding
their frames as MP4 shrinks them ~10-30x and unifies the schema with DROID so one transform stack (and
one action space) serves the whole mixture.

What it does, streaming the source parquet-by-parquet (v2.1 stores one episode per parquet, images inline
as PNG bytes):
  * decodes each episode's camera frames and encodes them into per-camera MP4 videos (all episodes
    concatenated per camera, each episode occupying a timestamp range) -- v3.0 layout;
  * writes the low-dim data parquet with DROID v3.0 column names, mapping the source's joint-velocity
    ``actions[0:7]`` -> ``action.joint_velocity`` and ``actions[7]`` -> ``action.gripper_position``,
    and ``joint_position``/``gripper_position`` -> ``observation.state.*``;
  * writes v3.0 meta (info.json, tasks.parquet, meta/episodes/... with the video timestamp mapping);
  * optionally pushes the result to a new HF dataset repo (needs your HF write token).

The output is directly consumable by ``StreamingLeRobotDataset`` (v3.0 path). Run per dataset:

    uv run python examples/droid/convert_lerobot_v21_to_v3_droid.py \
        --src-repo SamratSahoo/toys100_sim \
        --dst-repo SamratSahoo/toys100_sim_v3 \
        --out-dir /tmp/toys100_v3 --push

Use ``--max-episodes N`` to convert a small sample for a smoke test (skip ``--push``).
"""

import argparse
import json
from pathlib import Path

import av
import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training import streaming_dataset as _sd

# Source (v2.1 toys) key -> nothing; we hardcode the known toys schema and emit the DROID v3.0 schema.
_SRC_IMAGE_KEYS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
# Source image key -> DROID v3.0 video key.
_DST_VIDEO_KEY = {
    "exterior_image_1_left": "observation.images.exterior_1_left",
    "exterior_image_2_left": "observation.images.exterior_2_left",
    "wrist_image_left": "observation.images.wrist_left",
}
_H, _W = 180, 320


def _encode_open(path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = _W
    stream.height = _H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20"}  # high quality; training frames.
    return container, stream


def _encode_frame(container, stream, image: np.ndarray) -> None:
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def _encode_close(container, stream) -> None:
    for packet in stream.encode():  # flush
        container.mux(packet)
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo", required=True, help="Source v2.1 repo id, e.g. SamratSahoo/toys100_sim")
    parser.add_argument("--dst-repo", required=True, help="Destination v3.0 repo id to create/push to")
    parser.add_argument("--out-dir", required=True, help="Local directory to write the v3.0 dataset")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=None, help="For testing: convert only N episodes")
    parser.add_argument("--push", action="store_true", help="Push the result to --dst-repo (needs HF write token)")
    parser.add_argument(
        "--src-revision",
        default=None,
        help="Read the source from this git revision (e.g. a pre-conversion v2.1 commit for recovery).",
    )
    args = parser.parse_args()
    # hf:// reference for the source data files (optionally pinned to a revision).
    src_ref = args.src_repo if not args.src_revision else f"{args.src_repo}@{args.src_revision}"

    out = Path(args.out_dir)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

    api = HfApi()
    all_files = _sd.retry_call(
        lambda: api.list_repo_files(args.src_repo, repo_type="dataset", revision=args.src_revision),
        what="listing source files",
    )
    src_files = sorted(f for f in all_files if f.startswith("data/") and f.endswith(".parquet"))

    # Source totals, used as a completeness guard before pushing (so a truncated/failed conversion never
    # overwrites the original -- important when replacing the source repo in place).
    src_info_path = _sd.retry_call(
        lambda: hf_hub_download(args.src_repo, "meta/info.json", repo_type="dataset", revision=args.src_revision),
        what="downloading info.json",
    )
    src_info = json.loads(Path(src_info_path).read_text())
    # Idempotency: if the source is ALREADY v3.0 (e.g. re-running after a successful in-place replace),
    # there is nothing to do. Exit cleanly instead of trying to read the now-gone v2.1 episode parquets.
    if str(src_info.get("codebase_version", "")).startswith("v3"):
        print(f"{args.src_repo} is already v3.0 (codebase_version={src_info.get('codebase_version')}); nothing to do.")
        return
    src_total_frames = int(src_info.get("total_frames", 0))
    src_total_episodes = int(src_info.get("total_episodes", 0))

    # Source task_index -> task text (v2.1 stores this in meta/tasks.jsonl, not in the data parquet).
    src_tasks: dict[int, str] = {}
    tasks_path = _sd.retry_call(
        lambda: hf_hub_download(args.src_repo, "meta/tasks.jsonl", repo_type="dataset", revision=args.src_revision),
        what="downloading tasks.jsonl",
    )
    for raw_line in Path(tasks_path).read_text().splitlines():
        line = raw_line.strip()
        if line:
            row = json.loads(line)
            src_tasks[int(row["task_index"])] = str(row["task"])
    if args.max_episodes is not None:
        src_files = src_files[: args.max_episodes]
    print(f"Converting {len(src_files)} episodes from {args.src_repo} -> v3.0 at {out}")

    # Open one MP4 per camera (all episodes concatenated).
    video_paths = {dst: out / "videos" / dst / "chunk-000" / "file-000.mp4" for dst in _DST_VIDEO_KEY.values()}
    encoders = {}
    for src_key, dst_key in _DST_VIDEO_KEY.items():
        encoders[src_key] = _encode_open(video_paths[dst_key], args.fps)

    # Accumulators for the data parquet (low-dim) and per-episode meta.
    cols: dict[str, list] = {
        c: []
        for c in [
            "observation.state.joint_position",
            "observation.state.gripper_position",
            "observation.state",
            "action.joint_velocity",
            "action.gripper_position",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    }
    episode_rows = []  # per-episode meta
    global_index = 0
    frame_cursor = 0  # cumulative frames written to each MP4 (same for all cameras)

    for out_ep, rel in enumerate(src_files):
        ds = datasets.load_dataset(
            "parquet", data_files={"train": [f"hf://datasets/{src_ref}/{rel}"]}, split="train", streaming=True
        )
        rows = list(_sd.retry_call(lambda ds=ds: list(ds), what=f"reading {rel}"))
        length = len(rows)
        task_index = int(rows[0]["task_index"]) if rows else 0
        task_text = src_tasks.get(task_index, "")

        video_from_ts = frame_cursor / args.fps
        for local_i, row in enumerate(rows):
            for src_key in _SRC_IMAGE_KEYS:
                img = np.asarray(row[src_key])
                if img.dtype != np.uint8:
                    img = (
                        (255 * img).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img.astype(np.uint8)
                    )
                if img.shape[0] == 3:  # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))
                _encode_frame(*encoders[src_key], img)
            joint = np.asarray(row["joint_position"], dtype=np.float32).reshape(-1)[:7]
            gripper = np.atleast_1d(np.asarray(row["gripper_position"], dtype=np.float32)).reshape(-1)[:1]
            action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
            jv, ga = action[:7], action[7:8]
            cols["observation.state.joint_position"].append(joint.tolist())
            cols["observation.state.gripper_position"].append(gripper.tolist())
            cols["observation.state"].append(np.concatenate([joint, gripper]).tolist())
            cols["action.joint_velocity"].append(jv.tolist())
            cols["action.gripper_position"].append(ga.tolist())
            cols["action"].append(np.concatenate([jv, ga]).tolist())
            cols["timestamp"].append(float(local_i) / args.fps)
            cols["frame_index"].append(local_i)
            cols["episode_index"].append(out_ep)
            cols["index"].append(global_index)
            cols["task_index"].append(task_index)
            global_index += 1
        frame_cursor += length
        video_to_ts = frame_cursor / args.fps

        ep_meta = {
            "episode_index": out_ep,
            "length": length,
            "tasks": [task_text],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index - length,
            "dataset_to_index": global_index,
        }
        for dst_key in _DST_VIDEO_KEY.values():
            ep_meta[f"videos/{dst_key}/chunk_index"] = 0
            ep_meta[f"videos/{dst_key}/file_index"] = 0
            ep_meta[f"videos/{dst_key}/from_timestamp"] = video_from_ts
            ep_meta[f"videos/{dst_key}/to_timestamp"] = video_to_ts
        episode_rows.append(ep_meta)
        if (out_ep + 1) % 20 == 0:
            print(f"  converted {out_ep + 1}/{len(src_files)} episodes ({global_index} frames)")

    for src_key in _SRC_IMAGE_KEYS:
        _encode_close(*encoders[src_key])

    # Write data parquet.
    pq.write_table(pa.table(cols), out / "data" / "chunk-000" / "file-000.parquet")
    # Write meta/episodes.
    pq.write_table(pa.Table.from_pylist(episode_rows), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    # Write meta/tasks.parquet (preserving the source's task_index -> task text mapping).
    ordered = sorted(src_tasks.items())
    pq.write_table(
        pa.table({"task_index": [i for i, _ in ordered], "task": [t for _, t in ordered]}),
        out / "meta" / "tasks.parquet",
    )

    # Write meta/info.json (v3.0, DROID-compatible schema for the fields we emit).
    def _feat(dtype, shape):
        return {"dtype": dtype, "shape": list(shape), "names": None}

    features = {
        **{k: _feat("video", [_H, _W, 3]) for k in _DST_VIDEO_KEY.values()},
        "observation.state.joint_position": _feat("float32", [7]),
        "observation.state.gripper_position": _feat("float32", [1]),
        "observation.state": _feat("float32", [8]),
        "action.joint_velocity": _feat("float32", [7]),
        "action.gripper_position": _feat("float32", [1]),
        "action": _feat("float32", [8]),
        "timestamp": _feat("float32", [1]),
        "frame_index": _feat("int64", [1]),
        "episode_index": _feat("int64", [1]),
        "index": _feat("int64", [1]),
        "task_index": _feat("int64", [1]),
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": len(episode_rows),
        "total_frames": global_index,
        "total_tasks": len(src_tasks),
        "fps": args.fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # Validate every encoded MP4 actually opens and has the expected frame count. A parquet-only
    # completeness check would miss a truncated / non-finalized video (e.g. from an interrupted or
    # racy encode) -- av.open then fails with "Invalid data" at load time. Catch it BEFORE pushing.
    for vpath in video_paths.values():
        try:
            with av.open(str(vpath)) as container:
                n_frames = sum(1 for packet in container.demux(video=0) if packet.size)
        except Exception as exc:
            raise RuntimeError(f"Encoded video {vpath} is unreadable ({exc!r}); aborting before push.") from exc
        if n_frames != global_index:
            raise RuntimeError(
                f"Encoded video {vpath} has {n_frames} frames but expected {global_index}; corrupt/truncated."
            )
    print(f"Validated {len(video_paths)} videos ({global_index} frames each).")
    print(f"Done: {len(episode_rows)} episodes, {global_index} frames written to {out}")

    if args.push:
        # Completeness guard: never overwrite the source with a partial conversion. Skipped for
        # --max-episodes test runs.
        if args.max_episodes is None and (global_index != src_total_frames or len(episode_rows) != src_total_episodes):
            raise RuntimeError(
                f"Refusing to push: converted {len(episode_rows)} eps / {global_index} frames but source "
                f"{args.src_repo} has {src_total_episodes} eps / {src_total_frames} frames. Conversion incomplete."
            )
        replacing = args.dst_repo == args.src_repo
        print(f"Pushing to {args.dst_repo}{' (REPLACING v2.1 in place)' if replacing else ''} ...")
        api.create_repo(args.dst_repo, repo_type="dataset", exist_ok=True)
        # delete_patterns removes the old v2.1 files (data/meta/videos) not present in this upload, in the
        # SAME commit, so the repo atomically ends up as a clean v3.0 dataset (keeps .gitattributes).
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.dst_repo,
            repo_type="dataset",
            delete_patterns=["data/*", "meta/*", "videos/*"],
            commit_message=f"Convert to LeRobot v3.0 (video) from {args.src_repo}",
        )
        print("Pushed.")


if __name__ == "__main__":
    _sd.run_main(main)
