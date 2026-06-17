"""Merge two or more *already-converted* LeRobot DROID datasets into one.

Use this when both inputs are already in LeRobot format (local under `$HF_LEROBOT_HOME` or pullable
from the Hub) and you just want to concatenate them, e.g.

    samratsahoo/droid_100_joint  (DROID-100, 100 eps)
  + samratsahoo/droid-toys-no-collision  (tamp-vla toys, 20 eps)
  = samratsahoo/droid_100_extended  (120 eps)

This is different from `convert_combined_droid_toys_to_lerobot.py`, which *builds* a combined dataset
from the raw sources (DROID RLDS + raw toys trajectory folders). That script is the right tool when you
have the raw data; THIS script is the right tool when you already have two finished LeRobot datasets.

All sources must share the standard DROID **joint** schema (8-dim action = 7 joint velocities + gripper)
produced by the other converters in this directory, so the merged dataset loads directly through
`LeRobotDROIDDataConfig` (e.g. `pi05_droid100_extended_v2_lerobot`) and norm stats can be recomputed
with the same config. Episodes are copied verbatim (per-episode 15 Hz timeline rebuilt by `add_frame`);
each source's language instructions are carried over.

Usage:
  uv run examples/droid/merge_lerobot_datasets.py \
      --sources samratsahoo/droid_100_joint samratsahoo/droid-toys-no-collision \
      --repo-id samratsahoo/droid_100_extended --push-to-hub
"""

import shutil

import numpy as np
import tyro

# The DROID joint schema shared by every converter in this directory.
_FEATURES = {
    "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
    "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
    # Joint *velocity* (7) + gripper position (1); matches the action space pi05-DROID was pretrained on.
    "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
}
_IMG_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")


def _to_hwc_uint8(img) -> np.ndarray:
    """A LeRobot image frame (CHW float[0,1] tensor, or HWC array) -> HWC uint8, as add_frame wants."""
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return arr


def main(
    sources: list[str],
    repo_id: str,
    *,
    max_episodes_per_source: int | None = None,
    push_to_hub: bool = False,
    private: bool = False,
):
    if len(sources) < 2:
        raise ValueError("Pass at least two --sources to merge.")

    # Import heavy deps lazily so `--help` works without lerobot installed.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    out = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,  # DROID rate; every source converter writes 15 fps.
        features=_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_eps = 0
    for src_id in sources:
        src = LeRobotDataset(src_id)  # local cache if present, else downloaded from the Hub
        missing = [k for k in _FEATURES if k not in src.features]
        if missing:
            raise ValueError(
                f"Source '{src_id}' is missing {missing}; it is not a DROID-joint LeRobot dataset "
                "(expected the schema produced by the converters in examples/droid/)."
            )
        n_ep = src.num_episodes if max_episodes_per_source is None else min(max_episodes_per_source, src.num_episodes)
        print(f"Merging {n_ep}/{src.num_episodes} episodes from {src_id}")

        for ep in range(n_ep):
            start = int(src.episode_data_index["from"][ep])
            end = int(src.episode_data_index["to"][ep])
            task = "do something"
            for i in range(start, end):
                f = src[i]
                task = f["task"]
                frame = {
                    "joint_position": np.asarray(f["joint_position"], dtype=np.float32).reshape(7),
                    "gripper_position": np.asarray(f["gripper_position"], dtype=np.float32).reshape(1),
                    "actions": np.asarray(f["actions"], dtype=np.float32).reshape(8),
                    "task": task,
                }
                for k in _IMG_KEYS:
                    frame[k] = _to_hwc_uint8(f[k])
                out.add_frame(frame)
            out.save_episode()
            total_eps += 1
            print(f"  [{src_id} ep {ep + 1}/{n_ep}] {end - start} frames | task: {task!r}")

    print(f"Done. Wrote {total_eps} episodes from {len(sources)} sources to {output_path}")
    if push_to_hub:
        out.push_to_hub(
            tags=["droid", "panda", "tamp-vla", "merged"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
