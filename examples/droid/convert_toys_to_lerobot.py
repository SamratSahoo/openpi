"""Build a LeRobot dataset from *only* tamp-vla "toys" successful trajectories (no DROID-100 RLDS).

This is `convert_combined_droid_toys_to_lerobot.py` with the RLDS source removed: it writes just the
toys trajectories into one standalone LeRobot v3 dataset using the standard DROID **joint** schema
(8-dim action: 7 joint velocities + gripper), so the result loads directly through
`LeRobotDROIDDataConfig` and norm stats can be recomputed with the same config.

We read the real dense plan (`tiptop_plan.json`), NOT the pre-baked `_lerobot_raw.json` / `le-robot/`
export -- see the combined converter's module docstring for why (the pre-baked export subsampled the
plan to ~20 keyframes and produced joint velocities 15-70x too large with a constant gripper). The
dense-plan parsing, 50 Hz -> 15 Hz resampling, camera decode/resize/align, and frame schema are all
reused verbatim from `convert_combined_droid_toys_to_lerobot._iter_toys_episodes`.

Usage:
  uv run examples/droid/convert_toys_to_lerobot.py \
      --toys-dir /home/samrat/tamp-vla/tamp-vla-data/toys-no-collision-v2/success \
      --repo-id samratsahoo/<name> --push-to-hub
"""

import shutil

import tyro

from convert_combined_droid_toys_to_lerobot import _iter_toys_episodes


def main(
    repo_id: str,
    toys_dir: str,
    *,
    push_to_hub: bool = False,
    private: bool = False,
):
    # Import heavy deps lazily so `--help` works without lerobot installed.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,  # DROID is recorded at 15 fps; toys plans are resampled to match.
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            # Joint *velocity* (7) + gripper position (1); matches the action space pi05-DROID was pretrained on.
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_toys = 0
    for name, task, frames in _iter_toys_episodes(toys_dir):
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        n_toys += 1
        print(f"  [toys {n_toys}] {len(frames)} steps | {name} | task: {task!r}")

    print(f"Done. Wrote {n_toys} trajectory episodes to {output_path}")
    if push_to_hub:
        dataset.push_to_hub(tags=["droid", "panda", "tamp-vla"], private=private, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
