"""Build the toys LeRobot dataset from the v2 pre-baked `le-robot/` exports (NOT `tiptop_plan.json`).

Supersedes `convert_toys_to_lerobot.py` for the **v2** toys data. The v2 trajectory folders ship a
correct pre-baked `le-robot/` export per episode, so we no longer need to re-derive anything from the
dense plan. Compared to the plan-derived build, the v2 export fixes three things:

  * gripper is the *measured continuous* position (0..1), not a binary 0/1 reconstruction -- and this
    matches DROID's own continuous gripper, so it normalizes consistently with DROID-100;
  * `exterior_2_left` is a *real* second exterior camera (the toys rig has two), not a duplicate of
    `exterior_1_left` as the old plan-based converter assumed;
  * frames are dense at the recorded 15 Hz timeline (no 50->15 Hz plan resample).

We read each export's data parquet (joint_position[7], gripper_position[1], action[8]) plus its three
av1 videos (exterior_1_left / exterior_2_left / wrist_left), which are 1:1 aligned with the rows, and
re-emit them in the standard DROID **joint** schema so the result loads directly through
`LeRobotDROIDDataConfig` and merges with `samratsahoo/droid_100_joint` via merge_lerobot_datasets.py.
(observation.cartesian_position is present in the export but dropped here -- pi05-DROID does not use it
and droid_100_joint does not carry it.)

Usage:
  uv run examples/droid/convert_toys_lerobot_to_droid.py \
      --toys-dir /home/samrat/tamp-vla/tamp-vla-data/toys-no-collision-v2/success \
      --repo-id samratsahoo/droid-toys-no-collision --push-to-hub
"""

import glob
import json
import pathlib
import shutil

import numpy as np
import tyro

_FEATURES = {
    "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
    "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
    # Joint *velocity* (7) + gripper position (1); matches the action space pi05-DROID was pretrained on.
    "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
}


def _decode(path: str) -> list[np.ndarray]:
    """Decode an mp4 into an ordered list of HWC uint8 RGB frames (already 180x320 in the export)."""
    import av

    container = av.open(path)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return frames


def _one(cam_dir: pathlib.Path, cam: str) -> str:
    hits = glob.glob(str(cam_dir / f"observation.images.{cam}" / "**" / "*.mp4"), recursive=True)
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {cam} video, got {hits}")
    return hits[0]


def _task(traj: pathlib.Path) -> str:
    meta = traj / "metadata.json"
    if meta.exists():
        t = json.loads(meta.read_text()).get("task_instruction")
        if t:
            return t
    return json.loads((traj / "_lerobot_raw.json").read_text()).get("instruction", "do something")


def main(
    repo_id: str,
    toys_dir: str,
    *,
    push_to_hub: bool = False,
    private: bool = False,
):
    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    out = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,
        features=_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_eps = 0
    for traj in sorted(pathlib.Path(toys_dir).iterdir()):
        base = traj / "le-robot"
        if not (base / "meta" / "info.json").exists():
            continue
        df = pd.concat(
            [pd.read_parquet(p) for p in sorted(glob.glob(str(base / "data" / "**" / "*.parquet"), recursive=True))],
            ignore_index=True,
        )
        jp = np.stack(df["observation.joint_position"].values).astype(np.float32)   # (N,7)
        gp = np.stack(df["observation.gripper_position"].values).astype(np.float32)  # (N,1)
        if gp.ndim == 1:
            gp = gp[:, None]
        act = np.stack(df["action"].values).astype(np.float32)                       # (N,8): jvel[7]+gripper[1]
        ext1 = _decode(_one(base / "videos", "exterior_1_left"))
        ext2 = _decode(_one(base / "videos", "exterior_2_left"))
        wr = _decode(_one(base / "videos", "wrist_left"))
        n = len(df)
        if not (len(ext1) == len(ext2) == len(wr) == n):
            raise ValueError(f"{traj.name}: frame/row mismatch {len(ext1)},{len(ext2)},{len(wr)} vs {n}")

        task = _task(traj)
        for i in range(n):
            out.add_frame(
                {
                    "exterior_image_1_left": ext1[i],
                    "exterior_image_2_left": ext2[i],
                    "wrist_image_left": wr[i],
                    "joint_position": jp[i],
                    "gripper_position": gp[i],
                    "actions": act[i],
                    "task": task,
                }
            )
        out.save_episode()
        n_eps += 1
        print(f"  [toys {n_eps}] {n} frames | {traj.name} | task: {task!r}")

    print(f"Done. Wrote {n_eps} episodes to {output_path}")
    if push_to_hub:
        out.push_to_hub(tags=["droid", "panda", "tamp-vla"], private=private, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
