"""Viser replay of tamp-vla and DROID-100 trajectories on an FR3 + Robotiq 2F-85 robot.

Renders the Franka arm (Panda kinematics, identical 7-DOF to the FR3) with a Robotiq 2F-85 gripper.
The data source is chosen *in the viser UI* via a "Source" dropdown, no need to restart:
  * tamp-vla: every trajectory dir found under `--tamp-data-root` (default: the sibling
    tamp-vla-data/) is offered by its relative path, e.g. "toys-no-collision/success". Each rollout is
    replayed from its recorded LeRobot dataset (le-robot/) -- per-frame joint_position plus the
    *continuous* gripper_position and aligned videos -- falling back to the dense `tiptop_plan.json`
    (binary open/close gripper; see `TAMP_VLA_EXPORT_ISSUES.md`) when no le-robot/ dataset is present.
  * DROID-100: offered whenever an RLDS/TFDS build (builder `r2d2_faceblur`) is present at
    `--droid-rlds-dir` (default: ~/.cache/droid_rlds). Its steps already carry per-frame
    joint_position (7-DOF), gripper_position and 180x320 camera images at 15 Hz.
Sources load lazily on selection (so TensorFlow is imported only if you open DROID-100). The sidebar
also shows the two camera streams (exterior + wrist) synced to playback, plus play/scrub controls.

The robot model is assembled on the fly from `robot_descriptions` (Panda arm + Robotiq 2F-85),
downloaded/cached on first run. The gripper signal (0 = open, 1 = closed) drives the Robotiq
`finger_joint`; DROID is the same Panda + Robotiq 2F-85 hardware, so the same model drives both.

Run (deps are pulled in ad-hoc so they don't touch the project env). With no args it auto-discovers
sources under the sibling tamp-vla-data/ and DROID-100 at ~/.cache/droid_rlds:
  uv run --with viser --with yourdfpy --with robot_descriptions --group rlds \
      examples/droid/visualize_trajectories.py

  # point at a different data root / RLDS dir if needed:
  uv run ... examples/droid/visualize_trajectories.py \
      --tamp-data-root /path/to/tamp-vla-data --droid-rlds-dir /path/to/droid_rlds

Then open the printed http://localhost:8080 URL and pick a Source.
"""

import importlib.util
import pathlib
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import tyro

# Reuse the (already validated) plan->15Hz logic from the converter so replay matches the dataset.
_CONV = pathlib.Path(__file__).with_name("convert_combined_droid_toys_to_lerobot.py")
_spec = importlib.util.spec_from_file_location("_droid_conv", _CONV)
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)

# Robotiq 2F-85 driver joint range: 0 rad = open, ~0.725 rad = fully closed.
_ROBOTIQ_CLOSED_RAD = 0.725
# Fixed mount of the gripper on the Panda flange (panda_link8): same -45deg clocking as the Franka hand,
# small coupler offset along the approach (+z) axis.
_MOUNT_XYZ = "0 0 0.011"
_MOUNT_RPY = "0 0 -0.7853981633974483"


def _build_fr3_robotiq_urdf():
    """Assemble a Panda(FR3) arm + Robotiq 2F-85 gripper into one yourdfpy.URDF."""
    import yourdfpy
    from robot_descriptions import panda_description as pd
    from robot_descriptions import robotiq_2f85_description as rq

    cache_root = pathlib.Path(pd.REPOSITORY_PATH).parent  # ~/.cache/robot_descriptions

    def resolve(fname: str) -> str:
        # package://<pkg>/<rest>  ->  <cache_root>/<pkg>/<rest>
        if fname.startswith("package://"):
            return str(cache_root / fname[len("package://") :])
        return fname

    panda = ET.parse(pd.URDF_PATH).getroot()
    # Drop the Franka hand so we can mount the Robotiq gripper instead.
    drop_links = {"panda_hand", "panda_hand_tcp", "panda_leftfinger", "panda_rightfinger"}
    drop_joints = {"panda_hand_joint", "panda_hand_tcp_joint", "panda_finger_joint1", "panda_finger_joint2"}
    for el in list(panda):
        if el.tag == "link" and el.get("name") in drop_links:
            panda.remove(el)
        elif el.tag == "joint" and el.get("name") in drop_joints:
            panda.remove(el)

    # Append the Robotiq links/joints (and any materials it defines).
    existing_materials = {el.get("name") for el in panda.findall("material")}
    robotiq = ET.parse(rq.URDF_PATH).getroot()
    for el in list(robotiq):
        if el.tag in ("link", "joint"):
            panda.append(el)
        elif el.tag == "material" and el.get("name") not in existing_materials:
            panda.append(el)

    # Fixed joint: Panda flange -> Robotiq base.
    mount = ET.SubElement(panda, "joint", {"name": "panda_to_robotiq", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "panda_link8"})
    ET.SubElement(mount, "child", {"link": "robotiq_85_base_link"})
    ET.SubElement(mount, "origin", {"xyz": _MOUNT_XYZ, "rpy": _MOUNT_RPY})
    panda.set("name", "fr3_robotiq")

    import io

    urdf_bytes = ET.tostring(panda, encoding="utf-8")
    return yourdfpy.URDF.load(
        io.BytesIO(urdf_bytes), load_meshes=True, build_scene_graph=True, filename_handler=lambda fname: resolve(fname)
    )


def _load_trajectory(traj: pathlib.Path, img_hw=(180, 320)):
    """Return (positions[N,7], gripper[N], exterior[N,H,W,3], wrist[N,H,W,3], task) at 15 Hz.

    Prefers the recorded LeRobot dataset under le-robot/, which carries the *continuous* per-frame
    gripper signal (and aligned videos); falls back to the dense tiptop_plan.json (whose gripper is
    only a binary open/close event stream) when no le-robot/ dataset is present.
    """
    import json

    if list((traj / "le-robot" / "data").glob("chunk-*/file-*.parquet")):
        return _load_lerobot_trajectory(traj, img_hw)

    plan = json.loads((traj / "tiptop_plan.json").read_text())
    task = json.loads((traj / "_lerobot_raw.json").read_text()).get("instruction", "do something")
    pos50, _vel50, grip50 = _conv._dense_plan(plan)
    n = int(round(len(pos50) * _conv._PLAN_DT * _conv._DROID_FPS))
    sel = _conv._resample_indices(len(pos50), n)
    pos, grip = pos50[sel], grip50[sel]

    exterior = _conv._decode_resized(str(traj / "external_cam.mp4"), img_hw)
    wrist = _conv._decode_resized(str(traj / "hand_cam.mp4"), img_hw)
    ext = np.asarray(exterior)[_conv._resample_indices(len(exterior), n)]
    wr = np.asarray(wrist)[_conv._resample_indices(len(wrist), n)]
    return pos, grip, ext, wr, task


def _load_lerobot_trajectory(traj: pathlib.Path, img_hw=(180, 320)):
    """Load a trajectory from its LeRobot v3 dataset under le-robot/.

    Uses the recorded per-frame observation.joint_position (7) and the *continuous*
    observation.gripper_position (0 = open .. ~1 = closed, same convention as DROID), plus the aligned
    le-robot videos (exterior_1_left + wrist_left). Everything is already at 15 Hz, so -- unlike the
    tiptop_plan.json path -- there is no resample step and the gripper is the real continuous signal.
    """
    import json

    import pyarrow.parquet as pq

    lr = traj / "le-robot"
    data_file = sorted((lr / "data").glob("chunk-*/file-*.parquet"))[0]
    cols = pq.read_table(
        data_file, columns=["observation.joint_position", "observation.gripper_position"]
    ).to_pydict()
    pos = np.asarray(cols["observation.joint_position"], dtype=np.float64)  # (N, 7)
    grip = np.asarray(cols["observation.gripper_position"], dtype=np.float64).reshape(-1)  # (N,)
    n = len(pos)

    def _video(cam: str) -> np.ndarray:
        mp4 = sorted((lr / "videos" / f"observation.images.{cam}").glob("chunk-*/file-*.mp4"))[0]
        frames = np.asarray(_conv._decode_resized(str(mp4), img_hw))
        return frames[_conv._resample_indices(len(frames), n)]  # align to the parquet frame count

    ext = _video("exterior_1_left")
    wr = _video("wrist_left")
    task = json.loads((traj / "_lerobot_raw.json").read_text()).get("instruction", "do something")
    return pos, grip, ext, wr, task


def _droid_instruction(step: dict) -> str:
    """First non-empty language instruction in a DROID step (bytes decoded)."""
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        val = step.get(key, b"")
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if val and val.strip():
            return val.strip()
    return "do something"


def _load_droid_episode(ds, idx: int):
    """Return (positions[N,7], gripper[N], exterior[N,H,W,3], wrist[N,H,W,3], task) for DROID episode idx.

    DROID RLDS steps are already at 15 Hz with per-frame joint_position (the 7-DOF Panda arm),
    gripper_position (0 = open .. 1 = closed) and decoded 180x320 RGB images, so unlike tamp-vla there
    is no plan->resample step. `skip(idx)` re-reads from the start (O(idx)) but is fine interactively.
    """
    episode = next(iter(ds.skip(idx).take(1)))  # deterministic order (no shuffle)
    steps = list(episode["steps"].as_numpy_iterator())
    pos = np.asarray([s["observation"]["joint_position"] for s in steps], dtype=np.float64)
    grip = np.asarray([s["observation"]["gripper_position"] for s in steps], dtype=np.float64).reshape(-1)
    ext = np.asarray([s["observation"]["exterior_image_1_left"] for s in steps])
    wr = np.asarray([s["observation"]["wrist_image_left"] for s in steps])
    return pos, grip, ext, wr, _droid_instruction(steps[0])


def _tamp_loaders(data_dir: pathlib.Path) -> dict:
    """{traj name -> loader()} for every trajectory dir (with tiptop_plan.json) directly under data_dir."""
    trajs = sorted(p for p in data_dir.iterdir() if (p / "tiptop_plan.json").exists())
    return {p.name: (lambda t=p: _load_trajectory(t)) for p in trajs}


def _find_tamp_dirs(root: pathlib.Path) -> list:
    """Dirs at/under root that directly contain >=1 trajectory dir (a child with tiptop_plan.json).

    For the usual <experiment>/<split>/<timestamp>/ layout this surfaces the eval/failure/success dirs.
    """
    if not root.exists():
        return []
    out = []
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        try:
            if any((c / "tiptop_plan.json").exists() for c in d.iterdir() if c.is_dir()):
                out.append(d)
        except OSError:
            pass
    return sorted(out)


def _droid_loaders(rlds_dir: str, builder_name: str, version: str, max_episodes: int | None) -> dict:
    """{droid_NNN -> loader()} for a DROID-100 RLDS build. Imports TensorFlow lazily (only on use)."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # keep TF off the GPU so it doesn't clobber others
    import tensorflow_datasets as tfds

    builder = tfds.builder(builder_name, data_dir=rlds_dir, version=version)
    n_total = builder.info.splits["train"].num_examples
    n = n_total if max_episodes is None else min(max_episodes, n_total)
    ds = builder.as_dataset(split="train")
    return {f"droid_{i:03d}": (lambda i=i: _load_droid_episode(ds, i)) for i in range(n)}


def main(
    tamp_data_root: str | None = None,
    data_dir: str | None = None,
    droid_rlds_dir: str | None = None,
    builder_name: str = "r2d2_faceblur",
    version: str = "1.0.0",
    max_episodes: int | None = None,
    port: int = 8080,
):
    import viser
    from viser.extras import ViserUrdf

    # Default tamp-vla data root: the sibling tamp-vla-data/ next to this checkout.
    if tamp_data_root is None:
        tamp_data_root = str(pathlib.Path(__file__).resolve().parents[3].parent / "tamp-vla-data")
    root = pathlib.Path(tamp_data_root)

    # Each "source" is a lazy provider: label -> () -> {traj name: loader()}. Providers run (and
    # their result is cached) only when the source is picked in the UI, so e.g. TensorFlow is
    # imported only if you actually open DROID-100. Every loader() returns the same
    # (pos[N,7], grip[N], ext[N,H,W,3], wr[N,H,W,3], task) tuple consumed below.
    sources = {}
    for d in _find_tamp_dirs(root):
        sources[str(d.relative_to(root))] = lambda d=d: _tamp_loaders(d)
    # An explicit --data-dir is added as its own source (handy for a dir outside the root).
    if data_dir:
        dd = pathlib.Path(data_dir)
        sources[f"(--data-dir) {dd.name}"] = lambda dd=dd: _tamp_loaders(dd)
    # DROID-100 RLDS: offered whenever the dataset is present (default: ~/.cache/droid_rlds).
    droid_dir = droid_rlds_dir or str(pathlib.Path.home() / ".cache" / "droid_rlds")
    if (pathlib.Path(droid_dir) / builder_name).exists():
        sources["DROID-100"] = lambda: _droid_loaders(droid_dir, builder_name, version, max_episodes)

    if not sources:
        raise SystemExit(
            f"No trajectory sources found: no tamp-vla dirs under {root}, no DROID-100 at {droid_dir}."
        )
    source_names = list(sources)
    print("Available sources:")
    for s in source_names:
        print(f"  - {s}")

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

    print("Building FR3 + Robotiq robot model (first run downloads meshes)...")
    urdf = _build_fr3_robotiq_urdf()
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    joint_names = viser_urdf.get_actuated_joint_names()  # panda_joint1..7 + finger_joint
    arm_idx = [joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    grip_idx = joint_names.index("finger_joint")

    # ---- GUI ----
    server.gui.add_markdown("## Trajectory replay\nFR3 + Robotiq 2F-85")
    source_dd = server.gui.add_dropdown("Source", source_names, initial_value=source_names[0])
    traj_dd = server.gui.add_dropdown("Trajectory", ["(loading...)"], initial_value="(loading...)")
    task_text = server.gui.add_text("Task", initial_value="", disabled=True)
    with server.gui.add_folder("Playback"):
        play_cb = server.gui.add_checkbox("Play", initial_value=True)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)
        speed_sl = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        frame_sl = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        grip_text = server.gui.add_text("Gripper", initial_value="", disabled=True)
    with server.gui.add_folder("Cameras"):
        ext_img = server.gui.add_image(np.zeros((180, 320, 3), np.uint8), label="exterior", format="jpeg")
        wr_img = server.gui.add_image(np.zeros((180, 320, 3), np.uint8), label="wrist", format="jpeg")

    state = {"data": None, "n": 0, "loaders": {}}
    source_cache = {}  # label -> {traj name: loader()}, built on first selection
    suppress = {"traj": False}  # ignore traj_dd updates while we repopulate it programmatically
    lock = threading.Lock()

    def set_frame(i: int):
        data = state["data"]
        if data is None:
            return
        pos, grip, ext, wr, _ = data
        i = int(np.clip(i, 0, len(pos) - 1))
        cfg = np.zeros(len(joint_names))
        for k, j in enumerate(arm_idx):
            cfg[j] = pos[i][k]
        cfg[grip_idx] = float(grip[i]) * _ROBOTIQ_CLOSED_RAD
        viser_urdf.update_cfg(cfg)
        ext_img.image = ext[i]
        wr_img.image = wr[i]
        grip_text.value = "closed" if grip[i] > 0.5 else "open"

    def load(name: str):
        loaders = state["loaders"]
        if name not in loaders:
            return
        task_text.value = "loading..."
        with lock:
            data = loaders[name]()
            state["data"] = data
            state["n"] = len(data[0])
        frame_sl.max = max(1, state["n"] - 1)
        frame_sl.value = 0
        task_text.value = f"{data[4]}  ({state['n']} frames @15Hz)"
        set_frame(0)

    def select_source(label: str):
        task_text.value = f"loading source '{label}'..."
        if label not in source_cache:
            source_cache[label] = sources[label]()  # may import TF / build the dataset on first use
        loaders = source_cache[label]
        state["loaders"] = loaders
        names = list(loaders) or ["(empty)"]
        # Repopulate the trajectory dropdown without letting its on_update fire mid-swap.
        suppress["traj"] = True
        traj_dd.options = names
        traj_dd.value = names[0]
        suppress["traj"] = False
        load(names[0])

    @source_dd.on_update
    def _(_=None):
        select_source(source_dd.value)

    @traj_dd.on_update
    def _(_=None):
        if suppress["traj"]:
            return
        load(traj_dd.value)

    @frame_sl.on_update
    def _(_=None):
        # Only react to manual scrubs (playback updates .value too, but re-rendering is cheap/idempotent).
        set_frame(frame_sl.value)

    select_source(source_names[0])
    print(f"\n  Open  http://localhost:{port}  in your browser.\n")

    # Playback loop.
    while True:
        t0 = time.time()
        if play_cb.value and state["n"] > 1:
            nxt = frame_sl.value + 1
            if nxt > state["n"] - 1:
                nxt = 0 if loop_cb.value else state["n"] - 1
            frame_sl.value = nxt  # triggers on_update -> set_frame
        # 15 Hz nominal, scaled by speed.
        dt = 1.0 / (_conv._DROID_FPS * speed_sl.value)
        time.sleep(max(0.0, dt - (time.time() - t0)))


if __name__ == "__main__":
    tyro.cli(main)
