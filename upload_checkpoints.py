#!/usr/bin/env python
"""Upload the 19999 checkpoints to their corresponding HF repos (serial, resumable)."""
import sys
from huggingface_hub import HfApi

CKPT_ROOT = "/n/fs/tamp-vla/tamp-vla/openpi/checkpoints"

# (local checkpoint dir -> HF repo id)
MAPPING = [
    ("pi05droid-full-d100+toys100sim/d100_toys100sim_from_droid_full_ft/19999", "SamratSahoo/pi05droid_d100_toys100_sim"),
    ("pi05droid-full-d100+toys20sim/d100_toys20sim_from_droid_full_ft/19999",   "SamratSahoo/pi05droid_d100_toys20_sim"),
    ("pi05droid-toys100sim/toys100sim_from_droid_full_ft/19999",                "SamratSahoo/pi05droid_toys100_sim"),
    ("pi05droid-toys20sim/toys20sim_from_droid_full_ft/19999",                  "SamratSahoo/pi05droid_toys20_sim"),
]

api = HfApi()
for rel, repo_id in MAPPING:
    folder = f"{CKPT_ROOT}/{rel}"
    print(f"\n=== Uploading {folder}\n            -> {repo_id} ===", flush=True)
    api.upload_large_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"=== DONE {repo_id} ===", flush=True)

print("\nALL UPLOADS COMPLETE", flush=True)
