"""Upload a trained step-19999 openpi checkpoint dir to a HuggingFace model repo (hf_transfer)."""
import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from pathlib import Path

from huggingface_hub import HfApi

ckpt_dir = Path(sys.argv[1])
repo = sys.argv[2]
if not ckpt_dir.is_dir():
    sys.exit(f"checkpoint dir missing: {ckpt_dir} (training may not have saved step 19999)")
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
print(f"=== uploading {ckpt_dir} -> {repo} ===", flush=True)
# delete_patterns scoped to the checkpoint dirs: removes any STALE checkpoint files from a prior run
# (these repos may hold old checkpoints trained on the pre-fix binary-gripper data) while preserving
# a hand-written README.md / model card at the repo root.
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ckpt_dir),
                  commit_message="Add step-19999 full-finetune checkpoint (corrected data)",
                  delete_patterns=["params/**", "train_state/**", "assets/**", "_CHECKPOINT_METADATA", "_CHECKPOINT_METADATA/**"])
print(f"=== UPLOADED {ckpt_dir} -> https://huggingface.co/{repo} ===", flush=True)
