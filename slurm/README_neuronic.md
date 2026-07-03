# openpi slurm scripts (Neuronic, L40)

Two generic, **config-parameterized** jobs — pass any openpi `TrainConfig` name as the first argument:

```bash
cd /n/fs/tamp-vla/tamp-vla/openpi

# 1. Normalization stats (1x L40). Writes assets/<config>/<asset_id>/norm_stats.*
sbatch slurm/compute_norm_stats.slurm <config_name>
#    For big streaming configs, subsample with --max-frames, e.g.:
#    sbatch slurm/compute_norm_stats.slurm pi05base-droid+toys100sim-stream --max-frames 200000

# 2. Full fine-tune (whole 8x L40 node, FSDP=8). exp_name defaults to <config_name>.
sbatch slurm/run_training.slurm <config_name> [exp_name] [extra train.py args]
#    e.g. sbatch slurm/run_training.slurm pi05base-full-d100+toys100sim-stream
#         sbatch slurm/run_training.slurm pi05base-toys20sim my_run --batch-size 16
```

Run norm stats **before** training for the same config. To make training wait for it:

```bash
NS=$(sbatch --parsable slurm/compute_norm_stats.slurm <config>)
sbatch --dependency=afterok:$NS slurm/run_training.slurm <config>
```

## Neuronic settings (baked into both scripts)
- **L40 GPUs**: `--gres=gpu:l40:1` (norm stats) / `gpu:l40:8` (training); `--account=seas`, `--partition=all`.
- **FSDP=8 for training**: full FT of π₀.₅ needs >70 GB, but an L40 is only ~46 GB — so the whole 8×L40
  node is taken and the model + AdamW state are sharded across all 8 GPUs (~9 GB/GPU) via
  `--fsdp-devices 8`. Global batch stays 32 (4/device). If only 4 GPUs are free, edit `--gres` to
  `gpu:l40:4` and pass `--fsdp-devices 4` (on OOM also `--batch-size 16`).
- **Caches on the 1 TB project FS** (`/n/fs/tamp-vla/.cache/{uv,huggingface,openpi}`) — home is ~16 GB.
- `--exclude=neu306` (a node that reproducibly faults fsdp=8; drop once fixed).
- Logs land in `slurm/logs/`. W&B logs online once you've run `wandb login` on the login node
  (or uncomment `export WANDB_MODE=offline` in `run_training.slurm`).

## Other utilities (kept, not config-parameterized)
- `run_convert_droid_rlds_neuronic.slurm` — DROID RLDS → LeRobot (builds `SamratSahoo/d100`).
- `run_merge_datasets_v3_neuronic.slurm` — merge datasets.
- `upload_ckpt.slurm` / `upload_ckpt.py` / `upload_checkpoints_to_drive.sh` — checkpoint upload helpers.
