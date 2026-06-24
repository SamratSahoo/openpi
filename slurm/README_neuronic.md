# Neuronic slurm scripts (openpi full fine-tune of π₀.₅)

These `*_neuronic.slurm` scripts are the **Neuronic-cluster equivalents** of the Della
(`run_*.slurm`) scripts, set up to **full fine-tune π₀.₅** on the combined
**DROID-100 + toys-no-collision** dataset (`pi05droid-full-d100+toys`).

## What changed vs the Della scripts
- **Caches on the project FS.** Home (`/u/<user>`) has only ~16 GB, so all caches point at the
  1 TB `/n/fs/tamp-vla/.cache/{huggingface,openpi,uv}` (also exported from `~/.bashrc`).
- **GPUs are L40, not A100.** `--gres=gpu:l40:N`, no `--constraint=a100`. `--account=seas`,
  `--partition=all`.
- **No `module load proxy/default`** — Neuronic login/compute nodes have direct internet.
- **Full FT is sharded.** An L40 is ~46 GB but full FT needs >70 GB, so the train job takes the
  whole 8×L40 node and shards the model+optimizer with **FSDP across all 8 GPUs**
  (`--fsdp-devices 8`). Global batch stays 32 (4/device). If only 4 GPUs are free, use
  `--gres=gpu:l40:4 --fsdp-devices 4` (tighter but usually fits); on OOM also add `--batch-size 16`.

## Run order
```bash
cd /n/fs/tamp-vla/tamp-vla/openpi
# 1. DROID-100 raw RLDS -> LeRobot (downloads + converts -> SamratSahoo/d100)
sbatch slurm/run_convert_droid_rlds_neuronic.slurm
# 2. merge with toys-no-collision -> SamratSahoo/d100_toys20
sbatch slurm/run_merge_datasets_v3_neuronic.slurm
# 3. norm stats for the extended config
sbatch slurm/run_compute_norm_stats_v3_neuronic.slurm
# 4. full fine-tune (8×L40, FSDP=8)
sbatch slurm/run_train_v3_neuronic.slurm
# (optional) quick multi-GPU sanity check of the train loop
sbatch slurm/run_train_test_neuronic.slurm
```
Chain them so each waits for the previous: `sbatch --dependency=afterok:<JOBID> ...`.

Logs land in `slurm/logs/`. W&B logs online once you run `wandb login` on the login node;
otherwise uncomment `export WANDB_MODE=offline` in `run_train_v3_neuronic.slurm`.
