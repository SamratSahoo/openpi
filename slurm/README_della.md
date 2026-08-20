# openpi slurm scripts (Della, A100-80GB)

Della twins of the Neuronic scripts (`README_neuronic.md`). Same two generic, **config-parameterized**
jobs — pass any openpi `TrainConfig` name as the first argument:

```bash
cd /scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi

# 1. Normalization stats (1x A100). Writes assets/<config>/<asset_id>/norm_stats.*
sbatch slurm/compute_norm_stats_della.slurm <config_name>
#    For big streaming configs, subsample with --max-frames, e.g.:
#    sbatch slurm/compute_norm_stats_della.slurm pi05_droid100_lerobot --max-frames 200000

# 2. Full fine-tune (1x A100-80GB, FSDP=1). exp_name defaults to <config_name>.
sbatch slurm/run_training_della.slurm <config_name> [exp_name] [extra train.py args]
```

Chain training behind norm stats the same way:

```bash
NS=$(sbatch --parsable slurm/compute_norm_stats_della.slurm <config>)
sbatch --dependency=afterok:$NS slurm/run_training_della.slurm <config>
```

## Della settings (baked into both scripts)
- **1× A100-80GB** for training (`--gres=gpu:a100:1`, `--constraint=gpu80`, `--fsdp-devices 1`).
  Full FT of π₀.₅ needs ~78 GB, which fits on one 80 GB A100 with no sharding — this mirrors W&B run
  `llo47spj` (`droid100_joint_ft`), a single 80 GB card at global batch 32. `--constraint=gpu80`
  keeps the job off Della's 40 GB A100s, where it would OOM.
- **No `--partition`.** Della's submit plugin auto-routes GPU jobs by gres/constraint/time and
  **rejects** an explicit `--partition=...` ("This is not allowed"). Do not add one.
- **`--account=tsilver`**, `--time=3-00:00:00` (auto-maps to the `gpu-medium` QOS; ≤6 days needs
  `gpu-long`, ≤1 day gets `gpu-short`).
- **Caches on `/scratch/gpfs/TSILVER/ss1824/.cache/{uv,huggingface,openpi}`** — home is only ~50 GB.
- Logs land in `slurm/logs/`. W&B logs online once you've run `wandb login` on the login node
  (or uncomment `export WANDB_MODE=offline` in `run_training_della.slurm`).

The web launcher (`../../slurm/server.py`) submits these automatically when run on Della — see
`../../slurm/README.md`.
