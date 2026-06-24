#!/bin/bash
#SBATCH --job-name=droid-merge-extended-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/merge_extended_v3_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/merge_extended_v3_%j.err
#
# Neuronic-only step (no Della equivalent): builds the COMBINED DROID-100 + toys-no-collision
# dataset that the full-finetune config trains on (v3 build).
#
# README step 7.3: merge SamratSahoo/d100 (from run_convert_droid_rlds_neuronic.slurm)
# with SamratSahoo/toys20 (the corrected toys-no-collision-v2 build, pulled from
# the Hub) -> SamratSahoo/d100_toys20, the repo_id used by pi05droid-full-d100+toys.
# CPU/IO bound (video re-encode) -> no GPU. Add --push-to-hub to publish.
#
# NOTE: re-pull the latest toys version BEFORE this job (delete the stale cache so LeRobotDataset
# re-downloads from the Hub) -- see README step 7.2. The merge reads whatever is in $HF_LEROBOT_HOME.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Merging droid_100_joint + droid-toys-no-collision -> droid_100_extended_v3 on $(hostname) at $(date) ==="
uv run --frozen examples/droid/merge_lerobot_datasets.py \
    --sources SamratSahoo/d100 SamratSahoo/toys20 \
    --repo-id SamratSahoo/d100_toys20
echo "=== Done at $(date) ==="
