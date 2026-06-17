#!/bin/bash
#SBATCH --job-name=droid-merge-extended
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/merge_extended_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/merge_extended_%j.err
#
# Neuronic-only step (no Della equivalent): builds the COMBINED DROID-100 + toys-no-collision
# dataset that the full-finetune config trains on.
#
# README step 7.3: merge samratsahoo/droid_100_joint (from run_convert_droid_rlds_neuronic.slurm)
# with SamratSahoo/droid-toys-no-collision (pulled from the Hub automatically) ->
# samratsahoo/droid_100_extended_v2, the repo_id used by pi05_droid100_extended_v2_lerobot.
# CPU/IO bound (video re-encode) -> no GPU. Add --push-to-hub to publish.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Merging droid_100_joint + droid-toys-no-collision -> droid_100_extended_v2 on $(hostname) at $(date) ==="
uv run --frozen examples/droid/merge_lerobot_datasets.py \
    --sources samratsahoo/droid_100_joint samratsahoo/droid-toys-no-collision \
    --repo-id samratsahoo/droid_100_extended_v2
echo "=== Done at $(date) ==="
