#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/n/fs/prism-data/am9755/logs/launch_all_navigation_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/launch_all_navigation_%j.err

# Launch all navigation experiments: DiCo baselines, ADiCo, and DN-DiCo
# This reproduces all figures in the paper.

SLURM_DIR=/n/fs/prism-data/am9755/DiCo_Extended/slurm

echo "=== Launching DiCo baselines ==="
sbatch "$SLURM_DIR/launch_navigation.sbatch"

echo "=== Launching ADiCo experiments ==="
sbatch "$SLURM_DIR/launch_adico_navigation.sbatch"

echo "=== Launching DN-DiCo experiments ==="
sbatch "$SLURM_DIR/launch_dndico_navigation.sbatch"

echo "=== All jobs submitted ==="
