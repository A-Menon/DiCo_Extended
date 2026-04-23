#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/n/fs/prism-data/am9755/logs/launch_dndico_navigation_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/launch_dndico_navigation_%j.err

SBATCH_SCRIPT=/n/fs/prism-data/am9755/DiCo_Extended/slurm/dndico_navigation.sbatch

# Main DN-DiCo experiments: sweep over SND values and seeds at default alpha=0.5
for snd in 0.3 0.6 0.8 1 1.2; do
  for seed in 0 1 2 3 4; do
    sbatch --export=ALL,DESIRED_SND=$snd,SEED=$seed,ALPHA=0.5 "$SBATCH_SCRIPT"
  done
done

# Alpha ablation at SND=1.0 (near-optimal for navigation)
for alpha in 0 0.25 1.0 2.0; do
  for seed in 0 1 2 3 4; do
    sbatch --export=ALL,DESIRED_SND=1,SEED=$seed,ALPHA=$alpha "$SBATCH_SCRIPT"
  done
done
