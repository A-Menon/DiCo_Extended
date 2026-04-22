#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/n/fs/prism-data/am9755/logs/launch_adico_navigation_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/launch_adico_navigation_%j.err

SBATCH_SCRIPT=/n/fs/prism-data/am9755/DiCo_Extended/slurm/adico_navigation.sbatch

# Main ADiCo experiments: sweep over SND values and seeds at default alpha=1.0, beta=5.0
for snd in 0.3 0.6 0.8 1 1.2; do
  for seed in 0 1 2 3 4; do
    sbatch --export=ALL,DESIRED_SND=$snd,SEED=$seed,ALPHA=1.0,BETA=5.0 "$SBATCH_SCRIPT"
  done
done

# Alpha ablation at SND=1.0 (near-optimal for navigation)
for alpha in 0 0.5 2.0; do
  for seed in 0 1 2 3 4; do
    sbatch --export=ALL,DESIRED_SND=1,SEED=$seed,ALPHA=$alpha,BETA=5.0 "$SBATCH_SCRIPT"
  done
done

# Beta ablation at SND=1.0, alpha=1.0: beta=0.01 (effectively no learning progress gating)
for seed in 0 1 2 3 4; do
  sbatch --export=ALL,DESIRED_SND=1,SEED=$seed,ALPHA=1.0,BETA=0.01 "$SBATCH_SCRIPT"
done
