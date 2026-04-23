#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/n/fs/prism-data/am9755/logs/launch_cadico_anneal_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/launch_cadico_anneal_%j.err

SBATCH_SCRIPT=/n/fs/prism-data/am9755/DiCo_Extended/slurm/cadico_anneal_navigation.sbatch

for snd in 0.3 0.6 0.8 1 1.2; do
  for seed in 0 1 2 3 4; do
    sbatch --export=ALL,DESIRED_SND=$snd,SEED=$seed "$SBATCH_SCRIPT"
  done
done
