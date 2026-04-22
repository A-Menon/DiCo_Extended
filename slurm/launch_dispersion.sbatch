#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/n/fs/prism-data/am9755/logs/launch_dispersion_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/launch_dispersion_%j.err

for snd in -1 0 1 1.5 2 3 6; do
  for seed in 0 1 2 3; do
    sbatch --export=ALL,DESIRED_SND=$snd,SEED=$seed /n/fs/prism-data/am9755/DiCo_Extended/slurm/dico_dispersion.sbatch
  done
done
