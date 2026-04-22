#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/n/fs/prism-data/am9755/logs/dico_dispersion_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/dico_dispersion_%j.err

set -euo pipefail

ROOT=/n/fs/prism-data/am9755
REPO=$ROOT/DiCo_Extended
RUNS=$REPO/het_control/run_scripts
PYTHON=$ROOT/envs/dico/bin/python

: "${DESIRED_SND:?DESIRED_SND not set}"
: "${SEED:?SEED not set}"

if [[ "$DESIRED_SND" == "-1" ]]; then
    LABEL="unconstrained"
elif [[ "$DESIRED_SND" == "0" ]]; then
    LABEL="homogeneous"
else
    LABEL="snd_${DESIRED_SND//./p}"
fi

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export WANDB_DIR=$ROOT/cache/wandb
export TMPDIR=$ROOT/tmp
export TEMP=$ROOT/tmp
export TMP=$ROOT/tmp
export XDG_CACHE_HOME=$ROOT/cache
export PIP_CACHE_DIR=$ROOT/cache/pip
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions
mkdir -p "$WANDB_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$ROOT/logs"

SP=$ROOT/envs/dico/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cusparselt/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/nccl/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

OUTDIR=$ROOT/logs/dico_baselines/dispersion/${LABEL}/seed_${SEED}
mkdir -p "$OUTDIR"

RESTORE_ARG=""
if [[ -f "$OUTDIR/latest.pt" ]]; then
    RESTORE_ARG="experiment.restore_file=$OUTDIR/latest.pt"
    echo "Resuming from checkpoint: $OUTDIR/latest.pt"
fi


cd "$RUNS"
"$PYTHON" run_dispersion_maddpg.py \
  experiment.sampling_device=cuda \
  experiment.train_device=cuda \
  experiment.loggers='[csv]' \
  experiment.render=False \
  experiment.evaluation=True \
  experiment.checkpoint_interval=600000 \
  experiment.max_n_frames=10000000 \
  model.desired_snd=$DESIRED_SND \
  seed=$SEED \
  $RESTORE_ARG \
  hydra.run.dir="$OUTDIR"
