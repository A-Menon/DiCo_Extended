#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/n/fs/prism-data/am9755/logs/cadico_tag_%j.log
#SBATCH --error=/n/fs/prism-data/am9755/logs/cadico_tag_%j.err
set -euo pipefail
ROOT=/n/fs/prism-data/am9755; REPO=$ROOT/DiCo_Extended; PYTHON=$ROOT/envs/dico/bin/python
: "${DESIRED_SND:?}"; : "${SEED:?}"
if [[ "$DESIRED_SND" == "-1" ]]; then LABEL="unconstrained"; elif [[ "$DESIRED_SND" == "0" ]]; then LABEL="homogeneous"; else LABEL="snd_${DESIRED_SND//./p}"; fi
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1 WANDB_DIR=$ROOT/cache/wandb TMPDIR=$ROOT/tmp TEMP=$ROOT/tmp TMP=$ROOT/tmp XDG_CACHE_HOME=$ROOT/cache PIP_CACHE_DIR=$ROOT/cache/pip TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions
mkdir -p "$WANDB_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$ROOT/logs"
SP=$ROOT/envs/dico/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cusparselt/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/nccl/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
OUTDIR=$ROOT/logs/cadico_baselines/tag/${LABEL}/alpha_0p3/seed_${SEED}; mkdir -p "$OUTDIR"
RESTORE_ARG=""; [[ -f "$OUTDIR/latest.pt" ]] && RESTORE_ARG="experiment.restore_file=$OUTDIR/latest.pt"
cd "$REPO/het_control/run_scripts"
"$PYTHON" run_tag_ippo_cadico.py experiment.sampling_device=cuda experiment.train_device=cuda experiment.loggers='[csv]' experiment.render=False experiment.evaluation=True experiment.checkpoint_interval=600000 experiment.max_n_frames=12000000 model.desired_snd=$DESIRED_SND model.cadico_alpha=0.3 seed=$SEED $RESTORE_ARG hydra.run.dir="$OUTDIR"
