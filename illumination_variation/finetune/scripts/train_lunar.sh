#!/bin/bash
# Launch training of EfficientLoFTR on StereoLunar dataset

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../EfficientLoFTR"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
cd "$PROJECT_DIR"

EXP_NAME="lunar_finetune_rtx3050"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    EXP_NAME="$1"
    shift
fi
main_cfg_path="configs/loftr/eloftr_lunar.py"
data_cfg_path="configs/data/stereo_lunar_512.py"

# Use python from eloftr-lunar environment
PYTHON_BIN="/home/ojas/miniforge3/envs/eloftr-lunar/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "=========================================================="
echo " Starting EfficientLoFTR Fine-Tuning on StereoLunar"
echo " Experiment Name : ${EXP_NAME}"
echo " Main Config     : ${main_cfg_path}"
echo " Data Config     : ${data_cfg_path}"
echo " Python Binary   : ${PYTHON_BIN}"
echo "=========================================================="

$PYTHON_BIN -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${EXP_NAME} \
    --gpus=1 \
    --num_nodes=1 \
    --accelerator="ddp" \
    --batch_size=1 \
    --num_workers=2 \
    --pin_memory=true \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=5 \
    --flush_logs_every_n_steps=20 \
    --limit_val_batches=1.0 \
    --num_sanity_val_steps=1 \
    --max_epochs=20 \
    --thr 0.2 \
    --ckpt_path="${PROJECT_DIR}/weights/eloftr_outdoor.ckpt" \
    "$@"
