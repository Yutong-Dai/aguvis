#!/bin/bash
wandb login --relogin --host=https://salesforceairesearch.wandb.io local-43b3e79994791bf66817eaa5c5387869d27b0427 || { echo "Failed to login to wandb"; exit 1; }

export WANDB_PROJECT="text_to_flow"

LLM_VERSION=Qwen2-VL-7B-Instruct
LLM_PATH="Qwen/Qwen2-VL-7B-Instruct"
SFT_TASK="stage1_web"
SAVE_DIR=results/aguvis/
IMAGE_FOLDER=data/

SFT_DATA_YAML=data/${SFT_TASK}.yaml
SFT_RUN_NAME="${LLM_VERSION}-sft-${SFT_TASK}"
echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

WORLD_SIZE=1
RANK=0
MASTER_ADDR="localhost"
MASTER_PORT="29500"

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export ACCELERATE_CPU_AFFINITY=1

printenv

torchrun $DISTRIBUTED_ARGS train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path ${SFT_DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path $LLM_PATH \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR}/checkpoints/${SFT_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --freeze_visual_encoder True \
    --report_to wandb \
    --run_name $SFT_RUN_NAME
    # --report_to none
