#!/bin/bash
if [ $# -ne 11 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <PRETRAIN_VERSION> <FINETUNE_VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
CONV_VERSION="$7"
PRETRAIN_VERSION="$8"
FINETUNE_VERSION="$9"
TRAIN_RECIPE="${10}"
MODEL_MAX_LENGTH="${11}"

VT_VARIANT="${VT_VERSION##*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed llama3med/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path $HOME/checkpoints/llama3med-${LLM_VARIANT}-${VT_VARIANT}-${PRETRAIN_VERSION}-pretrain \
    --output_dir $HOME/checkpoints/llama3med-${LLM_VARIANT}-${VT_VARIANT}-${FINETUNE_VERSION}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name llama3med-${LLM_VARIANT}-${VT_VARIANT}-${FINETUNE_VERSION}-finetune
