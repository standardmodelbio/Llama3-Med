#!/bin/bash

MODEL_PATH="/home/user/checkpoints/llama3med-Meta-Llama-3.1-8B-Instruct-h-optimus-0-base-lora-zero2-r128-finetune/"
MODEL_NAME="llama3med-Meta-Llama-3.1-8B-Instruct-h-optimus-0-base-lora-zero2-r128-finetune"
EVAL_DIR="/home/user/cache"

python -m llama3med.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base $MODEL_NAME \
    --question-file $EVAL_DIR/vqa_path/pathvqa_test_questions.jsonl \
    --image-folder $EVAL_DIR/vqa_path/images \
    --answers-file $EVAL_DIR/vqa_path/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama3
    