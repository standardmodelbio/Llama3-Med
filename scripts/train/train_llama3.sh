DATA_PATH=/home/user/cache/pubmedvision/PubMedVision_Alignment_PATH_VQA.json #pretrain annotation file path
FINETUNE_DATA_PATH=/home/user/cache/pubmedvision/PubMedVision_InstructionTuning_PATH_VQA.json #finetune annotation file path
IMAGE_PATH=/home/user/cache/pubmedvision/ #pretrain image dir
FINETUNE_IMAGE_PATH=/home/user/cache/pubmedvision/ #finetune image dir

LLM_VERSION=meta-llama/Meta-Llama-3.1-8B-Instruct # llm path in huggingface
VT_VERSION=h-optimus-0 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=llama3 #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/lora/finetune_lora.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
