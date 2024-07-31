import os
from collections import OrderedDict

import torch
from transformers import (
    BitsAndBytesConfig,
)

from .configuration_llama3med import Llama3MedConfig
from .modeling_llama3med import Llama3MedForConditionalGeneration

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def load_base_ckp_for_lora(ckp_path):
    logger.info("loading ckpt...")
    ckp = torch.load(ckp_path, map_location=torch.device("cpu"), weights_only=True)
    new_ckp = OrderedDict()
    logger.info("mapping...")
    for k, v in ckp.items():
        new_k = k.replace(".base_layer", "")
        new_ckp[new_k] = v
    return new_ckp


def load_pretrained_model(
    model_name_or_path,
    load_type="hf",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    if model_name_or_path is not None and "lora" not in model_name_or_path:
        model = Llama3MedForConditionalGeneration.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        )

    elif model_name_or_path is not None and "lora" in model_name_or_path:
        if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
            logger.info("intialize Llama3MedConfig...")
            model_config = Llama3MedConfig.from_pretrained(model_name_or_path)
            logger.info(model_config)
            logger.debug("initialize Llama3Med Model...")
            model = Llama3MedForConditionalGeneration(model_config)

            # language model
            language_model_ckp_path = os.path.join(
                model_name_or_path, "language_model/pytorch_model.bin"
            )
            logger.info("loading base ckpt for lora...")
            language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
            logger.info("loading language model...")
            model.language_model.load_state_dict(language_model_ckp)
            del language_model_ckp

            # vision tower
            vision_tower_ckp_path = os.path.join(
                model_name_or_path, "vision_tower/pytorch_model.bin"
            )
            vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
            logger.info("loading vison tower...")
            model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
            del vision_tower_ckp

            # connector
            connector_ckp_path = os.path.join(
                model_name_or_path, "connector/pytorch_model.bin"
            )
            connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
            logger.info("loading connector...")
            model.connector.load_state_dict(connector_ckp)
            del connector_ckp

            # convert to fp16
            logger.info("model to bf16...")
            model.to(torch.bfloat16)
            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_name_or_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")

    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, "max_sequence_length", 3072)
    # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
    tokenizer = model.tokenizer
    # tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, image_processor, context_len
