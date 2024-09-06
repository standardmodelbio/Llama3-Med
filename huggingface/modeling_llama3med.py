import functools
import re
from typing import List, Optional, Tuple, Union

import timm
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import (
    AutoImageProcessor,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_llama3med import Llama3MedConfig
from .s2wrapper import forward as multiscale_forward

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


def tokenizer_and_post_load(tokenizer):
    # if tokenizer.unk_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # else:
    #     tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


class OptimusVisionTower(nn.Module):
    params = {
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "mlp_layer": functools.partial(
            timm.layers.mlp.GluMlp,
            act_layer=torch.nn.modules.activation.SiLU,
            gate_last=False,
        ),
        "act_layer": torch.nn.modules.activation.SiLU,
        "reg_tokens": 4,
        "no_embed_class": True,
        "img_size": 224,
        "num_classes": 0,
        "in_chans": 3,
        "weight_init": "skip",
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = timm.models.VisionTransformer(**self.params)
        self._image_processor = AutoImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self._image_processor.crop_size = 1344
        self._image_processor.size = 1344
        self._image_processor.image_mean = [0.707223, 0.578729, 0.703617]
        self._image_processor.image_std = [0.211883, 0.230117, 0.177517]

        self.s2_scales = getattr(cfg, "s2_scales", "224,672,1344")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        self.multiscale_forward = multiscale_forward

    @torch.no_grad()
    def forward_feature(self, x, **kwargs):
        # device = x.data.device
        # self.to(device)
        image_features = self._vision_tower.forward_features(x)

        if kwargs.get("vision_feature_select_strategy", "patch") == "patch":
            num_prefix_tokens = self.params["reg_tokens"] + 1
            image_features = image_features[:, num_prefix_tokens:]
        elif kwargs.get("vision_feature_select_strategy", "patch") == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}"
            )
        return image_features

    @torch.no_grad()
    def forward(self, images, **kwargs):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image,
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                    multi_images=True,
                )
                image_features.append(image_feature)  # [(num_images x h x w, c)]
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
                multi_images=False,
            )  # (batch, (h x w), c)
        return image_features


class MLPConnector(nn.Module):
    def __init__(self, config):
        super().__init__()

        act_type = config.connector_type.split("_")[-1]
        if act_type == "gelu":
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.connector_type)
            mlp_depth = int(mlp_gelu_match.group(1))
        elif act_type == "relu":
            mlp_relu_match = re.match(r"^mlp(\d+)x_relu$", config.connector_type)
            mlp_depth = int(mlp_relu_match.group(1))
        else:
            raise NotImplementedError("Currently only support ReLU and GELU.")
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))

        self._connector = nn.Sequential(*modules)

    def forward(self, x):
        if type(x) is list:
            features = []
            for sample in x:
                features.append(self._connector(sample.unsqueeze(0)).squeeze(0))
        else:
            features = self._connector(x)
        return features


class Llama3MedPreTrainedModel(PreTrainedModel):
    config_class = Llama3MedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class Llama3MedForConditionalGeneration(Llama3MedPreTrainedModel):
    def __init__(self, config: Llama3MedConfig):
        super().__init__(config)

        self.language_model = LlamaForCausalLM(config.text_config)
        self.vision_tower = OptimusVisionTower(config.vision_config)
        self.connector = MLPConnector(config)

        # self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def encode_images(self, images):
        kwargs = {}
        kwargs["vision_feature_layer"] = self.config.vision_feature_layer
        kwargs["vision_feature_select_strategy"] = (
            self.config.vision_feature_select_strategy
        )
        # images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(
                    cur_input_ids
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )
