import os

import torch
import torch.nn as nn
from loguru import logger
from transformers import PreTrainedModel
from .s2wrapper import forward as multiscale_forward


def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg
        
        self.s2_scales = getattr(cfg, "s2_scales", "224,672,1344")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        self.multiscale_forward = multiscale_forward

    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(
            kwargs, "pretrained_vision_tower_path"
        )
        if isinstance(self._vision_tower, PreTrainedModel):  # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(
                vision_tower_name, **kwargs
            )
        else:  # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(
                    os.path.join(pretrained_vision_tower_path, "pytorch_model.bin"),
                    map_location="cpu",
                )

                def get_w(weights, keyword):
                    return {
                        k.split(keyword + ".")[1]: v
                        for k, v in weights.items()
                        if keyword in k
                    }

                self._vision_tower.load_state_dict(vision_tower_weights)

        logger.info("Loading vision tower from ", vision_tower_name)

    @torch.no_grad()
    def forward_feature(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[
            kwargs.get("vision_feature_layer", -2)
        ]

        if kwargs.get("vision_feature_select_strategy", "patch") == "patch":
            image_features = image_features[:, 1:]
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

    @property
    def vision_tower(self):
        return self._vision_tower

    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
