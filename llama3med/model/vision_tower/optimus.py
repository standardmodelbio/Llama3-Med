import functools
import os

import timm
import torch
from transformers import AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower


def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


@register_vision_tower("optimus")
class OptimusVisionTower(VisionTower):
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
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = timm.models.VisionTransformer(**self.params)
        self._image_processor = AutoImageProcessor.from_pretrained(
            cfg.model_name_or_path
        )

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(
            kwargs, "pretrained_vision_tower_path"
        )
        if pretrained_vision_tower_path is not None:
            vision_tower_weights = torch.load(
                os.path.join(pretrained_vision_tower_path, "checkpoint.pth"),
                map_location="cpu",
            )
            self._vision_tower.load_state_dict(vision_tower_weights)
        else:
            raise ValueError

        print("Loading vision tower from ", vision_tower_name)

    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x)

        return image_features