import functools
import os

import timm
import torch
from transformers import AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower
from .s2wrapper import forward as multiscale_forward


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

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(
            kwargs, "pretrained_vision_tower_path"
        )
        pretrained_vision_tower_path = os.path.join(
            "/home/user/cache/checkpoints", vision_tower_name
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
                print(image.shape)
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image,
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                    multi_images=True
                )
                image_features.append(image_feature)  # [(num_images x h x w, c)]
        else:
            print(images.shape)
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
                multi_images=False
            ) # (batch, (h x w), c)
        return image_features
