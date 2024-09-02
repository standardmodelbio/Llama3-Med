import torch
import timm
from torchvision import transforms


if __name__ == "__main__":
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=False, dynamic_img_size=False, weight_init="skip"
    )

    print(model.pretrained_cfg)