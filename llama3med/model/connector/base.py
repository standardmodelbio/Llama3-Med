import os

import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file


class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    def load_model(self, **kwargs):
        pretrained_connector_path = kwargs.get("pretrained_connector_path", None)
        if pretrained_connector_path is not None:
            pretrained_connector_path = os.path.join(
                pretrained_connector_path, "model.safetensors"
            )
            connector_weights = load_file(pretrained_connector_path, device="cpu")

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self._connector.load_state_dict(get_w(connector_weights, "_connector"))
            logger.info(f"Loading connector from {pretrained_connector_path}...")

        for p in self._connector.parameters():
            p.requires_grad = False

    def forward(self, x):
        if type(x) is list:
            features = []
            for sample in x:
                features.append(self._connector(sample.unsqueeze(0)).squeeze(0))
        else:
            features = self._connector(x)
        return features
