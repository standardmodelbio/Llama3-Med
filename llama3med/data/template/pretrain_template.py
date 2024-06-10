import copy
from dataclasses import dataclass

from ...utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter


@register_template("pretrain")
@dataclass
class PretrainTemplate(Template):
    format_image_token: "Formatter" = EmptyFormatter(slot="")
    format_user: "Formatter" = EmptyFormatter(slot="<image>")
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}\n")
    system: "Formatter" = EmptyFormatter(slot="")
    separator: "Formatter" = EmptyFormatter(slot=["", ""])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        mask_len = len(self.tokenizer_image_token(DEFAULT_IMAGE_TOKEN, tokenizer))
        labels[:mask_len] = IGNORE_INDEX
        return labels
