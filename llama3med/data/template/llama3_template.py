from dataclasses import dataclass

import tokenizers
from packaging import version

from ...utils.constants import *
from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)

system = "You are a helpful language and vision assistant. "
"You are able to understand the visual content that the user provides, "
"and assist the user with a variety of tasks using natural language."


@register_template("llama3")
@dataclass
class Llama3Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(
        slot="<|start_header_id|>user<|end_header_id|>\n\n" + "{{content}}" + "<|eot_id|>"
    )
    format_assistant: "Formatter" = StringFormatter(
        slot="<|start_header_id|>assistant<|end_header_id|>\n\n" + "{{content}}" + "<|eot_id|>"
    )
    system: "Formatter" = EmptyFormatter(slot="<|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>")
    separator: "Formatter" = EmptyFormatter(slot=["<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"])

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1  # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = (
                len(self.tokenizer_image_token(rou, tokenizer))
                + eos_token_length
                - bos_token_length
            )
            instruction_len = (
                len(self.tokenizer_image_token(parts[0], tokenizer))
                - 1
                - bos_token_length
            )
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
