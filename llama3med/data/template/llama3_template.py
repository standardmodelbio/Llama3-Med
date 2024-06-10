import copy
import re
from dataclasses import dataclass

from ...utils.constants import *
from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter

system = "You are a helpful language and vision assistant. "
"You are able to understand the visual content that the user provides, "
"and assist the user with a variety of tasks using natural language."


@register_template("llama3")
@dataclass
class Llama3Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(
        slot="<|start_header_id|>user<|end_header_id|>\n\n"
        + "{{content}}"
        + "<|eot_id|>"
    )
    format_assistant: "Formatter" = StringFormatter(
        slot="<|start_header_id|>assistant<|end_header_id|>\n\n"
        + "{{content}}"
        + "<|eot_id|>"
    )
    system: "Formatter" = EmptyFormatter(
        slot="<|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>"
    )
    separator: "Formatter" = EmptyFormatter(
        slot=["<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"]
    )

    def _make_masks(self, labels, tokenizer, sep, rounds):
        cur_len = 1  # bos
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) == 2:
                parts[0] += sep
            round_len = (
                len(self.tokenizer_image_token(rou, tokenizer)) - bos_token_length
            )
            instruction_len = (
                len(self.tokenizer_image_token(parts[0], tokenizer)) - bos_token_length
            )

            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    def make_labels(self, input_ids, prompt, tokenizer):
        def separate_rounds(prompt):
            # Define a regular expression pattern to split the sequence
            parts = re.split(
                r"(<\|start_header_id\|>system<\|end_header_id\|>\n\n|<\|start_header_id\|>user<\|end_header_id\|>\n\n|<\|start_header_id\|>assistant<\|end_header_id\|>\n\n)",
                prompt,
            )

            # Reattach the split delimiters to the following text
            result = []
            for i in range(1, len(parts), 2):
                result.append(parts[i] + parts[i + 1].strip())

            return result

        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = separate_rounds(prompt)
        labels, cur_len = self._make_masks(labels, tokenizer, sep, rounds)
        if cur_len < tokenizer.model_max_length:
            # import time

            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                # print("number of rounds: ", len(rounds) - 1)
                # print("rounds: ", rounds[:-1])
                # print("prompt: ", prompt)
                # print(labels)
                # print(input_ids)
                # time.sleep(5)
                # labels[:] = IGNORE_INDEX
        return labels
