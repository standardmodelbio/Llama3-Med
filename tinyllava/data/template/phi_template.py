from dataclasses import dataclass

from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@register_template("phi")
@dataclass
class PhiTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(
        slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>"
    )
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT: ", "<|endoftext|>"])
