import copy
import re

import tokenizers
import torch
from packaging import version
from transformers import AutoTokenizer

IGNORE_INDEX = 128003
IMAGE_TOKEN_INDEX = 128002
DEFAULT_IMAGE_TOKEN = "<|reserved_special_token_0|>"
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)



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


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    def _insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def make_masks(labels, tokenizer, sep, eos_token_length, rounds):
    cur_len = 1  # bos
    eos_token_length = 1
    bos_token_length = 1
    labels[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
        if rou == "":
            break
        parts = rou.split(sep)
        if len(parts) == 2:
            parts[0] += sep
        round_len = (
            len(tokenizer_image_token(rou, tokenizer))
            - bos_token_length
        )
        instruction_len = (
            len(tokenizer_image_token(parts[0], tokenizer)) - bos_token_length
        )

        labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    labels[cur_len:] = IGNORE_INDEX
    print(cur_len)
    return labels, cur_len


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
# print(tokenizer("<|start_header_id|>system<|end_header_id|>\n\nYou are a pirate chatbot who always responds in pirate speak!"))
# print(tokenizer.bos_token)
# print(tokenizer.eos_token)
# tokenizer.eos_token = ""
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# print(tokenizer.pad_token)
# print(tokenizer.eos_token)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "What is of this image?"},
    # {"role": "assistant", "content": "It's a dog"},
    {
        "role": "user",
        "content": "How's the dog look like?",
    },
    # {"role": "assistant", "content": "ok."},
    {
        "role": "user",
        "content": "How does the dog look feel?",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    # add_generation_prompt=True,
    return_tensors="pt",
    tokenize=False,
)
prompt = "".join(prompt.split(tokenizer.bos_token)[1])
# print([prompt])
# prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a pirate chatbot who always responds in pirate speak!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_0|>\nWhat is of this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIt's a dog<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_0|>\nHow's the dog look like?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIt's small.<|eot_id|>"
# print(tokenizer(prompt))
# print([prompt])

input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
print(input_ids)

labels = copy.deepcopy(input_ids)
sep, eos_token = ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>")
total_len = int(labels.ne(tokenizer.pad_token_id).sum())
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    total_len += prompt.count(eos_token)
# rounds = prompt.split(sep)
print(total_len)
rounds = separate_rounds(prompt)
# print(rounds)
# eos_token_length = len(tokenizer.encode(eos_token))

masks = make_masks(labels, tokenizer, sep=sep, eos_token_length=1, rounds=rounds)
# print([tokenizer.decode(masks[0])])
print(masks)
# print([tokenizer.decode(tokenizer_image_token(prompt, tokenizer))])
# print([input_ids])

# print([tokenizer.decode([128000, 128006,   9125, 128007,    271,   2675,    527,    264,  55066,
#            6369,   6465,    889,   2744,  31680,    304,  55066,   6604,      0,
#          128009, 128006,    882, 128007,    271,  15546,    527,    499,     30,
#          128009, 128006,  78191, 128007,    271,     40,   2846,    459,  15592,
#              13, 128009])]
