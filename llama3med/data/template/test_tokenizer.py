import torch
from transformers import AutoTokenizer

IMAGE_TOKEN_INDEX = 128002
DEFAULT_IMAGE_TOKEN = "<|reserved_special_token_0|>"


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


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    {"role": "user", "content": "<|reserved_special_token_0|>\nWhat is of this image?"},
    {"role": "assistant", "content": "It's a dog"},
    {"role": "user", "content": "<|reserved_special_token_0|>\nHow's the dog look like?"},
    {"role": "assistant", "content": "It's small."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    # add_generation_prompt=True,
    return_tensors="pt",
    tokenize=False,
)

print(tokenizer_image_token(prompt, tokenizer))
print([tokenizer.decode(tokenizer_image_token(prompt, tokenizer))])
# print([input_ids])

# print([tokenizer.decode([128000, 128006,   9125, 128007,    271,   2675,    527,    264,  55066,
#            6369,   6465,    889,   2744,  31680,    304,  55066,   6604,      0,
#          128009, 128006,    882, 128007,    271,  15546,    527,    499,     30,
#          128009, 128006,  78191, 128007,    271,     40,   2846,    459,  15592,
#              13, 128009])]
