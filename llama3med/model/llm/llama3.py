from transformers import AutoTokenizer, LlamaForCausalLM

from . import register_llm


@register_llm("llama3")
def return_llama3class():
    def tokenizer_and_post_load(tokenizer):
        if tokenizer.unk_token is None:
            tokenizer.pad_token = "<|eot_id|>"
        else:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
