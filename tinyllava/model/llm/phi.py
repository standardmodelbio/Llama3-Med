from transformers import AutoTokenizer, PhiForCausalLM

from . import register_llm


@register_llm("phi")
def return_phiclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    return (PhiForCausalLM, (AutoTokenizer, tokenizer_and_post_load))
