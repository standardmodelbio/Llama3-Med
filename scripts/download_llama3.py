import torch
from transformers import AutoModelForCausalLM, AutoConfig


if __name__ == "__main__":
    def do_nothing_init(model):
        pass

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    text_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True
    )
    print(text_config)
    # Create the model instance without initializing weights
    model = AutoModelForCausalLM.from_pretrained(model_id)

    print(model)  # This will show the model architecture
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Verify that weights are uninitialized
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.sum().item()}")  # Should be 0 or very close to 0

    # quantized_model = AutoModelForCausalLM.from_config(text_config)

