<h2 align="center"> Repo for Llama3-Med </a><h5 align="center">

Code base inherited from [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory/tree/main)

## Contents

- [Installation](#installation-and-requirements)
- [Get Started](#get-started)
- [Model Zoo](#model-zoo)
- [Launch Demo Locally](#launch-demo-locally)
- [Custom Finetune](#Custom-Finetune)
- [Customize Your Own Large Multimodel Models](#customize-your-own-multimodel-models)


## Installation and Requirements

Please note that our environment requirements are different from LLaVA's environment requirements. We strongly recommend you create the environment from scratch as follows.

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/standardmodelbio/llama3-med.git
cd llama3-med
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n <env-name> python=3.10 -y
conda activate <env-name>
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages
```Shell
pip install -e ".[train]"

pip install flash-attn --no-build-isolation
```
#### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## Get Started

#### 1. Data Preparation

Please refer to the [Data Preparation](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html) section in our [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/).

#### 2. Train

Here's an example for training a LMM using Phi-2.

- Replace data paths with yours in `scripts/train/train_phi.sh`
- Replace `output_dir` with yours in `scripts/train/pretrain.sh`
- Replace `pretrained_model_path` and `output_dir` with yours in `scripts/train/finetune.sh`
- Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `scripts/train/pretrain.sh` and `scripts/train/finetune.sh`

```bash
bash scripts/train/train_phi.sh
```

Important hyperparameters used in pretraining and finetuning are provided below.

| Training Stage | Global Batch Size | Learning rate | conv_version |
| -------------- | :---------------: | :-----------: | :----------: |
| Pretraining    | 256               | 1e-3          | pretrain     |
| Finetuning     | 128               | 2e-5          | phi          |

**Tips:** 

Global Batch Size = num of GPUs * `per_device_train_batch_size` * `gradient_accumulation_steps`, we recommand you always keep global batch size and learning rate as above except for lora tuning your model.

`conv_version` is a hyperparameter used for choosing different chat templates for different LLMs. In the pretraining stage, `conv_version` is the same for all LLMs, using `pretrain`. In the finetuning stage, we use

`phi` for Phi-2, StableLM, Qwen-1.5

`llama` for TinyLlama, OpenELM

`gemma` for Gemma

#### 3. Evaluation

Please refer to the [Evaluation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html) section in our [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html).


## Launch Demo Locally

If you want to launch the model trained by yourself or us locally, here's an example.
<details>
<summary>Run inference with the model trained by yourself</summary>

```Python
from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/absolute/path/to/your/model/"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"
conv_mode = "phi" # or llama, gemma, etc

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)

"""
Output: 
XXXXXXXXXXXXXXXXX
"""
```
</details>

<details>
<summary>Run inference with the model trained by us using huggingface transformers</summary>

```Python
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = 'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
prompt="What are these?"
image_url="http://images.cocodataset.org/val2017/000000039769.jpg"
output_text, genertaion_time = model.chat(prompt=prompt, image=image_url, tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
```
</details>
