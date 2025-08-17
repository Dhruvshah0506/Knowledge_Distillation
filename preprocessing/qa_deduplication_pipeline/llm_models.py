# llm_models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_mistral(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def load_qwen(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    )
    return tokenizer, model
