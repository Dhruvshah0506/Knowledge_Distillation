# model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import torch

from huggingface_hub import login

login(token="your hf token")


def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device_index=3, task="text-generation"):
    """
    task: "text-generation" for LLaMA or "nli" for models like roberta-large-mnli
    """
    print(f"🚀 Loading model '{model_name}' on CUDA:{device_index} for task: {task}...")

    if task == "text-generation":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device_index},
            trust_remote_code=True
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    elif task == "nli":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device_index)

    else:
        raise ValueError("Unsupported task type. Use 'text-generation' or 'nli'.")

    print("✅ Model loaded successfully.")
    return pipe
