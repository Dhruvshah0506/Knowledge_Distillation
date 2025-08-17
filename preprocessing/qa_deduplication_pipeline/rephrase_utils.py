# rephrase_utils.py

from collections import defaultdict

def selective_rephrase(df, column_name, rephrase_fn):
    seen = defaultdict(bool)
    result = []
    for val in df[column_name]:
        if not seen[val]:
            seen[val] = True
            result.append(val)
        else:
            result.append(rephrase_fn(val))
    return result

def clean_output(text):
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.lower().startswith("or:")]
    return lines[0] if lines else text.strip()

def extract_qwen_response(raw_text):
    if "<|im_start|>assistant\n" in raw_text:
        response = raw_text.split("<|im_start|>assistant\n")[-1]
        for sep in ["<|im_start|>", "<|im_end|>", "system", "user", "assistant"]:
            response = response.split(sep)[0]
        return response.strip().strip('"').strip()
    return raw_text.strip()

def make_qwen_prompt(role, text):
    if role == "question":
        return f"Rephrase the following question using different words, but keep the meaning exactly the same. Give only one rephrased question:\n{text}"
    else:
        return f"Rephrase the following answer in one sentence using different words but keeping the same meaning. Do not include any question:\n{text}"

def make_mistral_prompt(role, text):
    if role == "question":
        return f"<s>[INST] Rephrase the following question clearly in one sentence without including multiple variations or adding 'Or:'. Do not include any answers:\n{text}\n[/INST]"
    else:
        return f"<s>[INST] Rephrase the following answer clearly in one sentence using different words. Do not add any questions or extra information:\n{text}\n[/INST]"
