import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login

from qa_deduplication_pipeline.constants import *
from qa_deduplication_pipeline.llm_models import load_mistral, load_qwen
from qa_deduplication_pipeline.rephrase_utils import (
    selective_rephrase,
    extract_qwen_response,
    make_qwen_prompt,
    make_mistral_prompt,
    clean_output
)

def run_llm_qa_refinement(
    input_path,
    output_path,
    hf_token,
    mistral_model_id,
    qwen_model_id,
    cuda_device,
    only_mistral_rephrased=None,
    only_qwen_rephrased=None,
    mistral_output=None,
    exact_duplicate_path=None,
    same_ans_diff_ques_path=None,
    same_ques_diff_ans_path=None
):
    # Hardcoded output path
    hardcoded_output_path = "/Projects/qa_cleaning__and_aug/content/outputs/deduplication_rephrasing.csv"

    # Initial setup
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    login(hf_token)
    tqdm.pandas()

    # Load dataset
    df = pd.read_csv(input_path)
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()
    original_total = len(df)
    print(f"Original rows: {original_total}")

    # === Save exact duplicate QA pairs ===
    exact_duplicates = df[df.duplicated(subset=["question", "answer"], keep=False)]
    exact_duplicates.to_csv(EXACT_DUPLICATE_PAIRS_CSV, index=False)
    print(f"Exact QA pairs found (total occurrences): {len(exact_duplicates)}")

    # === Save same question → different answers ===
    same_q_diff_a = df.groupby("question").filter(lambda x: x["answer"].nunique() > 1)
    same_q_diff_a.to_csv(SAME_Q_DIFF_A_CSV, index=False)
    print(f"Same Question → Different Answers: {len(same_q_diff_a)} rows")

    # === Save same answer → different questions ===
    same_a_diff_q = df.groupby("answer").filter(lambda x: x["question"].nunique() > 1)
    same_a_diff_q.to_csv(SAME_A_DIFF_Q_CSV, index=False)
    print(f"Same Answer → Different Questions: {len(same_a_diff_q)} rows")

    original_df = df.copy()

    # Mistral Phase
    print("\nRunning Mistral Phase...")
    mistral_pipe = load_mistral(mistral_model_id, cuda_device)

    def mistral_q(text):
        prompt = make_mistral_prompt("Question", text)
        result = mistral_pipe(prompt, max_new_tokens=64, temperature=0.7, top_p=0.9)[0]['generated_text']
        return result.split("[/INST]")[-1].split("</s>")[0].strip()

    def mistral_a(text):
        prompt = make_mistral_prompt("Answer", text)
        result = mistral_pipe(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9)[0]['generated_text']
        return result.split("[/INST]")[-1].split("</s>")[0].strip()

    df["Question"] = selective_rephrase(df, "Question", mistral_q)
    df["Answer"] = selective_rephrase(df, "Answer", mistral_a)

    # Save Mistral results
    mistral_changed = df.copy()
    mistral_changed["orig_question"] = original_df["Question"]
    mistral_changed["orig_answer"] = original_df["Answer"]
    only_mistral_changed = mistral_changed[
        (mistral_changed["Question"] != mistral_changed["orig_question"]) |
        (mistral_changed["Answer"] != mistral_changed["orig_answer"])
    ]
    if only_mistral_rephrased:
        only_mistral_changed.to_csv(only_mistral_rephrased, index=False)
    if mistral_output:
        df.to_csv(mistral_output, index=False)
    else:
        mistral_output = "temp_mistral_output.csv"
        df.to_csv(mistral_output, index=False)

    # Qwen Phase
    print("\nRunning Qwen Phase...")
    qwen_tokenizer, qwen_model = load_qwen(qwen_model_id, cuda_device)

    def qwen_chat(prompt):
        full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = qwen_tokenizer(full_prompt, return_tensors="pt").to(cuda_device)
        with torch.no_grad():
            output = qwen_model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.95)
        decoded = qwen_tokenizer.decode(output[0], skip_special_tokens=False)
        return extract_qwen_response(decoded)


    def qwen_q(text): return qwen_chat(make_qwen_prompt("question", text))
    def qwen_a(text): return qwen_chat(make_qwen_prompt("answer", text))

    df = pd.read_csv(mistral_output)
    df["question"] = df["question"].astype(str).str.strip().apply(clean_output)
    df["answer"] = df["answer"].astype(str).str.strip().apply(clean_output)
    original_df = df.copy()

    df = selective_rephrase(df, "question", qwen_q)
    df = selective_rephrase(df, "answer", qwen_a)

    # Save Qwen results
    qwen_changed = df.copy()
    qwen_changed["orig_question"] = original_df["question"]
    qwen_changed["orig_answer"] = original_df["answer"]
    only_qwen_changed = qwen_changed[
        (qwen_changed["question"] != qwen_changed["orig_question"]) |
        (qwen_changed["answer"] != qwen_changed["orig_answer"])
    ]
    if only_qwen_rephrased:
        only_qwen_changed.to_csv(only_qwen_rephrased, index=False)
    df.to_csv(output_path, index=False)

    # Final Conflicts
    conflict_q = df.groupby("question").filter(lambda x: x["answer"].nunique() > 1)
    conflict_a = df.groupby("answer").filter(lambda x: x["question"].nunique() > 1)

    def qwen_q(text): return qwen_chat(make_qwen_prompt("Question", text))
    def qwen_a(text): return qwen_chat(make_qwen_prompt("Answer", text))

    df = pd.read_csv(mistral_output)
    df["Question"] = df["Question"].astype(str).str.strip().apply(clean_output)
    df["Answer"] = df["Answer"].astype(str).str.strip().apply(clean_output)
    original_df = df.copy()

    df = selective_rephrase(df, "Question", qwen_q)
    df = selective_rephrase(df, "Answer", qwen_a)

    # Save Qwen results
    qwen_changed = df.copy()
    qwen_changed["orig_question"] = original_df["Question"]
    qwen_changed["orig_answer"] = original_df["Answer"]
    only_qwen_changed = qwen_changed[
        (qwen_changed["Question"] != qwen_changed["orig_question"]) |
        (qwen_changed["Answer"] != qwen_changed["orig_answer"])
    ]
    if only_qwen_rephrased:
        only_qwen_changed.to_csv(only_qwen_rephrased, index=False)
    # Save to both dynamic output path and hardcoded path
    df.to_csv(output_path, index=False)
    df.to_csv(hardcoded_output_path, index=False)

    # Final Conflicts
    conflict_q = df.groupby("Question").filter(lambda x: x["Answer"].nunique() > 1)
    conflict_a = df.groupby("Answer").filter(lambda x: x["Question"].nunique() > 1)


    print(f"\nRemaining Same Question → Diff Answers: {len(conflict_q)}")
    print(f"Remaining Same Answer → Diff Questions: {len(conflict_a)}")
    print(f"Final row count: {len(df)} (Expected: {original_total})")

    print(f"✅ Final output saved to '{output_path}' and '{hardcoded_output_path}'")

