import re
import pandas as pd
import torch
torch.cuda.set_device(3)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer, util
import logging
from typing import List, Tuple

from huggingface_hub import login

login(token="your hf token here")


from constants.constants import PROMPT_TEMPLATES, CONFIG
print("\n\nmcq_converter#################")

logger = logging.getLogger(__name__)

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_mcq(question_text: str) -> bool:
    """
    Detects whether a question is a multiple-choice question (MCQ) 
    based on structured option patterns.
    """
    if not isinstance(question_text, str):
        return False

    pattern_alpha = re.findall(r"\([a-dA-D]\)\s*[^\n\(\)]+", question_text)
    pattern_numeric = re.findall(r"\([1-9]\)\s*[^\n\(\)]+", question_text)
    pattern_roman_lower = re.findall(r"\((i{1,3}|iv|v|vi{0,3}|ix|x)\)\s*[^\n\(\)]+", question_text)
    pattern_roman_upper = re.findall(r"(?m)^(I{1,3}|IV|V|VI{0,3}|IX|X)[\.\)]\s+[^\n]+", question_text)

    total_matches = (
        len(pattern_alpha) +
        len(pattern_numeric) +
        len(pattern_roman_lower) +
        len(pattern_roman_upper)
    )

    return total_matches >= 2

def get_question_stem(text: str) -> str:
    """
    Extract the stem (main body) of an MCQ by removing options.
    """
    return re.split(r"\([a-dA-D1-9]\)", text)[0].strip()

def extract_options(question_text: str) -> List[Tuple[str, str]]:
    """
    Extract labeled options from an MCQ.
    """
    if not isinstance(question_text, str):
        return []

    options = []
    matches = re.findall(r"\(([a-dA-D])\)[\s\.]*([^\(\)\n]+)", question_text)
    if matches:
        return [(label.lower(), text.strip()) for label, text in matches]

    matches = re.findall(r"\(([1-9])\)[\s\.]*([^\(\)\n]+)", question_text)
    if matches:
        return matches

    matches = re.findall(r"\((i{1,3}|iv|v|vi{0,3}|ix|x)\)[\s\.]*([^\(\)\n]+)", question_text, re.IGNORECASE)
    if matches:
        return [(label.lower(), text.strip()) for label, text in matches]

    matches = re.findall(r"(?m)^(I{1,3}|IV|V|VI{0,3}|IX|X)[\.\)]\s+([^\n]+)", question_text)
    if matches:
        return [(label.upper(), text.strip()) for label, text in matches]

    return []

def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())

def convert_row_to_mcq_dict(row: dict) -> dict:
    question_text = row.get("question", "")
    answer_text = row.get("answer", "")

    if not question_text or not answer_text:
        return None

    extracted_options = extract_options(question_text)
    if not extracted_options:
        return None

    labels, option_texts = zip(*extracted_options)

    for i, label in enumerate(labels):
        pattern = rf"\(?{re.escape(label)}\)?[\.\)]?"
        if re.search(pattern, answer_text, re.IGNORECASE):
            return {
                "question": get_question_stem(question_text),
                "options": list(option_texts),
                "correct_option": i
            }

    norm_answer = normalize_text(answer_text)
    for i, option in enumerate(option_texts):
        if normalize_text(option) in norm_answer:
            return {
                "question": get_question_stem(question_text),
                "options": list(option_texts),
                "correct_option": i
            }

    try:
        embeddings = sbert_model.encode([answer_text] + list(option_texts), convert_to_tensor=True)
        scores = util.cos_sim(embeddings[0], embeddings[1:])
        best_score, best_index = scores[0].max(), scores[0].argmax().item()

        if best_score >= 0.70:
            return {
                "question": get_question_stem(question_text),
                "options": list(option_texts),
                "correct_option": best_index
            }
    except Exception as e:
        print(f"[MCQ] SBERT similarity failed: {e}")

    return None

def extract_qa_pairs_from_llm(llm_output: str, mcq_id: int) -> list:
    qa_pairs = []
    lines = llm_output.strip().splitlines()
    current_question = None
    current_answer = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("q:"):
            current_question = line[2:].strip()
        elif line.lower().startswith("a:") and current_question:
            current_answer = line[2:].strip()
            qa_pairs.append({
                "mcq_id": mcq_id,
                "question": current_question,
                "answer": current_answer
            })
            current_question = None
            current_answer = None

    return qa_pairs

class StopOnTokenString(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-len(self.stop_ids):].tolist() == self.stop_ids
        
        if len(input_ids[0]) < len(self.stop_ids):
            return False
        # Check for invalid token ids
        if (input_ids[0] < 0).any():
            print('\n **Invalid token id detected!')
            return False
        return input_ids[0].cpu()[-len(self.stop_ids):].tolist() == self.stop_ids


# main
def create_mcqs_questions(df: pd.DataFrame) -> pd.DataFrame:
    mcq_model_id = CONFIG['models'].get('mcq_model_id', "meta-llama/Llama-3.2-3B-Instruct")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(mcq_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        mcq_model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

        device_map={"": 3},
        quantization_config=bnb_config
    )
    
    model.eval()
    logger.info(f"[MCQ] Loaded MCQ model: {mcq_model_id}")

    df["is_mcq"] = df["Question"].apply(is_mcq)
    mcq_df = df[df["is_mcq"]].copy()
    logger.info(f"[MCQ] Identified {len(mcq_df)} MCQs")

    cleaned_mcqs = []
    failed = 0

    for idx, row in tqdm(mcq_df.iterrows(), total=len(mcq_df), desc="Processing MCQs"):
        mcq_obj = convert_row_to_mcq_dict({
            "question": row["Question"],
            "answer": row["Answer"]
        })
        if not mcq_obj:
            logger.warning(f"[MCQ] Could not parse MCQ at index {idx}")
            failed += 1
            continue


        labels = ['A', 'B', 'C', 'D']
        formatted_options = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(mcq_obj["options"]))
        correct_label = labels[mcq_obj["correct_option"]]


        import string
        labels = [string.ascii_uppercase[i] for i in range(len(mcq_obj["options"]))]
        formatted_options = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(mcq_obj["options"]))
        correct_label = labels[mcq_obj["correct_option"]]

        prompt = PROMPT_TEMPLATES["mcq_hybrid"].format(
            question=mcq_obj["question"],
            formatted_options=formatted_options,
            correct_label=correct_label
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")


        stop_token_ids = tokenizer.encode("###", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([StopOnTokenString(stop_token_ids)])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]

        qa_pairs = extract_qa_pairs_from_llm(decoded, idx)
        if qa_pairs:
            cleaned_mcqs.extend(qa_pairs)
        else:
            logger.warning(f"[MCQ] LLM returned no pairs for index {idx}")
            failed += 1

    cleaned_df = pd.DataFrame(cleaned_mcqs)
    non_mcq_df = df[~df["is_mcq"]].copy()

    final_df = pd.concat([
        non_mcq_df[["Question", "Answer"]],
        cleaned_df.rename(columns={"question": "Question", "answer": "Answer"})[["Question", "Answer"]]
    ], ignore_index=True)

    logger.info(f"[MCQ] Finished MCQ conversion: {len(cleaned_mcqs)} pairs extracted, {failed} failed")

    return final_df
