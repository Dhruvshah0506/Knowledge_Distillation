# === qa_pipeline/preprocessing/fitb_converter.py ===

import pandas as pd
import logging
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

from constants.constants import CONFIG, PROMPT_TEMPLATES

print("\nfitb_converter#######\n")

logger = logging.getLogger(__name__)

def make_fitb_prompt(question: str) -> str:
    """
    Build the LLM prompt for converting a fill-in-the-blank question to a normal question.
    """
    return PROMPT_TEMPLATES["fitb"].format(question=question)


def create_fitb_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects fill-in-the-blank questions in a dataframe and rephrases them using FLAN-T5.

    Args:
        df (pd.DataFrame): The input QA dataframe with a 'Question' column.

    Returns:
        pd.DataFrame: The dataframe with FITB questions rewritten.
    """

    # === Load model config ===
    model_id = CONFIG['models'].get('fitb_model_id', "google/flan-t5-large")

    logger.info(f"[FITB] Loading FLAN-T5 model: {model_id}")
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id).to("cuda:3")
    rephrase_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=3)

    # === Find fill-in-the-blank rows (detect underscores) ===
    fill_mask = df["Question"].astype(str).str.contains(r'_{2,}', regex=True)
    num_fitb = fill_mask.sum()
    logger.info(f"[FITB] Found {num_fitb} fill-in-the-blank questions to rephrase.")

    # === Rephrase each detected FITB ===
    for idx in df[fill_mask].index:
        original = df.at[idx, "Question"]
        prompt = make_fitb_prompt(original)

        try:
            result = rephrase_pipe(prompt, max_new_tokens=100)[0]['generated_text']
            df.at[idx, "Question"] = result.strip()
            logger.info(f"[FITB] Rephrased index {idx}: '{original}' -> '{result.strip()}'")

        except Exception as e:
            logger.warning(f"[FITB] Failed to rephrase at index {idx}: {e}")

    logger.info("[FITB] Rephrasing complete.")
    return df
