
import yaml
# System prompt templates
SYSTEM_PROMPT = """
You are a helpful AI assistant for cleaning and augmenting QA datasets.
"""

# Regex patterns
REGEX_PATTERNS = {
    "symbols": r"[!@#$%^&*()_+=\[\]{};:\"\\|,.<>\/?~`]",
    "unicode": r"[^\x00-\x7F]+",
    "multiple_spaces": r"\s+",
    "fill_in_blank": r'_{2,}'
}

# Default values
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 128

# Prompt templates
PROMPT_TEMPLATES = {
    "fitb": (
         "Convert this fill-in-the-blank into a clear, meaningful question "
        "that retains its original intent and does not give away the answer:\n\n"
        "{question}"
    ),

    "mcq_hybrid": (
        "<|begin_of_text|><|header_start|>system<|header_end|>\n"
        "You are a precise and concise tutor for accounting and business exams.\n"
        "Your task is to convert multiple-choice questions (MCQs) into structured question and answer (QA) pairs using a hybrid strategy.\n\n"
        "Use the correct answer to generate one main QA pair.\n"
        "From the incorrect options, generate additional QA pairs only if they are relevant to accounting or business.\n"
        "Ensure the Questions and Answers Generated have sufficient context.\n"
        "<|eot|><|header_start|>user<|header_end|>\n"
        "MCQ:\n\n"

        "Q: {question}\n"
        "{formatted_options}\n\n"

        "Correct option: {correct_label}\n\n"
        "ONLY DISPLAY THE QAs\n"
        "Return output in this format:\n\n"

        "- Hybrid QAs:\n"
        "  Q: ...\n"
        "  A: ...\n\n"

        "  Q: ...\n"
        "  A: ...\n\n"
        "<|eot|><|header_start|>assistant<|header_end|>\n"
    )
}

DEFAULTS = {
    "back_translation": {
        "en_to_fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr_to_en": "Helsinki-NLP/opus-mt-fr-en"
    },
    "paraphraser": "tuner007/pegasus_paraphrase",
    "mlm": "bert-base-uncased",
    "typo_noise": {
        "aug_char_min": 1,
        "aug_char_max": 1,
        "aug_char_p": 0.01
    }
}

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)
