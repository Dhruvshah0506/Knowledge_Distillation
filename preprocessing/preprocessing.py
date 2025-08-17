import pandas as pd
import re
import unicodedata
import logging
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from dotenv import load_dotenv
import yaml

from constants.constants import (
    PROMPT_TEMPLATES,
    REGEX_PATTERNS,
    DEFAULT_MAX_NEW_TOKENS
)

from mcq_converter import create_mcqs_questions
from fitb_converter import create_fitb_questions

# === Setup ===
load_dotenv()
with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

print("starting.....")

class preprocessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

        # === LLM Config for FITB ===
        model_id = CONFIG['models']['fitb_model_id']
        self.max_new_tokens = CONFIG['models'].get('fitb_max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        self.temperature = CONFIG['models'].get('fitb_temperature', 0.7)

        print("loading flan-t5")

        # === Load FLAN-T5 ===
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id).to("cuda:3")
        self.rephrase_pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=3 
        )

        print("loading data started")

    def validate_dataset(self, input_path):
        """Check if the dataset exists and contains the required columns."""
        if not os.path.exists(input_path):
            print("Dataset not found at the specified path. Please check the input path.")
            return False
        try:
            df = pd.read_csv(input_path)
            required_columns = ["Question", "Answer"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print("Dataset does not contain the required columns for processing. Missing columns:", missing_columns)
                return False
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return False
        return True

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        logger.info(f"[Preprocessing] Loaded data: {len(self.df)} rows")

    def save_data(self):
        self.df.to_csv(self.output_path, index=False)
        logger.info(f"[Preprocessing] Saved cleaned data to: {self.output_path}")

    def run_all(self):
        """Run the full preprocessing pipeline."""
        print("=== Starting preprocessing pipeline ===")
        
        # Validate dataset before any processing
        if not self.validate_dataset(self.input_path):
            print("Preprocessing pipeline aborted due to dataset issues.")
            return

        self.load_data()
        print("1. Data loaded.")
        self.drop_invalid_rows()
        print("2. Invalid rows dropped.")
        self.clean_symbols_unicode(columns=clean_columns)
        print("3. Symbols and unicode cleaned.")
        self.convert_fitb()
        print("4. Fill-in-the-blank questions converted.")
        self.convert_mcqs()
        print("5. MCQ questions converted.")
        self.save_data()
        print("6. Data saved after conversion steps.")

        # Semantic Outlier Filtering
        print("\n=== Starting semantic outlier filtering ===")
        from SOAP.semantic_outlier_filter import semantic_outlier_filter
        semantic_outlier_filter(
            input_csv_path=self.output_path,
            output_csv_path=self.output_path, 
        )
        print("7. Semantic outlier filtering complete.")

        print("\n\n=== starting duplication_rephrase===")
        from qa_deduplication_pipeline.llm_qa_refiner import run_llm_qa_refinement
        run_llm_qa_refinement(
            input_path=self.output_path,
            output_path=self.output_path,
            hf_token="your hf token here",
            mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
            qwen_model_id="Qwen/Qwen-7B-Chat",
            cuda_device="cuda:3",
            only_mistral_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_mistral_rephrased.csv",
            only_qwen_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_qwen_rephrased.csv",
            mistral_output="Projects/qa_cleaning__and_aug/content/outputs/rephrased_mistral_output.csv",
            exact_duplicate_path="Content/Duplicates_Count/exact_duplicate_qa_pairs.csv",
            same_ans_diff_ques_path="Content/Duplicates_Count/same_answer_diff_questions.csv",
            same_ques_diff_ans_path="Content/Duplicates_Count/same_question_diff_answers.csv"
        )

    def run_module(self, module_name, **kwargs):
        """Run a single module with custom params."""
        if self.df is None:
            self.load_data()
        func = getattr(self, f"{module_name}", None)
        if not func:
            raise ValueError(f"Module '{module_name}' not found.")
        func(**kwargs)
        self.save_data()

    # === Core steps ===

    def drop_invalid_rows(self):
        self.df['Classification'] = self.df['Classification'].astype(str).str.lower()
        self.df['Correction_Question'] = (
            self.df['Correction_Question'].fillna('').astype(str).str.lower()
        )
        mask = (self.df['Classification'] == 'no') | (self.df['Correction_Question'] == 'invalid')
        before = len(self.df)
        self.df = self.df[~mask].reset_index(drop=True)
        after = len(self.df)
        logger.info(f"[Preprocessing] Dropped invalid rows: {before - after} removed")

    def clean_symbols_unicode(self, columns=None):
        def clean_text(text):
            if pd.isnull(text):
                return text
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
            text = re.sub(r"[^\w\s.,;:!?\"'()\-]", '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        if columns is None:
            columns = ['Question', 'Answer']

        for col in columns:
            self.df[col] = self.df[col].astype(str).apply(clean_text)

        logger.info(f"[Preprocessing] Cleaned symbols/unicode for columns: {columns}")

    def convert_mcqs(self):
        self.df = create_mcqs_questions(self.df)
        MCQ_hardcoded_path= (
            "Projects/"
            "qa_cleaning__and_aug/content/outputs/converted_mcq.csv"
        )
        self.df.to_csv(MCQ_hardcoded_path, index=False)
        logger.info(f"[Preprocessing] Saved converted_mcq data to: {MCQ_hardcoded_path}")

    def convert_fitb(self):
        self.df = create_fitb_questions(self.df)
        fitb_hardcoded_path = (
            "Projects/"
            "qa_cleaning__and_aug/content/outputs/fitb_rephrased.csv"
        )
        self.df.to_csv(fitb_hardcoded_path, index=False)
        logger.info(f"[Preprocessing] Saved FITB-converted data to: {fitb_hardcoded_path}")

if __name__ == "__main__":
    # Load config
    with open("config/config.yaml") as f:
        CONFIG = yaml.safe_load(f)
    
    # Get paths from config
    input_path = CONFIG['paths']['input_path']
    output_path = CONFIG['paths']['cleaned_path']
    
    # Create and run preprocessing pipeline
    processor = preprocessing(input_path, output_path)
    processor.run_all()
