# run_pipeline.py

import logging
import yaml
from dotenv import load_dotenv
import os

# === Load environment ===
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load config ===
with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# === Import your modules ===
from preprocessing import preprocessing as Preprocessing
from augmentation.augmentation import Augmentation
from validation import run_validation_pipeline

def run_pipeline():
    logger.info("=== Starting Full QA Processing Pipeline ===")

    # === 1 Preprocessing ===
    logger.info("Step 1: Running Preprocessing...")
    input_path = CONFIG['paths']['input_path']
    cleaned_path = CONFIG['paths']['cleaned_path']

    preproc = Preprocessing(input_path=input_path, output_path=cleaned_path)
    preproc.run_all()

    logger.info(f"Preprocessing complete. Cleaned data: {cleaned_path}")

    # === 2 Augmentation ===
    logger.info("Step 2: Running Augmentation...")
    aug_output_path = CONFIG['paths']['augmented_path']

    aug = Augmentation(
        input_path=cleaned_path,
        output_path=aug_output_path,
        device='cuda:3'  # or adjust device
    )
    aug.run_all(columns=['Question', 'Answer'])

    logger.info(f"Augmentation complete. Augmented data: {aug_output_path}")

    # === 3 Validation ===
    logger.info("Step 3: Running Validation...")
    run_validation_pipeline()

    logger.info("Full pipeline complete. Final output in: data/outputs/final_validated_output.csv")


if __name__ == "__main__":
    run_pipeline()
