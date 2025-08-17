import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# --- Configuration ---
# Use a specific GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Model to be fine-tuned
model_name = "gpt2-large"
# Path to your dataset
csv_path = "final_qa.csv"
# Where the final model will be saved
output_model_path = "./gpt2-large-finetuned-accounting2"


# --- Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the pad token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token
# Ensure padding is done on the right side
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(model_name)

# --- IMPROVEMENT: Enable Gradient Checkpointing ---
#  saves a lot of VRAM, allowing for a larger batch size and faster training.
model.gradient_checkpointing_enable()


# --- Load and Prepare Dataset ---
df = pd.read_csv(csv_path)
df = df.dropna(subset=["question", "answer"])

# --- IMPROVEMENT: Use a more structured format ---
# This helps the model better learn the task structure.
def format_row(row):
    return {"text": f"Question: {row['question']}\nAnswer: {row['answer']}{tokenizer.eos_token}"}

dataset = Dataset.from_list(df.apply(format_row, axis=1).tolist())
dataset = dataset.train_test_split(test_size=0.1)

def tokenize(example):
    # Tokenize the text with padding and truncation
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=512, # Reduced max_length slightly to save memory
        padding="max_length"
    )
    # For language modeling, the labels are the same as the input_ids
    encoded["labels"] = encoded["input_ids"]
    return encoded

# --- FIX: Disable multiprocessing to prevent data corruption errors ---
# The num_proc=1 flag makes the mapping process more stable.
tokenized_dataset = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=["text"],
    num_proc=1 # Use a single process for tokenization
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# --- Set Training Arguments ---
# Check if bfloat16 is supported for better stability
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=output_model_path,
    # --- IMPROVEMENT: Increased batch size for speed ---
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
    # --- IMPROVEMENT: Adjusted epochs and warmup steps ---
    num_train_epochs=5,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    # Evaluation and saving strategies
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2, # Saves disk space
    load_best_model_at_end=True, # Loads the best model after training
    # Logging
    logging_dir="./logs",
    logging_steps=50,
    # --- IMPROVEMENT: Prefer bfloat16 for stability ---
    bf16=bf16_supported, # Use bfloat16 if available
    fp16=not bf16_supported, # Fallback to fp16 if not
    report_to="none",
)

# --- Initialize and Run Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting fine-tuning..")
trainer.train()

# --- Save the Final Model and Tokenizer ---
print("Saving final model and tokenizer")
trainer.save_model(output_model_path)
# model.save_pretrained(output_model_path) # trainer.save_model is preferred
# tokenizer.save_pretrained(output_model_path)

print("Fine-tuning complete")
