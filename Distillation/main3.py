import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, DatasetDict
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, math, getpass
from huggingface_hub import login
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter


# --- 2. Load & prepare the dataset ---
qa_df = pd.read_csv("distillation_02/final_qa.csv")  

def make_qa_prompt(row):
    return f"Question: {row['question']}\nAnswer:"

qa_df["text"] = qa_df.apply(make_qa_prompt, axis=1)
qa_df["target"] = qa_df["answer"]

# Split data (90% train, 5% val, 5% test)
train_df = qa_df.sample(frac=0.7, random_state=42)
remaining = qa_df.drop(train_df.index)
val_df = remaining.sample(frac=0.5, random_state=43)
test_df = remaining.drop(val_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

# --- 3. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("roshan0123/gpt2-large-accounting-finetuned")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    # Tokenize prompt and answer together
    full_text = example["text"] + example["target"]
    encoding = tokenizer(full_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    # Tokenize prompt separately to get its length
    prompt_encoding = tokenizer(example["text"], truncation=True, padding=False)
    prompt_length = len(prompt_encoding["input_ids"])
    # Create answer mask: 0 for prompt tokens, 1 for answer tokens
    answer_mask = torch.zeros(512, dtype=torch.long)
    answer_mask[prompt_length:] = 1  # 1 for answer tokens, 0 for prompt and padding
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "answer_mask": answer_mask
    }

tokenized_datasets = dataset.map(tokenize, batched=False, remove_columns=["text", "question", "answer", "target"])
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "answer_mask"])

# --- 4. Load teacher and student models ---
teacher = AutoModelForCausalLM.from_pretrained("roshan0123/gpt2-large-accounting-finetuned")
student = AutoModelForCausalLM.from_pretrained("distilgpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)

if teacher.config.vocab_size != student.config.vocab_size:
    print(f"Resizing student vocab from {student.config.vocab_size} to {teacher.config.vocab_size}")
    student.resize_token_embeddings(teacher.config.vocab_size)

optimizer = torch.optim.AdamW(student.parameters(), lr=6.795280793326053e-03)  #2e-4# Keeping original LR as requested

# --- 5. DataLoaders ---
batch_size = 32
train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)

# --- 6. Training / distillation loop ---
'''def cosine_temperature(epoch, total_epochs, initial_temp=0.5, final_temp=0.2):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return final_temp + (initial_temp - final_temp) * cosine_decay'''

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_mask = batch['answer_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            answer_mask = answer_mask[:, 1:].contiguous()
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            active_loss = answer_mask.view(-1) == 1
            loss = F.cross_entropy(logits_flat[active_loss], labels_flat[active_loss])
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)

num_epochs = 16
patience = 4
alpha_ce = 0.4664161814771782   #0.3
alpha_kl = 0.20806809509719452  #0.6
accum_steps = 8
save_every = 4
best_val_loss = float("inf")
patience_counter = 0

save_dir_base = "./distilled_checkpoints_distill_gpt2"
os.makedirs(save_dir_base, exist_ok=True)
student.train()
optimizer.zero_grad()
writer = SummaryWriter(log_dir='./logs') 

for epoch in range(num_epochs):
    total_loss = 0.0
    temperature =0.5     #cosine_temperature(epoch, num_epochs)
    print(f"Epoch {epoch + 1}/{num_epochs} - Temperature: {temperature:.2f}")
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_mask = batch['answer_mask'].to(device)

        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Shift logits and labels for Causal LM training
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        answer_mask = answer_mask[:, 1:].contiguous()

        # Flatten logits, labels and mask for loss calculation
        logits_flat = student_logits.view(-1, student_logits.size(-1))
        labels_flat = labels.view(-1)
        active_loss = answer_mask.view(-1) == 1

        # Calculate CE loss only on answer tokens
        loss_ce = F.cross_entropy(logits_flat[active_loss], labels_flat[active_loss])
        
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Reshape the full tensors to be 2D
        student_log_probs_flat = student_log_probs.view(-1, student_logits.size(-1))
        teacher_probs_flat = teacher_probs.view(-1, teacher_logits.size(-1))
        
        loss_kl = F.kl_div(
            input=student_log_probs_flat, # Use unmasked student logits
            target=teacher_probs_flat,    # Use unmasked teacher probabilities
            reduction='batchmean'
        ) * (temperature ** 2)
        # ####################################################################
        
        # Check for NaN losses before backpropagation
        if torch.isnan(loss_ce) or torch.isnan(loss_kl):
            print(f"Warning: NaN loss detected at step {step}. Skipping batch.")
            continue

        # Combine losses and scale for gradient accumulation
        loss = (alpha_ce * loss_ce + alpha_kl * loss_kl) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps # Unscale for logging
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
        writer.add_scalar('Train/Loss', total_loss / (step + 1), epoch * len(train_loader) + step)

    val_loss = evaluate_loss(student, val_loader, device)
    writer.add_scalar('Validation/Loss', val_loss, epoch)  # logging val loss
    print(f"Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print("Validation loss improved. Saving best model.")
        best_model_path = f"{save_dir_base}/best_model"
        student.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Saved best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break
    
    if (epoch + 1) % save_every == 0:
        checkpoint_path = f"{save_dir_base}/epoch_{epoch + 1}"
        student.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

writer.close()
# --- 7. Inference function ---
def infer_answer(question, model, tokenizer, device):
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=120,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,      # Activate sampling
        temperature=0.7,     # Control creativity
        top_k=50,            # Consider top 50 tokens
        top_p=0.95           # Nucleus sampling
    )
    # ####################################################################
    
    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "Answer:" in output_str:
        # Split and take the generated part
        return output_str.split("Answer:", 1)[1].strip()
    else:
        # If the prompt is somehow not in the output, return the whole thing
        return output_str

# --- 8. BLEU Score evaluation ---
# Load the best model for evaluation
print("\nLoading best model for final evaluation...")
best_model_path = f"{save_dir_base}/best_model"
student = AutoModelForCausalLM.from_pretrained(best_model_path)
student.to(device)
student.eval()

# Run example inference
example = qa_df.iloc[0]
print(f"\n--- Example Inference ---")
print(f"QUESTION: {example['question']}")
print("STUDENT MODEL ANSWER:", infer_answer(example['question'], student, tokenizer, device))
print("-------------------------\n")


# Evaluate on the validation set
print("Evaluating on the validation set using BLEU score...")
nltk.download('punkt', quiet=True)
smooth = SmoothingFunction().method4

def compute_bleu(reference, candidate):
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    if not cand_tokens: # Handle empty predictions
        return 0.0
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)

references = val_df['answer'].tolist()
bleu_scores = []

# Use tqdm for progress tracking on evaluation
for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc="Validation BLEU"):
    pred = infer_answer(row['question'], student, tokenizer, device)
    ref = row['answer']
    bleu_scores.append(compute_bleu(ref, pred))

print(f"\nAverage Validation BLEU Score: {sum(bleu_scores)/len(bleu_scores):.4f}")
