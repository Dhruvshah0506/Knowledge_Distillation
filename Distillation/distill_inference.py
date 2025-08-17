import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
teacher_path = "roshan0123/gpt2-large-accounting-finetuned"
distilled_path = "ShahDhruv/distillgpt2_accountant" #"/home/shah.dhruv/distillation_02/distilled_checkpoints_distill_gpt2/epoch_9"                           #"ShahDhruv/accountant_distillgpt2"
student_model_name = "distilgpt2"
# Load models and tokenizers with error handling
try:
    teacher_tok = AutoTokenizer.from_pretrained(teacher_path)
    teacher_mod = AutoModelForCausalLM.from_pretrained(teacher_path).to(device)
except Exception as e:
    print(f"Error loading teacher model: {e}")
    exit()

try:
    student_tok = AutoTokenizer.from_pretrained(student_model_name)
    student_mod = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
except Exception as e:
    print(f"Error loading student model: {e}")
    exit()

try:
    distilled_tok = AutoTokenizer.from_pretrained(distilled_path)
    distilled_mod = AutoModelForCausalLM.from_pretrained(distilled_path).to(device)
except Exception as e:
    print(f"Error loading distilled model: {e}")
    exit()

# Set pad tokens
teacher_tok.pad_token = teacher_tok.eos_token
student_tok.pad_token = student_tok.eos_token
distilled_tok.pad_token = distilled_tok.eos_token

# Test questions
questions = [
    #"What is the difference between accounts payable and accounts receivable?",
    "Explain the concept of double-entry bookkeeping.",
    "How is the merchandise expense calculated for a retailer?",
    "How do business cycles impact business strategy?",
    "What do you mean by accounting?"
]


def generate_response(model, tokenizer, question):
    try:
        inputs = tokenizer(question, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.4,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.2,
            min_length = 50,
            length_penalty = 1.2
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Generation error: {e}"

# Inference loop
for q in questions:
    print("\n\n=================================================================================")
    print(f"\nQ: {q}")
    print(f"Teacher (gpt2-large):\n{generate_response(teacher_mod, teacher_tok, q)}\n")
    print(f"Original Student (distilgpt2):\n{generate_response(student_mod, student_tok, q)}\n")
    print(f">> Distilled Student:\n{generate_response(distilled_mod, distilled_tok, q)}\n")