# constants.py

# Paths
INPUT_FILE = "Content/Dataset/input_dataset.csv"
FINAL_OUTPUT = "Content/Dataset/Final_Dataset.csv"
MISTRAL_OUTPUT = "Content/Phase_Wise_Output/rephrased_mistral_output.csv"
ONLY_MISTRAL_REPHRASED = "Content/Phase_Wise_Output/only_mistral_rephrased.csv"
ONLY_QWEN_REPHRASED = "Content/Phase_Wise_Output/only_qwen_rephrased.csv"

# HuggingFace login token
HF_TOKEN = "ENTER YOUR TOKEN"

# Models
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
QWEN_MODEL_ID = "Qwen/Qwen-7B-Chat"

# Device Settings
CUDA_DEVICE = "cuda:3"
