from tqdm import tqdm
<<<<<<< HEAD
from model_loader import load_model
=======
from SOAP.model_loader import load_model
>>>>>>> origin/complete_pipeline

def llm_validate_outliers(df, outlier_indices, prompt_template, model_name, device_index):
    llm = load_model(model_name=model_name, device_index=device_index)
    llm_validation = [""] * len(df)  # Pre-fill with empty strings

    for i in tqdm(outlier_indices):
<<<<<<< HEAD
        q, a = df.iloc[i]['question'], df.iloc[i]['answer']
=======
        q, a = df.iloc[i]['Question'], df.iloc[i]['Answer']
>>>>>>> origin/complete_pipeline
        prompt = prompt_template.format(question=q.strip(), answer=a.strip())

        res = llm(prompt, max_new_tokens=40, temperature=0.3, do_sample=False)[0]['generated_text']
        response = res.strip().lower()

        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            reply = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            reply = response

        if reply.startswith("no"):
            validated = "No"
        elif reply.startswith("yes"):
            validated = "Yes"
        else:
            validated = response

        llm_validation[i] = validated

    return llm_validation
