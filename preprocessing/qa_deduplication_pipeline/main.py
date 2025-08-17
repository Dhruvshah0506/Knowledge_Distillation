from qa_deduplication_pipeline.llm_qa_refiner import run_llm_qa_refinement

def main():
    run_llm_qa_refinement(
        input_path="Projects/qa_cleaning__and_aug/data/input/input_qa.csv",
        output_path="Projects/qa_cleaning__and_aug/data/processed_dataset/cleaned_qa.csv",
        hf_token="your hf token here",  
        mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
        qwen_model_id="Qwen/Qwen-7B-Chat",
        cuda_device="cuda:3",
        only_mistral_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_mistral_rephrased.csv",
        only_qwen_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_qwen_rephrased.csv",
        mistral_output="Projects/qa_cleaning__and_aug/content/outputs/mistral_output.csv",
        exact_duplicate_path="Content/Duplicates_Count/exact_duplicate_qa_pairs.csv",
        same_ans_diff_ques_path="Content/Duplicates_Count/same_answer_diff_questions.csv",
        same_ques_diff_ans_path="Content/Duplicates_Count/same_question_diff_answers.csv"
    )

if __name__ == "__main__":
    main()
