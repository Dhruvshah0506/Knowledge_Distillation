import sys
sys.path.append("Projects/qa_cleaning__and_aug/preprocessing")
#sys.path.append("Projects/qa_cleaning__and_aug/preprocessing")
from SOAP.semantic_outlier_filter import semantic_outlier_filter
from qa_deduplication_pipeline.llm_qa_refiner import run_llm_qa_refinement

if __name__ == "__main__":
    print("\n #### starting SOAP #### \n")
    semantic_outlier_filter(
        input_csv_path="Projects/qa_cleaning__and_aug/data/processed_dataset/cleaned_augmented_data.csv",
        output_csv_path="Projects/qa_cleaning__and_aug/data/processed_dataset/final_qa.csv",
        discarded_output_path="Projects/qa_cleaning__and_aug/content/outputs/discarded_rows_from_final.csv"
    )
    print("\n #### starting dedpulication and cleaning #### \n")

    run_llm_qa_refinement(
        input_path="Projects/qa_cleaning__and_aug/data/processed_dataset/final_qa.csv",
        output_path="Projects/qa_cleaning__and_aug/data/processed_dataset/final_qa.csv",
        hf_token="your hf tokens here",
        mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
        qwen_model_id="Qwen/Qwen-7B-Chat",
        cuda_device="cuda:3",
        only_mistral_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_mistral_rephrased_01.csv",
        only_qwen_rephrased="Projects/qa_cleaning__and_aug/content/outputs/only_qwen_rephrased_01.csv",
        mistral_output="Projects/qa_cleaning__and_aug/content/outputs/rephrased_mistral_output_01.csv"
    )

    print("\n\n Validation Complete ")
