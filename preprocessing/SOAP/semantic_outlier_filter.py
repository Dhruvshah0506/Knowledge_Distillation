import pandas as pd

from SOAP.helpers import OutlierDetector, rule_based_outlier_check
from SOAP.llm_vallidation import llm_validate_outliers
from SOAP.prompt import prompt_template


def semantic_outlier_filter(
    input_csv_path: str,
    output_csv_path: str, 
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_index: int = 3
):
    print(f"Loading dataset from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)
    df.dropna(subset=['question', 'answer'], inplace=True)
    qa_texts = [f"Question: {q.strip()} Answer: {a.strip()}" for q, a in zip(df['question'], df['answer'])
                

    # Hardcoded output paths
    final_output_path = "/home/shah.dhruv/Data_Augmentation/Internship-2025-Team2/Projects/qa_cleaning__and_aug/content/outputs/SOAP_QA.csv"
    discarded_output_path = "/home/shah.dhruv/Data_Augmentation/Internship-2025-Team2/Projects/qa_cleaning__and_aug/content/outputs/discarded_rows.csv"

    print(f"Loading dataset from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)
    df.dropna(subset=['Question', 'Answer'], inplace=True)
    qa_texts = [f"Question: {q.strip()} Answer: {a.strip()}" for q, a in zip(df['Question'], df['Answer'])]


    print("Generating sentence embeddings using all-MiniLM-L6-v2...")
    outlier_detector = OutlierDetector()
    embeddings = outlier_detector.embed_texts(qa_texts)

    print("Running HDBSCAN for clustering and semantic outlier detection...")
    cluster_labels, outlier_scores, detected_outliers = outlier_detector.detect_outliers(embeddings)

    print("Applying rule-based heuristics for common noise patterns...")
    rule_outliers = rule_based_outlier_check(df)

    outlier_indices = [i for i, (hdb, rule) in enumerate(zip(detected_outliers, rule_outliers)) if hdb or rule]

    print("Loading LLaMA for contextual verification of outliers...")
    llm_validation = llm_validate_outliers(df, outlier_indices, prompt_template, model_name, gpu_index)

    df['cluster_label'] = cluster_labels
    df['hdbscan_outlier_score'] = outlier_scores
    df['is_outlier'] = detected_outliers
    df['rule_outlier'] = rule_outliers
    df['Domain_related'] = llm_validation

    df['Domain_related'] = df['Domain_related'].str.strip().str.capitalize()

    keep_mask = ~(
        ((df['rule_outlier']) & (df['Domain_related'] == 'No')) |
        ((~df['rule_outlier']) & (df['is_outlier']) & (df['Domain_related'] == 'No'))
    )

    preprocessed_df = df.loc[keep_mask, ['question', 'answer']]
    preprocessed_df.to_csv(output_clean_csv_path, index=False)

    discarded_df = df.loc[~keep_mask, ['question', 'answer']]
    discarded_df.to_csv(output_discarded_csv_path, index=False)

    preprocessed_df = df.loc[keep_mask, ['Question', 'Answer']]
    discarded_df = df.loc[~keep_mask, ['Question', 'Answer']]

    # Save to both the dynamic output path and the hardcoded path
    preprocessed_df.to_csv(output_csv_path, index=False)
    preprocessed_df.to_csv(final_output_path, index=False)
    discarded_df.to_csv(discarded_output_path, index=False)

    rows_before = len(df)
    rows_after = len(preprocessed_df)
    rows_dropped = rows_before - rows_after
    print(f"\n🗑️ Dropped {rows_dropped} rows during semantic outlier filtering.")

    print(f"✅ Cleaned QA pairs saved to '{output_clean_csv_path}'")
    print(f"✅ Discarded rows saved to '{output_discarded_csv_path}'")
    print(f"✅ Cleaned QA pairs saved to '{output_csv_path}' and '{final_output_path}'")
    print(f"✅ Discarded rows saved to '{discarded_output_path}'")

