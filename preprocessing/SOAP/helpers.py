import hdbscan
from sentence_transformers import SentenceTransformer
import pandas as pd

class OutlierDetector:
    def __init__(self, min_cluster_size=6, min_samples=5, threshold=0.3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.threshold = threshold
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True)

    def embed_texts(self, texts):
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = st_model.encode(texts, show_progress_bar=True)
        return embeddings

    def detect_outliers(self, embeddings):
        cluster_labels = self.clusterer.fit_predict(embeddings)
        outlier_scores = self.clusterer.outlier_scores_
        detected_outliers = outlier_scores > self.threshold
        return cluster_labels, outlier_scores, detected_outliers

def rule_based_outlier_check(df):
    answer_keywords = ["chapter", "?", "syllabus", "unit", "paragraph", "para", "case", "act"]
    question_keywords = ["chapter","chapter?", "syllabus", "unit", "paragraph", "para", "case", "act"]

    rule_outliers = []

    for i in range(len(df)):

        question_words = df.iloc[i]['question'].lower().split()
        answer_words = df.iloc[i]['answer'].lower().split()
        question_words = df.iloc[i]['Question'].lower().split()
        answer_words = df.iloc[i]['Answer'].lower().split()

        question_has_noise = any(k in question_words for k in question_keywords)
        answer_has_noise = any(k in answer_words for k in answer_keywords)

        rule_outliers.append(question_has_noise or answer_has_noise)

    return rule_outliers
