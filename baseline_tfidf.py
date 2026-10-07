"""
baseline_tfidf.py
Baseline Legal Case Retrieval System using pure Python/NumPy TF-IDF and Cosine Similarity.
Self-contained, robust against Windows AppLocker policies, and highly performant.
"""

import os
import re
import math
import pickle
import pandas as pd
import numpy as np
from collections import Counter

CLEANED_DATA_PATH = "cleaned_criminal_cases.csv"
MODEL_PATH = "tfidf_baseline_model.pkl"

# Standard English stopwords
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", 
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

def tokenize(text: str):
    """Tokenizes text into words and generates unigrams and bigrams."""
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    clean_words = [w for w in words if w not in STOP_WORDS]
    
    tokens = list(clean_words)
    # Add adjacent bigrams for legal phrases
    for i in range(len(clean_words) - 1):
        tokens.append(f"{clean_words[i]}_{clean_words[i+1]}")
    return tokens

class NumpyTfidfVectorizer:
    """Pure NumPy TF-IDF Vectorizer with sublinear term-frequency and smooth IDF."""
    def __init__(self, max_features=5000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}          # token -> index
        self.idf = None          # ndarray of shape (vocab_size,)
        self.vocab_size = 0

    def fit(self, raw_documents):
        doc_count = len(raw_documents)
        df_counter = Counter()
        
        doc_tokens_list = []
        for doc in raw_documents:
            tokens = set(tokenize(doc))
            df_counter.update(tokens)
            doc_tokens_list.append(tokens)

        # Filter by min_df
        valid_tokens = [tok for tok, freq in df_counter.items() if freq >= self.min_df]
        
        # Sort by frequency and cap at max_features
        valid_tokens.sort(key=lambda tok: df_counter[tok], reverse=True)
        top_tokens = valid_tokens[:self.max_features]
        
        self.vocab = {tok: idx for idx, tok in enumerate(top_tokens)}
        self.vocab_size = len(self.vocab)
        
        # Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        self.idf = np.zeros(self.vocab_size, dtype=np.float32)
        for tok, idx in self.vocab.items():
            df = df_counter[tok]
            self.idf[idx] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0

        return self

    def transform(self, raw_documents):
        rows = len(raw_documents)
        matrix = np.zeros((rows, self.vocab_size), dtype=np.float32)
        
        for r_idx, doc in enumerate(raw_documents):
            tokens = tokenize(doc)
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                if tok in self.vocab:
                    col_idx = self.vocab[tok]
                    # Sublinear TF: 1 + log(cnt)
                    tf = 1.0 + math.log(cnt)
                    matrix[r_idx, col_idx] = tf * self.idf[col_idx]
                    
            # L2 normalization
            norm = np.linalg.norm(matrix[r_idx])
            if norm > 1e-8:
                matrix[r_idx] /= norm
                
        return matrix

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

class TfidfCaseRecommender:
    def __init__(self, data_path=CLEANED_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Corpus file {self.data_path} not found.")
        
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        
        if os.path.exists(MODEL_PATH):
            print(f"[*] Loading cached TF-IDF model from {MODEL_PATH}...", flush=True)
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = NumpyTfidfVectorizer(
                    max_features=data.get("max_features", 6000),
                    min_df=data.get("min_df", 2)
                )
                self.vectorizer.vocab = data["vocab"]
                self.vectorizer.idf = data["idf"]
                self.vectorizer.vocab_size = len(self.vectorizer.vocab)
                self.tfidf_matrix = data["matrix"]
        else:
            self.fit_and_save()

    def _build_corpus_texts(self):
        texts = []
        for _, row in self.df.iterrows():
            facts = str(row.get("Case Facts / Issue", ""))
            topic = str(row.get("Legal Topic / Law", ""))
            holding = str(row.get("Judgment / Holding", ""))
            combined = f"{topic} {topic} {facts} {holding[:400]}"
            texts.append(combined)
        return texts

    def fit_and_save(self):
        print(f"[*] Fitting pure NumPy TF-IDF Vectorizer on {len(self.df)} Sri Lankan criminal cases...", flush=True)
        texts = self._build_corpus_texts()
        self.vectorizer = NumpyTfidfVectorizer(max_features=6000, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "vocab": self.vectorizer.vocab,
                "idf": self.vectorizer.idf,
                "matrix": self.tfidf_matrix,
                "max_features": self.vectorizer.max_features,
                "min_df": self.vectorizer.min_df
            }, f)
            
        print(f"[+] TF-IDF Model saved to {MODEL_PATH} (Vocabulary Size: {self.vectorizer.vocab_size})", flush=True)

    def search(self, query: str, top_k: int = 5, category_filter: str = None) -> list:
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-8:
            return []

        # Cosine similarity: query_vec @ matrix.T (since matrix rows are L2 normalized)
        sim_scores = np.dot(self.tfidf_matrix, query_vec)

        if category_filter and category_filter != "All Topics":
            valid_indices = self.df[self.df["Legal Topic / Law"] == category_filter].index
            mask = np.zeros_like(sim_scores, dtype=bool)
            mask[valid_indices] = True
            sim_scores[~mask] = -1.0

        top_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            if score <= 0.0 or len(results) >= top_k:
                break
                
            row = self.df.iloc[idx]
            results.append({
                "rank": len(results) + 1,
                "dataset_id": row["Dataset ID"],
                "case_name": row["Case Name"],
                "court_no": row["Court Case No."],
                "year": row["Year"],
                "reporter": row["Reporter"],
                "legal_topic": row["Legal Topic / Law"],
                "similarity_score": round(score, 4),
                "similarity_pct": f"{round(score * 100, 1)}%",
                "case_facts": row["Case Facts / Issue"],
                "holding": row["Judgment / Holding"],
                "source_url": row["Source URL"]
            })

        return results

if __name__ == "__main__":
    recommender = TfidfCaseRecommender()
    test_query = "Accused was found carrying heroin in a hidden compartment of suitcase at airport without mens rea"
    print(f"\n--- Testing Baseline TF-IDF with query: '{test_query}' ---", flush=True)
    matches = recommender.search(test_query, top_k=3)
    for m in matches:
        print(f"\n[Rank {m['rank']}] ({m['similarity_pct']} match) - {m['case_name']} [{m['year']}]", flush=True)
        print(f"    Topic: {m['legal_topic']} | Court: {m['court_no']}", flush=True)
        print(f"    Facts: {m['case_facts'][:180]}...", flush=True)
        print(f"    Holding: {m['holding'][:120]}...", flush=True)
