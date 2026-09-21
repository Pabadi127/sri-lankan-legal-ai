"""
siamese_model.py
Pure NumPy implementation of the Siamese Neural Network for Legal Semantic Similarity.
Implements the exact twin-network architecture specified in Group 14's Project Proposal:
- Twin Subnetworks with Shared Weights
- Non-linear Dense Hidden Layers (4000 -> 512 -> 128)
- Contrastive Loss with Margin
- Adam Optimizer with Momentum
- L2-Normalized Metric Space for Cosine Similarity Retrieval
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from baseline_tfidf import tokenize, STOP_WORDS

EMBEDDINGS_MODEL_PATH = "siamese_case_embeddings.pkl"
CLEANED_DATA_PATH = "cleaned_criminal_cases.csv"

class SiameseProjectionNetwork:
    """
    Twin Neural Network with shared weights for mapping legal case facts
    into a dense 128-dimensional semantic embedding space.
    """
    def __init__(self, input_dim=4000, hidden_dim=512, embed_dim=128, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # He / Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

        # Adam optimizer state
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0

    def forward(self, X):
        """Forward pass: Dense -> ReLU -> Dense -> L2 Normalization"""
        H = np.dot(X, self.W1) + self.b1
        # ReLU activation
        H_relu = np.maximum(0, H)
        
        Z_unnorm = np.dot(H_relu, self.W2) + self.b2
        
        # L2 normalize each row to unit sphere
        norms = np.linalg.norm(Z_unnorm, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        Z = Z_unnorm / norms
        
        return H, H_relu, Z_unnorm, Z

    def encode(self, X):
        """Produces unit-normalized embeddings for input vectors."""
        _, _, _, Z = self.forward(X)
        return Z

    def train_step(self, X1, X2, Y, margin=1.0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Backpropagation for Siamese network with Contrastive Loss:
        Loss = Y * 0.5 * D^2 + (1 - Y) * 0.5 * max(0, margin - D)^2
        where D = ||Z1 - Z2||_2
        """
        batch_size = X1.shape[0]
        
        # Forward pass for both branches (shared weights)
        H1, H1_relu, Z1_raw, Z1 = self.forward(X1)
        H2, H2_relu, Z2_raw, Z2 = self.forward(X2)
        
        # Compute Euclidean distance D in normalized space
        diff = Z1 - Z2
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1e-8)
        
        # Contrastive loss computation
        loss_pos = Y * 0.5 * (dist ** 2)
        margin_diff = np.maximum(0.0, margin - dist)
        loss_neg = (1.0 - Y) * 0.5 * (margin_diff ** 2)
        loss = np.mean(loss_pos + loss_neg)
        
        # Gradient of loss w.r.t distance D
        dLoss_dDist = Y * dist - (1.0 - Y) * margin_diff
        
        # Gradient w.r.t Z1 and Z2
        dLoss_dZ1 = (dLoss_dDist[:, None] / dist[:, None]) * diff / batch_size
        dLoss_dZ2 = -dLoss_dZ1
        
        # Backprop through L2 norm for branch 1
        dLoss_dZ1_raw = self._grad_l2_norm(dLoss_dZ1, Z1_raw)
        dLoss_dZ2_raw = self._grad_l2_norm(dLoss_dZ2, Z2_raw)
        
        # Gradients for W2, b2
        grad_W2 = np.dot(H1_relu.T, dLoss_dZ1_raw) + np.dot(H2_relu.T, dLoss_dZ2_raw)
        grad_b2 = np.sum(dLoss_dZ1_raw + dLoss_dZ2_raw, axis=0)
        
        # Backprop to H_relu
        dH1_relu = np.dot(dLoss_dZ1_raw, self.W2.T)
        dH2_relu = np.dot(dLoss_dZ2_raw, self.W2.T)
        
        # Backprop through ReLU
        dH1 = dH1_relu * (H1 > 0)
        dH2 = dH2_relu * (H2 > 0)
        
        # Gradients for W1, b1
        grad_W1 = np.dot(X1.T, dH1) + np.dot(X2.T, dH2)
        grad_b1 = np.sum(dH1 + dH2, axis=0)
        
        # Adam parameter update
        self.t += 1
        for param, grad, m, v in [
            (self.W1, grad_W1, self.mW1, self.vW1),
            (self.b1, grad_b1, self.mb1, self.vb1),
            (self.W2, grad_W2, self.mW2, self.vW2),
            (self.b2, grad_b2, self.mb2, self.vb2)
        ]:
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** self.t)
            v_hat = v / (1.0 - beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
        return float(loss)

    def _grad_l2_norm(self, dZ, Z_raw):
        """Gradient of unit normalization operation Z = Z_raw / ||Z_raw||."""
        norm = np.linalg.norm(Z_raw, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        norm3 = norm ** 3
        dot = np.sum(dZ * Z_raw, axis=-1, keepdims=True)
        return (dZ / norm) - (Z_raw * dot / norm3)

class SiameseCaseRecommender:
    """
    High-level Inference and Case Retrieval engine using the trained Siamese Network.
    """
    def __init__(self, model_file=EMBEDDINGS_MODEL_PATH, data_file=CLEANED_DATA_PATH):
        self.model_file = model_file
        self.data_file = data_file
        self.df = None
        self.vocab = None
        self.network = None
        self.corpus_embeddings = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file {self.model_file} not found. Run train_siamese.py first.")
        
        print(f"[*] Loading trained Siamese Neural Network from {self.model_file}...", flush=True)
        with open(self.model_file, "rb") as f:
            data = pickle.load(f)
            self.network = data["network"]
            self.vocab = data["vocab"]
            self.corpus_embeddings = data["corpus_embeddings"]
            
        self.df = pd.read_csv(self.data_file, encoding='utf-8')
        print(f"[+] Loaded Siamese model with {len(self.corpus_embeddings)} indexed case embeddings (128-dim).", flush=True)

    def _text_to_vector(self, text: str):
        """Encodes text into normalized input feature vector for the network."""
        tokens = tokenize(text)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def search(self, query: str, top_k: int = 5, category_filter: str = None) -> list:
        if not query or not query.strip():
            return []

        q_vec = self._text_to_vector(query)[None, :]
        query_embed = self.network.encode(q_vec)[0]  # (128,)

        # Cosine similarity in normalized embedding space: dot product
        sim_scores = np.dot(self.corpus_embeddings, query_embed)

        if category_filter and category_filter != "All Topics":
            valid_indices = self.df[self.df["Legal Topic / Law"] == category_filter].index
            mask = np.zeros_like(sim_scores, dtype=bool)
            mask[valid_indices] = True
            sim_scores[~mask] = -1.0

        top_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            # Scale score to [0, 1] range for intuitive presentation
            # (since unit vector cosine ranges from -1 to 1, but legal similarity is >= 0)
            norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0))
            
            if len(results) >= top_k:
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
                "similarity_score": round(norm_score, 4),
                "similarity_pct": f"{round(norm_score * 100, 1)}%",
                "raw_cosine": round(score, 4),
                "case_facts": row["Case Facts / Issue"],
                "holding": row["Judgment / Holding"],
                "source_url": row["Source URL"]
            })

        return results
