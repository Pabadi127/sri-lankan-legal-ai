"""
train_siamese.py
Trains the Siamese Neural Network on 3,000 Sri Lankan criminal case-fact contrastive pairs
and precomputes dense 128-dimensional semantic embeddings for the entire corpus.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from baseline_tfidf import tokenize
from siamese_model import SiameseProjectionNetwork, EMBEDDINGS_MODEL_PATH, CLEANED_DATA_PATH

PAIRS_DATA_PATH = "siamese_training_pairs.csv"
INPUT_VOCAB_SIZE = 4000

def build_vocabulary(cases_df, max_vocab=INPUT_VOCAB_SIZE):
    """Builds token-to-index mapping from all case texts."""
    print(f"[*] Building vocabulary from {len(cases_df)} cases...", flush=True)
    counter = Counter()
    for _, row in cases_df.iterrows():
        text = f"{row['Legal Topic / Law']} {row['Case Facts / Issue']} {row['Judgment / Holding'][:300]}"
        tokens = tokenize(text)
        counter.update(tokens)
        
    top_tokens = [tok for tok, _ in counter.most_common(max_vocab)]
    vocab = {tok: idx for idx, tok in enumerate(top_tokens)}
    print(f"[+] Vocabulary created with {len(vocab)} unique legal features.", flush=True)
    return vocab

def text_to_bow(text, vocab):
    """Converts a string to an L2-normalized bag-of-words vector."""
    tokens = tokenize(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec

def train_siamese_network(epochs=15, batch_size=64, learning_rate=0.002):
    print("=" * 65, flush=True)
    print("[*] Training Siamese Neural Network on Legal Contrastive Pairs", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(PAIRS_DATA_PATH):
        raise FileNotFoundError(f"Pairs file {PAIRS_DATA_PATH} not found. Run create_siamese_pairs.py first.")

    cases_df = pd.read_csv(CLEANED_DATA_PATH, encoding='utf-8')
    pairs_df = pd.read_csv(PAIRS_DATA_PATH, encoding='utf-8')
    print(f"[*] Loaded {len(pairs_df)} contrastive pairs for training.", flush=True)

    vocab = build_vocabulary(cases_df, max_vocab=INPUT_VOCAB_SIZE)

    # Vectorize training pairs
    print("[*] Vectorizing training pairs into input matrices...", flush=True)
    N = len(pairs_df)
    X1 = np.zeros((N, len(vocab)), dtype=np.float32)
    X2 = np.zeros((N, len(vocab)), dtype=np.float32)
    Y = pairs_df["label"].values.astype(np.float32)

    for i, row in pairs_df.iterrows():
        X1[i] = text_to_bow(f"{row['category_a']} {row['case_a_facts']}", vocab)
        X2[i] = text_to_bow(f"{row['category_b']} {row['case_b_facts']}", vocab)

    # Initialize Siamese Twin Network
    network = SiameseProjectionNetwork(input_dim=len(vocab), hidden_dim=512, embed_dim=128)

    print(f"\n[*] Starting Mini-Batch Optimization (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate})...", flush=True)
    start_time = time.time()

    indices = np.arange(N)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []
        
        for start_idx in range(0, N, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            b_X1 = X1[batch_idx]
            b_X2 = X2[batch_idx]
            b_Y = Y[batch_idx]
            
            loss = network.train_step(b_X1, b_X2, b_Y, margin=1.0, lr=learning_rate)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        print(f"    Epoch [{epoch:2d}/{epochs:2d}] - Contrastive Loss: {avg_loss:.5f}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[+] Optimization complete in {elapsed:.2f}s!", flush=True)

    # Precompute dense embeddings for all 771 cases in the corpus
    print(f"\n[*] Precomputing 128-dimensional dense embeddings for all {len(cases_df)} cases...", flush=True)
    corpus_X = np.zeros((len(cases_df), len(vocab)), dtype=np.float32)
    for idx, row in cases_df.iterrows():
        text = f"{row['Legal Topic / Law']} {row['Legal Topic / Law']} {row['Case Facts / Issue']} {row['Judgment / Holding'][:300]}"
        corpus_X[idx] = text_to_bow(text, vocab)

    corpus_embeddings = network.encode(corpus_X)
    print(f"[+] Corpus Embeddings Shape: {corpus_embeddings.shape}", flush=True)

    # Save artifacts
    save_data = {
        "network": network,
        "vocab": vocab,
        "corpus_embeddings": corpus_embeddings
    }
    with open(EMBEDDINGS_MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
        
    print(f"[+] Siamese Network and Corpus Embeddings saved to: {EMBEDDINGS_MODEL_PATH}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    train_siamese_network(epochs=15, batch_size=64, learning_rate=0.002)
