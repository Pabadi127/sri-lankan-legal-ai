"""
create_siamese_pairs.py
Generates balanced positive (label=1.0) and negative (label=0.0) case-fact pairs
from the Sri Lankan Criminal Cases Corpus for Siamese Neural Network fine-tuning.
"""

import os
import random
import pandas as pd
from itertools import combinations

CLEANED_DATA_PATH = "cleaned_criminal_cases.csv"
PAIRS_OUTPUT_PATH = "siamese_training_pairs.csv"

def generate_siamese_pairs(target_pairs_per_class=1500, random_seed=42):
    random.seed(random_seed)
    
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned corpus {CLEANED_DATA_PATH} not found.")

    df = pd.read_csv(CLEANED_DATA_PATH, encoding='utf-8')
    print(f"[*] Loaded {len(df)} cases for Siamese pair generation.")

    # Group cases by legal topic
    topic_groups = {}
    for _, row in df.iterrows():
        topic = row["Legal Topic / Law"]
        facts = str(row["Case Facts / Issue"]).strip()
        d_id = row["Dataset ID"]
        if len(facts) < 60:
            continue
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append({"id": d_id, "topic": topic, "facts": facts})

    print(f"[*] Topic categories found: {list(topic_groups.keys())}")

    # 1. Generate Positive Pairs (Same Category)
    positive_pairs = []
    for topic, cases in topic_groups.items():
        if len(cases) < 2:
            continue
        # Generate all combinations within category
        combs = list(combinations(cases, 2))
        random.shuffle(combs)
        
        # Proportional sampling per category
        quota = min(len(combs), max(50, target_pairs_per_class // len(topic_groups)))
        for c1, c2 in combs[:quota]:
            positive_pairs.append({
                "case_a_id": c1["id"],
                "case_b_id": c2["id"],
                "case_a_facts": c1["facts"][:1200],
                "case_b_facts": c2["facts"][:1200],
                "category_a": c1["topic"],
                "category_b": c2["topic"],
                "label": 1.0
            })

    # If we need more positive pairs to hit target
    all_positive_candidates = []
    for topic, cases in topic_groups.items():
        if len(cases) >= 2:
            for c1, c2 in combinations(cases, 2):
                all_positive_candidates.append((c1, c2))
    random.shuffle(all_positive_candidates)
    
    for c1, c2 in all_positive_candidates:
        if len(positive_pairs) >= target_pairs_per_class:
            break
        pair_dict = {
            "case_a_id": c1["id"],
            "case_b_id": c2["id"],
            "case_a_facts": c1["facts"][:1200],
            "case_b_facts": c2["facts"][:1200],
            "category_a": c1["topic"],
            "category_b": c2["topic"],
            "label": 1.0
        }
        if pair_dict not in positive_pairs:
            positive_pairs.append(pair_dict)

    print(f"[+] Generated {len(positive_pairs)} positive contrastive pairs (label=1.0).")

    # 2. Generate Negative Pairs (Different Categories)
    negative_pairs = []
    categories = list(topic_groups.keys())
    all_cases_flat = [c for cases in topic_groups.values() for c in cases]
    
    attempts = 0
    max_attempts = target_pairs_per_class * 10
    while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
        attempts += 1
        c1 = random.choice(all_cases_flat)
        c2 = random.choice(all_cases_flat)
        
        if c1["topic"] != c2["topic"] and c1["id"] != c2["id"]:
            pair_dict = {
                "case_a_id": c1["id"],
                "case_b_id": c2["id"],
                "case_a_facts": c1["facts"][:1200],
                "case_b_facts": c2["facts"][:1200],
                "category_a": c1["topic"],
                "category_b": c2["topic"],
                "label": 0.0
            }
            if pair_dict not in negative_pairs:
                negative_pairs.append(pair_dict)

    print(f"[+] Generated {len(negative_pairs)} negative contrastive pairs (label=0.0).")

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(PAIRS_OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\n[+] Total Siamese training pairs created: {len(pairs_df)}")
    print(f"[+] Saved dataset to: {PAIRS_OUTPUT_PATH}")
    return pairs_df

if __name__ == "__main__":
    generate_siamese_pairs(target_pairs_per_class=1500)
