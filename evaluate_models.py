"""
evaluate_models.py
Empirical Evaluation Framework for Sri Lankan Criminal Case Law Recommendation.
Evaluates retrieval accuracy across 15 realistic benchmark legal queries
using Precision@5, Hit Rate@5, and Mean Reciprocal Rank (MRR).
"""

import os
import pandas as pd
import numpy as np

BENCHMARK_QUERIES = [
    {
        "id": "BQ-01",
        "query": "Accused intercepted at Katunayake airport carrying heroin concealed in luggage false bottom and shoes without mens rea",
        "expected_topics": ["Poisons, Opium & Dangerous Drugs", "Penal Code Offences", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["heroin", "poisons", "dangerous drugs", "katunayake", "mens rea", "concealed"]
    },
    {
        "id": "BQ-02",
        "query": "Murder conviction under Section 296 based solely on circumstantial evidence and recovery of the weapon",
        "expected_topics": ["Penal Code Offences", "Evidence Ordinance & Proof"],
        "core_keywords": ["murder", "circumstantial", "296", "weapon", "killing", "death"]
    },
    {
        "id": "BQ-03",
        "query": "Public surveyor demanding gratification or bribe from land permit holder to issue favorable report caught in trap",
        "expected_topics": ["Bribery & Public Corruption"],
        "core_keywords": ["bribery", "gratification", "trap", "public servant", "bribe", "officer"]
    },
    {
        "id": "BQ-04",
        "query": "Magistrate made remand order detaining suspect without recording reasons or grounds under Section 75",
        "expected_topics": ["Bail Law & Remand", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["bail", "remand", "section 75", "custody", "magistrate", "grounds"]
    },
    {
        "id": "BQ-05",
        "query": "Accused made unsworn statement from the dock instead of giving evidence from the witness box under affirmation",
        "expected_topics": ["Evidence Ordinance & Proof", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["dock", "unsworn", "affirmation", "witness box", "statement", "evidence"]
    },
    {
        "id": "BQ-06",
        "query": "Trial proceeded against accused without formal charge being framed and without plea being recorded",
        "expected_topics": ["Criminal Procedure & Jurisdiction", "Penal Code Offences"],
        "core_keywords": ["plea", "charge", "irregularity", "primary court", "magistrate", "trial"]
    },
    {
        "id": "BQ-07",
        "query": "Undergoing weapons training as a member of an illegal organization under Prevention of Terrorism Act PTA",
        "expected_topics": ["Emergency Regulations & Prevention of Terrorism", "Penal Code Offences"],
        "core_keywords": ["prevention of terrorism", "pta", "weapons training", "confession", "detention"]
    },
    {
        "id": "BQ-08",
        "query": "Decoy used by price control inspector to purchase goods with marked currency note and corroboration needed",
        "expected_topics": ["Evidence Ordinance & Proof", "Penal Code Offences"],
        "core_keywords": ["decoy", "corroboration", "price controller", "marked note", "accomplice"]
    },
    {
        "id": "BQ-09",
        "query": "Robbery of gold jewellery and murder forming part of the same transaction with recent unexplained possession of stolen goods",
        "expected_topics": ["Penal Code Offences", "Evidence Ordinance & Proof"],
        "core_keywords": ["robbery", "murder", "possession", "stolen", "same transaction", "jewellery"]
    },
    {
        "id": "BQ-10",
        "query": "Police officer assaulted and removed in military vehicle during clash with army officers claiming fundamental rights breach",
        "expected_topics": ["Penal Code Offences", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["police", "assault", "army", "fundamental rights", "affidavit", "injuries"]
    },
    {
        "id": "BQ-11",
        "query": "Questioned handwriting document forwarded to Examiner of Questioned Documents EQD for comparison and opinion evidence",
        "expected_topics": ["Evidence Ordinance & Proof", "Penal Code Offences"],
        "core_keywords": ["handwriting", "eqd", "document", "expert", "comparison", "section 73"]
    },
    {
        "id": "BQ-12",
        "query": "Trial judge failed to direct jury on matters in favour of accused on capital murder charge denying fair trial",
        "expected_topics": ["Penal Code Offences", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["fair trial", "jury", "murder", "misdirection", "favour", "capital"]
    },
    {
        "id": "BQ-13",
        "query": "Search warrant issued by Magistrate on credible information challenged for fundamental rights infringement",
        "expected_topics": ["Criminal Procedure & Jurisdiction", "Bail Law & Remand"],
        "core_keywords": ["search warrant", "magistrate", "fundamental rights", "police", "search"]
    },
    {
        "id": "BQ-14",
        "query": "Corroboration of mother evidence in maintenance proceedings concerning time of sexual intimacy",
        "expected_topics": ["Evidence Ordinance & Proof", "Criminal Procedure & Jurisdiction"],
        "core_keywords": ["corroboration", "maintenance", "intimacy", "evidence ordinance", "conception"]
    },
    {
        "id": "BQ-15",
        "query": "Application for revision in Court of Appeal against order of High Court where preliminary objections were raised",
        "expected_topics": ["Criminal Procedure & Jurisdiction"],
        "core_keywords": ["revision", "court of appeal", "high court", "preliminary objection", "order"]
    }
]

def evaluate_recommender(recommender_instance, recommender_name="Recommender"):
    """
    Computes Precision@1, Precision@5, Hit Rate@5, and Mean Reciprocal Rank (MRR)
    across the standard 15 benchmark queries.
    """
    print(f"\n=======================================================")
    print(f"[*] Running Benchmark Evaluation: {recommender_name}")
    print(f"=======================================================")
    
    query_eval_records = []
    
    p1_list = []
    p5_list = []
    hit5_list = []
    rr_list = []

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        query_text = item["query"]
        expected_topics = item["expected_topics"]
        core_keywords = item["core_keywords"]

        # Retrieve Top 5 recommendations
        results = recommender_instance.search(query_text, top_k=5)
        
        relevant_in_top5 = 0
        first_relevant_rank = None

        for rank, res in enumerate(results, start=1):
            topic = res.get("legal_topic", "")
            facts = (res.get("case_facts", "") + " " + res.get("holding", "")).lower()
            
            # Determine relevance: topic match OR keyword overlap
            topic_match = topic in expected_topics
            keyword_overlap = any(kw in facts for kw in core_keywords)
            
            is_relevant = topic_match and keyword_overlap
            
            if is_relevant:
                relevant_in_top5 += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # Compute metrics
        p1 = 1.0 if (first_relevant_rank == 1) else 0.0
        p5 = relevant_in_top5 / 5.0
        hit5 = 1.0 if (relevant_in_top5 > 0) else 0.0
        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        p1_list.append(p1)
        p5_list.append(p5)
        hit5_list.append(hit5)
        rr_list.append(rr)

        query_eval_records.append({
            "Query ID": q_id,
            "Factual Query": query_text[:65] + "...",
            "Top-1 Relevant": "Yes" if p1 == 1.0 else "No",
            "Precision@5": f"{p5 * 100:.1f}%",
            "Hit@5": "Hit" if hit5 == 1.0 else "Miss",
            "Reciprocal Rank": f"{rr:.2f}"
        })

    avg_p1 = np.mean(p1_list)
    avg_p5 = np.mean(p5_list)
    avg_hit5 = np.mean(hit5_list)
    mrr = np.mean(rr_list)

    summary_metrics = {
        "Model": recommender_name,
        "Precision@1": f"{avg_p1 * 100:.2f}%",
        "Precision@5": f"{avg_p5 * 100:.2f}%",
        "Hit Rate@5": f"{avg_hit5 * 100:.2f}%",
        "MRR": f"{mrr:.4f}"
    }

    print("\n--- Summary Evaluation Metrics ---", flush=True)
    for k, v in summary_metrics.items():
        print(f"  {k}: {v}", flush=True)
        
    eval_df = pd.DataFrame(query_eval_records)
    return summary_metrics, eval_df

def run_comparative_evaluation():
    from baseline_tfidf import TfidfCaseRecommender
    from siamese_model import SiameseCaseRecommender

    print("=================================================================", flush=True)
    print("[*] SRI LANKAN CRIMINAL LEGAL CASE RECOMMENDATION - BENCHMARK EVALUATION", flush=True)
    print("=================================================================", flush=True)

    # 1. Evaluate Baseline TF-IDF
    tfidf_rec = TfidfCaseRecommender()
    tfidf_summary, tfidf_df = evaluate_recommender(tfidf_rec, "TF-IDF + Cosine Similarity (Baseline)")

    # 2. Evaluate Proposed Siamese Neural Network
    siamese_rec = SiameseCaseRecommender()
    siamese_summary, siamese_df = evaluate_recommender(siamese_rec, "Siamese Neural Network (Proposed)")

    # Build Comparative Table
    comparison_df = pd.DataFrame([tfidf_summary, siamese_summary])
    
    print("\n" + "=" * 70, flush=True)
    print("[*] HEAD-TO-HEAD MODEL COMPARISON SUMMARY (15 BENCHMARK QUERIES):", flush=True)
    print("=" * 70, flush=True)
    print(comparison_df.to_string(index=False), flush=True)
    print("=" * 70, flush=True)

    comparison_df.to_csv("evaluation_comparison.csv", index=False)
    
    # Save combined details
    tfidf_df["Model"] = "TF-IDF Baseline"
    siamese_df["Model"] = "Siamese Neural Network"
    combined_details = pd.concat([tfidf_df, siamese_df], ignore_index=True)
    combined_details.to_csv("evaluation_details.csv", index=False)
    print("[+] Saved evaluation results to 'evaluation_comparison.csv' and 'evaluation_details.csv'.", flush=True)

if __name__ == "__main__":
    run_comparative_evaluation()

