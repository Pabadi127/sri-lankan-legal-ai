"""
data_preprocessing.py
Preprocesses, cleans, and standardizes the Sri Lankan Criminal Cases Corpus.
Normalizes text, strips OCR artifacts, validates metadata, and classifies cases into
canonical Sri Lankan criminal law domains.
"""

import os
import re
import csv
import shutil
import pandas as pd

SRC_FILE = "sri_lanka_criminal_cases_corpus_v2.csv"
OUTPUT_CLEAN_FILE = "cleaned_criminal_cases.csv"
CORPUS_MIRROR_FILE = "sri_lanka_criminal_cases_corpus.csv"

# Canonical Legal Domains in Sri Lankan Criminal Jurisprudence
CANONICAL_TOPICS = [
    ("Penal Code Offences", [
        "penal code", "murder", "theft", "robbery", "cheating", "rape", "hurt", 
        "culpable homicide", "grievous hurt", "criminal breach of trust", "forgery",
        "extortion", "unlawful assembly", "mischief", "house-breaking"
    ]),
    ("Criminal Procedure & Jurisdiction", [
        "criminal procedure", "indictment", "framing charge", "revision", "summary trial",
        "magistrate", "discharge", "acquittal", "preliminary inquiry", "jurisdiction",
        "administration of justice"
    ]),
    ("Evidence Ordinance & Proof", [
        "evidence ordinance", "confession", "dying declaration", "accomplice",
        "circumstantial evidence", "decoy", "burden of proof", "hearsay", "identification parade"
    ]),
    ("Bail Law & Remand", [
        "bail", "remand", "custody", "anticipatory bail", "section 75", "surety"
    ]),
    ("Bribery & Public Corruption", [
        "bribery", "corruption", "gratification", "public servant", "commission to investigate"
    ]),
    ("Poisons, Opium & Dangerous Drugs", [
        "poisons", "opium", "dangerous drugs", "heroin", "cannabis", "narcotics", "psychotropic"
    ]),
    ("Emergency Regulations & Prevention of Terrorism", [
        "prevention of terrorism", "pta", "emergency regulations", "detention order", "state of emergency"
    ])
]

def clean_text_field(text: str) -> str:
    """Sanitizes raw OCR text, removes boilerplate and excess whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URL remnants and page headers
    t = re.sub(r'https?://\S+', '', text)
    t = re.sub(r'Sri Lanka Law Reports\s*\(\d{4}\)\s*\d+\s*Sri\s*L\.\s*R\.?', '', t, flags=re.I)
    t = re.sub(r'New Law Reports\s*\(\d{4}\)\s*Vol\.?\s*\d+\s*N\.?L\.?R\.?', '', t, flags=re.I)
    t = re.sub(r'\[\d+\s*N\.L\.R\.\]', '', t)
    
    # Strip common leading/trailing quotation artifacts
    t = t.strip('"\' \t\r\n')
    
    # Normalize multiple whitespace, newlines, and tabs to a single space
    t = re.sub(r'\s+', ' ', t)
    
    return t.strip()

def standardize_case_name(name: str, dataset_id: str) -> str:
    """Ensures clean, readable case titles."""
    if not isinstance(name, str) or not name.strip() or name.lower() == "unknown":
        return f"Criminal Case ({dataset_id})"
    
    n = clean_text_field(name)
    # Remove leading numbers/court initials if misplaced
    n = re.sub(r'^(?:\d+\s*CA|\d+\s*SC|sc|CA\s*[\.\-]?)\s*', '', n, flags=re.I)
    n = re.sub(r'\s*\b(?:J\.|C\.J\.|S\.P\.J\.)\b.*$', '', n)
    if len(n) < 4:
        return f"Criminal Case ({dataset_id})"
    return n[:120].strip()

def classify_canonical_topic(raw_topic: str, facts: str) -> str:
    """Classifies a case into one of the 7 standard legal categories."""
    combined_text = f"{raw_topic} {facts}".lower()
    
    for category_name, keywords in CANONICAL_TOPICS:
        for kw in keywords:
            if kw in combined_text:
                return category_name
            
    return "Penal Code Offences"

def preprocess_corpus():
    print("=" * 60)
    print("[*] Preprocessing Sri Lankan Criminal Cases Corpus")
    print(f"[*] Input File: {SRC_FILE}")
    print("=" * 60)
    
    if not os.path.exists(SRC_FILE):
        raise FileNotFoundError(f"Source file {SRC_FILE} not found!")

    df = pd.read_csv(SRC_FILE, encoding='utf-8')
    initial_count = len(df)
    print(f"[*] Loaded {initial_count} raw cases from {SRC_FILE}.")

    # Drop duplicates by Dataset ID or Source URL
    df = df.drop_duplicates(subset=['Dataset ID']).drop_duplicates(subset=['Source URL'])
    print(f"[*] After duplicate removal: {len(df)} unique cases.")

    records = []
    for _, row in df.iterrows():
        d_id = str(row.get("Dataset ID", "")).strip()
        raw_name = str(row.get("Case Name", "")).strip()
        court_no = clean_text_field(str(row.get("Court Case No.", "")))
        year = str(row.get("Year", "")).strip()
        reporter = str(row.get("Reporter", "")).strip()
        raw_topic = str(row.get("Legal Topic / Law", "")).strip()
        raw_facts = str(row.get("Case Facts / Issue", ""))
        raw_holding = str(row.get("Judgment / Holding", ""))
        source_url = str(row.get("Source URL", "")).strip()

        # Clean text
        clean_facts = clean_text_field(raw_facts)
        clean_holding = clean_text_field(raw_holding)
        case_name = standardize_case_name(raw_name, d_id)
        
        # Ensure year fallback
        if not year or year.lower() == "nan" or not re.match(r'^(19|20)\d\d$', year):
            y_search = re.search(r'\b(19\d\d|20\d\d)\b', reporter + " " + d_id)
            year = y_search.group(1) if y_search else "Unknown"

        # Quality filter: case must have substantial facts
        if len(clean_facts) < 50:
            continue

        if not clean_holding or clean_holding.lower() == "nan":
            clean_holding = "Judgment and conviction affirmed or set aside on legal grounds stated in the full opinion."

        # Standardize topic
        canonical_topic = classify_canonical_topic(raw_topic, clean_facts)

        records.append({
            "Dataset ID": d_id,
            "Case Name": case_name,
            "Court Case No.": court_no if court_no and court_no.lower() != "nan" else "Unspecified Court Record",
            "Year": year,
            "Reporter": reporter,
            "Legal Topic / Law": canonical_topic,
            "Case Facts / Issue": clean_facts,
            "Judgment / Holding": clean_holding,
            "Source URL": source_url
        })

    clean_df = pd.DataFrame(records)
    print(f"[+] Cleaned Corpus Count: {len(clean_df)} cases.")
    
    # Display category distribution
    print("\n[*] Canonical Legal Topic Breakdown:")
    topic_counts = clean_df["Legal Topic / Law"].value_counts()
    for topic, count in topic_counts.items():
        print(f"    - {topic}: {count} cases ({count / len(clean_df) * 100:.1f}%)")

    # Save cleaned corpus
    clean_df.to_csv(OUTPUT_CLEAN_FILE, index=False, encoding='utf-8')
    print(f"\n[+] Saved cleaned corpus to: {OUTPUT_CLEAN_FILE}")

    # Safely sync to CORPUS_MIRROR_FILE
    try:
        clean_df.to_csv(CORPUS_MIRROR_FILE, index=False, encoding='utf-8')
        print(f"[+] Synced to master corpus file: {CORPUS_MIRROR_FILE}")
    except Exception as e:
        print(f"[!] Note on {CORPUS_MIRROR_FILE}: {e} (Active file is {OUTPUT_CLEAN_FILE})")

    print("=" * 60)
    print("[+] Preprocessing Complete!")
    print("=" * 60)
    return clean_df

if __name__ == "__main__":
    preprocess_corpus()
