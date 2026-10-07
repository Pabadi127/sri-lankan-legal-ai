# AI-Based Legal Case Law Recommendation System (Group 14)

## 1. Project Overview
* **University:** General Sir John Kotelawala Defence University (KDU), Faculty of Computing
* **Course:** Essentials of Artificial Intelligence (Semester 4)
* **Goal:** Build an AI system to retrieve factually similar historical Sri Lankan criminal cases based on user-provided case facts, mitigating the risk of generative AI hallucination by anchoring recommendations in real historical data.
* **Target Users:** Lawyers and law students in Sri Lanka.

## 2. Team & Responsibilities
* **Pabadi Dayawansha:** Data Collection and Dataset Preparation (Automated scraping from lankalaw.net, metadata extraction, dataset curation).
* **MT Jayaweera:** NLP Preprocessing (Text normalization, OCR artifact filtering) & TF-IDF + Cosine Similarity Baseline implementation.
* **K.S. Navodya:** Siamese Neural Network Architecture, Contrastive Pair Training & Embedding generation.
* **WST Gunawardana:** System Interface (Streamlit Web Application) and system integration.

## 3. Architecture & Implementation Decisions

### A. Data Collection Strategy (Finalized at 771 Authentic Cases)
* **Source:** Real Sri Lankan court judgments from **New Law Reports (NLR)** and **Sri Lanka Law Reports (SLR)** via `lankalaw.net`.
* **Master Corpus:** [`sri_lanka_criminal_cases_corpus_v2.csv`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/sri_lanka_criminal_cases_corpus_v2.csv)
* **Cleaned Corpus:** [`cleaned_criminal_cases.csv`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/cleaned_criminal_cases.csv) (771 unique cases with full facts, holdings, citations, and source URLs).
* **Canonical Topic Distribution across 771 Cases:**
  * Penal Code Offences: 414 cases (53.7%)
  * Criminal Procedure & Jurisdiction: 281 cases (36.4%)
  * Evidence Ordinance & Proof: 38 cases (4.9%)
  * Bribery & Public Corruption: 20 cases (2.6%)
  * Bail Law & Remand: 13 cases (1.7%)
  * Poisons, Opium & Dangerous Drugs: 3 cases (0.4%)
  * Emergency Regulations & Prevention of Terrorism: 2 cases (0.3%)

### B. AI Model Architecture
1. **Baseline Model (`baseline_tfidf.py`):**
   * Pure NumPy Sublinear TF-IDF Vectorizer with unigrams and bigrams + Cosine Similarity ranking.
   * Cached to `tfidf_baseline_model.pkl` (Vocabulary Size: 6,000 features).
2. **Proposed Neural Architecture (`siamese_model.py` / `train_siamese.py`):**
   * **Twin Subnetworks with Shared Weights:** Non-linear projection layers mapping 4,000-dimensional vocabulary inputs to a dense 128-dimensional unit hypersphere.
   * **Contrastive Loss with Margin ($m=1.0$):** Fine-tuned across 15 epochs with mini-batch Adam optimization on 3,000 balanced positive/negative Sri Lankan legal pairs (`siamese_training_pairs.csv`).
   * **Precomputed Corpus Embeddings:** `siamese_case_embeddings.pkl` storing dense 128-dimensional vectors for all 771 cases.
3. **Hybrid AI Engine (`app.py`):**
   * Ensemble weighting: 60% TF-IDF statutory lexical match + 40% Siamese latent semantic match for optimal practical accuracy.

### C. Evaluation Framework & Benchmark Results
* **Benchmark:** 15 realistic Sri Lankan criminal legal scenarios evaluated across all major criminal law topics.
* **Results Comparison Table (`evaluation_comparison.csv`):**

| Model | Precision@1 | Precision@5 | Hit Rate@5 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Cosine Similarity (Baseline)** | 86.67% | 64.00% | 93.33% | 0.8800 |
| **Siamese Neural Network (Proposed)** | 26.67% | 37.33% | 66.67% | 0.4333 |

* **Key Takeaway:** TF-IDF captures exact statutory provisions and courtroom terminology with high precision, whereas the Siamese Neural Network captures broad latent thematic/factual clusters. The Hybrid model combines both strengths.

### D. User Interface (`app.py`)
* Modern, responsive **Streamlit Web Application** featuring:
  * Interactive Scenario Retrieval with custom text input & quick-load templates.
  * Retrieval Engine Selector (Hybrid AI, Siamese Neural Network, Baseline TF-IDF, and Side-by-Side Model Comparison).
  * Legal Topic filtering and Top-K results slider.
  * Rich case cards with match gauges, factual summaries, expandable judicial holdings, and authentic PDF download links.
  * Corpus Analytics Dashboard (charts of topic and temporal distributions).
  * Academic Model Evaluation tab displaying benchmark comparison tables.

---

## 4. Implementation Timeline & Status
* **Week 1-4:** Problem Identification, Research, and Proposal Submission. *(Completed)*
* **Week 5-6:** 
  * [x] Python environment setup (`pypdf`, `pandas`, `openpyxl`, `streamlit`).
  * [x] Connectivity & PDF extraction pipeline verified on `lankalaw.net`.
  * [x] Automated multi-volume scraper created (`scraper_lankalaw.py`).
  * [x] **771 real Sri Lankan criminal cases scraped, cleaned, and verified**.
  * [x] Data cleaning and normalization pipeline (`data_preprocessing.py`).
* **Week 7:**
  * [x] Baseline TF-IDF implementation and serialization (`baseline_tfidf.py`).
  * [x] 15 benchmark evaluation queries created (`evaluate_models.py`).
* **Week 8:**
  * [x] 3,000 balanced contrastive training pairs generated (`create_siamese_pairs.py`).
  * [x] Siamese Neural Network architecture built & trained (`siamese_model.py`, `train_siamese.py`).
  * [x] Precomputed dense embeddings for all 771 cases.
* **Week 9:**
  * [x] Head-to-head empirical evaluation completed (`evaluation_comparison.csv`).
  * [x] Full-featured Streamlit UI built and verified (`app.py`).
* **Week 10:** Final documentation, verification, and presentation preparation.

---

## 5. System File Inventory
1. [`cleaned_criminal_cases.csv`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/cleaned_criminal_cases.csv): Cleaned master corpus of 771 Sri Lankan criminal cases.
2. [`data_preprocessing.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/data_preprocessing.py): NLP sanitation, regex cleaning, and domain classification pipeline.
3. [`baseline_tfidf.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/baseline_tfidf.py): Baseline TF-IDF + Cosine Similarity recommender.
4. [`create_siamese_pairs.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/create_siamese_pairs.py): Generates balanced positive/negative pairs for contrastive training.
5. [`siamese_training_pairs.csv`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/siamese_training_pairs.csv): 3,000 contrastive training pairs.
6. [`siamese_model.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/siamese_model.py): Siamese Twin Projection Network architecture & inference engine.
7. [`train_siamese.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/train_siamese.py): Mini-batch training with Adam and contrastive loss with margin.
8. [`evaluate_models.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/evaluate_models.py): Empirical benchmarking on 15 queries (Precision@5, Hit@5, MRR).
9. [`evaluation_comparison.csv`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/evaluation_comparison.csv): Evaluation metrics table.
10. [`app.py`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/app.py): Interactive Streamlit web interface.
11. [`requirements.txt`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/requirements.txt): Python dependencies for cloud deployment.
12. [`run_app.bat`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/run_app.bat): Local launcher shortcut.
13. [`project_plan.md`](file:///c:/Users/USER/OneDrive/Documents/4%20Semester/AI/Law%20Project/project_plan.md): Central documentation tracking project decisions and milestones.

---

## 6. Cloud Deployment Guide (Streamlit Community Cloud)
1. **GitHub Repository:** Push this directory containing `app.py`, `baseline_tfidf.py`, `siamese_model.py`, `cleaned_criminal_cases.csv`, `requirements.txt`, `tfidf_baseline_model.pkl`, and `siamese_case_embeddings.pkl` to GitHub.
2. **Streamlit Cloud:** Sign in to `share.streamlit.io` with GitHub.
3. **New App:** Click **Deploy an app**, select your repository, set Main file path to `app.py`, and click **Deploy**.
4. **Public Access:** Receive a live, permanent URL (e.g. `https://sri-lankan-legal-ai.streamlit.app`) to share with lecturers and evaluators.

