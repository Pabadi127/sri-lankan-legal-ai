# Sri Lankan Legal Case Law Recommender
### AI-Based Precedent Retrieval System

An AI-powered tool designed to help law students, legal professionals, and researchers find relevant past Sri Lankan criminal cases. Users can describe a legal situation in their own words and receive a ranked list of related judgments from the **New Law Reports (NLR)** and **Sri Lanka Law Reports (SLR)**.

[Streamlit App](https://share.streamlit.io/) · [Python](https://www.python.org/) · [System Architecture](https://github.com/Pabadi127/sri-lankan-legal-ai/blob/main/README.md#system-architecture--ai-methodology)

---

## Table of Contents

1. [Project Background & Motivation](#project-background--motivation)
2. [Live Application Access](#live-application-access)
3. [Local Installation & Setup](#local-installation--setup)
4. [Evaluator's Guide: How to Use the App](#evaluators-guide-how-to-use-the-app)
5. [Sample Test Scenarios for Evaluation](#sample-test-scenarios-for-evaluation)
6. [System Architecture & AI Methodology](#system-architecture--ai-methodology)
7. [Corpus & Dataset Overview](#corpus--dataset-overview)
8. [Repository Structure](#repository-structure)

---

## Project Background & Motivation

In Sri Lanka, lawyers and judges often need to find earlier court decisions that are relevant to a current case. This principle is known as **stare decisis**, or judicial precedent. However, legal research can take time because:

- Case judgments are spread across printed and digital copies of the **New Law Reports (NLR)** and **Sri Lanka Law Reports (SLR)**.
- A keyword search may miss relevant cases when the same issue is described using different words. For example, *"car theft"* and *"dishonest misappropriation of a motor vehicle"* may describe related legal situations.
- Traditional search tools may not capture the meaning of the facts described in a case.

### Our Solution

The project uses a **hybrid AI approach** that combines two methods:

1. **Siamese Neural Network:** Compares the meaning and context of a user's description with existing cases.
2. **TF-IDF:** Identifies important words and legal references, such as *Section 296 of the Penal Code* and *Section 75 of the Administration of Justice Law*.

The interface is designed to be straightforward, presenting search results in organized case cards that are easy to review.

---

## Live Application Access

The application is hosted on **Streamlit Community Cloud** and can be opened in a web browser without installing it locally.

**Live application:** https://sri-lankan-legal-ai.streamlit.app/

---

## Local Installation & Setup

To run the application on your computer, follow these steps.

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pabadi127/sri-lankan-legal-ai.git
cd sri-lankan-legal-ai
```

### Step 2: Install the Required Packages

Run:

```bash
pip install -r requirements.txt
```

The main dependencies are `streamlit`, `pandas`, and `numpy`.

### Step 3: Run the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

On Windows, you can also run `run_app.bat` from the project folder.

Once the app starts, it should open in your default browser at `http://localhost:8501`.

---

## Evaluator's Guide: How to Use the App

The application has two main sections.

### 1. Find Similar Cases (`Similar Case Laws` tab)

1. **Enter the case facts:** In the **Case Details** box, describe the situation, incident, possible legal violations, or legal issues in plain English.
2. **Choose a legal area (optional):** Use the sidebar to select an area, such as *Penal Code Offences*, *Bail Law & Remand*, or *Narcotics & Dangerous Drugs*. You can also leave the filter set to `"All Areas of Law"`.
3. **Choose how many results to show (optional):** Use the slider to select between 3 and 10 results.
4. **Search:** Click **Find Similar Case Laws**. The system checks the query using both the TF-IDF and Siamese Neural Network components, combines their scores, and ranks the results.
5. **Review the results:**
   - **Similarity score:** Shows the calculated match score for each result.
   - **NLR citation / Dataset ID:** Shows the case identifier, such as `NLR-80-013`.
   - **Case summary:** Summarizes the facts and legal issues.
   - **Detailed holding:** Select **View Full Case Analysis** to read the case details and the court's decision (*ratio decidendi*).
6. **View search history:** Previous searches are saved in the sidebar for the current session. Select a past search to load it again with its results.

### 2. Browse the Database (`Browse Database` tab)

- Search through all **771 verified Sri Lankan criminal judgments**.
- Filter by case name, facts, legal topic, or court case number.

---

## Sample Test Scenarios for Evaluation

You can test the application using the example prompts below or select the quick templates in the sidebar. Enter a prompt in the **Case Details** box.

### Scenario A: Narcotics Concealment and Knowledge

> "The accused was stopped by customs at Bandaranaike International Airport while carrying heroin hidden in a false bottom of luggage and in shoes. The accused claimed to have no knowledge of the drugs."

- **Expected legal topic:** Narcotics & Dangerous Drugs
- **What this tests:** Cases involving concealed drugs and the question of whether the accused knew they were carrying them.

### Scenario B: Murder Based on Circumstantial Evidence

> "The accused was convicted of murder under Section 296 based only on circumstantial evidence. The case included the recovery of a weapon under Section 27 and evidence that the accused was last seen with the deceased."

- **Expected legal topic:** Penal Code Offences / Murder & Evidence
- **What this tests:** Legal principles relating to circumstantial evidence in murder cases and Section 27 of the Evidence Ordinance.

### Scenario C: Remand Without Stated Reasons

> "A Magistrate ordered a suspect to remain in custody without stating the reasons for the decision, as required under Section 75."

- **Expected legal topic:** Bail Law & Remand
- **What this tests:** The legal requirements for giving reasons when ordering remand under Section 75 of the Administration of Justice Law.

---

## System Architecture & AI Methodology

The system uses both keyword-based and meaning-based matching. It combines the results to rank relevant cases.

```text
User's Legal Scenario
        |
        +--> TF-IDF keyword matching ------> Lexical score (60%)
        |
        +--> Siamese Neural Network -------> Semantic score (40%)
                                              |
                                              v
                                    Combined similarity score
                                              |
                                              v
                                     Ranked case results
```

### 1. Siamese Neural Network (Semantic Matching)

- **How it works:** Two matching neural-network branches turn text into comparable numerical representations in a shared **128-dimensional space**.
- **Training:** The network was trained using contrastive loss with a margin of `1.0` and 3,000 case-fact pairs: 1,500 similar pairs and 1,500 dissimilar pairs.
- **Precomputed representations:** The system stores representations for all 771 cases in `siamese_case_embeddings.pkl`, helping it compare cases quickly during a search.

### 2. TF-IDF and Cosine Similarity (Keyword Matching)

- The system represents legal text using TF-IDF, with sublinear term-frequency scaling and inverse document frequency (IDF). The vocabulary contains 6,000 single words and two-word phrases.
- This helps give distinctive legal terms and references—such as *Section 296*, *Section 75*, and *Section 27*—greater importance during matching.

### 3. Combining the Scores

The final score combines the two matching methods:

```text
Final Score = (0.60 × TF-IDF Score) + (0.40 × Siamese Score)
```

This gives more weight to exact legal terms while also considering the meaning of the user's description.

### 4. A Lightweight Implementation

The TF-IDF and Siamese Neural Network components were implemented using Python and NumPy. This kept the project lightweight and helped avoid relying on some external machine-learning libraries that can be difficult to install in restricted environments.

---

## Corpus & Dataset Overview

| Metric | Details |
|---|---|
| **Verified judgments** | 771 cases |
| **Year range** | 1974–2015 |
| **Law reports included** | New Law Reports (NLR) and Sri Lanka Law Reports (SLR) |
| **Legal categories** | 7: Penal Code Offences, Narcotics, Bail Law, Bribery & Corruption, Criminal Procedure, Evidence, and Prevention of Terrorism |
| **Training pairs** | 3,000 pairs: 1,500 similar and 1,500 dissimilar |
| **Mean Reciprocal Rank (MRR)** | 0.8800 across 15 benchmark queries |

---

## Repository Structure

```text
sri-lankan-legal-ai/
├── app.py                         # Streamlit web application
├── baseline_tfidf.py              # TF-IDF and cosine-similarity search
├── siamese_model.py               # Siamese Neural Network
├── train_siamese.py               # Training process for the Siamese network
├── create_siamese_pairs.py        # Creates 3,000 training pairs
├── data_preprocessing.py          # Cleans text and maps legal topics
├── scraper_lankalaw.py            # Collects case information from LankaLaw
├── evaluate_models.py             # Tests retrieval using 15 benchmark queries
├── print_evidence.py              # Formats model results for reports
├── requirements.txt               # Required Python packages
├── run_app.bat                    # Starts the app on Windows
│
├── cleaned_criminal_cases.csv     # Cleaned dataset of 771 cases
├── siamese_training_pairs.csv     # 3,000 labeled training pairs
├── tfidf_baseline_model.pkl       # Saved TF-IDF model data
└── siamese_case_embeddings.pkl    # Saved representations of the 771 cases
```

---

## Academic Context

This project was developed for the **Essentials of Artificial Intelligence** module in Semester 4, Faculty of Computing, **General Sir John Kotelawala Defence University (KDU)**.
