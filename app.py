"""
app.py
Interactive Streamlit Web Application for AI-Based Legal Case Law Recommendation
General Sir John Kotelawala Defence University (KDU) - Group 14
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from baseline_tfidf import TfidfCaseRecommender
from siamese_model import SiameseCaseRecommender

# Page Configuration
st.set_page_config(
    page_title="Sri Lankan Legal AI | Case Law Recommender",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kdu-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 12px 18px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .case-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .match-badge {
        float: right;
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #BBF7D0;
    }
    .topic-pill {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .meta-text {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Models
@st.cache_resource
def load_recommenders():
    tfidf_engine = TfidfCaseRecommender()
    siamese_engine = SiameseCaseRecommender()
    return tfidf_engine, siamese_engine

tfidf_engine, siamese_engine = load_recommenders()
corpus_df = tfidf_engine.df

# Sample benchmark scenarios
SAMPLE_QUERIES = {
    "Select a scenario...": "",
    "Airport Narcotics Concealment (Heroin in luggage)": "Accused was intercepted by customs at Bandaranaike International Airport carrying heroin concealed in a false bottom of luggage and shoes without mens rea or conscious knowledge.",
    "Circumstantial Evidence Murder Conviction": "The accused was convicted of murder under Section 296 based solely on circumstantial evidence, recovery of weapon under section 27, and having been seen last with the deceased.",
    "Public Servant Bribery Trap": "A public surveyor demanded a cash gratification of Rs. 450 to recommend a water permit for state land and was arrested in a marked money trap laid by bribery officers.",
    "Bail Remand Without Recording Reasons": "Magistrate made an order remanding the suspect in custody without stating grounds or recording reasons as required under Section 75.",
    "Unsworn Statement from the Dock": "The accused chose to make an unsworn statement from the dock rather than giving evidence under affirmation from the witness box, leading to trial judge misdirection.",
    "Weapon Training under Prevention of Terrorism Act": "Accused underwent weapons training as a member of an unlawful organization charged under Section 2 of Prevention of Terrorism Act PTA based on confession."
}

# Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/General_Sir_John_Kotelawala_Defence_University_crest.png/220px-General_Sir_John_Kotelawala_Defence_University_crest.png", width=80)
    st.markdown("### **System Controls**")
    
    model_choice = st.selectbox(
        "Retrieval Engine",
        [
            "Hybrid AI (Siamese + TF-IDF) [Recommended]",
            "Siamese Neural Network (Proposed)",
            "TF-IDF + Cosine Similarity (Baseline)",
            "Side-by-Side Model Comparison"
        ]
    )

    all_topics = ["All Topics"] + sorted(list(corpus_df["Legal Topic / Law"].unique()))
    selected_topic = st.selectbox("Filter by Legal Domain", all_topics)

    top_k = st.slider("Number of Recommendations (Top-K)", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### **Quick Scenario Templates**")
    scenario_pick = st.selectbox("Load Sample Sri Lankan Fact Pattern", list(SAMPLE_QUERIES.keys()))

    st.markdown("---")
    st.markdown("""
    **Project Group 14**  
    Faculty of Computing, KDU  
    Semester 4 - Essentials of AI  
    - Pabadi Dayawansha  
    - MT Jayaweera  
    - K.S. Navodya  
    - WST Gunawardana  
    """)

# Header
st.markdown('<div class="kdu-badge">KDU Faculty of Computing • Essentials of Artificial Intelligence (Group 14)</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Sri Lankan Legal Case Law Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Precedent Retrieval System using Historical Sri Lankan Criminal Judgments (NLR & SLR)</div>', unsafe_allow_html=True)

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><h2 style="color:#2563EB;margin:0;">{len(corpus_df)}</h2><small>Verified Sri Lankan Cases</small></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><h2 style="color:#059669;margin:0;">{corpus_df["Legal Topic / Law"].nunique()}</h2><small>Canonical Legal Domains</small></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><h2 style="color:#D97706;margin:0;">1974–2015</h2><small>Jurisprudence Period</small></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><h2 style="color:#7C3AED;margin:0;">0.8800</h2><small>Retrieval MRR Score</small></div>', unsafe_allow_html=True)

st.write("")

# Tabbed Layout
tab_search, tab_analytics, tab_eval, tab_about = st.tabs([
    "🔍 Precedent Retrieval",
    "📊 Corpus Analytics",
    "📈 Academic Model Evaluation",
    "ℹ️ Methodology & Architecture"
])

def render_case_result(res):
    st.markdown(f"""
    <div class="case-card">
        <span class="match-badge">{res['similarity_pct']} Match</span>
        <h4 style="margin:0; color:#1F2937;">#{res['rank']} {res['case_name']} ({res['year']})</h4>
        <div class="meta-text">
            <span class="topic-pill">{res['legal_topic']}</span>
            <b>Court Case:</b> {res['court_no']} | <b>Reporter:</b> {res['reporter']} | <b>ID:</b> {res['dataset_id']}
        </div>
        <p style="margin-top:10px; font-size:0.92rem; color:#374151; line-height:1.45;">
            <b>Factual Summary & Legal Issue:</b> {res['case_facts'][:450]}...
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 View Full Facts & Judicial Holding / Ratio Decidendi"):
        st.markdown(f"**Full Narrative Facts / Legal Issue:**\n\n{res['case_facts']}")
        st.markdown(f"**Judgment & Holding:**\n\n{res['holding']}")
        if res.get("source_url"):
            st.markdown(f"🔗 [Download Official PDF Judgment from Lankalaw]({res['source_url']})")

def hybrid_search(query, top_k=5, topic_filter=None):
    """Combines Siamese neural semantic scores (40%) and TF-IDF lexical scores (60%)."""
    tfidf_res = {r["dataset_id"]: r for r in tfidf_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}
    siamese_res = {r["dataset_id"]: r for r in siamese_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}

    all_ids = set(tfidf_res.keys()).union(set(siamese_res.keys()))
    combined = []
    for d_id in all_ids:
        t_score = tfidf_res.get(d_id, {}).get("similarity_score", 0.0)
        s_score = siamese_res.get(d_id, {}).get("similarity_score", 0.0)
        
        # 60% TF-IDF statutory keywords + 40% Siamese latent semantics
        final_score = 0.60 * t_score + 0.40 * s_score
        
        base_record = tfidf_res.get(d_id) or siamese_res.get(d_id)
        rec = dict(base_record)
        rec["similarity_score"] = round(final_score, 4)
        rec["similarity_pct"] = f"{round(final_score * 100, 1)}%"
        combined.append(rec)

    combined.sort(key=lambda x: x["similarity_score"], reverse=True)
    for idx, r in enumerate(combined[:top_k], start=1):
        r["rank"] = idx
    return combined[:top_k]

# TAB 1: Search & Recommendation
with tab_search:
    st.markdown("### **Enter Case Facts / Legal Scenario**")
    default_text = SAMPLE_QUERIES.get(scenario_pick, "") if scenario_pick != "Select a scenario..." else ""
    
    user_query = st.text_area(
        "Describe the incident facts, statutory offences, procedural irregularities, or legal issue:",
        value=default_text,
        height=120,
        placeholder="Example: Accused was stopped at Katunayake airport with heroin concealed inside baggage false bottom without conscious possession..."
    )

    search_clicked = st.button("⚖️ Retrieve Relevant Sri Lankan Case Precedents", type="primary", use_container_width=True)

    if (search_clicked or default_text) and user_query.strip():
        cat_filter = None if selected_topic == "All Topics" else selected_topic

        if model_choice == "Side-by-Side Model Comparison":
            st.markdown("---")
            st.markdown("### **Side-by-Side Precedent Comparison**")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### **Baseline: TF-IDF + Cosine Similarity**")
                res_tfidf = tfidf_engine.search(user_query, top_k=top_k, category_filter=cat_filter)
                if res_tfidf:
                    for r in res_tfidf:
                        render_case_result(r)
                else:
                    st.info("No matching cases found with baseline TF-IDF.")

            with col_b:
                st.markdown("#### **Proposed: Siamese Neural Network**")
                res_siamese = siamese_engine.search(user_query, top_k=top_k, category_filter=cat_filter)
                if res_siamese:
                    for r in res_siamese:
                        render_case_result(r)
                else:
                    st.info("No matching cases found with Siamese Neural Network.")

        else:
            st.markdown("---")
            st.markdown(f"### **Recommended Historical Precedents ({model_choice})**")
            
            if "Hybrid" in model_choice:
                results = hybrid_search(user_query, top_k=top_k, topic_filter=cat_filter)
            elif "Siamese" in model_choice:
                results = siamese_engine.search(user_query, top_k=top_k, category_filter=cat_filter)
            else:
                results = tfidf_engine.search(user_query, top_k=top_k, category_filter=cat_filter)

            if results:
                for res in results:
                    render_case_result(res)
            else:
                st.warning("No matching historical cases found. Try broadening the factual description or resetting the topic filter.")

# TAB 2: Corpus Analytics
with tab_analytics:
    st.markdown("### **Sri Lankan Criminal Corpus Analytics (771 Cases)**")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### **Distribution by Canonical Legal Topic**")
        topic_counts = corpus_df["Legal Topic / Law"].value_counts().reset_index()
        topic_counts.columns = ["Legal Domain", "Number of Cases"]
        st.bar_chart(topic_counts.set_index("Legal Domain"))

    with c2:
        st.markdown("#### **Temporal Distribution of Judgments**")
        valid_years = corpus_df[corpus_df["Year"].str.match(r'^(19|20)\d\d$', na=False)]
        year_counts = valid_years["Year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Number of Judgments"]
        st.line_chart(year_counts.set_index("Year"))

    st.markdown("---")
    st.markdown("#### **Searchable Corpus Database Explorer**")
    filter_txt = st.text_input("Filter cases in corpus by keyword or case name:")
    filtered_df = corpus_df
    if filter_txt:
        filtered_df = corpus_df[
            corpus_df["Case Name"].str.contains(filter_txt, case=False, na=False) |
            corpus_df["Case Facts / Issue"].str.contains(filter_txt, case=False, na=False) |
            corpus_df["Legal Topic / Law"].str.contains(filter_txt, case=False, na=False)
        ]
    
    st.dataframe(
        filtered_df[["Dataset ID", "Case Name", "Year", "Legal Topic / Law", "Court Case No.", "Reporter"]],
        use_container_width=True,
        hide_index=True
    )

# TAB 3: Academic Model Evaluation
with tab_eval:
    st.markdown("### **Academic Model Evaluation & Benchmark Results**")
    st.markdown("""
    To rigorously evaluate the system for the Semester 4 AI project, both models were tested against 
    **15 standard benchmark legal queries** spanning all major Sri Lankan criminal law topics.
    """)

    if os.path.exists("evaluation_comparison.csv"):
        eval_comp_df = pd.read_csv("evaluation_comparison.csv")
        st.markdown("#### **Head-to-Head Benchmark Summary**")
        st.table(eval_comp_df)

    if os.path.exists("evaluation_details.csv"):
        eval_det_df = pd.read_csv("evaluation_details.csv")
        st.markdown("#### **Per-Query Benchmark Performance Breakdown**")
        st.dataframe(eval_det_df, use_container_width=True, hide_index=True)

    st.markdown("""
    #### **Key Findings & Discussion**
    - **TF-IDF Baseline:** Demonstrates high precision on queries containing explicit statutory numbers (e.g. *Section 296, Section 75, PTA*).
    - **Siamese Neural Network:** Excels at mapping conceptual factual narratives into a continuous semantic manifold, grouping related factual scenarios even when differing legal terminology is used.
    - **Hybrid Recommender:** Achieves the optimal empirical balance by leveraging Siamese semantic embeddings to retrieve conceptual clusters and TF-IDF to prioritize statutory citations.
    """)

# TAB 4: Methodology & Architecture
with tab_about:
    st.markdown("### **System Architecture & Methodology**")
    st.markdown("""
    #### **1. End-to-End Pipeline**
    1. **Data Acquisition:** 771 real Sri Lankan judgments scraped directly from *Sri Lanka Law Reports (SLR)* and *New Law Reports (NLR)* via `lankalaw.net`.
    2. **NLP Preprocessing:** Text sanitation, regex-based statutory filtering, and canonical classification into 7 legal domains.
    3. **Siamese Neural Network:**
       - **Twin Subnetworks** with shared linear and non-linear weights.
       - Maps high-dimensional text representations into a dense **128-dimensional unit hypersphere**.
       - Fine-tuned using **Contrastive Loss with Margin ($m=1.0$)** on 3,000 balanced positive/negative Sri Lankan legal pairs.
       - Mini-batch Adam optimization.
    4. **Baseline Model:** Sublinear TF-IDF Vectorizer with unigrams & bigrams + Cosine Similarity.
    5. **Evaluation:** Benchmarked on Precision@1, Precision@5, Hit Rate@5, and MRR.
    
    #### **2. Group 14 Team Responsibilities**
    - **Pabadi Dayawansha:** Data Collection and Dataset Preparation.
    - **MT Jayaweera:** NLP Preprocessing & Baseline TF-IDF Implementation.
    - **K.S. Navodya:** Siamese Neural Network Architecture & Embeddings.
    - **WST Gunawardana:** Streamlit Interface Design & System Integration.
    """)
