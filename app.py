"""
app.py
Interactive Streamlit Web Application for AI-Based Legal Case Law Recommendation
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from baseline_tfidf import TfidfCaseRecommender
from siamese_model import SiameseCaseRecommender

# Page Configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "loaded_history_query" not in st.session_state:
    st.session_state.loaded_history_query = ""

# Custom Styling (Deep dark blue theme, glassmorphic accents, clean navigation header)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Open+Sans:wght@400;600&display=swap');

    /* Keep header clean & transparent: hide Share, GitHub, Star, Edit, keeping ONLY the theme settings and sidebar toggle */
    header[data-testid="stHeader"] {
        background: rgba(1, 4, 10, 0.6) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    [data-testid="stToolbarActions"],
    [data-testid="stHeader"] .stActionButton,
    [data-testid="stHeader"] a,
    [data-testid="manage-app-button"],
    .stDeployButton,
    button[title="Share this app"],
    button[title="View source on GitHub"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 15% 20%, #071938 0%, #01040a 65%, #020914 100%);
    }

    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }
    p, span, div, input, button, .stMarkdown {
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Hero header */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        font-family: 'Montserrat', sans-serif !important;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Interactive Popping Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #0b1933 0%, #102a54 100%);
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.08);
        transition: all 0.25s ease-in-out;
        border: 1px solid rgba(56, 189, 248, 0.25);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 30px -5px rgba(0, 0, 0, 0.6), 0 0 25px rgba(56, 189, 248, 0.22);
        border-color: rgba(56, 189, 248, 0.6);
    }
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin: 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        color: #cbd5e1 !important;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.25rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }

    /* Sidebar Specific Button Styling */
    section[data-testid="stSidebar"] .stButton>button {
        padding: 0.45rem 0.9rem !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    /* Download Memo Button */
    .stDownloadButton>button {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOGIN PORTAL (Displayed when user is not authenticated)
# -------------------------------------------------------------
if not st.session_state.logged_in:
    col_l1, col_login, col_l2 = st.columns([1, 2, 1])
    with col_login:
        st.write("")
        st.write("")
        st.markdown('<div class="main-header" style="text-align: center; font-size: 2.6rem;">Legal AI Portal</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #94a3b8; margin-bottom: 25px;">Sri Lankan Jurisprudence & Precedent Retrieval System</p>', unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("### Practitioner & Student Sign In")
            st.caption("Authorized access for academic evaluators, law students, and legal researchers.")
            
            login_user = st.text_input("Username or Academic Email", value="student@kdu.ac.lk")
            login_pwd = st.text_input("Password", type="password", value="legalai2026")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("Sign In", use_container_width=True):
                    if login_user and login_pwd:
                        st.session_state.logged_in = True
                        st.session_state.user_name = login_user.split('@')[0].capitalize()
                        st.rerun()
                    else:
                        st.error("Please enter valid credentials.")
            with col_b2:
                if st.button("Evaluator / Guest Access", use_container_width=True):
                    st.session_state.logged_in = True
                    st.session_state.user_name = "Academic Evaluator (Guest)"
                    st.rerun()
            
            st.divider()
            st.markdown(
                "<div style='text-align: center; font-size: 0.8rem; color: #64748b;'>"
                "<b>Evaluator Pass:</b> Click <i>Evaluator / Guest Access</i> to bypass authentication instantly."
                "</div>",
                unsafe_allow_html=True
            )
    st.stop()

# -------------------------------------------------------------
# AUTHENTICATED APPLICATION
# -------------------------------------------------------------

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
    st.markdown(f"""
        <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Current Session</div>
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.92rem;">{st.session_state.user_name}</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.rerun()

    st.markdown("""
        <div style="text-align: center; padding: 5px 0 15px 0;">
            <h2 style="font-family: 'Montserrat', sans-serif; margin-top: 5px; color: #38bdf8;">Legal AI Assistant</h2>
            <p style="color: #94a3b8; font-size: 0.85rem;">Sri Lankan Jurisprudence Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Search Parameters")

    all_topics = ["All Areas of Law"] + sorted(list(corpus_df["Legal Topic / Law"].unique()))
    selected_topic = st.selectbox("Filter by Legal Domain", all_topics)

    top_k = st.slider("Number of cases to retrieve", min_value=3, max_value=10, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick Examples")
    scenario_pick = st.selectbox("Load a sample scenario:", list(SAMPLE_QUERIES.keys()))

    st.markdown("---")
    st.markdown("### History")
    if st.session_state.search_history:
        for idx, h_item in enumerate(reversed(st.session_state.search_history[-8:])):
            short_q = h_item["query"][:28] + "..." if len(h_item["query"]) > 28 else h_item["query"]
            if st.button(f"{short_q}", key=f"hist_sb_{idx}", help=f"{h_item['timestamp']}\n\n{h_item['query']}", use_container_width=True):
                st.session_state.loaded_history_query = h_item["query"]
                st.rerun()
        if st.button("Clear History", key="clear_hist_btn", use_container_width=True):
            st.session_state.search_history = []
            st.session_state.loaded_history_query = ""
            st.rerun()
    else:
        st.caption("No past searches yet.")

# Hero Section
st.markdown('<div class="main-header">Legal AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Semantic Precedent Search across 771 Sri Lankan Criminal Law Judgments</div>', unsafe_allow_html=True)

# Top Metrics Row (Popped up Toggle Buttons)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{len(corpus_df)}</p>
            <p class="metric-label">Cases Processed</p>
        </div>
    ''', unsafe_allow_html=True)
with m2:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{corpus_df["Legal Topic / Law"].nunique()}</p>
            <p class="metric-label">Legal Domains</p>
        </div>
    ''', unsafe_allow_html=True)
with m3:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">1974-2015</p>
            <p class="metric-label">Time Period</p>
        </div>
    ''', unsafe_allow_html=True)

st.write("")
st.divider()

# Tabbed Layout
tab_search, tab_analytics = st.tabs([
    "Similar Case Laws",
    "Browse Database"
])

def generate_memo_text(res: dict) -> str:
    """Formats a case recommendation into a structured legal research memo."""
    return f"""================================================================================
SRI LANKAN LEGAL CASE LAW RESEARCH MEMO
Generated via Sri Lankan Legal AI Precedent Retrieval System
Date: {datetime.now().strftime("%B %d, %Y - %I:%M %p")}
================================================================================

1. CASE IDENTIFICATION:
   Case Title          : {res['case_name']}
   Official Citation   : {res['dataset_id']} ({res['reporter']})
   Year of Decision    : {res['year']}
   Court & Case Record : {res['court_no']}
   Legal Subject Area  : {res['legal_topic']}
   Relevance Match     : {res['similarity_pct']}

2. FACTUAL SUMMARY & LEGAL ISSUES:
{res['case_facts']}

3. JUDICIAL HOLDING & COURT'S DECISION (RATIO DECIDENDI):
{res['holding']}

================================================================================
DISCLAIMER: This memorandum is generated for academic and legal research purposes.
Official verification with the New Law Reports / Sri Lanka Law Reports is advised.
================================================================================
"""

def render_case_result(res: dict) -> None:
    """
    Renders a single case result in the Streamlit UI with a clean, properly aligned card aesthetic
    and an Export Legal Research Memo button.
    """
    with st.container(border=True):
        # Header Row: Title & Similarity Badge with proper vertical alignment
        col_title, col_badge = st.columns([3.8, 1.2])
        with col_title:
            st.markdown(f"### {res['rank']}. {res['case_name']}")
        with col_badge:
            st.markdown(
                f"<div style='text-align: right; padding-top: 4px;'>"
                f"<span style='background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; box-shadow: 0 0 10px rgba(16, 185, 129, 0.25);'>{res['similarity_pct']} Match</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Tags / Metadata row with consistent padding & alignment
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f"<div style='background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 6px; padding: 7px 10px; font-size: 0.85rem; color: #e2e8f0;'>"
            f"<b>NLR No:</b> <span style='color: #38bdf8; font-weight: 700;'>{res['dataset_id']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        col2.markdown(
            f"<div style='background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 6px; padding: 7px 10px; font-size: 0.85rem; color: #e2e8f0;'>"
            f"<b>Reporter:</b> {res['reporter']}"
            f"</div>",
            unsafe_allow_html=True
        )
        col3.markdown(
            f"<div style='background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 6px; padding: 7px 10px; font-size: 0.85rem; color: #e2e8f0;'>"
            f"<b>Domain:</b> {res['legal_topic']}"
            f"</div>",
            unsafe_allow_html=True
        )
        col4.markdown(
            f"<div style='background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 6px; padding: 7px 10px; font-size: 0.85rem; color: #e2e8f0;'>"
            f"<b>Court:</b> {res['court_no']}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.write("")
        st.markdown(f"> **Case Summary:** {res['case_facts'][:450]}...")
        
        # Clean Full-Width Expander for Case Details
        with st.expander("View Full Case Analysis & Court Decision"):
            st.markdown(f"**NLR Number / Official Citation:** `{res['dataset_id']}` ({res['reporter']})")
            st.divider()
            st.markdown(f"**Full Narrative Facts & Issues:**\n\n{res['case_facts']}")
            st.divider()
            st.markdown(f"**The Court's Decision (Holding / Ratio Decidendi):**\n\n{res['holding']}")
            
        # Clean Action Bar: Export Legal Research Memo button
        memo_content = generate_memo_text(res)
        st.download_button(
            label="Export Legal Research Memo",
            data=memo_content,
            file_name=f"Legal_Memo_{res['dataset_id']}.txt",
            mime="text/plain",
            key=f"dl_memo_{res['dataset_id']}_{res['rank']}"
        )

def hybrid_search(query: str, top_k: int = 5, topic_filter: str = None) -> list:
    """
    Combines Siamese neural semantic scores (40%) and TF-IDF lexical scores (60%).
    """
    tfidf_res = {r["dataset_id"]: r for r in tfidf_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}
    siamese_res = {r["dataset_id"]: r for r in siamese_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}

    all_ids = set(tfidf_res.keys()).union(set(siamese_res.keys()))
    combined = []
    for d_id in all_ids:
        t_score = tfidf_res.get(d_id, {}).get("similarity_score", 0.0)
        s_score = siamese_res.get(d_id, {}).get("similarity_score", 0.0)
        
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
    st.write("")
    st.subheader("Case Details")
    
    # Priority for query: Loaded from history > quick scenario pick > empty
    default_text = ""
    if st.session_state.loaded_history_query:
        default_text = st.session_state.loaded_history_query
    elif scenario_pick != "Select a scenario...":
        default_text = SAMPLE_QUERIES.get(scenario_pick, "")
    
    user_query = st.text_area(
        "Enter the facts of the case, statutory offenses, or the legal issue:",
        value=default_text,
        height=150,
        placeholder="Example: The accused was caught with narcotics at the airport, but claims they didn't know it was in their bag...",
        label_visibility="collapsed"
    )

    search_clicked = st.button("Find Similar Case Laws", use_container_width=True, type="primary")

    if (search_clicked or default_text) and user_query.strip():
        cat_filter = None if selected_topic == "All Areas of Law" else selected_topic
        results = hybrid_search(user_query, top_k=top_k, topic_filter=cat_filter)

        # Record search query and results into session state history
        clean_q = user_query.strip()
        if not st.session_state.search_history or st.session_state.search_history[-1]["query"] != clean_q:
            st.session_state.search_history.append({
                "query": clean_q,
                "timestamp": datetime.now().strftime("%b %d, %I:%M %p"),
                "results": results
            })

        st.write("")
        st.subheader("Similar Case Laws Found:")
        st.write("")

        if results:
            # Clean summary metrics header
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric("Cases Recommended", len(results))

            with summary_col2:
                st.metric("Highest Similarity", results[0]["similarity_pct"])

            st.caption(
                "Similarity scores indicate textual and semantic similarity, "
                "not legal applicability or legal advice."
            )

            st.divider()

            for res in results:
                render_case_result(res)
        else:
            st.warning(
                "No historical cases matching that description were found. "
                "Try modifying the input or clearing the legal domain filter."
            )
        
# TAB 2: Corpus Database Explorer
with tab_analytics:
    st.write("")
    st.subheader("Database Explorer")
    filter_txt = st.text_input("Type a keyword or case name to search the entire database:", label_visibility="collapsed", placeholder="Search by keyword, case name, or domain...")
    
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
